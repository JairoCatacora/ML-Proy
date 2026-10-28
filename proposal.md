# Predicción de la duración de viajes de taxi amarillo en Nueva York

### Integrantes
- **Integrante 1:**
- **Integrante 2:**
- **Integrante 3:**

## Dataset elegido
Se utilizará la **Opción A: NYC Taxi Trip Records**, publicada por **New York City Taxi & Limousine Commission (NYC TLC)**. Para la entrega previa se trabajará con los registros de **Yellow Taxi** de los meses de enero, febrero y marzo de 2026

**Fuente oficial:**  
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

**Diccionario de datos:**  
https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

**Formato:** archivos mensuales Apache Parquet.

**Acceso:** público y sin autenticación para los archivos Parquet publicados por NYC TLC. Los archivos de datos crudos no se incluyen en Git debido a su tamaño; el notebook los descarga automáticamente desde la fuente oficial.

La TLC indica que estos registros son enviados por proveedores tecnológicos y que no garantiza completamente su exactitud. Por ello, la calidad de los datos se tratará como parte explícita del proyecto.

## Pregunta predictiva
**¿Cuál será la duración, en minutos, de un viaje de taxi amarillo al momento de iniciarse el viaje, utilizando únicamente información que razonablemente estaría disponible antes o al inicio del trayecto?**

El objetivo práctico es obtener una estimación de tiempo de viaje evitando utilizar información conocida únicamente después de que el viaje haya finalizado.

## Variable objetivo
La variable objetivo es `duration_min` y se calculará de la forma: `duration_min = tpep_dropoff_datetime - tpep_pickup_datetime`, expresada en minutos. La variable objetivo no existe directamente como una columna del archivo original; se deriva de las marcas de tiempo de inicio y fin.

## Unidad de predicción

La unidad de predicción será **un viaje individual de taxi amarillo**. Cada fila del dataset representa un viaje.

## Variables disponibles antes de la predicción
Se considerarán variables que puedan conocerse antes o al inicio del viaje:
- `tpep_pickup_datetime`: fecha y hora de inicio.
- `PULocationID`: zona de recogida.
- `DOLocationID`: zona de destino, bajo el supuesto de que el pasajero comunica el destino al inicio.
- `passenger_count`: número de pasajeros, cuando se encuentre disponible.
- `VendorID`: proveedor tecnológico que generó el registro, si se justifica su inclusión.
- Variables derivadas de la hora de inicio:
  - hora del día;
  - día de la semana;
  - fin de semana;
  - mes.

### Variable con tratamiento especial
`DOLocationID` representa el destino final registrado. Para utilizarla se define explícitamente el instante de predicción como **el inicio del viaje una vez que el destino ha sido indicado**. Si en una aplicación real el destino no estuviera disponible al inicio, esta variable debería excluirse y el problema predictivo tendría que reformularse.

## Riesgos de leakage
Se evitarán variables que contienen información generada durante o después del viaje.

### Leakage directo
- `tpep_dropoff_datetime`: forma parte de la definición del objetivo.
- `trip_distance`: corresponde a la distancia efectivamente recorrida y es reportada por el taxímetro al finalizar el trayecto.

### Variables posteriores al viaje o potencialmente contaminadas
- `fare_amount`
- `tip_amount`
- `tolls_amount`
- `total_amount`
- `payment_type`
- `extra`
- `mta_tax`
- `improvement_surcharge`
- `congestion_surcharge`
- `airport_fee`
- `cbd_congestion_fee`
- `RatecodeID`, porque el diccionario la describe como la tarifa final vigente al término del viaje.
- `store_and_fwd_flag`, porque describe el proceso de almacenamiento/transmisión del registro y no una condición necesaria para estimar la duración antes del viaje.

Estas variables podrán utilizarse únicamente para análisis exploratorio o control de calidad cuando resulte útil, pero **no como predictores del modelo principal**.

## Métrica principal y métrica secundaria

### Métrica principal: MAE
Se utilizará **MAE (Mean Absolute Error)** expresado en minutos.

Justificación:
- Es fácil de interpretar;
- Permite comunicar el error promedio en unidades del problema;
- Es menos sensible a valores extremos que RMSE.

### Métrica secundaria: RMSE
Se utilizará **RMSE (Root Mean Squared Error)**, también en minutos.

Justificación:
- Penaliza con mayor intensidad los errores grandes;
- Permite detectar si el modelo falla de forma importante en determinados viajes.

Se reportará adicionalmente `R²` como métrica diagnóstica, pero no será la única medida de desempeño.

## Plan de validación

Debido al carácter temporal de los datos, **no se realizará un split aleatorio como estrategia principal**.

### Entrega previa
- **Entrenamiento:** enero y febrero de 2026.
- **Validación:** marzo de 2026.

Esto simula una situación real en la que se entrena con información pasada y se predice un período posterior.

### Entrega final
Si se mantienen disponibles los datos publicados por TLC, se propone:
- **Entrenamiento:** enero a mayo de 2026.
- **Validación:** junio de 2026.
- **Test final:** julio de 2026.

El conjunto de test final se mantendrá sin utilizar para tomar decisiones de modelado hasta la evaluación final.

## Modelo baseline
El primer baseline es un **DummyRegressor con estrategia de mediana**.

Procedimiento:
1. Calcular la mediana de la duración utilizando únicamente los viajes de entrenamiento;
2. Predecir esa misma duración para cada viaje del período de validación;
3. Evaluar MAE, RMSE y R² en marzo de 2026.

Este baseline es deliberadamente simple y permite responder una pregunta básica:

**¿El modelo desarrollado posteriormente mejora realmente respecto de una predicción constante y honesta?**

Como siguiente baseline se evaluará una regresión lineal con variables disponibles al inicio del viaje.

## Riesgos técnicos
1. **Volumen de datos.** Cada mes contiene millones de viajes y los tres meses pueden superar ampliamente los diez millones de filas. Se procesarán archivos Parquet y se utilizarán muestras únicamente para visualizaciones pesadas, manteniendo los cálculos críticos sobre los datos completos cuando sea viable.
2. **Valores faltantes.** Algunas variables pueden presentar datos nulos y deberán cuantificarse antes de decidir imputación o exclusión.
3. **Outliers y errores de registro.** Pueden existir duraciones negativas, nulas o extremadamente altas, distancias anómalas, tarifas negativas o fechas fuera del período esperado.
4. **Leakage.** Varias columnas se generan al terminar el viaje y podrían inflar artificialmente la métrica.
5. **Cambio temporal.** Los patrones de tráfico y demanda pueden variar entre meses, estaciones y eventos.
6. **Sesgo geográfico.** La distribución de viajes puede no ser uniforme entre las distintas zonas de Nueva York.
7. **Representatividad.** El dataset representa viajes registrados por el sistema de taxis amarillos y no toda la movilidad de la ciudad.
8. **Calidad de origen.** TLC advierte que los datos son proporcionados por terceros y no garantiza completamente su exactitud.
9. **Definición del destino.** El uso de `DOLocationID` supone que el destino es conocido al momento de predecir; esta suposición deberá mantenerse explícita.
10. **Costo computacional.** Los modelos más complejos deberán diseñarse para trabajar de forma eficiente con millones de observaciones.

## Plan de trabajo para las semanas restantes (semana 9 a 16)

El plan está organizado para cubrir, en orden, los artefactos y secciones exigidos en la entrega final: pipeline reproducible en `src/`, al menos 3 familias de modelos comparadas contra los baselines, búsqueda de hiperparámetros sin tocar el test final, análisis de errores e interpretabilidad, discusión de riesgos éticos y limitaciones, y el informe final con sus 15 secciones.

| Semana | Actividad | Artefacto / sección de la entrega final que cubre |
|---|---|---|
| 9 | Ingeniería de variables (temporales, geográficas, codificación de zonas) y construcción del pipeline reproducible en `src/features.py` y `src/data.py`. | Limpieza y preparación de datos (informe, punto 4); `notebooks/02_limpieza_features.ipynb` |
| 10 | Implementar y entrenar la primera familia adicional de modelos: regresión lineal/regularizada, comparando contra el baseline de mediana con las mismas particiones y reglas de calidad. | Baselines (punto 7); Modelado y baselines (rúbrica) |
| 11 | Entrenar una segunda y tercera familia de modelos (árboles/ensambles y gradient boosting) para cumplir el mínimo de 3 familias comparables. | Modelos entrenados (punto 8); `notebooks/03_modelos.ipynb` |
| 12 | Búsqueda controlada de hiperparámetros (validación temporal, sin usar el test final para decidir) y selección del mejor modelo por familia. | Búsqueda de hiperparámetros (punto 9); Validación y métricas (rúbrica) |
| 13 | Evaluación final sobre el conjunto de test (solo una vez, sin retocar el modelo después) y análisis de errores por hora, día, zona y duración; interpretabilidad (importancia de variables, coeficientes o SHAP). | Evaluación final e Interpretación (puntos 10-11); `notebooks/04_analisis_errores.ipynb` |
| 14 | Análisis de casos problemáticos por segmento, discusión de riesgos éticos, sesgos (geográfico, temporal, de representatividad) y limitaciones identificadas a lo largo del proyecto. | Análisis de errores y Riesgos éticos, sesgos y limitaciones (puntos 12-13) |
| 15 | Refactorización final del código (`src/train.py`, `src/evaluate.py`), fijar semillas, verificar reproducibilidad desde cero, y redacción del informe final (`reports/informe_final.md`) con resumen ejecutivo, trabajo futuro e instrucciones de reproducción. | Reproducibilidad y calidad del código; Trabajo futuro e Instrucciones de reproducción (puntos 14-15) |
| 16 | Revisión completa del pipeline ejecutado desde cero, guardado de resultados en `outputs/` y `reports/figures/`, preparación y ensayo de la presentación breve. | Comunicación, informe y presentación (rúbrica); Artefactos obligatorios completos |
