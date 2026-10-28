# Avance técnico hasta semana 12

## Semana 9
Pipeline de limpieza y variables temporales, geográficas y pasajero. Variables posteriores al viaje excluidas para evitar fuga de información. Separación temporal: enero-febrero para entrenamiento y marzo para validación.

## Semana 10
Baseline de mediana, regresión lineal y Ridge. Comparación con MAE, RMSE y R².

## Semana 11
Árbol de decisión, Random Forest y Gradient Boosting. Se incluye una comparación homogénea de las familias.

## Semana 12
Búsqueda controlada con entrenamiento enero, selección febrero y evaluación de configuraciones seleccionadas en marzo; sin acceso al test final.

## Estado de resultados
**Pendiente de ejecución con los datos oficiales:** el ZIP inicial no contiene los archivos Parquet. Los notebooks descargan los meses necesarios al ejecutarse con internet. No se han inventado métricas ni se afirma haber entrenado modelos.

## Reproducibilidad
Desde la raíz del repositorio: `pip install -r requirements.txt`; ejecutar `notebooks/02_limpieza_features.ipynb` y luego `notebooks/03_modelos.ipynb`. El parámetro `MAX_ROWS_PER_MONTH` limita el uso de RAM. Los archivos de resultados se generan en `outputs/`.

## Limitaciones
La muestra por mes reduce el costo computacional; los resultados podrían diferir de los obtenidos con los datos completos. El destino se supone conocido al inicio del viaje. La búsqueda de hiperparámetros es acotada y no constituye una optimización exhaustiva.
