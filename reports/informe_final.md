# Informe final — Predicción de duración de viajes NYC Yellow Taxi

**Proyecto:** Machine Learning — semanas 9 a 16  
**Equipo:** Miguel Adrián Espinoza Arnero, Alair Jairo Catacora Tupa, Enzo Coronado Salcedo  
**Corte del informe:** 22 de noviembre de 2026  
**Estado de los resultados:** cifras obtenidas previamente en la rama `feature/semanas-13-14`. Este entregable añade el código para una **reproducción independiente**, pero la nueva reejecución sobre los Parquet oficiales debe realizarse en una máquina con acceso a dichos archivos. No se presenta una nueva corrida como ejecutada.

## 1. Resumen ejecutivo

Se desarrolló un modelo de regresión que estima la duración de un viaje de taxi amarillo de Nueva York al inicio del trayecto, suponiendo que el destino es conocido. Se compararon baselines y tres familias de modelos, manteniendo separación temporal entre entrenamiento (enero–mayo de 2026), validación (junio de 2026) y test final (julio de 2026). El modelo elegido mediante validación fue **HistGradientBoostingRegressor**. En 200 000 viajes de test, el MAE publicado en el avance de semanas 13–14 fue **4.202 min**, RMSE **6.400 min** y R² **0.729**, frente a MAE **8.626 min** del baseline de mediana. La mejora relativa del MAE es **51.3 %**. El modelo no usa datos conocidos únicamente después del viaje.

## 2. Problema y motivación

La duración se necesita antes de salir para estimar tiempos de llegada y apoyar decisiones operativas. Pregunta: ¿cuántos minutos durará un viaje individual de taxi amarillo con información disponible al momento de iniciarse?

## 3. Fuente, universo y unidad de análisis

NYC Taxi & Limousine Commission (TLC), archivos Yellow Taxi Parquet de enero a julio de 2026, disponibles en https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. Cada observación es un viaje. Los registros están sujetos a la cobertura y calidad del sistema de taxis amarillos; no representan todos los viajes en la ciudad.

## 4. Preparación y limpieza

Se deriva `duration_min` de la diferencia entre fecha de fin e inicio; se conservan duraciones entre **1 y 180 minutos**, viajes cuya fecha de inicio corresponde al mes y zonas de origen/destino informadas. La muestra histórica fue de **200 000 viajes por mes** con semilla 42. Para reproducir con el universo completo se emplea `--max-rows-per-month 0`. Las características incluidas son hora, día, fin de semana, codificaciones cíclicas, número de pasajeros, proveedor y características geográficas por zona de origen, destino y par origen-destino.

## 5. Prevención de fuga de datos

Quedan excluidos de los predictores `tpep_dropoff_datetime`, `trip_distance`, importes de tarifa, pagos y otros campos posteriores a la finalización. La zona destino **solo es válida** si está disponible antes de la predicción. La codificación supervisada de zonas se aprende dentro de un `Pipeline` exclusivamente en entrenamiento. Las medias de zona calculadas con la propia muestra de entrenamiento pueden introducir optimismo en el ajuste del train; para reducirlo en futuras versiones conviene evaluar codificación out-of-fold, sin usar etiquetas de validación/test.

## 6. Diseño experimental y particiones

| Conjunto | Periodo | Filas de la muestra reportada |
|:--|:--|--:|
| Entrenamiento | enero–mayo 2026 | 1 000 000 |
| Validación | junio 2026 | 200 000 |
| Test final | julio 2026 | 200 000 |

La selección de parámetros se realizó en junio. Julio se reservó para evaluación final. El script `src/reproduce.py` conserva **los mismos parámetros fijados antes del test**; una reejecución del test verifica resultados, pero no autoriza retocar el modelo por las métricas obtenidas.

## 7. Baseline

`DummyRegressor(strategy='median')`: predice siempre la mediana de la duración de entrenamiento. MAE de test **8.626 min**, RMSE **12.729 min**, R² **−0.070**. Constituye la referencia de mejora.

## 8. Familias de modelos

Se evaluaron regresión lineal/regularizada (`LinearRegression`, `Ridge`), árboles (`DecisionTree`, `RandomForest`) y gradient boosting (`HistGradientBoostingRegressor`), además del baseline. La representación espacial usa medias suavizadas de las duraciones históricas por zonas y por pares.

## 9. Búsqueda de hiperparámetros

La búsqueda reportada en semanas 13–14 examinó **13 configuraciones** sobre la validación temporal: tres de Ridge, seis de árbol y cuatro de boosting. Mejores candidatos por familia:

| Modelo | Parámetros elegidos | MAE validación (min) |
|:--|:--|--:|
| HistGradientBoostingRegressor | max_iter=150; max_leaf_nodes=31 | **4.282** |
| DecisionTreeRegressor | max_depth=16; min_samples_leaf=80 | 4.374 |
| Ridge | alpha=100 | 4.968 |
| Baseline mediana | sin ajuste | 8.951 |

El criterio fue MAE. **No se realiza una nueva búsqueda en las semanas 15–16.**

## 10. Evaluación final

| Modelo | MAE (min) | RMSE (min) | R² |
|:--|--:|--:|--:|
| HistGradientBoosting final | **4.202** | **6.400** | **0.729** |
| Baseline mediana | 8.626 | 12.729 | −0.070 |

Reducción relativa de MAE: `(8.626457 - 4.201626)/8.626457 × 100 = 51.3 %`. El RMSE mayor que el MAE indica penalización adicional de errores altos. Estas cifras son de los archivos históricos `outputs/final_test_metrics.json` y `reports/avance_semana13_14.md`.

## 11. Interpretabilidad

La importancia por permutación en validación mostró como característica dominante `pair_mean` (impacto aproximado **7.605 min** en MAE al permutar), seguida de `pu_zone_mean` (**0.343**), `hour_cos` (**0.315**), `do_zone_mean` (**0.270**) y `slot` (**0.256**). El modelo depende intensamente de la ruta histórica, sin medir la distancia futura real. Las importancias son predictivas, **no** efectos causales; además pueden distribuirse entre variables correlacionadas.

## 12. Análisis de errores

Los viajes de **40–180 min** tuvieron MAE **12.35 min** y sesgo **−8.26 min** (predicción menos duración real), mientras que los viajes **1–5 min** tuvieron sesgo **+3.71 min**. El modelo tiende a sobreestimar viajes muy cortos y subestimar viajes largos. La franja de mayor MAE reportada fue **15:00**, y el día más problemático fue jueves. Un par origen-destino difícil fue **132→265**, con MAE **21.18 min** en **459** viajes. Conviene controlar tamaño muestral e incertidumbre antes de comparar grupos minoritarios.

## 13. Ética, sesgos y limitaciones

**Geografía:** la frecuencia de trayectos varía por zona, lo que puede crear errores desiguales. **Temporalidad:** patrones de tráfico, festividades y estacionalidad cambian. **Representatividad:** se incluyen taxis amarillos, no toda la movilidad urbana. **Supuesto de destino:** la variable DOLocationID es usable solo cuando se conoce al inicio. **Filtrado:** se excluyen viajes de duración extrema, que también pueden ser relevantes. **Ausencia de tráfico en tiempo real:** causa inevitable de error. No inferir equidad individual a partir de métricas por zona, ni utilizar predicciones como promesas garantizadas.

## 14. Trabajo futuro

1. Incorporar información legalmente utilizable sobre tráfico, festivos, clima y obras, conocida en tiempo de inferencia.
2. Evaluar codificación *out-of-fold* de zonas en entrenamiento para disminuir optimismo del target encoding.
3. Repetir el entrenamiento con todas las filas, medir variación temporal y analizar intervalos de predicción.
4. Auditar errores por segmentos con umbrales mínimos de soporte e incertidumbre.
5. Empaquetar modelo con esquema de entrada, monitoreo de drift y política de recalibración.

## 15. Instrucciones de reproducción y calidad del código

Desde la raíz del repositorio con Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.reproduce --max-rows-per-month 200000
```

La primera ejecución necesita conexión a la fuente Parquet oficial; si ya existen archivos `data/yellow_tripdata_YYYY-MM.parquet`, se reutilizan. La corrida escribe `outputs/semana15_16/reproduccion_metricas.json`, tablas por segmento, `modelo_final.joblib` y `manifest.json` con hashes SHA-256, además de `reports/figures/12_reproduccion_segmentos.png`. Para usar **todas** las observaciones: `python -m src.reproduce --max-rows-per-month 0`, con mayor costo de memoria. Los resultados pueden cambiar por versiones de bibliotecas, actualizaciones de origen y número de filas.

**Control de calidad:** semillas fijas, separación temporal, cero uso de etiquetas de test para selección, pruebas sintéticas unitarias, métricas y artefactos registrados. Las pruebas unitarias no equivalen a repetir las métricas del dataset real. No deben interpretarse como validación independiente de la exactitud de TLC ni como una auditoría de producción.

### Referencias y procedencia

- NYC TLC, Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Propuesta del equipo: `proposal.md`.
- Avance y resultados históricos: `reports/avance_semana13_14.md`, `outputs/final_test_metrics.json`, `outputs/comparacion_familias_validacion.csv`, `outputs/error_por_duracion.csv`.
