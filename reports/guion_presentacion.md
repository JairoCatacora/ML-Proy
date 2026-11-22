# Presentación final — Guion de 8 diapositivas (6–8 minutos)

## 1. Problema (40 s)
Predecir en minutos la duración del viaje cuando se conoce el origen, destino y horario de salida. Caso de uso: estimación previa del tiempo de llegada.

## 2. Datos y seguridad de la predicción (50 s)
NYC TLC, enero–julio de 2026; duración calculada de inicio/fin SOLO como etiqueta. Prohibidos como entradas: fin del viaje, distancia realmente recorrida, tarifa. Aclarar que el destino debe ser conocido desde el inicio.

## 3. Features y pipeline (50 s)
Variables temporales cíclicas, pasajeros, proveedor, mismo barrio, medias históricas de zona y par. Codificación geográfica ajustada exclusivamente con train.

## 4. Partición temporal (45 s)
Train enero–mayo (1 M), validación junio (200 k), test julio (200 k), muestreo reproducible semilla 42. El test no decide hiperparámetros.

## 5. Comparación y selección (60 s)
MAE validación: baseline 8.951; Ridge 4.968; árbol 4.374; boosting 4.282 min. Elegido boosting con max_iter=150 y max_leaf_nodes=31.

## 6. Evaluación final (60 s)
Test: boosting MAE 4.202, RMSE 6.400, R² 0.729; baseline MAE 8.626. Reducción MAE 51.3 %. Explicar diferencia entre MAE y RMSE y que R² no sustituye error en minutos.

## 7. Errores e interpretación (60 s)
Importancia de `pair_mean`, viajes largos subestimados (40–180 min, sesgo −8.26); cortos sobreestimados (1–5 min, sesgo +3.71). Zonas y horas de mayor error; advertir correlación no implica causalidad.

## 8. Reproducibilidad, ética y cierre (60 s)
Mostrar `python -m unittest discover -s tests -v` y `python -m src.reproduce`. Explicar que esta última reejecución debe verificarse con los siete Parquet. Limites: cobertura de taxis amarillos, destino conocido y ausencia de tráfico. Futuro: tráfico y recalibración.

### Preguntas probables

**¿Por qué no usaron trip_distance?** Porque la distancia real registrada está disponible después del viaje y usarla produce fuga de datos.

**¿Por qué no split aleatorio?** Porque se busca simular pronóstico de un periodo futuro.

**¿Qué es pair_mean?** Promedio suavizado de duración por par de zonas, aprendido solo en el conjunto de entrenamiento.

**¿Por qué los viajes largos fallan más?** Por congestión y variaciones no observadas; el estimador tiende a regresar hacia duraciones habituales.

**¿Qué hace la semana 15?** Centraliza ejecución y métricas, agrega tests y fija configuración final.

**¿Qué hace la semana 16?** Define corrida limpia, manifiesto de artefactos, resultados reproducibles y material de presentación.
