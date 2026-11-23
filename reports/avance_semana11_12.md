# Avance semanas 11 y 12

Detalle completo, código y figuras: `notebooks/03_modelos.ipynb` (secciones "Semana 11" y "Semana 12").
Resultados: `outputs/modelos_sem11_12.json`, `outputs/busqueda_hiperparametros_sem12.csv`,
`reports/figures/09b_familias_modelos.png`, `reports/figures/09c_busqueda_hiperparametros.png`.

## Configuración común

- **Partición temporal**: entrenamiento enero-febrero 2026, validación marzo 2026. El test final (julio 2026) no se carga.
- **Muestra reproducible** de 200 000 viajes por mes (`MAX_ROWS_PER_MONTH`, semilla 42): 400 000 filas de entrenamiento y 200 000 de validación. Random Forest sobre los ~7 M de filas completas no es viable en un portátil. Para que la comparación sea justa, **el baseline y la familia lineal se reentrenan sobre la misma muestra**.
- **Mismas reglas de limpieza y variables** que en las semanas 9 y 10 (`src/data.py`, `TripFeatureBuilder`). La codificación de zonas por duración media se ajusta solo en entrenamiento, dentro del pipeline.
- Los pipelines están en `src/train.py` (`all_model_families`, `make_tree_pipeline`).

## Semana 11: árboles, ensambles y gradient boosting

| Modelo | Familia | MAE mar | RMSE mar | R² | Ajuste |
|---|---|---|---|---|---|
| Random Forest | ensamble (bagging) | **4.42** | 7.66 | 0.717 | 26.5 s |
| HistGradientBoosting | gradient boosting | 4.43 | **7.55** | **0.725** | 1.3 s |
| Árbol de decisión | árbol | 4.63 | 7.96 | 0.693 | 1.7 s |
| Ridge + hora×finde | lineal | 5.36 | 9.08 | 0.602 | 0.9 s |
| Baseline mediana | baseline | 9.19 | 14.79 | −0.057 | — |

- Las familias no lineales reducen el MAE ~17 % frente a la mejor lineal con los mismos datos, y ~52 % frente al baseline.
- Random Forest y HistGradientBoosting empatan en MAE. HGB gana en RMSE y R² y entrena unas 20 veces más rápido.
- HGB entrenado con la muestra (MAE 4.43) supera al mejor lineal entrenado con todos los registros en la semana 10 (MAE 4.71).
- El error se concentra en los viajes de más de 40 min (MAE ~16 min en todas las familias).

## Semana 12: búsqueda controlada de hiperparámetros

Para no elegir hiperparámetros y reportar el resultado sobre el mismo período:

| Paso | Entrena con | Se puntúa en |
|---|---|---|
| Búsqueda (17 candidatos, 4 familias) | enero | febrero |
| Confirmación de las configuraciones elegidas | enero + febrero | marzo |

| Familia | Elegida | MAE feb | MAE mar | MAE mar por defecto |
|---|---|---|---|---|
| HistGradientBoosting | `max_iter=150, max_leaf_nodes=31` | 4.73 | **4.40** | 4.43 |
| Random Forest | `n_estimators=60, max_depth=16, min_samples_leaf=20` | 4.79 | 4.42 | 4.42 |
| Árbol de decisión | `max_depth=12, min_samples_leaf=20` | 5.00 | 4.62 | 4.63 |
| Ridge | `alpha=100` | 5.68 | 5.41 | 5.41 |

- El orden de las familias elegido con febrero se mantiene en marzo, que no participó en la decisión.
- El ajuste fino aporta poco (≤ 0.03 min). La elección de familia pesa mucho más que los hiperparámetros.
- **Mejor candidato al cierre de la semana 12: HistGradientBoosting.** En la semana 13 se repite la misma rejilla sobre la partición final (ene-may → jun) y se congela el modelo antes de evaluar en julio.

## Limitaciones

- Se trabaja con una muestra. La familia lineal empeora con menos datos (MAE 4.71 → 5.36), porque depende de medias por par origen-destino que se vuelven más ruidosas.
- La búsqueda es acotada. No se exploran `learning_rate` ni la regularización de HGB.
- Se mantiene el supuesto de la propuesta: el destino (`DOLocationID`) se conoce al inicio del viaje.
- La codificación de zonas incluye el propio viaje en su media dentro de entrenamiento (optimismo leve en pares raros, mitigado por el suavizado). Una codificación *out-of-fold* queda como trabajo futuro.

## Reproducción

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/03_modelos.ipynb
```

Descarga enero-marzo 2026 si no están en `data/`. Requiere unos 16 GB de RAM, porque la sección de la semana 10 usa todos los registros. Probado con Python 3.15, pandas 3.1.1, scikit-learn 1.9.1, numpy 2.5.5 y pyarrow 26.0.1.
