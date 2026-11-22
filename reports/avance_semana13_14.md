# Avance técnico — semanas 13 y 14

## Resumen

Se completó la evaluación final del proyecto sobre el conjunto de test (julio de 2026), el análisis de errores por hora, día, zona y duración, la interpretabilidad del modelo y la discusión de casos problemáticos, riesgos éticos, sesgos y limitaciones. Todo se ejecutó de forma reproducible en `notebooks/04_analisis_errores.ipynb`.

La familia ganadora fue **gradient boosting** (`HistGradientBoostingRegressor`, `max_iter=150`, `max_leaf_nodes=31`), seleccionada únicamente con datos de ene–may (entrenamiento) y junio (validación). El test de julio se evaluó después de congelar el modelo y no se realizaron cambios en el modelo a partir de ese resultado. El notebook se re-ejecutó una vez para corregir un error en la celda de interpretabilidad; la familia y los hiperparámetros no se modificaron en ningún momento.

---

## 10. Evaluación final

### Partición

| Conjunto | Meses | Filas (muestra reproducible) |
|---|---|---|
| Entrenamiento | ene–may 2026 | 1 000 000 |
| Validación (selección) | jun 2026 | 200 000 |
| Test final (evaluación única) | jul 2026 | 200 000 |

Se aplican las mismas reglas de calidad de las semanas anteriores: duración entre 1 y 180 minutos, fecha de recogida dentro del mes y zonas de recogida/destino no nulas. El muestreo es de 200 000 filas por mes con semilla 42; `MAX_ROWS_PER_MONTH = None` permite usar el 100 % de los registros.

### Selección en validación

| Familia | Configuración | MAE (min) | RMSE (min) | R² |
|---|---|---:|---:|---:|
| HistGradientBoosting | `max_iter=150, max_leaf_nodes=31` | **4.282** | 6.637 | 0.739 |
| DecisionTree | `max_depth=16, min_samples_leaf=80` | 4.374 | 6.789 | 0.727 |
| Ridge | `alpha=100` | 4.968 | 7.452 | 0.671 |
| Baseline mediana | — | 8.951 | 13.477 | −0.075 |

La búsqueda evaluó 13 configuraciones (3 de Ridge, 6 de árbol y 4 de boosting) con `ParameterGrid`, ajustando siempre en ene–may y puntuando en junio. Julio no intervino en ninguna decisión.

### Resultado en test (julio 2026)

| Modelo | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| **HistGradientBoosting (final)** | **4.202** | **6.400** | **0.729** |
| Baseline mediana | 8.626 | 12.729 | −0.070 |

El modelo final reduce el MAE del baseline en un **51.3 %** (de 8.626 a 4.202 minutos). El RMSE (6.400 min) es mayor que el MAE (4.202 min), coherente con la cola derecha de la distribución de duraciones: unos pocos viajes muy largos penalizan más el error cuadrático. El R² de 0.729 se reporta como diagnóstico, no como criterio de selección.

---

## 11. Interpretación

Se usaron dos vistas complementarias, sin instalar `shap` (el proposal admite "importancia de variables, coeficientes o SHAP"):

**Permutation importance** del modelo final sobre validación (caída de MAE al desordenar cada variable):

| Variable | Caída de MAE (min) |
|---|---:|
| `pair_mean` | 7.605 |
| `pu_zone_mean` | 0.343 |
| `hour_cos` | 0.315 |
| `do_zone_mean` | 0.270 |
| `slot` | 0.256 |
| `pickup_hour` | 0.209 |

**Coeficientes del modelo lineal de referencia** (Ridge, unidades de desviación estándar): `pair_mean` es el coeficiente positivo más grande (+11.85), seguido por `dow_sin` (+0.50) y `pickup_is_weekend` (+0.39); las componentes de hora (`hour_cos` −2.06, `hour_sin` −1.03) y `vendor_id` (−0.90) son negativas.

**Lectura.** El par origen–destino (`pair_mean`) concentra casi toda la señal: actúa como sustituto de `trip_distance`, que no puede utilizarse porque el taxímetro lo reporta al finalizar el viaje. Las zonas y la hora aportan el resto. Los coeficientes no son causalmente atribuibles: `pair_mean` se construye suavizando las medias de zona, por lo que está correlacionada con `pu_zone_mean` y `do_zone_mean` por diseño.

---

## 12. Análisis de errores

### Por duración real

| Rango (min) | n | MAE | RMSE | Sesgo (pred − real) |
|---|---:|---:|---:|---:|
| 1–5 | 16 155 | 3.72 | 4.47 | +3.71 |
| 5–10 | 44 685 | 2.64 | 3.57 | +2.15 |
| 10–20 | 77 020 | 3.29 | 4.36 | +0.86 |
| 20–40 | 51 434 | 5.38 | 7.08 | −1.48 |
| 40–180 | 10 706 | **12.35** | 17.44 | **−8.26** |

El patrón es claro: el error crece con la duración y el sesgo cambia de signo. El modelo **sobreestima** los viajes cortos y **subestima** los largos — una regresión a la media esperable cuando no se dispone de información de tráfico en tiempo real. El sesgo global es +0.29 min (casi nulo), pero oculta estos desvios sistemáticos por segmento.

### Por tiempo y geografía

- Peor hora: **15:00**; peor día: **jueves** (día 4).
- Peor combinación hora × día: **15:00 del jueves** (1 772 viajes, MAE 5.93 min), seguida por 14:00 del jueves (MAE 5.69 min).
- Zonas con mayor MAE: recogida **6**, destino **44**.
- Peor par origen → destino: **132 → 265** (459 viajes, MAE 21.18 min, sesgo −4.00 min). Le siguen 138 → 265 (MAE 11.52) y 164 → 132 (MAE 10.50).

Estos segmentos coinciden con viajes largos y franjas de congestión. El modelo conoce el destino y la hora de inicio, pero no el estado del tráfico, de modo que no puede anticipar los desvíos que alargan el trayecto.

---

## 13. Riesgos éticos, sesgos y limitaciones

### Riesgos éticos
- **Uso operativo.** Una estimación de duración puede alimentar despacho de flotas o promesas de tiempo al pasajero; un error sistemático en ciertas zonas podría traducirse en peor servicio para esos barrios.
- **Atributos protegidos ausentes.** El dataset no incluye ingresos, etnia ni género, pero las zonas están correlacionadas con características socioeconómicas. El modelo puede reproducir desigualdades históricas sin que esto sea explícito.
- **Suposición del destino.** Se asume que `DOLocationID` se conoce al inicio. Si no es así en una aplicación real, el modelo no es utilizable y el problema debe reformularse.
- **Transparencia.** Las codificaciones de zona aprenden promedios históricos: son correlación, no causalidad.

### Sesgos
- **Geográfico.** La demanda se concentra en pocas zonas; las zonas raras se suavizan hacia la media, lo que puede perjudicar recorridos poco frecuentes (ver el par 132 → 265).
- **Temporal.** Entrenar con ene–may y evaluar en julio expone al modelo a cambio estacional y eventos no vistos. `pickup_month` se excluyó deliberadamente porque una tendencia lineal extrapolaría mal.
- **De representatividad.** Solo se consideran taxis amarillos registrados; quedan fuera ride-hailing, transporte público y viajes no registrados.
- **De selección.** El filtro de 1–180 min descarta viajes atípicos, que son justamente los más problemáticos en la práctica.

### Limitaciones
- **Muestreo.** Se entrena con 200 000 filas por mes para controlar memoria y tiempo; los resultados pueden diferir de los obtenidos con el 100 % de los registros.
- **Sin tráfico en tiempo real.** El error en viajes largos tiene un piso estructural que ningún modelo con estas variables puede eliminar.
- **Métricas globales.** MAE y RMSE resumen el error, pero las tablas por segmento muestran que el desempeño no es homogéneo.
- **Búsqueda acotada.** La grid es pequeña y no constituye una optimización exhaustiva.
- **Calidad de origen.** TLC advierte que los datos provienen de terceros y no garantiza exactitud total.

---

## Artefactos generados

- `notebooks/04_analisis_errores.ipynb` — notebook ejecutado con salidas.
- `outputs/split_final.json`, `outputs/busqueda_hiperparametros.csv`, `outputs/hiperparametros_finales.json`.
- `outputs/comparacion_familias_validacion.csv`, `outputs/final_test_metrics.json`.
- `outputs/error_por_hora.csv`, `error_por_dia.csv`, `error_por_duracion.csv`, `error_por_zona_recogida.csv`, `error_por_zona_destino.csv`, `outputs/error_analysis.json`.
- `outputs/casos_problematicos.json`, `outputs/coeficientes_lineales.csv`, `outputs/importancia_variables.csv`.
- `reports/figures/10_errores_por_segmento.png`, `reports/figures/11_interpretabilidad.png`.

## Reproducibilidad

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 notebooks/04_analisis_errores.ipynb
```

Semilla: `RANDOM_STATE = 42`. Los archivos Parquet se descargan automáticamente desde la fuente oficial si no están en `data/`.
