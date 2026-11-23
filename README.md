# Proyecto Final - Machine Learning

## Fuente de datos

NYC Taxi & Limousine Commission (TLC):

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Diccionario de Yellow Taxi:

https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

## Estructura del repositorio

```text
proyecto-final-nyc-taxi/
├── README.md
├── proposal.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_limpieza_features.ipynb
│   ├── 03_modelos.ipynb
│   └── 04_analisis_errores.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
└── reports/
    ├── avance_semana13_14.md
    └── figures/
```

## Requisitos

- Python 3.10 o superior
- Acceso a internet para la descarga de los Parquet oficiales
- Aproximadamente 1 GB de espacio libre para trabajar con comodidad
- Memoria suficiente para procesar un archivo mensual a la vez

## Instalación

### 1. Crear un entorno virtual

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Abrir Jupyter

```bash
jupyter notebook
```

Abrir:

```text
notebooks/01_exploracion_inicial.ipynb
```

### 4. Ejecutar el notebook completo

En Jupyter usar:

**Kernel -> Restart & Run All**

La primera ejecución descargará automáticamente:

```text
data/yellow_tripdata_2026-01.parquet
data/yellow_tripdata_2026-02.parquet
data/yellow_tripdata_2026-03.parquet
```

Después de la descarga, las siguientes ejecuciones reutilizarán los archivos locales.

## Qué reproduce el notebook

El notebook genera automáticamente:

1. carga de los tres meses de datos;
2. número de filas, columnas y tipos de datos;
3. porcentaje de valores faltantes por variable;
4. construcción y distribución de la variable objetivo `duration_min`;
5. seis visualizaciones exploratorias;
6. identificación y cuantificación de registros sospechosos;
7. discusión de sesgos, leakage y limitaciones;
8. split temporal enero-febrero vs. marzo;
9. baseline con `DummyRegressor(strategy="median")`;
10. MAE, RMSE y R² del baseline;
11. interpretación escrita debajo de cada resultado y cada visualización;
12. archivos de resumen en `outputs/` (`eda_summary.json`, `baseline_metrics.json`).

## Limpieza, features y modelos (semanas 9 a 12)

Ejecutar en orden, desde la raíz del repositorio:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/02_limpieza_features.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/03_modelos.ipynb
```

- `02_limpieza_features.ipynb` (semana 9): reglas de limpieza, faltantes, variables temporales y codificación de zonas (`src/data.py`, `src/features.py`).
- `03_modelos.ipynb`:
  - semana 10: familia lineal con todos los registros → `outputs/linear_metrics.json`;
  - semana 11: árbol, Random Forest y HistGradientBoosting sobre una muestra de 200 000 viajes/mes → `outputs/modelos_sem11_12.json`;
  - semana 12: búsqueda de hiperparámetros (ene → feb) y confirmación en marzo → `outputs/busqueda_hiperparametros_sem12.csv`.
- Resumen de resultados: `reports/avance_semana11_12.md`.

Julio 2026 (test final) no se usa en estos notebooks. La sección de la semana 10 carga ~11 M de filas: se recomiendan unos 16 GB de RAM. Versiones probadas: Python 3.15, pandas 3.1.1, scikit-learn 1.9.1, numpy 2.5.5, pyarrow 26.0.1.

## Reproducibilidad

Se utiliza:

```python
RANDOM_STATE = 42
```

Las visualizaciones utilizan una muestra reproducible para evitar que la representación gráfica de millones de registros sea innecesariamente lenta.

El baseline se evalúa con datos de un período posterior al entrenamiento.

## Evaluación final (semanas 13 y 14)

`notebooks/04_analisis_errores.ipynb` ejecuta la parte final del proyecto: búsqueda de hiperparámetros con validación en junio, evaluación del modelo congelado sobre el test de julio, análisis de errores por hora/día/zona/duración, interpretabilidad y discusión de riesgos éticos y limitaciones. El detalle está en `reports/avance_semana13_14.md`.

Para reproducirlo desde cero:

```bash
python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 notebooks/04_analisis_errores.ipynb
```

Por defecto se usa una muestra reproducible de 200 000 filas por mes. Para entrenar con todos los registros, cambiar `MAX_ROWS_PER_MONTH = None` en la primera celda (requiere más memoria y tiempo).