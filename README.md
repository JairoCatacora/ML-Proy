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
│   └── 03_modelos.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
└── reports/
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

## Reproducibilidad

Se utiliza:

```python
RANDOM_STATE = 42
```

Las visualizaciones utilizan una muestra reproducible para evitar que la representación gráfica de millones de registros sea innecesariamente lenta.

El baseline se evalúa con datos de un período posterior al entrenamiento.