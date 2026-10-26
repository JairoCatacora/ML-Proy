from pathlib import Path
import requests
import pandas as pd

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DEFAULT_MONTHS = ("2026-01", "2026-02", "2026-03")


def yellow_taxi_url(month: str) -> str:
    """Devuelve la URL oficial del Parquet de Yellow Taxi para YYYY-MM."""
    return f"{BASE_URL}/yellow_tripdata_{month}.parquet"


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Descarga un archivo de forma transaccional: .part -> archivo final."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    temp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Descargando {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    temp_path.replace(destination)
    print(f"Guardado en {destination}")
    return destination


def download_yellow_taxi_months(months=DEFAULT_MONTHS, data_dir="data"):
    """Descarga los meses indicados y devuelve un dict {mes: ruta}."""
    data_dir = Path(data_dir)
    paths = {}
    for month in months:
        destination = data_dir / f"yellow_tripdata_{month}.parquet"
        paths[month] = download_file(yellow_taxi_url(month), destination)
    return paths


def load_month(path, columns=None):
    """Carga un archivo Parquet mensual con pandas."""
    return pd.read_parquet(path, columns=columns)


# Columnas mínimas necesarias: solo información disponible al inicio del viaje
# más el instante de bajada, que se usa únicamente para derivar el objetivo.
MODEL_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
]


def load_clean_month(path, month=None, min_minutes=1, max_minutes=180):
    """Carga un mes, deriva `duration_min` y aplica las reglas de calidad.

    Reglas (las mismas del baseline de la semana 8):
    - duración entre `min_minutes` y `max_minutes`;
    - fecha de recogida dentro del mes esperado (si se indica `month`);
    - zonas de recogida y destino no nulas.
    """
    from src.features import add_duration, valid_duration_mask

    df = add_duration(load_month(path, columns=MODEL_COLUMNS))
    mask = valid_duration_mask(df, min_minutes, max_minutes)
    mask &= df["PULocationID"].notna() & df["DOLocationID"].notna()
    if month is not None:
        mask &= df["tpep_pickup_datetime"].dt.strftime("%Y-%m") == month
    return df.loc[mask].reset_index(drop=True)


def load_clean_months(months, data_dir="data", **kwargs):
    """Concatena varios meses ya limpios (descarga los que falten)."""
    paths = download_yellow_taxi_months(months, data_dir)
    parts = [load_clean_month(paths[m], month=m, **kwargs) for m in months]
    return pd.concat(parts, ignore_index=True)
