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
