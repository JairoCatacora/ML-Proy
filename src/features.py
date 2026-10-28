import numpy as np
import pandas as pd


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega duración del viaje en minutos."""
    out = df.copy()
    out["tpep_pickup_datetime"] = pd.to_datetime(
        out["tpep_pickup_datetime"], errors="coerce"
    )
    out["tpep_dropoff_datetime"] = pd.to_datetime(
        out["tpep_dropoff_datetime"], errors="coerce"
    )
    out["duration_min"] = (
        out["tpep_dropoff_datetime"] - out["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return out


def add_pickup_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas exclusivamente de la fecha/hora de recogida."""
    out = df.copy()
    pickup = pd.to_datetime(out["tpep_pickup_datetime"], errors="coerce")
    out["pickup_hour"] = pickup.dt.hour
    out["pickup_dayofweek"] = pickup.dt.dayofweek
    out["pickup_month"] = pickup.dt.month
    out["pickup_is_weekend"] = out["pickup_dayofweek"].isin([5, 6]).astype("int8")
    return out


def valid_duration_mask(df: pd.DataFrame, min_minutes=1, max_minutes=180):
    """Máscara de duración plausible para el baseline."""
    return (
        df["duration_min"].notna()
        & np.isfinite(df["duration_min"])
        & df["duration_min"].between(min_minutes, max_minutes, inclusive="both")
    )
