"""Pipeline reproducible de semanas 9 a 12; solo variables al inicio."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUMERIC = ["pickup_hour", "pickup_dayofweek", "pickup_month", "pickup_is_weekend", "passenger_count"]
CATEGORICAL = ["PULocationID", "DOLocationID"]
FEATURES = NUMERIC + CATEGORICAL

def prepare(df):
    from src.features import add_duration, add_pickup_time_features, valid_duration_mask
    df = add_pickup_time_features(add_duration(df))
    df = df.loc[valid_duration_mask(df)].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Sin destino o origen no se puede representar el trayecto solicitado
    df = df.dropna(subset=["PULocationID", "DOLocationID", "tpep_pickup_datetime"])
    for col in CATEGORICAL:
        df[col] = df[col].astype("int64").astype(str)
    return df

def preprocessing():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])

def models(seed=42):
    return {
        "baseline_mediana": DummyRegressor(strategy="median"),
        "regresion_lineal": Pipeline([("prep", preprocessing()), ("model", LinearRegression())]),
        "ridge": Pipeline([("prep", preprocessing()), ("model", Ridge(alpha=10.0))]),
        "arbol": Pipeline([("prep", preprocessing()), ("model", DecisionTreeRegressor(max_depth=12, min_samples_leaf=30, random_state=seed))]),
        "random_forest": Pipeline([("prep", preprocessing()), ("model", RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_leaf=20, n_jobs=-1, random_state=seed))]),
        "gradient_boosting": Pipeline([("prep", preprocessing()), ("model", HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=31, random_state=seed))]),
    }

def metrics(y, pred):
    return {"MAE_min": float(mean_absolute_error(y,pred)), "RMSE_min": float(np.sqrt(mean_squared_error(y,pred))), "R2": float(r2_score(y,pred))}

def sample_month(df, max_rows=100000, seed=42):
    return df.sample(n=min(len(df),max_rows),random_state=seed) if max_rows else df
