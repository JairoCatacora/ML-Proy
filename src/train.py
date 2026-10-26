"""Pipelines de modelos (semana 10: familia lineal)."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import NUMERIC_BASE, TripFeatureBuilder

RANDOM_STATE = 42
ZONE_COLS = ["pu_zone_mean", "do_zone_mean", "pair_mean"]
NUMERIC_COLS = NUMERIC_BASE + ZONE_COLS


def make_baseline_median():
    return DummyRegressor(strategy="median")


def make_linear_pipeline(model, one_hot_slot: bool = False):
    """TripFeatureBuilder -> escalado (+ one-hot hora x finde) -> modelo."""
    transformers = [("num", StandardScaler(), NUMERIC_COLS)]
    if one_hot_slot:
        transformers.append(("slot", OneHotEncoder(handle_unknown="ignore"), ["slot"]))
    return Pipeline([
        ("features", TripFeatureBuilder()),
        ("prep", ColumnTransformer(transformers)),
        ("model", model),
    ])


def linear_models():
    """Familia lineal comparada en la semana 10."""
    return {
        "LinearRegression": make_linear_pipeline(LinearRegression()),
        "Ridge(alpha=10)": make_linear_pipeline(Ridge(alpha=10, random_state=RANDOM_STATE)),
        "Ridge + hora x finde (one-hot)": make_linear_pipeline(
            Ridge(alpha=10, solver="sparse_cg", random_state=RANDOM_STATE), one_hot_slot=True
        ),
    }
