"""Construcción centralizada y determinista de pipelines sin fuga del objetivo."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from src.features import NUMERIC_BASE,TripFeatureBuilder

RANDOM_STATE=42
ZONE_COLS=['pu_zone_mean','do_zone_mean','pair_mean']
NUMERIC_COLS=NUMERIC_BASE+ZONE_COLS

class _DummyFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):return self
    def transform(self,X):return np.zeros((len(X),1),dtype=np.int8)

def make_baseline_median():return DummyRegressor(strategy='median')
def make_baseline_pipeline():
    return Pipeline([('dummy_x',_DummyFeatures()),('model',make_baseline_median())])
def make_linear_pipeline(model,one_hot_slot=False):
    transformers=[('num',StandardScaler(),NUMERIC_COLS)]
    if one_hot_slot:transformers.append(('slot',OneHotEncoder(handle_unknown='ignore'),['slot']))
    return Pipeline([('features',TripFeatureBuilder()),('prep',ColumnTransformer(transformers)),('model',model)])
def linear_models():
    return {'LinearRegression':make_linear_pipeline(LinearRegression()),
      'Ridge(alpha=10)':make_linear_pipeline(Ridge(alpha=10,random_state=42)),
      'Ridge + hora x finde (one-hot)':make_linear_pipeline(Ridge(alpha=10,solver='sparse_cg',random_state=42),True)}
def make_tree_pipeline(model):
    return Pipeline([('features',TripFeatureBuilder()),('model',model)])
def tree_models(seed=42):
    return {'DecisionTree':make_tree_pipeline(DecisionTreeRegressor(max_depth=12,min_samples_leaf=50,random_state=seed)),
     'RandomForest':make_tree_pipeline(RandomForestRegressor(n_estimators=60,max_depth=16,min_samples_leaf=20,n_jobs=-1,random_state=seed)),
     'HistGradientBoosting':make_tree_pipeline(HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=31,learning_rate=.1,random_state=seed))}
def all_model_families(seed=42):
    models={'Baseline mediana':make_baseline_pipeline()}
    models.update(linear_models());models.update(tree_models(seed));return models

def frozen_final_model():
    """Modelo ganador establecido antes del test de julio; sin reoptimización."""
    return make_tree_pipeline(HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=31,learning_rate=.1,random_state=42))
