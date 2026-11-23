"""Features disponibles al inicio y codificación de zonas ajustada solo en train."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_BASE = ['pickup_hour','pickup_dayofweek','pickup_is_weekend','passenger_count',
                'passenger_missing','vendor_id','same_zone','hour_sin','hour_cos','dow_sin','dow_cos']

def add_duration(df):
    out=df.copy()
    for c in ('tpep_pickup_datetime','tpep_dropoff_datetime'):
        out[c]=pd.to_datetime(out[c],errors='coerce')
    out['duration_min']=(out['tpep_dropoff_datetime']-out['tpep_pickup_datetime']).dt.total_seconds()/60
    return out

def valid_duration_mask(df,min_minutes=1,max_minutes=180):
    return df['duration_min'].notna() & np.isfinite(df['duration_min']) & df['duration_min'].between(min_minutes,max_minutes)

def add_pickup_time_features(df):
    out=df.copy()
    dt=pd.to_datetime(out['tpep_pickup_datetime'],errors='coerce')
    out['pickup_hour']=dt.dt.hour
    out['pickup_dayofweek']=dt.dt.dayofweek
    out['pickup_month']=dt.dt.month
    out['pickup_is_weekend']=out['pickup_dayofweek'].isin([5,6]).astype('int8')
    return out

def add_basic_features(df):
    out=add_pickup_time_features(df)
    hour,dow=out['pickup_hour'],out['pickup_dayofweek']
    out['hour_sin']=np.sin(2*np.pi*hour/24)
    out['hour_cos']=np.cos(2*np.pi*hour/24)
    out['dow_sin']=np.sin(2*np.pi*dow/7)
    out['dow_cos']=np.cos(2*np.pi*dow/7)
    out['passenger_missing']=out['passenger_count'].isna().astype('int8')
    out['passenger_count']=out['passenger_count'].fillna(1).clip(1,6)
    out['vendor_id']=out['VendorID'].fillna(-1).astype('int16')
    out['same_zone']=(out['PULocationID']==out['DOLocationID']).astype('int8')
    return out

class ZoneTargetEncoder:
    def __init__(self,smoothing=20.0):self.smoothing=smoothing
    def fit(self,df,y):
        y=pd.Series(np.asarray(y,dtype='float64'),index=df.index)
        k=self.smoothing
        self.global_mean_=float(y.mean())
        def smooth(stats,prior):return (stats['sum']+k*prior)/(stats['count']+k)
        def agg(keys):return y.groupby([df[c] for c in keys]).agg(['sum','count'])
        pu,do=agg(['PULocationID']),agg(['DOLocationID'])
        self.pu_=smooth(pu,self.global_mean_)
        self.do_=smooth(do,self.global_mean_)
        pair=agg(['PULocationID','DOLocationID'])
        prior=(self.pu_.reindex(pair.index.get_level_values(0)).to_numpy()+
               self.do_.reindex(pair.index.get_level_values(1)).to_numpy())/2
        self.pair_=smooth(pair,prior)
        return self
    def transform(self,df):
        pu=df['PULocationID'].map(self.pu_).fillna(self.global_mean_)
        do=df['DOLocationID'].map(self.do_).fillna(self.global_mean_)
        idx=pd.MultiIndex.from_arrays([df['PULocationID'],df['DOLocationID']])
        pair=pd.Series(self.pair_.reindex(idx).to_numpy(),index=df.index)
        pair=pair.fillna((pu+do)/2)
        return pd.DataFrame({'pu_zone_mean':pu,'do_zone_mean':do,'pair_mean':pair},index=df.index).astype('float32')

def build_feature_matrix(df,encoder):
    base=add_basic_features(df)[NUMERIC_BASE].astype('float32')
    return pd.concat([base,encoder.transform(df)],axis=1)

class TripFeatureBuilder(BaseEstimator,TransformerMixin):
    def __init__(self,smoothing=20.0):self.smoothing=smoothing
    def fit(self,X,y):
        self.encoder_=ZoneTargetEncoder(self.smoothing).fit(X,y)
        return self
    def transform(self,X):
        out=build_feature_matrix(X,self.encoder_)
        out['slot']=(out['pickup_is_weekend']*24+out['pickup_hour']).astype('int16')
        return out
