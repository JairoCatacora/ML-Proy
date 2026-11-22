"""Descarga atómica y carga mensual de registros oficiales TLC."""
from pathlib import Path
import pandas as pd
import requests
BASE_URL='https://d37ci6vzurychx.cloudfront.net/trip-data'
DEFAULT_MONTHS=('2026-01','2026-02','2026-03')
MODEL_COLUMNS=['VendorID','tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','PULocationID','DOLocationID']
def yellow_taxi_url(month):return f'{BASE_URL}/yellow_tripdata_{month}.parquet'
def download_file(url,destination,chunk_size=1048576):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.stat().st_size>0:return destination
    tmp=destination.with_suffix(destination.suffix+'.part')
    with requests.get(url,stream=True,timeout=120) as response:
        response.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:f.write(chunk)
    tmp.replace(destination)
    return destination
def download_yellow_taxi_months(months=DEFAULT_MONTHS,data_dir='data'):
    return {month:download_file(yellow_taxi_url(month),Path(data_dir)/f'yellow_tripdata_{month}.parquet') for month in months}
def load_month(path,columns=None):return pd.read_parquet(path,columns=columns)
def load_clean_month(path,month=None,min_minutes=1,max_minutes=180):
    from src.features import add_duration,valid_duration_mask
    df=add_duration(load_month(path,columns=MODEL_COLUMNS))
    mask=valid_duration_mask(df,min_minutes,max_minutes)
    mask &= df['PULocationID'].notna() & df['DOLocationID'].notna()
    if month is not None:mask &= df['tpep_pickup_datetime'].dt.strftime('%Y-%m')==month
    return df.loc[mask].reset_index(drop=True)
def load_clean_months(months,data_dir='data',max_rows_per_month=None,seed=42,**kwargs):
    paths=download_yellow_taxi_months(months,data_dir)
    parts=[]
    for month in months:
        df=load_clean_month(paths[month],month=month,**kwargs)
        if max_rows_per_month is not None and len(df)>max_rows_per_month:
            df=df.sample(n=max_rows_per_month,random_state=seed)
        parts.append(df.reset_index(drop=True))
    return pd.concat(parts,ignore_index=True)
