"""Métricas y análisis de sesgo por segmentos, sin usar predictores post-viaje."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """MAE y RMSE en minutos; R² diagnóstico. Rechaza entradas inválidas."""
    yt, yp = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    if len(yt) != len(yp) or not len(yt) or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError('y_true/y_pred deben tener misma longitud no nula y valores finitos')
    return {'MAE_minutes':float(mean_absolute_error(yt,yp)),
            'RMSE_minutes':float(np.sqrt(mean_squared_error(yt,yp))),
            'R2':float(r2_score(yt,yp)) if len(yt)>1 else None}


def segment_metrics(frame, y_true, y_pred, group_values):
    """Métricas no causales por segmento; sesgo = predicción menos observado."""
    yt, yp = np.asarray(y_true,dtype=float), np.asarray(y_pred,dtype=float)
    if not (len(frame)==len(yt)==len(yp)==len(group_values)):
        raise ValueError('Longitudes inconsistentes')
    name = group_values.name or 'segmento'
    values = pd.DataFrame({name:np.asarray(group_values), 'error':yp-yt})
    values['error_abs']=values['error'].abs()
    values['error_sq']=values['error']**2
    out = values.groupby(name, observed=True, dropna=False).agg(
        n=('error','size'),MAE_minutes=('error_abs','mean'),
        MSE=('error_sq','mean'),bias_pred_minus_real=('error','mean')).reset_index()
    out['RMSE_minutes']=np.sqrt(out.pop('MSE'))
    return out
