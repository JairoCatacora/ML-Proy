"""Reproduce el modelo congelado de las semanas 13-14 sin volver a optimizarlo.
Ejecutar desde la raiz: python -m src.reproduce --max-rows-per-month 200000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import load_clean_months
from src.evaluate import regression_metrics, segment_metrics
from src.train import RANDOM_STATE, make_baseline_pipeline, make_tree_pipeline

TRAIN_MONTHS = ('2026-01','2026-02','2026-03','2026-04','2026-05')
VAL_MONTHS = ('2026-06',)
TEST_MONTHS = ('2026-07',)
MODEL_PARAMS = {'max_iter': 150, 'max_leaf_nodes': 31, 'learning_rate': 0.1, 'random_state': RANDOM_STATE}
INPUT_COLS = ['VendorID','tpep_pickup_datetime','passenger_count','PULocationID','DOLocationID']


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def hashes(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def run(data_dir='data', output_dir='outputs/semana15_16', figures_dir='reports/figures', max_rows_per_month=200000, seed=RANDOM_STATE):
    """Ejecuta un nuevo cálculo de reproducibilidad; no altera el test histórico de semana 13.
    Se mantiene el conjunto de test separado de cualquier selección de hiperparámetros.
    """
    if seed != RANDOM_STATE:
        raise ValueError('La configuración congelada utiliza exclusivamente seed=42')
    output_dir, figures_dir = Path(output_dir), Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, months in [('train',TRAIN_MONTHS),('validation',VAL_MONTHS),('test',TEST_MONTHS)]:
        print(f'Cargando {name}: {months}', flush=True)
        splits[name] = load_clean_months(months, data_dir=data_dir,
                           max_rows_per_month=max_rows_per_month, seed=seed)
        if splits[name].empty:
            raise ValueError(f'Conjunto {name} vacío')
    train, val, test = (splits[k] for k in ('train','validation','test'))
    X_train, y_train = train[INPUT_COLS], train['duration_min']
    X_val, y_val = val[INPUT_COLS], val['duration_min']
    X_test, y_test = test[INPUT_COLS], test['duration_min']

    # Parámetros FROZEN antes del test. Ajuste SOLO sobre entrenamiento ene-may.
    model = make_tree_pipeline(HistGradientBoostingRegressor(**MODEL_PARAMS))
    baseline = make_baseline_pipeline()
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    base_test = baseline.predict(X_test)
    val_metrics = regression_metrics(y_val, pred_val)
    test_metrics = regression_metrics(y_test, pred_test)
    baseline_metrics = regression_metrics(y_test, base_test)
    improvement = 100*(1-test_metrics['MAE_minutes']/baseline_metrics['MAE_minutes'])

    hour = segment_metrics(test, y_test, pred_test,
                           test['tpep_pickup_datetime'].dt.hour.rename('pickup_hour'))
    day = segment_metrics(test, y_test, pred_test,
                          test['tpep_pickup_datetime'].dt.dayofweek.rename('pickup_dayofweek'))
    duration_bins = pd.cut(y_test, bins=[1,5,10,20,40,180],
             labels=['1-5','5-10','10-20','20-40','40-180'],include_lowest=True)
    duration = segment_metrics(test, y_test, pred_test, duration_bins.rename('duration_bucket'))
    pu = segment_metrics(test, y_test, pred_test, test['PULocationID'].rename('PULocationID'))
    do = segment_metrics(test, y_test, pred_test, test['DOLocationID'].rename('DOLocationID'))
    for key, frame in {'error_por_hora':hour,'error_por_dia':day,'error_por_duracion':duration,
                       'error_por_zona_recogida':pu,'error_por_zona_destino':do}.items():
        frame.to_csv(output_dir/f'{key}.csv', index=False)

    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(hour['pickup_hour'], hour['MAE_minutes'], marker='o')
    axes[0].set(xlabel='Hora del viaje',ylabel='MAE (min)',title='Error por hora (test julio)')
    axes[0].grid(alpha=.25)
    axes[1].bar(duration['duration_bucket'].astype(str),duration['MAE_minutes'])
    axes[1].set(xlabel='Duración real (min)',ylabel='MAE (min)',title='Error por duración (test julio)')
    fig.tight_layout()
    fig.savefig(figures_dir/'12_reproduccion_segmentos.png',dpi=160)
    plt.close(fig)

    summary = {
      'estado': 'reejecucion independiente del pipeline congelado; no nueva seleccion de modelo',
      'modelo': 'HistGradientBoostingRegressor', 'parametros': MODEL_PARAMS,
      'semilla': seed, 'muestra_por_mes': max_rows_per_month,
      'split': {'train':list(TRAIN_MONTHS),'validation':list(VAL_MONTHS),'test':list(TEST_MONTHS)},
      'filas':{k:len(v) for k,v in splits.items()},
      'validation_metrics':val_metrics,'test_metrics':test_metrics,
      'baseline_test_metrics':baseline_metrics,'reduccion_mae_baseline_pct':float(improvement),
      'software':{'python':platform.python_version(),'numpy':np.__version__,
                  'pandas':pd.__version__,'sklearn':sklearn.__version__},
      'fecha_utc':datetime.now(timezone.utc).isoformat(),
    }
    model_file = output_dir/'modelo_final.joblib'
    joblib.dump(model, model_file, compress=3)
    summary['modelo_sha256'] = hashes(model_file)
    save_json(output_dir/'reproduccion_metricas.json', summary)
    manifest = {'tipo':'manifiesto de artefactos','files':{}}
    for p in sorted(list(output_dir.glob('*')) + [figures_dir/'12_reproduccion_segmentos.png']):
        if p.is_file() and p.name != 'manifest.json':
            manifest['files'][str(p)] = {'bytes':p.stat().st_size,'sha256':hashes(p)}
    save_json(output_dir/'manifest.json',manifest)
    print(json.dumps({k:summary[k] for k in ('filas','test_metrics','baseline_test_metrics','reduccion_mae_baseline_pct')},indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--output-dir',default='outputs/semana15_16')
    parser.add_argument('--figures-dir',default='reports/figures')
    parser.add_argument('--max-rows-per-month',type=int,default=200000,
                        help='0 para usar todos los registros')
    parser.add_argument('--seed',type=int,default=42)
    a = parser.parse_args()
    if a.max_rows_per_month < 0: parser.error('max-rows-per-month debe ser >=0')
    run(data_dir=a.data_dir,output_dir=a.output_dir,figures_dir=a.figures_dir,
        max_rows_per_month=a.max_rows_per_month or None,seed=a.seed)

if __name__ == '__main__':
    main()
