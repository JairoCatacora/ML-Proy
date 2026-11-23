"""Pruebas rápidas sintéticas: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
import pandas as pd
from src.evaluate import regression_metrics, segment_metrics
from src.train import make_tree_pipeline, make_baseline_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

class PipelineTests(unittest.TestCase):
    def setUp(self):
        n=96
        self.X=pd.DataFrame({
            'VendorID': [1,2]*48,
            'tpep_pickup_datetime':pd.date_range('2026-01-01',periods=n,freq='h'),
            'passenger_count':[1,np.nan,2,3]*24,
            'PULocationID':[100,101,102,103]*24,
            'DOLocationID':[101,102,103,100]*24,
        })
        self.y=pd.Series(np.linspace(5,40,n))
    def test_metrics(self):
        m=regression_metrics([1,3],[2,4])
        self.assertAlmostEqual(m['MAE_minutes'],1)
        self.assertAlmostEqual(m['RMSE_minutes'],1)
    def test_invalid(self):
        with self.assertRaises(ValueError): regression_metrics([1,2],[1])
    def test_segments(self):
        g=pd.Series(['a','a','b'],name='grupo')
        z=segment_metrics(pd.DataFrame(index=range(3)),[2,4,6],[3,3,6],g)
        self.assertEqual(z['n'].sum(),3)
        self.assertAlmostEqual(z.query('grupo == "a"')['MAE_minutes'].iloc[0],1)
    def test_model_determinism(self):
        m1=make_tree_pipeline(HistGradientBoostingRegressor(max_iter=10,max_leaf_nodes=5,random_state=42))
        m2=make_tree_pipeline(HistGradientBoostingRegressor(max_iter=10,max_leaf_nodes=5,random_state=42))
        a=m1.fit(self.X,self.y).predict(self.X)
        b=m2.fit(self.X,self.y).predict(self.X)
        np.testing.assert_allclose(a,b)
    def test_baseline(self):
        p=make_baseline_pipeline().fit(self.X,self.y)
        np.testing.assert_allclose(p.predict(self.X),np.median(self.y))

if __name__=='__main__': unittest.main()
