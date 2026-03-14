import numpy as np
import pandas as pd
from jaredis_backend.ml_models.feature_engineer import FeatureEngineer, FeatureConfig

np.random.seed(0)
dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
close = 1.0750 + np.random.randn(500).cumsum() * 0.0001
open_ = close + np.random.randn(500) * 0.00005
high = close + np.abs(np.random.randn(500) * 0.0001)
low = close - np.abs(np.random.randn(500) * 0.0001)

df = pd.DataFrame({
    'open': open_,
    'high': high,
    'low': low,
    'close': close,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

engineer = FeatureEngineer(FeatureConfig())
features_df = engineer.engineer_features(df)

print('features_df shape', features_df.shape)
print('nan count', features_df.isna().sum().sum())
print('rows with any nan', features_df.isna().any(axis=1).sum())
