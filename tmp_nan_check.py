import numpy as np
import pandas as pd
from jaredis_backend.ml_models.feature_engineer import FeatureEngineer, FeatureConfig

np.random.seed(0)
dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
close = 1.0750 + np.random.randn(500).cumsum() * 0.0001
open_ = close + np.random.randn(500) * 0.00005
high = close + np.abs(np.random.randn(500) * 0.0001)
low = close - np.abs(np.random.randn(500) * 0.0001)

df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': np.random.randint(1000, 10000, 500)}, index=dates)
engineer = FeatureEngineer(FeatureConfig())
features_df = engineer.engineer_features(df)

nan_cols = [c for c in features_df.columns if features_df[c].isna().any()]
print('columns with NaN count:', len(nan_cols))
print('top 20 columns by NaN count:')
print(features_df.isna().sum().sort_values(ascending=False).head(20))
