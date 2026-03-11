"""
Example: Data loading and feature engineering
"""

import numpy as np
from jaredis_backend.data_processing import DataLoader, FeatureEngineer, Preprocessor


def example_load_data():
    """Example of loading market data"""
    
    loader = DataLoader(data_dir="data")
    
    # Load sample CSV data
    candles = loader.load_csv("data/sample_eurusd.csv", symbol="EURUSD")
    
    if candles:
        print(f"Loaded {len(candles)} candles")
        print(f"First candle: {candles[0]}")
        print(f"Last candle: {candles[-1]}")
        
        # Validate data
        is_valid = Preprocessor.validate_data(candles)
        print(f"Data validation: {'PASS' if is_valid else 'FAIL'}")
    else:
        print("No data loaded")


def example_feature_engineering():
    """Example of creating ML features"""
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 100
    base_price = 100
    
    closes = np.cumsum(np.random.randn(n)) + base_price
    opens = closes + np.random.randn(n) * 0.5
    highs = np.maximum(closes, opens) + np.abs(np.random.randn(n) * 0.5)
    lows = np.minimum(closes, opens) - np.abs(np.random.randn(n) * 0.5)
    volumes = np.random.rand(n) * 10000
    
    print("Created synthetic OHLCV data:")
    print(f"  Closes: {closes[:5]}")
    print(f"  Volumes: {volumes[:5]}")
    
    # Create features
    features = FeatureEngineer.create_feature_set(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes
    )
    
    print(f"\nCreated {len(features)} features:")
    for key, value in features.items():
        print(f"  {key}: shape {value.shape if hasattr(value, 'shape') else 'scalar'}")
    
    # Show sample values
    print(f"\nRSI values (last 10): {features['rsi'][-10:]}")
    print(f"ATR values (last 10): {features['atr'][-10:]}")


def example_preprocessing():
    """Example of data preprocessing"""
    
    # Create sample data with missing values
    data = np.array([100, 102, np.nan, 104, 105])
    
    print(f"Original data: {data}")
    
    # Handle missing values
    filled = Preprocessor.handle_missing_values(data, method="forward_fill")
    print(f"After forward fill: {filled}")
    
    # Normalize
    normalized, params = Preprocessor.normalize(filled, method="minmax")
    print(f"Normalized: {normalized}")
    print(f"Normalization params: {params}")
    
    # Denormalize
    denormalized = Preprocessor.denormalize(normalized, params)
    print(f"Denormalized: {denormalized}")
    
    # Create sequences for LSTM
    data = np.arange(10)
    X, y = Preprocessor.create_sequences(data, seq_length=3)
    print(f"\nSequences (input/output pairs):")
    for i in range(3):
        print(f"  Input: {X[i]} -> Output: {y[i]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Loading Data")
    print("=" * 60)
    example_load_data()
    
    print("\n" + "=" * 60)
    print("Example 2: Feature Engineering")
    print("=" * 60)
    example_feature_engineering()
    
    print("\n" + "=" * 60)
    print("Example 3: Data Preprocessing")
    print("=" * 60)
    example_preprocessing()
