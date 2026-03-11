"""
Example: Training an ML model with historical data
"""

import numpy as np
from pathlib import Path

from jaredis_backend.ml_models import ModelManager, PricePredictor
from jaredis_backend.data_processing import DataLoader, FeatureEngineer, Preprocessor


def example_train_model():
    """Example of training and saving a model"""
    
    # Initialize components
    model_manager = ModelManager(models_dir="models")
    data_loader = DataLoader(data_dir="data")
    
    # Create synthetic training data (in practice, load real historical data)
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000)) + 100  # Random walk
    
    # Create features
    features = FeatureEngineer.create_feature_set(
        opens=prices,
        highs=prices + np.abs(np.random.randn(1000)),
        lows=prices - np.abs(np.random.randn(1000)),
        closes=prices,
        volumes=np.random.rand(1000) * 10000
    )
    
    print(f"Created {len(features)} features")
    print(f"Feature keys: {list(features.keys())}")
    
    # Initialize model
    predictor = PricePredictor(lookback_period=60, forecast_period=5)
    
    # Train (simplified - just initialize in this example)
    training_result = predictor.train(prices[:800])
    print(f"Training result: {training_result}")
    
    # Register and save model
    model_manager.register_model(
        "price_predictor_v1",
        predictor,
        metadata={
            "type": "LSTM",
            "lookback": 60,
            "forecast": 5,
            "training_samples": 800,
            "status": "trained"
        }
    )
    
    saved_path = model_manager.save_model("price_predictor_v1")
    print(f"Model saved to: {saved_path}")
    
    # Load model and verify
    loaded_model = model_manager.load_model("price_predictor_v1", saved_path)
    print(f"Model loaded successfully: {loaded_model}")


def example_predict():
    """Example of making predictions with trained model"""
    
    model_manager = ModelManager(models_dir="models")
    
    # Assume model exists (from training example)
    try:
        predictor = model_manager.get_model("price_predictor_v1")
        
        if predictor:
            # Get recent data
            recent_prices = np.array([100, 101, 102, 103, 104])  # Example prices
            
            # Make prediction
            predictions, confidence = predictor.predict(recent_prices)
            
            print(f"Recent prices: {recent_prices}")
            print(f"Predicted prices (next 5 candles): {predictions}")
            print(f"Confidence intervals: {confidence}")
    except FileNotFoundError:
        print("Model not found - run training example first")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Training and Saving Models")
    print("=" * 60)
    example_train_model()
    
    print("\n" + "=" * 60)
    print("Example 2: Making Predictions")
    print("=" * 60)
    example_predict()
