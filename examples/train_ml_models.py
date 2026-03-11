"""
Complete Example: Training & Using Advanced ML Models
Shows how to train Random Forest, XGBoost, and LSTM models
and use them with the ensemble system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from jaredis_backend.data_processing import DataLoader, FeatureEngineer
from jaredis_backend.advanced_models import (
    RandomForestPredictor,
    GradientBoostingPredictor,
    LSTMPredictor,
)
from jaredis_backend.ensemble import EnsemblePredictor, MarketRegimeDetector


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for demonstration.
    In production, you would load real historical data.
    """
    logger.info(f"Generating {n_samples} synthetic market data samples...")

    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="5min")
    close_prices = 1.0950 + np.cumsum(np.random.randn(n_samples) * 0.0005)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices + np.random.randn(n_samples) * 0.0002,
            "high": close_prices + np.abs(np.random.randn(n_samples) * 0.0003),
            "low": close_prices - np.abs(np.random.randn(n_samples) * 0.0003),
            "close": close_prices,
            "volume": np.random.randint(1000, 100000, n_samples),
        }
    )

    return data


def engineer_features(data: pd.DataFrame) -> tuple:
    """
    Create technical features for ML models.
    Returns feature matrix (X) and target labels (y).
    """
    logger.info("Engineering features from market data...")

    df = data.copy()

    # Technical indicators
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    # RSI
    df["rsi"] = FeatureEngineer.calculate_rsi(df["close"].values, period=14)

    # ATR
    df["atr"] = FeatureEngineer.calculate_atr(
        df["high"].values,
        df["low"].values,
        df["close"].values,
        period=14,
    )

    # Momentum indicators
    df["momentum"] = df["close"].pct_change()
    df["volume_ma"] = df["volume"].rolling(20).mean()

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + std * 2
    df["bb_lower"] = df["bb_middle"] - std * 2

    # Target: price goes up in next candle
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop rows with NaN values
    df = df.dropna()

    # Select feature columns (exclude target and price data)
    feature_cols = [
        "ma_5",
        "ma_20",
        "ma_50",
        "rsi",
        "atr",
        "momentum",
        "volume_ma",
        "macd",
        "macd_signal",
        "bb_middle",
        "bb_upper",
        "bb_lower",
    ]

    X = df[feature_cols].values
    y = df["target"].values

    logger.info(f"Created features: {len(feature_cols)} columns, {len(X)} samples")
    logger.info(f"Target distribution: {np.mean(y)*100:.1f}% ups, {(1-np.mean(y))*100:.1f}% downs")

    return X, y, df, feature_cols


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Split data into train and test sets"""
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> EnsemblePredictor:
    """Train all ML models and create ensemble"""
    logger.info("=" * 70)
    logger.info("TRAINING ML MODELS")
    logger.info("=" * 70)

    ensemble = EnsemblePredictor()

    # Random Forest
    logger.info("\n1. Training Random Forest...")
    try:
        rf_model = RandomForestPredictor(n_trees=100, max_depth=10)
        rf_model.train(X_train, y_train)
        ensemble.add_model("random_forest", rf_model, weight=1.0)
        logger.info("✓ Random Forest trained successfully")
    except Exception as e:
        logger.error(f"✗ Failed to train Random Forest: {e}")

    # XGBoost
    logger.info("\n2. Training XGBoost...")
    try:
        gb_model = GradientBoostingPredictor(max_depth=5, learning_rate=0.1)
        gb_model.train(X_train, y_train)
        ensemble.add_model("xgboost", gb_model, weight=1.2)
        logger.info("✓ XGBoost trained successfully")
    except Exception as e:
        logger.error(f"✗ Failed to train XGBoost: {e}")

    # LSTM (optional - requires TensorFlow)
    logger.info("\n3. Training LSTM...")
    try:
        # Reshape data for LSTM (2D -> 3D: samples, timesteps, features)
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        lstm_model = LSTMPredictor(input_shape=(1, X_train.shape[1]))
        lstm_model.train(X_train_lstm, y_train, epochs=10, batch_size=32)
        ensemble.add_model("lstm", lstm_model, weight=0.8)
        logger.info("✓ LSTM trained successfully")
    except Exception as e:
        logger.error(f"✗ LSTM not available (TensorFlow required): {e}")

    logger.info(f"\n✓ Ensemble ready with {len(ensemble.models)} models")
    return ensemble


def evaluate_models(
    ensemble: EnsemblePredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate ensemble performance on test data"""
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING MODEL PERFORMANCE")
    logger.info("=" * 70)

    predictions = []
    confidences = []

    for i in range(len(X_test)):
        result = ensemble.predict(X_test[i : i + 1])
        predictions.append(result["prediction"])
        confidences.append(result["confidence"])

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # Calculate metrics
    correct = (predictions == y_test).sum()
    accuracy = correct / len(y_test)

    # Precision (of positive predictions)
    pos_pred = predictions == 1
    pos_correct = (y_test[pos_pred] == 1).sum()
    precision = pos_correct / pos_pred.sum() if pos_pred.sum() > 0 else 0

    # Recall
    pos_true = y_test == 1
    recall = pos_correct / pos_true.sum() if pos_true.sum() > 0 else 0

    # F1 score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_confidence": np.mean(confidences),
        "high_confidence_accuracy": np.mean(y_test[confidences > 0.7] == predictions[confidences > 0.7])
        if (confidences > 0.7).sum() > 0
        else 0,
    }

    logger.info("\nModel Performance Metrics:")
    logger.info(f"  Accuracy:           {metrics['accuracy']:.2%}")
    logger.info(f"  Precision:          {metrics['precision']:.2%}")
    logger.info(f"  Recall:             {metrics['recall']:.2%}")
    logger.info(f"  F1 Score:           {metrics['f1_score']:.3f}")
    logger.info(f"  Avg Confidence:     {metrics['avg_confidence']:.2%}")
    logger.info(f"  High Conf Accuracy: {metrics['high_confidence_accuracy']:.2%}")

    return metrics


def demonstrate_predictions(
    ensemble: EnsemblePredictor,
    X_test: np.ndarray,
    feature_cols: list,
    n_samples: int = 10,
):
    """Show example predictions from the ensemble"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE PREDICTIONS")
    logger.info("=" * 70)

    for i in range(min(n_samples, len(X_test))):
        result = ensemble.predict(X_test[i : i + 1])

        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Features:")
        for j, col in enumerate(feature_cols[:5]):  # Show first 5 features
            logger.info(f"    {col}: {X_test[i, j]:.4f}")
        logger.info(f"  ...")
        logger.info(f"  Prediction:    {'UP' if result['prediction'] == 1 else 'DOWN'}")
        logger.info(f"  Confidence:    {result['confidence']:.1%}")
        logger.info(f"  Consensus:     {result['consensus']:.2f}")
        logger.info(f"  Disagreement:  {result['disagreement']:.2f}")


def demonstrate_regime_detection(data: pd.DataFrame):
    """Show market regime detection"""
    logger.info("\n" + "=" * 70)
    logger.info("MARKET REGIME DETECTION")
    logger.info("=" * 70)

    regime_detector = MarketRegimeDetector()

    # Detect regime for different price patterns
    closes = data["close"].values[-100:]
    volumes = data["volume"].values[-100:]

    regime = regime_detector.detect_regime(closes, volumes)

    logger.info(f"\nDetected Regime: {regime['regime']}")
    logger.info(f"Confidence:      {regime['confidence']:.1%}")
    logger.info(f"Strategy:        {regime.get('strategy', 'N/A')}")


def main():
    """Main execution"""
    logger.info("\n" + "=" * 70)
    logger.info("JAREDIS SMART - ML TRAINING EXAMPLE")
    logger.info("=" * 70)

    # 1. Generate synthetic market data
    data = generate_synthetic_data(n_samples=2000)

    # 2. Engineer features
    X, y, df, feature_cols = engineer_features(data)

    # 3. Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # 4. Train models
    ensemble = train_models(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_models(ensemble, X_test, y_test)

    # 6. Show predictions
    demonstrate_predictions(ensemble, X_test, feature_cols, n_samples=5)

    # 7. Regime detection
    demonstrate_regime_detection(data)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - Models Ready for Production")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Save trained models: ensemble.save()")
    logger.info("2. Deploy to production: bot.ensemble = loaded_ensemble")
    logger.info("3. Monitor performance in real trading")


if __name__ == "__main__":
    main()
