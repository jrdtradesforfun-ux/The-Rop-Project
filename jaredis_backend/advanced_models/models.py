"""
Advanced ML Models for Trading
Professional-grade predictive models
"""

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class RandomForestPredictor:
    """
    Random Forest classifier for price direction prediction.
    Best for noisy financial data with many indicators.
    """

    def __init__(self, n_trees: int = 100, max_depth: int = 10):
        """Initialize Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            self.is_trained = False
        except ImportError:
            logger.error("scikit-learn not installed")
            self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model"""
        if self.model is None:
            return {"error": "scikit-learn not available"}

        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Random Forest model trained")
        return {"status": "trained", "samples": len(X)}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict price direction.
        
        Returns:
            (predictions, probabilities)
            predictions: 0=down, 1=up
            probabilities: confidence [0-1]
        """
        if not self.is_trained:
            return np.array([]), np.array([])

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X).max(axis=1)
        return predictions, probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        return dict(zip(range(len(self.model.feature_importances_)), 
                       self.model.feature_importances_))


class GradientBoostingPredictor:
    """
    XGBoost model for advanced price prediction.
    More powerful than Random Forest but requires tuning.
    """

    def __init__(self, max_depth: int = 6, learning_rate: float = 0.1):
        """Initialize XGBoost model"""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.is_trained = False
        except ImportError:
            logger.warning("XGBoost not installed, using fallback")
            self.model = None

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model"""
        if self.model is None:
            return {"error": "XGBoost not available"}

        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Gradient Boosting model trained")
        return {"status": "trained", "samples": len(X)}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with XGBoost"""
        if not self.is_trained:
            return np.array([]), np.array([])

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X).max(axis=1)
        return predictions, probabilities


class LSTMPredictor:
    """
    LSTM neural network for time-series prediction.
    Good for sequential market data patterns.
    """

    def __init__(self, sequence_length: int = 60, lstm_units: int = 50):
        """Initialize LSTM model"""
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.is_trained = False

        try:
            import tensorflow as tf
            self.tf_available = True
        except ImportError:
            logger.warning("TensorFlow not installed, LSTM unavailable")
            self.tf_available = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train LSTM model.
        X should be shape (samples, sequence_length, features)
        """
        if not self.tf_available:
            return {"error": "TensorFlow not installed"}

        try:
            from tensorflow import keras
            
            self.model = keras.Sequential([
                keras.layers.LSTM(self.lstm_units, input_shape=(self.sequence_length, X.shape[2])),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            self.is_trained = True
            logger.info("LSTM model trained")
            return {"status": "trained", "samples": len(X)}
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return {"error": str(e)}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with LSTM"""
        if not self.is_trained or self.model is None:
            return np.array([]), np.array([])

        try:
            probabilities = self.model.predict(X, verbose=0).flatten()
            predictions = (probabilities > 0.5).astype(int)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return np.array([]), np.array([])
