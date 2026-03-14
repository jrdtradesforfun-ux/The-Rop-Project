"""
Ensemble Prediction System
Combines multiple models for robust predictions
"""

import logging
from typing import Dict, Tuple, List, Any
import numpy as np

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Combines predictions from multiple models.
    Uses voting and weighted averaging for robustness.
    """

    def __init__(self):
        """Initialize ensemble"""
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.total_weight = 0.0

    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0
    ) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Model object with predict method
            weight: Weight in ensemble (higher = more influence)
        """
        self.models[name] = model
        self.model_weights[name] = weight
        self.total_weight = sum(self.model_weights.values())
        logger.info(f"Added model '{name}' to ensemble with weight {weight}")

    def replace_model(
        self,
        name: str,
        new_model: Any,
        weight: Optional[float] = None
    ) -> bool:
        """
        Replace an existing model in the ensemble.
        
        Args:
            name: Model identifier to replace
            new_model: New model object
            weight: New weight (if None, keeps existing weight)
            
        Returns:
            True if replaced, False if model not found
        """
        if name not in self.models:
            logger.warning(f"Model '{name}' not found in ensemble")
            return False
            
        self.models[name] = new_model
        if weight is not None:
            self.model_weights[name] = weight
            self.total_weight = sum(self.model_weights.values())
        
        logger.info(f"Replaced model '{name}' in ensemble")
        return True

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate ensemble prediction.
        
        Returns:
            {
                'prediction': 0 or 1,
                'confidence': 0-1,
                'model_votes': {...},
                'disagreement': 0-1 (consensus score)
            }
        """
        if not self.models:
            logger.warning("No models in ensemble")
            return {"error": "No models available"}

        predictions = {}
        confidences = {}
        total_weighted_pred = 0.0
        total_confidence_weighted = 0.0

        for name, model in self.models.items():
            try:
                pred, conf = model.predict(X)
                predictions[name] = pred
                confidences[name] = conf
                
                weight = self.model_weights.get(name, 1.0) / self.total_weight
                total_weighted_pred += pred * weight
                total_confidence_weighted += np.mean(conf) * weight
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
                continue

        # Ensemble prediction (weighted average)
        ensemble_pred = (total_weighted_pred > 0.5).astype(int)
        ensemble_confidence = min(abs(total_weighted_pred - 0.5) * 2, 1.0)

        # Calculate disagreement (lower = more consensus)
        model_predictions = [p.flatten()[0] if hasattr(p, 'flatten') else p 
                            for p in predictions.values()]
        disagreement = np.std(model_predictions) if model_predictions else 0

        return {
            "prediction": ensemble_pred,
            "confidence": ensemble_confidence,
            "weighted_confidence": total_confidence_weighted,
            "model_predictions": predictions,
            "model_confidences": confidences,
            "disagreement": float(disagreement),
            "consensus": 1.0 - min(disagreement, 1.0),
            "num_models": len(self.models),
        }


class MarketRegimeDetector:
    """
    Detects current market regime (trending, ranging, volatile).
    Switches strategy intelligently based on regime.
    """

    def __init__(self, lookback_period: int = 50):
        """Initialize regime detector"""
        self.lookback_period = lookback_period
        self.current_regime = None

    def detect_regime(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Detect market regime from OHLC data.
        
        Returns:
            {
                'regime': 'trending' | 'ranging' | 'volatile',
                'trend_strength': 0-1,
                'volatility': value,
                'liquidity': value,
                'confidence': 0-1
            }
        """
        if len(prices) < self.lookback_period:
            return {"regime": "unknown", "confidence": 0}

        recent_prices = prices[-self.lookback_period:]
        recent_volumes = volumes[-self.lookback_period:]

        # Calculate trend strength
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        price_std = np.std(recent_prices)
        mean_price = np.mean(recent_prices)
        volatility = price_std / mean_price if mean_price > 0 else 0

        # Average volume indicator
        avg_volume = np.mean(recent_volumes)
        current_volume = recent_volumes[-1]
        liquidity = current_volume / avg_volume if avg_volume > 0 else 0

        # Regime classification
        if abs(price_momentum) > 0.02:  # > 2% move
            regime = "trending"
            confidence = min(abs(price_momentum) * 10, 1.0)
        elif volatility > 0.003:  # High volatility
            regime = "volatile"
            confidence = min(volatility * 100, 1.0)
        else:
            regime = "ranging"
            confidence = 0.7

        self.current_regime = regime

        return {
            "regime": regime,
            "trend_strength": abs(price_momentum),
            "volatility": volatility,
            "liquidity": liquidity,
            "confidence": confidence,
            "direction": "up" if price_momentum > 0 else "down",
        }

    def get_strategy_suggestion(self, regime: str) -> Dict[str, Any]:
        """
        Suggest strategy based on detected regime.
        
        Returns strategy parameters for current market condition.
        """
        strategies = {
            "trending": {
                "strategy": "trend_following",
                "use_ma_crossover": True,
                "position_size": 1.0,
                "stop_loss_pips": 50,
                "take_profit_pips": 150,
                "description": "Use trend-following strategies with wider stops"
            },
            "ranging": {
                "strategy": "mean_reversion",
                "use_bollinger_bands": True,
                "position_size": 0.8,
                "stop_loss_pips": 30,
                "take_profit_pips": 60,
                "description": "Use mean reversion at support/resistance"
            },
            "volatile": {
                "strategy": "breakout",
                "use_volatility_stops": True,
                "position_size": 0.5,
                "stop_loss_pips": 80,
                "take_profit_pips": 120,
                "description": "Reduce size in volatile markets"
            },
            "unknown": {
                "strategy": "wait",
                "position_size": 0,
                "description": "Not enough data, wait for clarity"
            }
        }
        return strategies.get(regime, strategies["unknown"])
