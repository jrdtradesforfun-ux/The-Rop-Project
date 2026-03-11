"""
Jaredis Smart Trading Backend
ML-powered trading engine with MQL5 integration
"""

__version__ = "1.0.0"
__author__ = "Jaredis Trading"

from .trading_engine.engine import TradingEngine
from .ml_models.model_manager import ModelManager

__all__ = ["TradingEngine", "ModelManager"]
