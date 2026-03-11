"""
Test ML Models
"""

import numpy as np
import pytest
from jaredis_backend.ml_models import ModelManager, PricePredictor, TrendAnalyzer


class TestModelManager:
    """Tests for ModelManager"""

    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ModelManager()

    def test_register_model(self):
        """Test registering a model"""
        model = {"name": "test_model"}
        self.manager.register_model("test", model)

        assert "test" in self.manager.list_models()
        assert self.manager.get_model("test") == model

    def test_metadata_update(self):
        """Test updating model metadata"""
        model = {}
        self.manager.register_model("test", model, metadata={"version": "1.0"})
        self.manager.update_metadata("test", {"accuracy": 0.95})

        metadata = self.manager.get_metadata("test")
        assert metadata["version"] == "1.0"
        assert metadata["accuracy"] == 0.95


class TestPricePredictor:
    """Tests for PricePredictor"""

    def setup_method(self):
        """Setup test fixtures"""
        self.predictor = PricePredictor(lookback_period=60, forecast_period=5)

    def test_initialization(self):
        """Test predictor initialization"""
        assert self.predictor.lookback_period == 60
        assert self.predictor.forecast_period == 5

    def test_training(self):
        """Test model training"""
        prices = np.random.rand(100)
        result = self.predictor.train(prices)

        assert result["samples"] == 100
        assert result["lookback"] == 60
        assert result["forecast"] == 5

    def test_prediction(self):
        """Test price prediction"""
        recent_prices = np.random.rand(60)
        predictions, confidence = self.predictor.predict(recent_prices)

        assert len(predictions) == 5
        assert confidence.shape == (5, 2)


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = TrendAnalyzer(window_size=20)

    def test_trend_analysis(self):
        """Test trend analysis"""
        prices = np.linspace(100, 110, 100)
        volumes = np.random.rand(100) * 1000

        result = self.analyzer.analyze_trend(prices, volumes)

        assert "trend_direction" in result
        assert "trend_strength" in result
        assert 0 <= result["trend_strength"] <= 1

    def test_pattern_classification(self):
        """Test candlestick pattern classification"""
        candles = np.random.rand(10, 4)  # OHLC
        result = self.analyzer.classify_pattern(candles)

        assert "pattern" in result
        assert "pattern_confidence" in result
        assert "bullish" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
