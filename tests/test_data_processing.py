"""
Test Data Processing
"""

import numpy as np
import pytest
from jaredis_backend.data_processing import DataLoader, FeatureEngineer, Preprocessor


class TestDataLoader:
    """Tests for DataLoader"""

    def setup_method(self):
        """Setup test fixtures"""
        self.loader = DataLoader()

    def test_initialization(self):
        """Test data loader initialization"""
        assert self.loader.data_dir.exists()
        assert len(self.loader.loaded_data) == 0

    def test_to_numpy(self):
        """Test conversion to numpy array"""
        candles = [
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
            {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100},
        ]

        arr = self.loader.to_numpy(candles)

        assert arr.shape == (2, 5)
        assert arr[0, 0] == 100  # First open


class TestFeatureEngineer:
    """Tests for FeatureEngineer"""

    def test_calculate_returns(self):
        """Test return calculation"""
        prices = np.array([100, 102, 101, 103])
        returns = FeatureEngineer.calculate_returns(prices)

        assert len(returns) == 3
        assert isinstance(returns[0], np.floating)

    def test_calculate_rsi(self):
        """Test RSI calculation"""
        prices = np.random.rand(100) * 10 + 100  # 100-110 range
        rsi = FeatureEngineer.calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)
        assert np.all((rsi >= 0) | np.isnan(rsi))
        assert np.all((rsi <= 100) | np.isnan(rsi))

    def test_calculate_ema(self):
        """Test EMA calculation"""
        prices = np.linspace(100, 110, 100)
        ema = FeatureEngineer.calculate_ema(prices, period=10)

        assert len(ema) == len(prices)
        assert not np.any(np.isnan(ema))

    def test_calculate_macd(self):
        """Test MACD calculation"""
        prices = np.linspace(100, 110, 100)
        macd = FeatureEngineer.calculate_macd(prices)

        assert "macd" in macd
        assert "signal" in macd
        assert "histogram" in macd
        assert len(macd["macd"]) == len(prices)


class TestPreprocessor:
    """Tests for Preprocessor"""

    def test_normalize_minmax(self):
        """Test min-max normalization"""
        data = np.array([10, 20, 30, 40, 50])
        normalized, params = Preprocessor.normalize(data, method="minmax")

        assert np.min(normalized) == 0
        assert np.max(normalized) == 1
        assert params["method"] == "minmax"

    def test_denormalize(self):
        """Test denormalization"""
        data = np.array([10, 20, 30, 40, 50])
        normalized, params = Preprocessor.normalize(data, method="minmax")
        denormalized = Preprocessor.denormalize(normalized, params)

        np.testing.assert_array_almost_equal(denormalized, data)

    def test_handle_missing_values(self):
        """Test missing value handling"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        filled = Preprocessor.handle_missing_values(data, method="forward_fill")

        assert not np.any(np.isnan(filled))

    def test_create_sequences(self):
        """Test sequence creation"""
        data = np.arange(10)
        X, y = Preprocessor.create_sequences(data, seq_length=3)

        assert X.shape == (7, 3)
        assert y.shape == (7,)
        np.testing.assert_array_equal(X[0], [0, 1, 2])
        assert y[0] == 3

    def test_validate_data(self):
        """Test data validation"""
        valid_candles = [
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
        ]

        assert Preprocessor.validate_data(valid_candles) is True

    def test_validate_data_invalid_ohlc(self):
        """Test validation rejects invalid OHLC"""
        invalid_candles = [
            {"open": 100, "high": 98, "low": 99, "close": 101, "volume": 1000},
        ]

        assert Preprocessor.validate_data(invalid_candles) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
