"""
Feature Engineer: Creates ML features from market data
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates technical and ML features from OHLCV data
    """

    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns"""
        return np.diff(np.log(prices))

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    @staticmethod
    def calculate_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, np.ndarray]:
        """Calculate MACD"""
        ema_fast = FeatureEngineer.calculate_ema(prices, fast)
        ema_slow = FeatureEngineer.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = FeatureEngineer.calculate_ema(macd, signal)
        histogram = macd - signal_line

        return {
            "macd": macd,
            "signal": signal_line,
            "histogram": histogram
        }

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.zeros_like(prices, dtype=float)
        multiplier = 2 / (period + 1)

        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)

        return ema

    @staticmethod
    def calculate_bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = np.convolve(prices, np.ones(period) / period, mode='valid')
        std = np.std(prices[-len(sma):], axis=0)

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return {
            "upper": upper,
            "middle": sma,
            "lower": lower
        }

    @staticmethod
    def calculate_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate Average True Range"""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close[:-1]),
                np.abs(low - close[:-1])
            )
        )

        atr = np.zeros_like(high)
        atr[:period] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i + 1] = (atr[i] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def create_feature_set(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Create comprehensive feature set for ML"""
        features = {
            "returns": FeatureEngineer.calculate_returns(closes),
            "rsi": FeatureEngineer.calculate_rsi(closes),
            "atr": FeatureEngineer.calculate_atr(highs, lows, closes),
        }

        macd = FeatureEngineer.calculate_macd(closes)
        features.update({
            f"macd_{k}": v for k, v in macd.items()
        })

        bb = FeatureEngineer.calculate_bollinger_bands(closes)
        features.update({
            f"bb_{k}": v for k, v in bb.items()
        })

        # Volume features
        sma_volume = np.convolve(volumes, np.ones(20) / 20, mode='valid')
        features["volume_sma_ratio"] = volumes[-len(sma_volume):] / sma_volume

        return features
