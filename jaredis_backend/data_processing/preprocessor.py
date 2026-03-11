"""
Preprocessor: Data cleaning and normalization
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Handles data cleaning, normalization, and preparation
    """

    @staticmethod
    def remove_outliers(data: np.ndarray, std_threshold: float = 3.0) -> np.ndarray:
        """
        Remove outliers using standard deviation method
        
        Args:
            data: Input data
            std_threshold: Standard deviation threshold
            
        Returns:
            Cleaned data
        """
        mean = np.mean(data)
        std = np.std(data)
        
        mask = np.abs((data - mean) / std) < std_threshold
        return data[mask]

    @staticmethod
    def normalize(data: np.ndarray, method: str = "minmax") -> Tuple[np.ndarray, Dict]:
        """
        Normalize data
        
        Args:
            data: Input data
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Tuple of (normalized_data, scaling_params)
        """
        if method == "minmax":
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val)
            params = {"min": min_val, "max": max_val, "method": "minmax"}
        else:  # zscore
            mean = np.mean(data)
            std = np.std(data)
            normalized = (data - mean) / std
            params = {"mean": mean, "std": std, "method": "zscore"}

        return normalized, params

    @staticmethod
    def denormalize(
        data: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Reverse normalization
        
        Args:
            data: Normalized data
            params: Scaling parameters from normalize()
            
        Returns:
            Denormalized data
        """
        method = params.get("method")
        
        if method == "minmax":
            return data * (params["max"] - params["min"]) + params["min"]
        else:  # zscore
            return data * params["std"] + params["mean"]

    @staticmethod
    def handle_missing_values(data: np.ndarray, method: str = "forward_fill") -> np.ndarray:
        """
        Handle missing values (NaN)
        
        Args:
            data: Input data with possible NaN values
            method: Handling method ('forward_fill', 'backward_fill', 'interpolate')
            
        Returns:
            Data with missing values handled
        """
        if method == "forward_fill":
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.size), 0)
            np.maximum.accumulate(idx, out=idx)
            return data[idx]
        elif method == "backward_fill":
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.size), mask.size - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            return data[idx]
        else:  # interpolate
            y = np.arange(len(data))
            valid = ~np.isnan(data)
            return np.interp(y, y[valid], data[valid])

    @staticmethod
    def create_sequences(
        data: np.ndarray,
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Input data
            seq_length: Sequence length
            
        Returns:
            Tuple of (input_sequences, target_values)
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)

    @staticmethod
    def validate_data(candles: List[Dict]) -> bool:
        """
        Validate market data integrity
        
        Args:
            candles: List of OHLCV candles
            
        Returns:
            True if data is valid, False otherwise
        """
        if not candles:
            logger.warning("Empty candle list")
            return False

        for candle in candles:
            # Check required fields
            required = ["open", "high", "low", "close", "volume"]
            if not all(k in candle for k in required):
                logger.warning(f"Missing required fields in candle: {candle}")
                return False

            # Check OHLC relationship
            if not (candle["low"] <= candle["close"] <= candle["high"] and
                    candle["low"] <= candle["open"] <= candle["high"]):
                logger.warning(f"Invalid OHLC relationship in candle: {candle}")
                return False

            # Check for negative values
            if any(v < 0 for k, v in candle.items() if k != "timestamp"):
                logger.warning(f"Negative values in candle: {candle}")
                return False

        return True
