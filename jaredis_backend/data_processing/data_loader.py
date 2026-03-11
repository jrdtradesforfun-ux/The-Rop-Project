"""
Data Loader: Loads market data from various sources
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and manages market data for ML training and inference
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing market data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.loaded_data: Dict[str, List] = {}

    def load_csv(self, filepath: str, symbol: Optional[str] = None) -> List[Dict]:
        """
        Load OHLCV data from CSV file
        
        Args:
            filepath: Path to CSV file
            symbol: Optional symbol identifier
            
        Returns:
            List of OHLCV candles
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return []

        candles = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    candle = {
                        "timestamp": row.get("timestamp") or row.get("datetime"),
                        "open": float(row.get("open")),
                        "high": float(row.get("high")),
                        "low": float(row.get("low")),
                        "close": float(row.get("close")),
                        "volume": float(row.get("volume", 0))
                    }
                    candles.append(candle)

            if symbol:
                self.loaded_data[symbol] = candles

            logger.info(f"Loaded {len(candles)} candles from {filepath}")
            return candles

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []

    def load_from_mt5(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Load historical data from MT5 (placeholder for MT5 API integration)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
            start_date: Start date
            end_date: End date
            
        Returns:
            List of OHLCV candles
        """
        logger.info(f"Loading {symbol} data from MT5: {timeframe}")
        # MT5 API integration would go here
        return []

    def get_latest_candles(self, symbol: str, count: int = 100) -> List[Dict]:
        """
        Get latest N candles for a symbol
        
        Args:
            symbol: Trading symbol
            count: Number of candles
            
        Returns:
            List of recent candles
        """
        if symbol not in self.loaded_data:
            logger.warning(f"No data loaded for {symbol}")
            return []

        return self.loaded_data[symbol][-count:]

    def get_symbol_data(self, symbol: str) -> List[Dict]:
        """Get all loaded data for a symbol"""
        return self.loaded_data.get(symbol, [])

    def to_numpy(self, candles: List[Dict]) -> np.ndarray:
        """
        Convert candles to numpy array for ML
        
        Args:
            candles: List of OHLCV candles
            
        Returns:
            Numpy array [open, high, low, close, volume]
        """
        if not candles:
            return np.array([])

        return np.array([
            [c["open"], c["high"], c["low"], c["close"], c["volume"]]
            for c in candles
        ])

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear loaded data"""
        if symbol:
            self.loaded_data.pop(symbol, None)
        else:
            self.loaded_data.clear()
