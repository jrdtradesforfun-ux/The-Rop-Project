"""
Advanced Feature Engineering Pipeline
Produces technical indicators and features for ML models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_period: int = 50
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_price_patterns: bool = True
    n_lags: int = 5


class FeatureEngineer:
    """Advanced feature engineering with technical indicators"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data
        
        Args:
            df: DataFrame with OHLC data (open, high, low, close, volume)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Trend indicators
        df = self._add_trend_features(df)
        
        # Momentum indicators
        if self.config.include_momentum:
            df = self._add_momentum_features(df)
        
        # Volatility indicators
        if self.config.include_volatility:
            df = self._add_volatility_features(df)
        
        # Volume indicators
        if self.config.include_volume:
            df = self._add_volume_features(df)
        
        # Price patterns
        if self.config.include_price_patterns:
            df = self._add_pattern_features(df)
        
        # Lag features
        df = self._add_lag_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ranges
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['close']
        
        # Position in range
        df['position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend indicators: Moving Averages"""
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Distance from MA
            df[f'dist_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators: RSI, Stochastic, etc."""
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'], 14)
        
        # Momentum (Price change)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_21'] = df['close'] - df['close'].shift(21)
        
        # Rate of Change
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['roc_21'] = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators: ATR, Bollinger Bands, etc."""
        # ATR
        df['atr_14'] = self._calculate_atr(df['high'], df['low'], df['close'], 14)
        df['atr_21'] = self._calculate_atr(df['high'], df['low'], df['close'], 21)
        
        # Volatility (Standard Deviation)
        df['volatility_10'] = df['close'].rolling(window=10).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        df['volatility_30'] = df['close'].rolling(window=30).std()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility regime
        df['volatility_ma'] = df['volatility_20'].rolling(window=20).mean()
        df['volatility_regime'] = pd.cut(df['volatility_20'], bins=3, labels=['low', 'medium', 'high'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        if 'volume' not in df.columns:
            return df
        
        # Volume moving average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # OBV
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        
        # Volume trend
        df['volume_trend'] = df['volume'].pct_change()
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price action pattern features"""
        # Higher High / Higher Low pattern
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        
        df['higher_high'] = df['high'] > df['prev_high']
        df['lower_low'] = df['low'] < df['prev_low']
        
        # Consecutive closes above/below moving average
        sma_20 = df['close'].rolling(window=20).mean()
        df['closes_above_sma'] = (df['close'] > sma_20).astype(int)
        
        # Consecutive up/down days
        df['returns_sign'] = np.sign(df['returns'])
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag features for sequence learning"""
        for lag in range(1, self.config.n_lags + 1):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            if 'rsi_14' in df.columns:
                df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                             period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        L = low.rolling(window=period).min()
        H = high.rolling(window=period).max()
        K = 100 * ((close - L) / (H - L))
        K_ma = K.rolling(window=k_period).mean()
        D = K_ma.rolling(window=d_period).mean()
        return K_ma, D
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.where(close > close.shift(1), volume, 
               np.where(close < close.shift(1), -volume, 0))
        return pd.Series(obv).cumsum()


class DataValidator:
    """Validate data quality before training"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate OHLCV data quality
        
        Returns:
            Dictionary of validation checks
        """
        checks = {
            'sufficient_rows': len(df) >= 100,
            'no_missing': df[['open', 'high', 'low', 'close']].isnull().sum().sum() == 0,
            'positive_prices': (df[['open', 'high', 'low', 'close']] > 0).all().all(),
            'hlc_consistency': (df['high'] >= df['low']).all(),
            'ohlc_consistency': (
                (df['high'] >= df[['open', 'close']].max(axis=1)) &
                (df['low'] <= df[['open', 'close']].min(axis=1))
            ).all(),
            'no_duplicates': not df.index.duplicated().any(),
            'monotonic_time': df.index.is_monotonic_increasing if hasattr(df.index, 'is_monotonic_increasing') else True
        }
        
        return checks
    
    @staticmethod
    def log_validation(checks: Dict[str, bool]) -> None:
        """Log validation results"""
        passed = sum(checks.values())
        total = len(checks)
        logger.info(f"Data validation: {passed}/{total} checks passed")
        
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check_name}")


class LabelGenerator:
    """Generate labels for supervised learning"""
    
    @staticmethod
    def triple_barrier_labels(df: pd.DataFrame, 
                             profit_target: float = 0.001,
                             stop_loss: float = 0.002,
                             horizon: int = 5) -> pd.Series:
        """
        Triple barrier labeling method
        
        Args:
            df: DataFrame with close prices
            profit_target: Profit target threshold (0.001 = 0.1%)
            stop_loss: Stop loss threshold
            horizon: Number of bars ahead to look
            
        Returns:
            Series with labels (1=long, -1=short, 0=hold)
        """
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - horizon):
            current_price = df['close'].iloc[i]
            future_window = df['close'].iloc[i+1:i+horizon+1]
            
            max_up = (future_window.max() - current_price) / current_price
            max_down = (future_window.min() - current_price) / current_price
            
            # Triple barrier logic
            if max_up >= profit_target and max_down > -stop_loss:
                labels.iloc[i] = 1  # Long
            elif max_down <= -profit_target and max_up < stop_loss:
                labels.iloc[i] = -1  # Short
            # else: 0 (no clear direction)
        
        return labels
    
    @staticmethod
    def simple_direction_labels(df: pd.DataFrame, threshold: float = 0.001, horizon: int = 5) -> pd.Series:
        """
        Simple directional labels based on future returns
        
        Args:
            df: DataFrame with close prices
            threshold: Threshold for classification
            horizon: Number of bars ahead
            
        Returns:
            Series with labels (1=up, -1=down, 0=neutral)
        """
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        
        labels = pd.Series(0, index=df.index)
        labels[future_return > threshold] = 1
        labels[future_return < -threshold] = -1
        
        return labels
