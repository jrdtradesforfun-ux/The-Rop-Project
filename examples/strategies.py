"""
Example Trading Strategy
Base class for creating custom strategies
"""

from typing import Dict


class BaseStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str):
        """
        Initialize strategy
        
        Args:
            name: Strategy identifier
        """
        self.name = name

    def generate_signal(self, market_data: Dict) -> Dict:
        """
        Generate trading signal from market data
        
        Args:
            market_data: Current market data
            
        Returns:
            Signal dictionary with direction, entry, SL, TP, etc.
        """
        raise NotImplementedError


class SimpleMomentumStrategy(BaseStrategy):
    """
    Simple momentum-based strategy
    Uses price action and volume
    """

    def __init__(self, lookback: int = 20):
        """
        Initialize strategy
        
        Args:
            lookback: Lookback period in candles
        """
        super().__init__("SimpleMomentum")
        self.lookback = lookback

    def generate_signal(self, market_data: Dict) -> Dict:
        """Generate signal based on momentum"""
        # Extract OHLCV
        candles = market_data.get("candles", [])
        
        if len(candles) < self.lookback:
            return {"signal": "no_signal"}

        # Simple momentum: compare close prices
        recent_close = candles[-1]["close"]
        prior_close = candles[-(self.lookback + 1)]["close"]
        
        momentum = (recent_close - prior_close) / prior_close
        
        if momentum > 0.01:  # 1% threshold for uptrend
            return {
                "direction": "long",
                "symbol": market_data.get("symbol"),
                "entry_price": recent_close,
                "stop_loss": recent_close * 0.98,  # 2% below entry
                "take_profit": recent_close * 1.03,  # 3% above entry
                "confidence": abs(momentum),
                "signal_type": "momentum_buy"
            }
        elif momentum < -0.01:  # -1% threshold for downtrend
            return {
                "direction": "short",
                "symbol": market_data.get("symbol"),
                "entry_price": recent_close,
                "stop_loss": recent_close * 1.02,  # 2% above entry
                "take_profit": recent_close * 0.97,  # 3% below entry
                "confidence": abs(momentum),
                "signal_type": "momentum_sell"
            }
        
        return {"signal": "no_signal"}


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy
    Trades against extreme moves
    """

    def __init__(self, period: int = 20, std_threshold: float = 2.0):
        """
        Initialize strategy
        
        Args:
            period: Period for mean calculation
            std_threshold: Std deviation threshold
        """
        super().__init__("MeanReversion")
        self.period = period
        self.std_threshold = std_threshold

    def generate_signal(self, market_data: Dict) -> Dict:
        """Generate mean reversion signal"""
        candles = market_data.get("candles", [])
        
        if len(candles) < self.period:
            return {"signal": "no_signal"}

        # Calculate mean and std
        closes = [c["close"] for c in candles[-self.period:]]
        mean = sum(closes) / len(closes)
        variance = sum((x - mean) ** 2 for x in closes) / len(closes)
        std = variance ** 0.5
        
        current_price = closes[-1]
        distance_from_mean = (current_price - mean) / std if std > 0 else 0
        
        # Generate signal if price is sufficiently far from mean
        if distance_from_mean < -self.std_threshold:
            return {
                "direction": "long",
                "symbol": market_data.get("symbol"),
                "entry_price": current_price,
                "stop_loss": current_price * 0.98,
                "take_profit": mean,
                "confidence": min(abs(distance_from_mean) / self.std_threshold, 1.0),
                "signal_type": "reversion_buy"
            }
        elif distance_from_mean > self.std_threshold:
            return {
                "direction": "short",
                "symbol": market_data.get("symbol"),
                "entry_price": current_price,
                "stop_loss": current_price * 1.02,
                "take_profit": mean,
                "confidence": min(abs(distance_from_mean) / self.std_threshold, 1.0),
                "signal_type": "reversion_sell"
            }
        
        return {"signal": "no_signal"}
