"""
Core Trading Engine: Main decision making and strategy execution
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .position_manager import PositionManager
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine coordinating ML signals with position management
    """

    def __init__(self, account_size: float, risk_per_trade: float = 0.02):
        """
        Initialize trading engine
        
        Args:
            account_size: Account size in base currency
            risk_per_trade: Risk percentage per trade (0-1)
        """
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(account_size=account_size)
        self.trade_history: List[Dict] = []
        self.active_strategies: Dict = {}

    def register_strategy(self, name: str, strategy) -> None:
        """
        Register a trading strategy
        
        Args:
            name: Strategy identifier
            strategy: Strategy object with signal() method
        """
        self.active_strategies[name] = strategy
        logger.info(f"Strategy '{name}' registered")

    def evaluate_signals(self, market_data: Dict) -> Dict:
        """
        Evaluate signals from all active strategies
        
        Args:
            market_data: Current market data (OHLCV, etc.)
            
        Returns:
            Aggregated signals and recommendations
        """
        signals = {
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data,
            "strategy_signals": {},
            "recommended_action": None,
            "confidence": 0.0
        }

        # Collect signals from each strategy
        for strategy_name, strategy in self.active_strategies.items():
            try:
                signal = strategy.generate_signal(market_data)
                signals["strategy_signals"][strategy_name] = signal
            except Exception as e:
                logger.error(f"Error in strategy '{strategy_name}': {e}")

        return signals

    def execute_trade(self, signal: Dict, symbol: str) -> Optional[Dict]:
        """
        Execute a trade based on signal
        
        Args:
            signal: Trade signal with direction and parameters
            symbol: Trading instrument symbol
            
        Returns:
            Trade execution result or None if no trade
        """
        # Risk validation
        if not self.risk_manager.validate_trade(signal):
            logger.warning("Trade rejected by risk manager")
            return None

        # Create position
        position = self.position_manager.open_position(
            symbol=symbol,
            direction=signal.get("direction"),
            size=signal.get("size"),
            entry_price=signal.get("entry_price"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit")
        )

        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "position_id": position.get("id"),
            "signal": signal,
            "status": "open"
        }

        self.trade_history.append(trade_record)
        logger.info(f"Trade executed: {position}")

        return position

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Update all open positions with current market prices
        
        Args:
            current_prices: Current prices by symbol
            
        Returns:
            List of closed positions
        """
        closed_positions = self.position_manager.update_positions(current_prices)
        
        for position in closed_positions:
            logger.info(f"Position closed: {position['id']} with P&L: {position.get('pnl')}")

        return closed_positions

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            "account_size": self.account_size,
            "open_positions": self.position_manager.get_open_positions(),
            "total_equity": self.position_manager.calculate_total_equity(),
            "unrealized_pnl": self.position_manager.calculate_unrealized_pnl()
        }

    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get trade history"""
        history = self.trade_history
        if limit:
            history = history[-limit:]
        return history
