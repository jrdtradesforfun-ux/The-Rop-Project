"""
Risk Manager: Position sizing, risk validation, and portfolio protection
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk parameters and validates trades
    """

    def __init__(
        self,
        account_size: float,
        max_risk_per_trade: float = 0.02,
        max_positions: int = 5,
        max_daily_loss: float = 0.05
    ):
        """
        Initialize risk manager
        
        Args:
            account_size: Account size
            max_risk_per_trade: Max risk per trade (0-1)
            max_positions: Maximum concurrent positions
            max_daily_loss: Maximum daily loss before stopping (0-1)
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.daily_loss = 0.0
        self.open_position_count = 0

    def validate_trade(self, signal: Dict) -> bool:
        """
        Validate if a trade meets risk criteria
        
        Args:
            signal: Trade signal to validate
            
        Returns:
            True if trade passes risk checks, False otherwise
        """
        # Check position count
        if self.open_position_count >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached")
            return False

        # Check risk amount
        risk_amount = signal.get("risk_amount", 0)
        max_risk = self.account_size * self.max_risk_per_trade
        
        if risk_amount > max_risk:
            logger.warning(f"Risk amount {risk_amount} exceeds max {max_risk}")
            return False

        # Check daily loss limit
        if self.daily_loss >= self.account_size * self.max_daily_loss:
            logger.warning("Daily loss limit reached, trading halted")
            return False

        return True

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Risk percentage (uses default if None)
            
        Returns:
            Position size
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade

        risk_per_point = abs(entry_price - stop_loss)
        risk_amount = self.account_size * risk_percent
        
        if risk_per_point == 0:
            return 0

        position_size = risk_amount / risk_per_point

        logger.info(
            f"Calculated position size: {position_size} "
            f"(risk: {risk_amount}, per point: {risk_per_point})"
        )

        return position_size

    def update_daily_loss(self, pnl: float) -> None:
        """Update daily loss accumulator"""
        if pnl < 0:
            self.daily_loss += abs(pnl)
            logger.info(f"Daily loss updated: {self.daily_loss}")

    def reset_daily_loss(self) -> None:
        """Reset daily loss at end of session"""
        self.daily_loss = 0.0
        logger.info("Daily loss reset")

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            "account_size": self.account_size,
            "max_risk_per_trade": self.max_risk_per_trade,
            "daily_loss": self.daily_loss,
            "daily_loss_limit": self.account_size * self.max_daily_loss,
            "open_positions": self.open_position_count,
            "max_positions": self.max_positions
        }
