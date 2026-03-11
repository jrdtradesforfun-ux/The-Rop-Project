"""
Position Manager: Manages open positions and P&L calculations
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages trading positions, P&L, and position updates
    """

    def __init__(self):
        """Initialize position manager"""
        self.positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []

    def open_position(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            
        Returns:
            Position details
        """
        position_id = str(uuid4())[:8]
        
        position = {
            "id": position_id,
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opening_time": datetime.now().isoformat(),
            "current_price": entry_price,
            "status": "open"
        }

        self.positions[position_id] = position
        logger.info(f"Position opened: {position_id} - {symbol} {direction} {size}@{entry_price}")

        return position

    def update_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Update positions with current prices and check for exits
        
        Args:
            current_prices: Current prices by symbol
            
        Returns:
            List of closed positions
        """
        closed = []

        for pos_id, position in list(self.positions.items()):
            symbol = position["symbol"]
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            position["current_price"] = current_price

            # Check stop loss or take profit
            should_close = False
            reason = None

            if position["direction"] == "long":
                if position["stop_loss"] and current_price <= position["stop_loss"]:
                    should_close = True
                    reason = "stop_loss"
                elif position["take_profit"] and current_price >= position["take_profit"]:
                    should_close = True
                    reason = "take_profit"
            else:  # short
                if position["stop_loss"] and current_price >= position["stop_loss"]:
                    should_close = True
                    reason = "stop_loss"
                elif position["take_profit"] and current_price <= position["take_profit"]:
                    should_close = True
                    reason = "take_profit"

            if should_close:
                closed_pos = self.close_position(pos_id, current_price, reason)
                closed.append(closed_pos)

        return closed

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Dict:
        """
        Close an open position
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            reason: Reason for closing (sl, tp, manual, etc.)
            
        Returns:
            Closed position with P&L
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.positions.pop(position_id)
        
        # Calculate P&L
        if position["direction"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["size"]

        pnl_percent = (pnl / (position["entry_price"] * position["size"])) * 100

        closed_position = {
            **position,
            "exit_price": exit_price,
            "closing_time": datetime.now().isoformat(),
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "close_reason": reason,
            "status": "closed"
        }

        self.closed_positions.append(closed_position)
        logger.info(f"Position closed: {position_id} P&L: {pnl:.2f} ({pnl_percent:.2f}%)")

        return closed_position

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return list(self.positions.values())

    def get_closed_positions(self, limit: Optional[int] = None) -> List[Dict]:
        """Get closed positions history"""
        positions = self.closed_positions
        if limit:
            positions = positions[-limit:]
        return positions

    def calculate_total_equity(self) -> float:
        """Calculate total unrealized equity"""
        total = 0.0
        for position in self.positions.values():
            if position["direction"] == "long":
                pnl = (position["current_price"] - position["entry_price"]) * position["size"]
            else:
                pnl = (position["entry_price"] - position["current_price"]) * position["size"]
            total += pnl

        return total

    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return self.calculate_total_equity()
