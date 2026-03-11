"""
Signal Communicator: Formats and sends trading signals to MQL5
"""

import logging
from typing import Dict, Optional

from .mql5_connector import MQL5Connector

logger = logging.getLogger(__name__)


class SignalCommunicator:
    """
    Handles signal formatting and communication with MT5
    """

    def __init__(self, mql5_connector: MQL5Connector):
        """
        Initialize signal communicator
        
        Args:
            mql5_connector: MQL5Connector instance
        """
        self.connector = mql5_connector

    def send_buy_signal(
        self,
        symbol: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        volume: float,
        comment: str = ""
    ) -> bool:
        """
        Send buy signal to MT5
        
        Args:
            symbol: Trading instrument
            entry: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            volume: Position size
            comment: Signal comment/description
            
        Returns:
            True if signal sent successfully
        """
        signal = {
            "type": "order",
            "order_type": "buy",
            "symbol": symbol,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "volume": volume,
            "comment": comment
        }

        return self.connector.send_signal(signal)

    def send_sell_signal(
        self,
        symbol: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        volume: float,
        comment: str = ""
    ) -> bool:
        """
        Send sell signal to MT5
        
        Args:
            symbol: Trading instrument
            entry: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            volume: Position size
            comment: Signal comment/description
            
        Returns:
            True if signal sent successfully
        """
        signal = {
            "type": "order",
            "order_type": "sell",
            "symbol": symbol,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "volume": volume,
            "comment": comment
        }

        return self.connector.send_signal(signal)

    def send_close_signal(self, symbol: str, ticket: Optional[int] = None) -> bool:
        """
        Send close signal to MT5
        
        Args:
            symbol: Trading instrument
            ticket: Optional specific position ticket
            
        Returns:
            True if signal sent successfully
        """
        signal = {
            "type": "close",
            "symbol": symbol,
            "ticket": ticket
        }

        return self.connector.send_signal(signal)

    def send_modify_signal(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Send modify signal to MT5 (update SL/TP)
        
        Args:
            ticket: Position ticket
            stop_loss: New stop loss
            take_profit: New take profit
            
        Returns:
            True if signal sent successfully
        """
        signal = {
            "type": "modify",
            "ticket": ticket,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

        return self.connector.send_signal(signal)

    def format_signal_for_mt5(self, signal: Dict) -> Dict:
        """
        Format trading signal for MT5
        
        Args:
            signal: Raw signal from trading engine
            
        Returns:
            Formatted signal for MQL5
        """
        return {
            "type": "order",
            "order_type": signal.get("direction", "buy").lower(),
            "symbol": signal.get("symbol"),
            "entry": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "volume": signal.get("volume"),
            "magic_number": signal.get("magic_number", 12345),
            "comment": signal.get("comment", "Jaredis Smart"),
            "confidence": signal.get("confidence")
        }
