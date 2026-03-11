"""
JustMarkets Broker Integration
Connects to broker through MetaTrader via PyTrader API
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from ..mql5_bridge import PytraderConnector

logger = logging.getLogger(__name__)


class JustMarketsBroker:
    """
    JustMarkets broker connector via MetaTrader.
    
    Setup:
    1. Create demo account at JustMarkets
    2. Install MetaTrader 5
    3. Login to JustMarkets account in MT5
    4. Deploy PyTrader EA in MT5
    5. Run this connector
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        account_type: str = "Demo",
    ):
        """
        Initialize broker connector.
        
        Args:
            host: MT5 server host (localhost if local, or VPS IP for live)
            port: EA socket port
            account_type: "Demo" or "Live"
        """
        self.host = host
        self.port = port
        self.account_type = account_type
        self.connector = PytraderConnector(
            host=host,
            port=port,
            license_type=account_type,
        )
        self.connected = False
        self.broker_name = "JustMarkets"

    def connect(self) -> bool:
        """Establish connection to broker via MetaTrader"""
        if self.connector.connect():
            self.connected = True
            logger.info(f"Connected to {self.broker_name} via MetaTrader")
            try:
                account = self.connector.get_account_info()
                logger.info(f"Account: {account}")
            except Exception as e:
                logger.warning(f"Could not retrieve account info: {e}")
            return True
        else:
            self.connected = False
            logger.error(f"Failed to connect to {self.broker_name}")
            return False

    def disconnect(self) -> None:
        """Disconnect from broker"""
        self.connector.disconnect()
        self.connected = False
        logger.info(f"Disconnected from {self.broker_name}")

    def get_account_balance(self) -> float:
        """Get current account balance"""
        if not self.connected:
            return 0.0
        try:
            return self.connector.get_balance()
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def get_account_equity(self) -> float:
        """Get current account equity"""
        try:
            info = self.connector.get_account_info()
            return info.get("equity", 0.0)
        except Exception as e:
            logger.error(f"Error fetching equity: {e}")
            return 0.0

    def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols on broker"""
        try:
            return self.connector.get_instruments()
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []

    def get_tick(self, symbol: str) -> Dict[str, Any]:
        """Get last tick for a symbol"""
        try:
            return self.connector.get_last_tick(symbol)
        except Exception as e:
            logger.error(f"Error fetching tick for {symbol}: {e}")
            return {}

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
    ) -> Optional[Any]:
        """
        Get historical bars.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe ('M1', 'M5', 'H1', 'D1', etc.)
            count: Number of bars
            
        Returns:
            Bar data or None on error
        """
        try:
            return self.connector.get_bars(symbol, timeframe)
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}/{timeframe}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        order_type: str,  # 'buy' or 'sell'
        volume: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
        magic: int = 0,
    ) -> Dict[str, Any]:
        """
        Place a trade order.
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            volume: Position size (in lots)
            entry_price: Entry price (0 for market order)
            stop_loss: Stop loss level
            take_profit: Take profit level
            comment: Order comment
            magic: Magic number for identification
            
        Returns:
            Order result dict
        """
        try:
            result = self.connector.open_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=entry_price or 0.0,
                sl=stop_loss or 0.0,
                tp=take_profit or 0.0,
                comment=comment or f"Jaredis Smart {datetime.now().isoformat()}",
                magic=magic or 12345,
            )
            logger.info(f"Order placed: {symbol} {order_type} {volume} lots - {result}")
            return result
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close a position by ticket number"""
        try:
            result = self.connector.close_position(ticket)
            logger.info(f"Position closed: ticket {ticket}")
            return result
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"error": str(e)}

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update SL/TP on an open position"""
        try:
            result = self.connector.modify_position(ticket, stop_loss, take_profit)
            logger.info(f"Position modified: ticket {ticket}")
            return result
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return {"error": str(e)}
