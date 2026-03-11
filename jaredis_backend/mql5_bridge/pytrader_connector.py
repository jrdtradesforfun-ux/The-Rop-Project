"""
PytraderConnector: wrapper around Pytrader_API for enhanced MT4/MT5 support
"""

from typing import Optional, Dict, Any, List
import logging

from ..pytrader import Pytrader_API

logger = logging.getLogger(__name__)


class PytraderConnector:
    """
    Connector that uses the Pytrader_API class to interact with
    both MT4 and MT5 platforms. This provides extended market data
    and order management capabilities beyond the simple socket-based
    MQL5Connector.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        timeout: int = 60,
        license_type: str = "Demo",
    ):
        """
        Initialize the connector.
        Args:
            host: address of the license EA socket server
            port: port number
            timeout: seconds to wait for socket responses
            license_type: license mode used by the EA
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.api = Pytrader_API()
        self.api.license = license_type
        self.connected = False

    def connect(self) -> bool:
        """Establish connection to the EA"""
        try:
            result = self.api.Connect(self.host, self.port)
            self.connected = result
            logger.info(f"Pytrader connected: {result}")
            return result
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect the TCP connection"""
        try:
            self.api.Disconnect()
        except Exception:
            pass
        self.connected = False
        logger.info("Pytrader disconnected")

    def is_connected(self) -> bool:
        """Check if API is connected"""
        return self.api.IsConnected()

    # Market data methods
    def get_instruments(self) -> List[str]:
        """Retrieve list of instruments known to the broker"""
        return self.api.Get_instruments()

    def get_last_tick(self, symbol: str) -> Dict[str, Any]:
        """Retrieve last tick information for a symbol"""
        return self.api.Get_last_tick_info(symbol)

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        """Fetch historical bar data from the server"""
        if start and end:
            return self.api.Get_specific_bar(symbol, timeframe, start, end)
        return self.api.Get_last_x_bars_from_now(symbol, timeframe, self.api.max_bars)

    # Order/trading helpers
    def open_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
        magic: int = 0,
    ) -> Dict:
        """Open a new market or pending order"""
        direction = 0 if order_type.lower() == "buy" else 1
        return self.api.Open_order(symbol, direction, volume, price, sl, tp, comment, magic)

    def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket number"""
        return self.api.Close_position_by_ticket(ticket)

    def modify_position(
        self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None
    ) -> Dict:
        """Modify SL/TP on an existing position"""
        return self.api.Set_sl_and_tp_for_position(ticket, sl, tp)

    # Utility
    def get_account_info(self) -> Dict:
        """Get static account information"""
        return self.api.Get_static_account_info()

    def get_balance(self) -> float:
        info = self.api.Get_dynamic_account_info()
        return info.get("balance", 0.0)
