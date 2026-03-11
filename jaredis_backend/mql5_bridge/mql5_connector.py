"""
MQL5 Connector: Manages communication with MetaTrader 5 platform
"""

import json
import logging
import socket
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MQL5Connector:
    """
    Connects to MetaTrader 5 EA via socket communication
    """

    def __init__(self, host: str = "localhost", port: int = 5000, timeout: float = 5.0):
        """
        Initialize MQL5 connector
        
        Args:
            host: MT5 host address
            port: MT5 listening port
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        """
        Establish connection to MQL5 EA
        
        Returns:
            True if connected successfully
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to MQL5 at {self.host}:{self.port}")
            return True
        except (socket.error, OSError) as e:
            logger.error(f"Failed to connect to MQL5: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from MQL5 EA"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        logger.info("Disconnected from MQL5")

    def send_signal(self, signal: Dict) -> bool:
        """
        Send trading signal to MQL5
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            True if sent successfully
        """
        if not self.connected:
            logger.error("Not connected to MQL5")
            return False

        try:
            message = json.dumps(signal)
            self.socket.sendall(message.encode('utf-8') + b'\n')
            logger.info(f"Signal sent: {signal}")
            return True
        except (socket.error, OSError) as e:
            logger.error(f"Failed to send signal: {e}")
            self.connected = False
            return False

    def receive_update(self) -> Optional[Dict]:
        """
        Receive market update from MQL5
        
        Returns:
            Market data dictionary or None if error
        """
        if not self.connected:
            return None

        try:
            data = self.socket.recv(4096).decode('utf-8')
            if data:
                update = json.loads(data)
                logger.debug(f"Received update: {update}")
                return update
        except (socket.error, json.JSONDecodeError) as e:
            logger.error(f"Failed to receive update: {e}")
            self.connected = False

        return None

    def heartbeat(self) -> bool:
        """
        Send heartbeat to keep connection alive
        
        Returns:
            True if heartbeat sent successfully
        """
        heartbeat_msg = {"type": "heartbeat", "timestamp": "current_time"}
        return self.send_signal(heartbeat_msg)

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected
