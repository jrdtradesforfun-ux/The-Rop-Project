"""
ZeroMQ Bridge for Low-Latency MT5-Python Communication
Provides real-time market data and order execution via ZeroMQ sockets
"""

import logging
import asyncio
import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    import zmq
    import zmq.asyncio
    ZEROMQ_AVAILABLE = True
except ImportError:
    ZEROMQ_AVAILABLE = False
    logging.warning("ZeroMQ not installed. Install with: pip install pyzmq")

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """Trade action types"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    MODIFY = "MODIFY"


@dataclass
class Tick:
    """Market tick data"""
    symbol: str
    bid: float
    ask: float
    time: datetime
    volume: int
    spread: float
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'time': self.time.isoformat(),
            'volume': self.volume,
            'spread': self.spread
        }


@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MT5ZeroMQBridge:
    """
    High-performance ZeroMQ bridge to MetaTrader 5
    
    Provides:
    - REQ/REP socket for command-response (trade orders)
    - PUB/SUB socket for price feed (real-time data)
    - Heartbeat monitoring for connection health
    """
    
    def __init__(self,
                 cmd_host: str = "localhost",
                 cmd_port: int = 5555,
                 pub_host: str = "localhost",
                 pub_port: int = 5556,
                 timeout_ms: int = 5000):
        """
        Initialize ZeroMQ bridge
        
        Args:
            cmd_host: Command socket host
            cmd_port: Command socket port (REQ/REP)
            pub_host: Price feed socket host
            pub_port: Price feed socket port (PUB/SUB)
            timeout_ms: Socket timeout in milliseconds
        """
        if not ZEROMQ_AVAILABLE:
            raise ImportError("ZeroMQ not installed. Install with: pip install pyzmq")
        
        self.cmd_address = f"tcp://{cmd_host}:{cmd_port}"
        self.pub_address = f"tcp://{pub_host}:{pub_port}"
        self.timeout_ms = timeout_ms
        
        # Async context
        self.context = zmq.asyncio.Context()
        
        # Sockets
        self.cmd_socket: Optional[zmq.asyncio.Socket] = None
        self.pub_socket: Optional[zmq.asyncio.Socket] = None
        
        # State
        self.connected = False
        self.last_heartbeat = None
        self.price_cache: Dict[str, Tick] = {}
        self.request_count = 0
        self.error_count = 0
    
    async def connect(self) -> bool:
        """
        Initialize ZeroMQ connections
        
        Returns:
            True if connection successful
        """
        try:
            # Command socket (REQ pattern)
            self.cmd_socket = self.context.socket(zmq.REQ)
            self.cmd_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.cmd_socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.cmd_socket.connect(self.cmd_address)
            
            # Price feed socket (SUB pattern)
            self.pub_socket = self.context.socket(zmq.SUB)
            self.pub_socket.setsockopt_string(zmq.SUBSCRIBE, "PRICE|")
            self.pub_socket.connect(self.pub_address)
            
            self.connected = True
            logger.info(f"✅ Connected to MT5 Bridge")
            logger.info(f"  Command: {self.cmd_address}")
            logger.info(f"  Price Feed: {self.pub_address}")
            
            # Start background tasks
            asyncio.create_task(self._price_feed_loop())
            asyncio.create_task(self._heartbeat_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Clean shutdown of ZeroMQ connections"""
        self.connected = False
        
        if self.cmd_socket:
            self.cmd_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        
        self.context.term()
        logger.info("Disconnected from MT5 Bridge")
    
    async def _send_command(self, command: str) -> str:
        """
        Send command to MT5 and receive response
        
        Args:
            command: Command string
            
        Returns:
            Response from MT5
        """
        if not self.connected or not self.cmd_socket:
            raise ConnectionError("Not connected to MT5")
        
        try:
            self.request_count += 1
            await self.cmd_socket.send_string(command)
            response = await self.cmd_socket.recv_string()
            return response
            
        except zmq.Again:
            self.error_count += 1
            raise TimeoutError(f"MT5 command timeout: {command}")
        except Exception as e:
            self.error_count += 1
            raise
    
    async def trade(self,
                   action: TradeAction,
                   symbol: str,
                   volume: float,
                   sl_points: Optional[float] = None,
                   tp_points: Optional[float] = None,
                   comment: str = "ZMQ_BOT") -> TradeResult:
        """
        Execute trade via MT5
        
        Args:
            action: BUY or SELL
            symbol: Trading symbol
            volume: Position size
            sl_points: Stop loss in points
            tp_points: Take profit in points
            comment: Trade comment
            
        Returns:
            Trade result with order ID or error
        """
        sl = sl_points if sl_points else 0
        tp = tp_points if tp_points else 0
        
        cmd = f"TRADE|{action.value}|{symbol}|{volume}|{sl}|{tp}|{comment}"
        
        try:
            response = await self._send_command(cmd)
            parts = response.split("|")
            
            if parts[0] == "OK" and len(parts) >= 3:
                return TradeResult(
                    success=True,
                    order_id=int(parts[2])
                )
            else:
                error_msg = "|".join(parts[1:]) if len(parts) > 1 else "Unknown error"
                return TradeResult(
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResult(
                success=False,
                error=str(e)
            )
    
    async def close_position(self, ticket: int) -> TradeResult:
        """
        Close specific position
        
        Args:
            ticket: Position ticket number
            
        Returns:
            Close result
        """
        cmd = f"CLOSE|{ticket}"
        
        try:
            response = await self._send_command(cmd)
            parts = response.split("|")
            
            if parts[0] == "OK":
                return TradeResult(success=True)
            else:
                return TradeResult(
                    success=False,
                    error="|".join(parts[1:])
                )
        except Exception as e:
            return TradeResult(success=False, error=str(e))
    
    async def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """
        Modify stop loss and take profit
        
        Args:
            ticket: Position ticket
            sl: New stop loss
            tp: New take profit
            
        Returns:
            Success status
        """
        cmd = f"MODIFY|{ticket}|{sl}|{tp}"
        
        try:
            response = await self._send_command(cmd)
            return response.startswith("OK")
        except Exception as e:
            logger.error(f"Modification failed: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, float]:
        """
        Fetch account information
        
        Returns:
            Account details (balance, equity, margin, etc.)
        """
        try:
            response = await self._send_command("INFO|ACCOUNT")
            parts = response.split("|")
            
            if parts[0] == "OK" and len(parts) >= 6:
                return {
                    "name": parts[1],
                    "balance": float(parts[2]),
                    "equity": float(parts[3]),
                    "margin": float(parts[4]),
                    "margin_free": float(parts[5])
                }
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict]:
        """
        Get open positions
        
        Returns:
            List of open positions
        """
        try:
            response = await self._send_command("INFO|POSITIONS")
            parts = response.split("|")
            
            positions = []
            if len(parts) > 1 and parts[1]:
                for pos_str in parts[1].split(";"):
                    if not pos_str:
                        continue
                    vals = pos_str.split(",")
                    if len(vals) >= 6:
                        positions.append({
                            "ticket": int(vals[0]),
                            "symbol": vals[1],
                            "volume": float(vals[2]),
                            "price_open": float(vals[3]),
                            "sl": float(vals[4]),
                            "tp": float(vals[5])
                        })
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def _price_feed_loop(self) -> None:
        """Background task: receive and cache price updates"""
        while self.connected:
            try:
                if self.pub_socket:
                    msg = await asyncio.wait_for(
                        self.pub_socket.recv_string(),
                        timeout=5.0
                    )
                    
                    # Format: PRICE|SYMBOL|BID|ASK|TIME|VOLUME|SPREAD
                    try:
                        parts = msg.split("|")
                        if len(parts) >= 7 and parts[0] == "PRICE":
                            tick = Tick(
                                symbol=parts[1],
                                bid=float(parts[2]),
                                ask=float(parts[3]),
                                time=datetime.fromtimestamp(int(parts[4])),
                                volume=int(parts[5]),
                                spread=float(parts[6])
                            )
                            self.price_cache[tick.symbol] = tick
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse price message: {msg}")
                        
            except asyncio.TimeoutError:
                await asyncio.sleep(0.1)
            except Exception as e:
                if self.connected:
                    logger.error(f"Price feed error: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_loop(self) -> None:
        """Monitor connection health via heartbeat"""
        while self.connected:
            try:
                response = await self._send_command("HEARTBEAT")
                if response.startswith("OK"):
                    self.last_heartbeat = datetime.now()
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                await asyncio.sleep(1)
    
    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get cached tick data for symbol"""
        return self.price_cache.get(symbol)
    
    def get_cached_ticks(self) -> Dict[str, Tick]:
        """Get all cached ticks"""
        return self.price_cache.copy()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection health status"""
        return {
            "connected": self.connected,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "cached_symbols": len(self.price_cache),
            "error_rate": self.error_count / max(self.request_count, 1)
        }


class ZeroMQConnectionPool:
    """Manage multiple MT5 connections for resilience"""
    
    def __init__(self, connections: List[Dict[str, str]]):
        """
        Initialize connection pool
        
        Args:
            connections: List of connection configs
                [{"cmd_host": "localhost", "cmd_port": 5555}, ...]
        """
        self.connections = connections
        self.bridges = []
        self.active_bridge = None
    
    async def connect_all(self) -> bool:
        """Connect all bridges"""
        for config in self.connections:
            bridge = MT5ZeroMQBridge(**config)
            if await bridge.connect():
                self.bridges.append(bridge)
                if self.active_bridge is None:
                    self.active_bridge = bridge
        
        logger.info(f"Connected {len(self.bridges)}/{len(self.connections)} bridges")
        return len(self.bridges) > 0
    
    async def trade(self, *args, **kwargs) -> TradeResult:
        """Execute trade with failover"""
        for bridge in self.bridges:
            if bridge == self.active_bridge:
                try:
                    return await bridge.trade(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Primary bridge failed: {e}")
                    self.active_bridge = None
        
        # Failover to backup
        for bridge in self.bridges:
            if bridge != self.active_bridge:
                try:
                    result = await bridge.trade(*args, **kwargs)
                    self.active_bridge = bridge
                    return result
                except Exception:
                    continue
        
        return TradeResult(success=False, error="All bridges failed")
