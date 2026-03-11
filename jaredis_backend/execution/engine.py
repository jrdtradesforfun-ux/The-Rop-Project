"""
Advanced Execution Engine
Intelligent order routing and execution
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Professional-grade execution engine.
    Handles order creation, routing, and execution logic.
    """

    def __init__(self, broker, risk_manager):
        """
        Initialize execution engine.
        
        Args:
            broker: Broker connector (e.g., JustMarketsBroker)
            risk_manager: Risk management module
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.pending_orders: Dict[str, Dict] = {}
        self.execution_history: list = []

    def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Execute a trading signal with full validation.
        
        Signal format:
        {
            'symbol': 'EURUSD',
            'direction': 'long',  # 'long' or 'short'
            'entry_price': 1.0950,
            'stop_loss': 1.0900,
            'take_profit': 1.1000,
            'volume': 0.1,
            'confidence': 0.85,
            'signal_type': 'ml_prediction',
            'timestamp': '2024-01-01T12:00:00'
        }
        """
        # Risk validation
        if not self.risk_manager.validate_trade(signal):
            logger.warning(f"Signal rejected by risk manager: {signal}")
            return None

        # Calculate position size with risk management
        position_size = self._calculate_position_size(signal)
        if position_size <= 0:
            logger.warning(f"Invalid position size calculated: {position_size}")
            return None

        # Build order
        order = self._build_order(signal, position_size)

        # Execute order
        execution_result = self._execute_order(order)

        if execution_result and "error" not in execution_result:
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "signal": signal,
                "execution": execution_result,
                "position_size": position_size,
            })
            logger.info(f"Signal executed successfully: {execution_result}")
            return execution_result
        else:
            logger.error(f"Execution failed: {execution_result}")
            return None

    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on risk parameters"""
        entry = signal.get("entry_price", 0)
        sl = signal.get("stop_loss", 0)
        
        if entry <= 0 or sl <= 0:
            return 0

        position_size = self.risk_manager.calculate_position_size(
            entry_price=entry,
            stop_loss=sl,
            risk_percent=0.02,  # 2% risk per trade
        )
        return position_size

    def _build_order(self, signal: Dict, position_size: float) -> Dict:
        """Build complete order from signal"""
        direction_str = "buy" if signal.get("direction") == "long" else "sell"
        
        return {
            "symbol": signal.get("symbol"),
            "direction": direction_str,
            "volume": position_size,
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "comment": f"Jaredis ML Signal ({signal.get('signal_type', 'unknown')})",
            "magic": 12345,
            "confidence": signal.get("confidence", 0),
            "timestamp": signal.get("timestamp", datetime.now().isoformat()),
        }

    def _execute_order(self, order: Dict) -> Optional[Dict]:
        """Send order to broker"""
        try:
            result = self.broker.place_order(
                symbol=order["symbol"],
                order_type=order["direction"],
                volume=order["volume"],
                entry_price=order["entry_price"],
                stop_loss=order["stop_loss"],
                take_profit=order["take_profit"],
                comment=order["comment"],
                magic=order["magic"],
            )
            return result
        except Exception as e:
            logger.error(f"Broker execution error: {e}")
            return None

    def close_position(self, ticket: int) -> Optional[Dict]:
        """Close a position"""
        try:
            return self.broker.close_position(ticket)
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Dict]:
        """Modify position SL/TP"""
        try:
            return self.broker.modify_position(ticket, stop_loss, take_profit)
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return None

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total = len(self.execution_history)
        successful = sum(
            1 for ex in self.execution_history 
            if "error" not in ex.get("execution", {})
        )
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
        }
