"""
Monitoring and Metrics System
Real-time performance tracking and alerting
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Track trading performance metrics in real-time"""

    def __init__(self, alert_threshold_drawdown: float = 0.05):
        """
        Initialize monitor.
        
        Args:
            alert_threshold_drawdown: Alert if drawdown exceeds this (default 5%)
        """
        self.trades: List[Dict] = []
        self.alert_threshold_drawdown = alert_threshold_drawdown
        self.alerts: List[Dict] = []
        self.session_start_time = datetime.now()
        self.peak_equity = 0.0
        self.current_drawdown = 0.0

    def record_trade(self, trade: Dict) -> None:
        """Record completed trade"""
        self.trades.append({
            "timestamp": datetime.now().isoformat(),
            **trade
        })
        logger.info(f"Trade recorded: {trade}")

    def update_equity(self, current_equity: float, initial_equity: float) -> None:
        """Update equity and calculate drawdown"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        if self.current_drawdown > self.alert_threshold_drawdown:
            self._generate_alert("DRAWDOWN", f"Drawdown exceeded threshold: {self.current_drawdown:.2%}")

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "total_pnl": 0,
                "sharpe_ratio": 0,
            }

        pnls = [t.get("pnl", 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

        # Simple Sharpe ratio approximation
        import numpy as np
        returns = np.diff(np.cumsum(pnls))
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 0 and np.std(returns) > 0 else 0

        return {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "sharpe_ratio": float(sharpe),
            "current_drawdown": self.current_drawdown,
        }

    def _generate_alert(self, alert_type: str, message: str) -> None:
        """Generate system alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": "HIGH" if alert_type in ["DRAWDOWN", "ERROR"] else "MEDIUM",
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")

    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return self.alerts[-limit:]

    def generate_session_report(self) -> Dict:
        """Generate complete session report"""
        metrics = self.get_metrics()
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600

        return {
            "session_duration_hours": session_duration,
            "start_time": self.session_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "performance_metrics": metrics,
            "total_alerts": len(self.alerts),
            "recent_alerts": self.get_alerts(5),
        }


class SystemMonitor:
    """Monitor system health and connectivity"""

    def __init__(self):
        self.connection_failures: List[Dict] = []
        self.data_errors: List[Dict] = []
        self.execution_errors: List[Dict] = []
        self.system_healthy = True

    def record_connection_error(self, error: str) -> None:
        """Record connection failure"""
        self.connection_failures.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
        })
        self.system_healthy = False
        logger.error(f"Connection error: {error}")

    def record_data_error(self, error: str) -> None:
        """Record data processing error"""
        self.data_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
        })
        logger.error(f"Data error: {error}")

    def record_execution_error(self, error: str) -> None:
        """Record execution error"""
        self.execution_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
        })
        logger.error(f"Execution error: {error}")

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "healthy": self.system_healthy,
            "connection_failures": len(self.connection_failures),
            "data_errors": len(self.data_errors),
            "execution_errors": len(self.execution_errors),
            "timestamp": datetime.now().isoformat(),
        }
