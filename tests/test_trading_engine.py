"""
Test Trading Engine
"""

import pytest
from jaredis_backend.trading_engine import TradingEngine, PositionManager, RiskManager


class TestTradingEngine:
    """Tests for TradingEngine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = TradingEngine(account_size=10000, risk_per_trade=0.02)

    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        assert self.engine.account_size == 10000
        assert self.engine.risk_per_trade == 0.02
        assert len(self.engine.trade_history) == 0

    def test_portfolio_status(self):
        """Test portfolio status calculation"""
        status = self.engine.get_portfolio_status()
        assert status["account_size"] == 10000
        assert status["total_equity"] == 0


class TestPositionManager:
    """Tests for PositionManager"""

    def setup_method(self):
        """Setup test fixtures"""
        self.pm = PositionManager()

    def test_open_position(self):
        """Test opening a position"""
        position = self.pm.open_position(
            symbol="EURUSD",
            direction="long",
            size=1.0,
            entry_price=1.0950,
            stop_loss=1.0900,
            take_profit=1.1000
        )

        assert position["symbol"] == "EURUSD"
        assert position["direction"] == "long"
        assert position["status"] == "open"

    def test_close_position(self):
        """Test closing a position"""
        position = self.pm.open_position(
            symbol="EURUSD",
            direction="long",
            size=1.0,
            entry_price=1.0950,
            stop_loss=1.0900,
            take_profit=1.1000
        )

        pos_id = position["id"]
        closed = self.pm.close_position(pos_id, exit_price=1.1000)

        assert closed["status"] == "closed"
        assert closed["pnl"] > 0  # Profit since exit > entry
        assert len(self.pm.get_open_positions()) == 0


class TestRiskManager:
    """Tests for RiskManager"""

    def setup_method(self):
        """Setup test fixtures"""
        self.rm = RiskManager(account_size=10000)

    def test_risk_validation(self):
        """Test risk validation"""
        signal = {
            "direction": "buy",
            "entry_price": 1.0950,
            "stop_loss": 1.0900,
            "risk_amount": 200,  # 2% of 10000
        }

        assert self.rm.validate_trade(signal) is True

    def test_excessive_risk_rejection(self):
        """Test rejection of excessive risk"""
        self.rm.open_position_count = 5  # Max positions

        signal = {
            "direction": "buy",
            "risk_amount": 100,
        }

        assert self.rm.validate_trade(signal) is False

    def test_position_size_calculation(self):
        """Test position sizing"""
        size = self.rm.calculate_position_size(
            entry_price=1.0950,
            stop_loss=1.0900,
            risk_percent=0.02
        )

        assert size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
