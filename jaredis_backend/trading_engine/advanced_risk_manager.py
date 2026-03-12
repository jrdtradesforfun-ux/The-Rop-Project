"""
Advanced Risk Management System
Implements professional-grade risk controls and position management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk alert levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class RiskLimits:
    """Risk management parameters"""
    # Position sizing
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_position_size: float = 1.0  # 1 lot max
    min_position_size: float = 0.01
    
    # Portfolio limits
    max_open_positions: int = 5
    max_positions_same_symbol: int = 2
    max_correlation: float = 0.8
    
    # Account limits
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_drawdown: float = 0.10  # 10% maximum drawdown
    min_margin_level: float = 1.5  # 150% minimum margin level
    
    # Trade filters
    min_win_rate: float = 0.45  # Minimum expected win rate
    min_profit_factor: float = 1.2  # Min profit factor
    
    # Time-based limits
    max_trades_per_day: int = 20
    trading_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour) in UTC
    halt_on_news: bool = True


@dataclass
class AccountMetrics:
    """Real-time account metrics"""
    balance: float
    equity: float
    free_margin: float
    used_margin: float
    margin_level: float
    open_positions: int
    daily_pnl: float
    total_pnl: float
    update_time: datetime = None
    
    def __post_init__(self):
        if self.update_time is None:
            self.update_time = datetime.now()


class AdvancedRiskManager:
    """Production-grade risk management system"""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager
        
        Args:
            limits: RiskLimits configuration
        """
        self.limits = limits or RiskLimits()
        self.account_metrics = None
        self.trade_history: List[Dict] = []
        self.daily_trades = []
        self.position_tracking = {}
        self.risk_alerts = []
        self.is_trading_allowed = True
        self.halt_reason = None
    
    def update_account(self, account_info: Dict) -> Optional[AccountMetrics]:
        """
        Update account metrics
        
        Args:
            account_info: Account data from broker
            
        Returns:
            Updated metrics
        """
        metrics = AccountMetrics(
            balance=account_info.get('balance', 0),
            equity=account_info.get('equity', 0),
            free_margin=account_info.get('margin_free', 0),
            used_margin=account_info.get('margin', 0),
            margin_level=account_info.get('margin_level', 0),
            open_positions=account_info.get('open_positions', 0),
            daily_pnl=account_info.get('daily_pnl', 0),
            total_pnl=account_info.get('total_pnl', 0)
        )
        
        self.account_metrics = metrics
        self._check_risk_limits()
        
        return metrics
    
    def validate_trade(self, symbol: str, action: str, volume: float,
                      entry_price: float, sl: float, tp: float) -> Tuple[bool, Optional[str]]:
        """
        Validate trade against risk limits
        
        Args:
            symbol: Trading symbol
            action: BUY or SELL
            volume: Position size
            entry_price: Entry price
            sl: Stop loss price
            tp: Take profit price
            
        Returns:
            (is_valid, error_message)
        """
        if not self.is_trading_allowed:
            return False, f"Trading halted: {self.halt_reason}"
        
        # Check 1: Position size limits
        if volume < self.limits.min_position_size:
            return False, f"Position too small: {volume} < {self.limits.min_position_size}"
        
        if volume > self.limits.max_position_size:
            return False, f"Position too large: {volume} > {self.limits.max_position_size}"
        
        # Check 2: Open positions limit
        if self.account_metrics and self.account_metrics.open_positions >= self.limits.max_open_positions:
            return False, f"Max open positions reached: {self.limits.max_open_positions}"
        
        # Check 3: Same symbol positions
        same_symbol_positions = sum(1 for p in self.position_tracking.values() 
                                   if p.get('symbol') == symbol)
        if same_symbol_positions >= self.limits.max_positions_same_symbol:
            return False, f"Too many {symbol} positions: {same_symbol_positions}"
        
        # Check 4: Risk per trade
        risk_amount = self._calculate_risk(volume, entry_price, sl)
        risk_percent = risk_amount / self.account_metrics.balance if self.account_metrics else 0
        
        if risk_percent > self.limits.max_risk_per_trade:
            return False, f"Risk too high: {risk_percent:.2%} > {self.limits.max_risk_per_trade:.2%}"
        
        # Check 5: Daily loss limit
        if self.account_metrics and self.account_metrics.daily_pnl < -self.account_metrics.balance * self.limits.max_daily_loss:
            return False, f"Daily loss limit reached: {self.account_metrics.daily_pnl:.2f}"
        
        # Check 6: Margin level
        if self.account_metrics and self.account_metrics.margin_level < self.limits.min_margin_level:
            return False, f"Insufficient margin: {self.account_metrics.margin_level:.2f}"
        
        # Check 7: Daily trade count
        trades_today = [t for t in self.daily_trades 
                       if (datetime.now() - t['time']).days == 0]
        if len(trades_today) >= self.limits.max_trades_per_day:
            return False, f"Max daily trades reached: {self.limits.max_trades_per_day}"
        
        # Check 8: Trading hours
        if self.limits.trading_hours:
            current_hour = datetime.utcnow().hour
            start, end = self.limits.trading_hours
            if not (start <= current_hour < end):
                return False, f"Trading outside allowed hours: {start}-{end} UTC"
        
        return True, None
    
    def calculate_position_size(self, entry_price: float, sl: float,
                               risk_percent: Optional[float] = None) -> float:
        """
        Calculate optimal position size using risk management
        
        Args:
            entry_price: Entry price
            sl: Stop loss price
            risk_percent: Risk percentage (default from limits)
            
        Returns:
            Position size
        """
        if not self.account_metrics:
            return 0
        
        risk_pct = risk_percent or self.limits.max_risk_per_trade
        risk_amount = self.account_metrics.balance * risk_pct
        
        # Risk = Position Size * (Entry - SL) * Pip Value
        pips_at_risk = abs(entry_price - sl) * 10000  # For 4-decimal pairs
        
        position_size = risk_amount / pips_at_risk
        
        # Clamp to limits
        position_size = np.clip(position_size, 
                               self.limits.min_position_size,
                               self.limits.max_position_size)
        
        return round(position_size, 2)
    
    def calculate_kelly_criterion(self, win_rate: float, profit_factor: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Historical win rate (0-1)
            profit_factor: Average profit / average loss
            
        Returns:
            Fraction of bankroll to risk (0-1)
        """
        if profit_factor <= 1:
            return 0
        
        # Kelly% = (win_rate * profit_factor - (1 - win_rate)) / profit_factor
        kelly = (win_rate * profit_factor - (1 - win_rate)) / profit_factor
        
        # Apply kelly fraction (typically 0.25 for safety)
        kelly_fraction = 0.25
        kelly_pos = kelly * kelly_fraction
        
        return np.clip(kelly_pos, 0, self.limits.max_risk_per_trade)
    
    def add_trade(self, ticket: int, symbol: str, action: str, 
                  volume: float, entry_price: float, sl: float, tp: float) -> None:
        """Record executed trade"""
        trade = {
            'ticket': ticket,
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'time': datetime.now(),
            'status': 'open'
        }
        
        self.position_tracking[ticket] = trade
        self.trade_history.append(trade)
        self.daily_trades.append(trade)
        
        logger.info(f"Trade recorded: {symbol} {action} {volume} @ {entry_price}")
    
    def close_trade(self, ticket: int, close_price: float, pnl: float) -> None:
        """Record closed trade"""
        if ticket in self.position_tracking:
            self.position_tracking[ticket]['status'] = 'closed'
            self.position_tracking[ticket]['close_price'] = close_price
            self.position_tracking[ticket]['pnl'] = pnl
            self.position_tracking[ticket]['close_time'] = datetime.now()
            
            logger.info(f"Trade closed: Ticket{ticket} PnL={pnl:.2f}")
    
    def _check_risk_limits(self) -> None:
        """Check account-level risk limits and enforce trading halt if needed"""
        if not self.account_metrics:
            return
        
        alerts = []
        
        # Drawdown check
        initial_balance = self.account_metrics.balance
        current_equity = self.account_metrics.equity
        drawdown = 1 - (current_equity / initial_balance) if initial_balance > 0 else 0
        
        if drawdown > self.limits.max_drawdown:
            self._set_trading_halt(f"Maximum drawdown exceeded: {drawdown:.2%}")
            alerts.append((RiskLevel.HALT, f"Drawdown: {drawdown:.2%}"))
        elif drawdown > self.limits.max_drawdown * 0.8:
            alerts.append((RiskLevel.CRITICAL, f"Drawdown critical: {drawdown:.2%}"))
        
        # Daily loss check
        if self.account_metrics.daily_pnl < -initial_balance * self.limits.max_daily_loss:
            self._set_trading_halt("Daily loss limit exceeded")
            alerts.append((RiskLevel.HALT, f"Daily loss: {self.account_metrics.daily_pnl:.2f}"))
        
        # Margin check
        if self.account_metrics.margin_level < self.limits.min_margin_level:
            self._set_trading_halt("Insufficient margin")
            alerts.append((RiskLevel.HALT, f"Margin level: {self.account_metrics.margin_level:.2f}"))
        elif self.account_metrics.margin_level < self.limits.min_margin_level * 1.2:
            alerts.append((RiskLevel.CRITICAL, f"Margin low: {self.account_metrics.margin_level:.2f}"))
        
        self.risk_alerts = alerts
    
    def _set_trading_halt(self, reason: str) -> None:
        """Halt trading"""
        self.is_trading_allowed = False
        self.halt_reason = reason
        logger.critical(f"⛔ TRADING HALTED: {reason}")
    
    def _calculate_risk(self, volume: float, entry_price: float, stop_loss: float) -> float:
        """Calculate risk amount in base currency"""
        pips_at_risk = abs(entry_price - stop_loss) * 10000
        risk = volume * pips_at_risk * 0.0001  # Pip value for standard pairs
        return risk
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary"""
        return {
            'trading_allowed': self.is_trading_allowed,
            'halt_reason': self.halt_reason,
            'account_metrics': self.account_metrics.__dict__ if self.account_metrics else None,
            'risk_alerts': [(level.value, msg) for level, msg in self.risk_alerts],
            'open_positions': len(self.position_tracking),
            'daily_trades': len(self.daily_trades),
            'total_trades': len(self.trade_history)
        }


class CorrelationMonitor:
    """Monitor correlation between open positions"""
    
    def __init__(self, correlation_limit: float = 0.8):
        self.correlation_limit = correlation_limit
        self.price_data: Dict[str, pd.Series] = {}
    
    def add_symbol_data(self, symbol: str, prices: np.ndarray) -> None:
        """Add price data for symbol"""
        self.price_data[symbol] = pd.Series(prices)
    
    def check_correlation(self, new_symbol: str, existing_symbols: List[str]) -> Tuple[bool, Optional[float]]:
        """
        Check if new symbol correlates too much with existing positions
        
        Returns:
            (is_allowed, max_correlation_found)
        """
        if new_symbol not in self.price_data:
            return True, None
        
        max_corr = 0
        for symbol in existing_symbols:
            if symbol in self.price_data:
                corr = self.price_data[new_symbol].corr(self.price_data[symbol])
                max_corr = max(max_corr, abs(corr))
        
        is_allowed = max_corr <= self.correlation_limit
        return is_allowed, max_corr if max_corr > 0 else None
