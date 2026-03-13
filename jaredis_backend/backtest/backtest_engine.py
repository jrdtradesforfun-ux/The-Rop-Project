"""Backtest engine for the Jaredis trading system.

This module provides a reusable backtest engine that simulates trades using:
- ML model signals (XGBoost / ensemble)
- Strategy signals (VWAP / momentum)
- Risk rules matching phase 1-4

The backtester is designed to be used with historical OHLCV data and can
be used as the core engine for phase-based performance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from jaredis_backend.trading_engine.advanced_risk_manager import AdvancedRiskManager, AccountMetrics, RiskLevel, RiskLimits
from jaredis_backend.ml_models.xgb_trainer import XGBoostTradingModel


@dataclass
class Trade:
    ticket: int
    entry_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    closed: bool = False
    reason: Optional[str] = None


@dataclass
class BacktestResult:
    balance_curve: pd.Series
    trades: List[Trade]
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    total_return: float
    metrics: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Backtesting engine that uses the same risk rules as the live bot."""

    def __init__(self,
                 model: XGBoostTradingModel,
                 starting_balance: float = 100.0,
                 phase: str = 'MICRO',
                 risk_limits: Optional[RiskLimits] = None,
                 confidence_threshold: float = 0.65,
                 max_hold_bars: int = 2):
        self.model = model
        self.starting_balance = starting_balance
        self.phase = phase
        self.confidence_threshold = confidence_threshold
        self.max_hold_bars = max_hold_bars

        if risk_limits is None:
            risk_limits = RiskLimits()
        self.risk_manager = AdvancedRiskManager(risk_limits)

        self.trades: List[Trade] = []
        self.balance_series: List[float] = []

    def _phase_limits(self) -> Dict:
        """Map phase to backtest limits (matching Phase 1-4 checklist)."""
        phase = self.phase.upper()
        if phase == 'MICRO':
            return {
                'max_positions': 1,
                'max_daily_trades': 1,
                'risk_per_trade': 0.02,
                'daily_loss_limit': 0.06,
                'kill_drawdown': 0.20,
                'max_hold_bars': 2
            }
        if phase == 'BUILDING':
            return {
                'max_positions': 3,
                'max_daily_trades': 5,
                'risk_per_trade': 0.015,
                'daily_loss_limit': 0.06,
                'kill_drawdown': 0.15,
                'max_hold_bars': 4
            }
        if phase == 'GROWING':
            return {
                'max_positions': 6,
                'max_daily_trades': 10,
                'risk_per_trade': 0.01,
                'daily_loss_limit': 0.06,
                'kill_drawdown': 0.10,
                'max_hold_bars': 8
            }
        # PROP
        return {
            'max_positions': 10,
            'max_daily_trades': 15,
            'risk_per_trade': 0.01,
            'daily_loss_limit': 0.06,
            'kill_drawdown': 0.05,
            'max_hold_bars': 12
        }

    def simulate(self, df: pd.DataFrame) -> BacktestResult:
        """Simulate trading over historical data."""
        df = df.copy()
        df = df.sort_index()
        df = df.dropna()

        params = self._phase_limits()
        balance = self.starting_balance
        start_balance = balance
        self.trades = []
        open_trades: List[Trade] = []

        # Balance series per bar
        balance_series = []
        daily_trades = 0
        daily_loss = 0.0
        current_day = None

        # Ensure model is trained
        if self.model.model is None:
            raise RuntimeError("Model is not trained. Cannot backtest.")

        # Model needs feature columns; prepare features once
        X, y, feature_cols = self.model.prepare_data(df)
        prediction_df = self.model.predict_with_confidence(X)
        prediction_df = prediction_df.reindex(df.index).fillna(0)

        for idx, row in df.iterrows():
            # Reset daily counters at new day
            day = idx.date()
            if current_day is None or day != current_day:
                current_day = day
                daily_trades = 0
                daily_loss = 0.0

            # Close trades if stop/target hit or max hold reached
            to_close = []
            for trade in open_trades:
                if trade.closed:
                    continue

                bar_high = row['high']
                bar_low = row['low']
                bar_close = row['close']
                direction = trade.direction

                stop_hit = False
                target_hit = False

                if direction == 'long':
                    if bar_low <= trade.stop_loss:
                        stop_hit = True
                    elif bar_high >= trade.take_profit:
                        target_hit = True
                else:
                    if bar_high >= trade.stop_loss:
                        stop_hit = True
                    elif bar_low <= trade.take_profit:
                        target_hit = True

                if stop_hit or target_hit:
                    trade.exit_time = idx
                    trade.exit_price = trade.stop_loss if stop_hit else trade.take_profit
                    trade.closed = True
                    trade.reason = 'stop' if stop_hit else 'target'
                    trade.pnl = (trade.exit_price - trade.entry_price) * (1 if direction == 'long' else -1) * trade.volume * 10000
                    balance += trade.pnl
                    daily_loss += min(trade.pnl, 0)
                    to_close.append(trade)
                    continue

                # Max hold bars
                if (idx - trade.entry_time).days >= 1 or (idx - trade.entry_time).seconds >= params['max_hold_bars'] * 60 * 15:
                    trade.exit_time = idx
                    trade.exit_price = bar_close
                    trade.closed = True
                    trade.reason = 'timeout'
                    trade.pnl = (trade.exit_price - trade.entry_price) * (1 if direction == 'long' else -1) * trade.volume * 10000
                    balance += trade.pnl
                    daily_loss += min(trade.pnl, 0)
                    to_close.append(trade)

            for closed in to_close:
                open_trades.remove(closed)
                self.trades.append(closed)

            # Kill-switch checks
            if daily_loss <= -start_balance * params['daily_loss_limit']:
                break
            if balance <= start_balance * (1 - params['kill_drawdown']):
                break

            # Skip entering new trades if max positions reached or daily trades reached
            if len(open_trades) >= params['max_positions'] or daily_trades >= params['max_daily_trades']:
                balance_series.append(balance)
                continue

            # Generate signal from model
            signal = prediction_df.loc[idx]
            if not signal.get('valid_signal', False):
                balance_series.append(balance)
                continue

            if signal['confidence'] < self.confidence_threshold:
                balance_series.append(balance)
                continue

            direction = 'long' if signal['prediction'] == 1 else 'short'

            # Determine stop/target based on phase (fixed percent)
            stop_pct = 0.005 if self.phase == 'MICRO' else 0.0075 if self.phase == 'BUILDING' else 0.01
            rr = 2.0 if self.phase == 'MICRO' else 1.5
            entry_price = row['close']
            stop_loss = entry_price * (1 - stop_pct) if direction == 'long' else entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 + stop_pct * rr) if direction == 'long' else entry_price * (1 - stop_pct * rr)

            # Determine position size using risk manager
            self.risk_manager.account_metrics = AccountMetrics(
                balance=balance,
                equity=balance,
                free_margin=balance,
                used_margin=0,
                margin_level=10,
                open_positions=len(open_trades),
                daily_pnl=daily_loss,
                total_pnl=balance - self.starting_balance
            )

            position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, risk_percent=params['risk_per_trade'])
            is_valid, reason = self.risk_manager.validate_trade(
                symbol='EURUSD',
                action=direction.upper(),
                volume=position_size,
                entry_price=entry_price,
                sl=stop_loss,
                tp=take_profit
            )

            if not is_valid:
                balance_series.append(balance)
                continue

            # Open trade
            ticket = len(self.trades) + len(open_trades) + 1
            trade = Trade(
                ticket=ticket,
                entry_time=idx,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=position_size
            )
            open_trades.append(trade)
            daily_trades += 1

            balance_series.append(balance)

        # Close remaining open trades at final close
        if open_trades:
            last_close = df['close'].iloc[-1]
            for trade in open_trades:
                trade.exit_time = df.index[-1]
                trade.exit_price = last_close
                trade.closed = True
                trade.reason = 'end'
                trade.pnl = (trade.exit_price - trade.entry_price) * (1 if trade.direction == 'long' else -1) * trade.volume * 10000
                balance += trade.pnl
                self.trades.append(trade)

        balance_series.append(balance)

        balance_series = pd.Series(balance_series, index=df.index[: len(balance_series)])
        returns = balance_series.pct_change().fillna(0)

        total_return = (balance / start_balance) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        max_dd = (balance_series / balance_series.cummax() - 1).min()

        wins = [t for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.trades if t.pnl and t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else float('inf')

        return BacktestResult(
            balance_curve=balance_series,
            trades=self.trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe=sharpe,
            max_drawdown=max_dd,
            total_return=total_return,
            metrics={
                'total_trades': len(self.trades),
                'wins': len(wins),
                'losses': len(losses)
            }
        )
