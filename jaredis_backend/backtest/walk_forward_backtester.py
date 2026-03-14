"""Walk-forward backtester for realistic ML model evaluation.

This module implements a walk-forward optimization backtester that:
- Trains models on historical data windows
- Tests on future out-of-sample data
- Rolls forward through time
- Provides more realistic performance estimates than single-window backtests

Key features:
- Rolling training windows (e.g., 6-12 months)
- Forward testing periods (e.g., 1-3 months)
- Model retraining at regular intervals
- Performance aggregation across all test periods
- Risk management matching live phases
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

from jaredis_backend.backtest.backtest_engine import BacktestEngine, BacktestResult
from jaredis_backend.ml_models.xgb_trainer import XGBoostTradingModel
from jaredis_backend.trading_engine.advanced_risk_manager import RiskLimits

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    model_path: Optional[str] = None
    backtest_result: Optional[BacktestResult] = None


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward analysis."""
    windows: List[WalkForwardWindow]
    overall_balance_curve: pd.Series
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    num_windows: int
    avg_window_return: float
    window_returns: List[float]
    metrics: Dict[str, float] = field(default_factory=dict)


class WalkForwardBacktester:
    """
    Walk-forward backtester for realistic ML model evaluation.
    
    Simulates real-world deployment where models are retrained periodically
    and evaluated on future unseen data.
    """

    def __init__(
        self,
        train_window_months: int = 6,
        test_window_months: int = 1,
        step_months: int = 1,
        starting_balance: float = 100.0,
        phase: str = 'MICRO',
        confidence_threshold: float = 0.65,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize walk-forward backtester.
        
        Args:
            train_window_months: Months of data for training each model
            test_window_months: Months of data for testing each model
            step_months: Months to advance between windows
            starting_balance: Starting account balance
            phase: Trading phase (MICRO, BUILDING, GROWING, PROP)
            confidence_threshold: Minimum confidence for trades
            parallel: Whether to run windows in parallel
            max_workers: Max parallel workers (None = CPU count)
        """
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.starting_balance = starting_balance
        self.phase = phase
        self.confidence_threshold = confidence_threshold
        self.parallel = parallel
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Will be set during run
        self.base_model = None
        self.windows: List[WalkForwardWindow] = []

    def _generate_windows(self, df: pd.DataFrame) -> List[WalkForwardWindow]:
        """Generate walk-forward windows from data."""
        df = df.sort_index()
        start_date = df.index[0]
        end_date = df.index[-1]
        
        windows = []
        current_train_end = start_date + pd.DateOffset(months=self.train_window_months)
        
        while current_train_end + pd.DateOffset(months=self.test_window_months) <= end_date:
            train_start = current_train_end - pd.DateOffset(months=self.train_window_months)
            test_end = current_train_end + pd.DateOffset(months=self.test_window_months)
            
            window = WalkForwardWindow(
                train_start=train_start.to_pydatetime(),
                train_end=current_train_end.to_pydatetime(),
                test_start=current_train_end.to_pydatetime(),
                test_end=test_end.to_pydatetime()
            )
            windows.append(window)
            
            # Advance to next window
            current_train_end += pd.DateOffset(months=self.step_months)
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def _train_and_test_window(self, window: WalkForwardWindow, df: pd.DataFrame) -> WalkForwardWindow:
        """Train model on window and test on future data."""
        try:
            # Extract training data
            train_mask = (df.index >= window.train_start) & (df.index <= window.train_end)
            train_df = df[train_mask].copy()
            
            if len(train_df) < 100:  # Minimum training samples
                logger.warning(f"Insufficient training data for window {window.train_start}")
                return window
            
            # Train model
            model = XGBoostTradingModel()
            model.train(train_df)
            
            # Save model temporarily (in production, save to disk)
            window.model_path = f"temp_model_{window.train_start.strftime('%Y%m%d')}.pkl"
            model.save_model(window.model_path)
            
            # Extract test data
            test_mask = (df.index >= window.test_start) & (df.index <= window.test_end)
            test_df = df[test_mask].copy()
            
            if len(test_df) < 10:  # Minimum test samples
                logger.warning(f"Insufficient test data for window {window.test_start}")
                return window
            
            # Run backtest
            backtester = BacktestEngine(
                model=model,
                starting_balance=self.starting_balance,
                phase=self.phase,
                confidence_threshold=self.confidence_threshold
            )
            
            result = backtester.simulate(test_df)
            window.backtest_result = result
            
            logger.info(f"Window {window.train_start.date()}: Return={result.total_return:.2%}, "
                       f"WinRate={result.win_rate:.1%}, Trades={len(result.trades)}")
            
        except Exception as e:
            logger.error(f"Error processing window {window.train_start}: {e}")
        
        return window

    def run_walk_forward(self, df: pd.DataFrame) -> WalkForwardResult:
        """
        Run complete walk-forward analysis.
        
        Args:
            df: Historical OHLCV data with features
            
        Returns:
            Aggregated walk-forward results
        """
        logger.info("Starting walk-forward backtest...")
        logger.info(f"Train window: {self.train_window_months} months, "
                   f"Test window: {self.test_window_months} months, "
                   f"Step: {self.step_months} months")
        
        # Generate windows
        self.windows = self._generate_windows(df)
        if not self.windows:
            raise ValueError("No valid windows generated from data")
        
        # Process windows
        if self.parallel and len(self.windows) > 1:
            logger.info(f"Running {len(self.windows)} windows in parallel...")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._train_and_test_window, window, df) 
                          for window in self.windows]
                
                completed_windows = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    completed_windows.append(future.result())
                
                self.windows = sorted(completed_windows, key=lambda w: w.test_start)
        else:
            logger.info(f"Running {len(self.windows)} windows sequentially...")
            for window in tqdm(self.windows):
                self._train_and_test_window(window, df)
        
        # Aggregate results
        return self._aggregate_results()

    def _aggregate_results(self) -> WalkForwardResult:
        """Aggregate results across all windows."""
        if not self.windows:
            raise ValueError("No windows to aggregate")
        
        # Collect all balance curves
        all_curves = []
        window_returns = []
        
        for window in self.windows:
            if window.backtest_result and window.backtest_result.balance_curve is not None:
                curve = window.backtest_result.balance_curve.copy()
                # Normalize to starting balance for each window
                curve = curve / curve.iloc[0] * self.starting_balance
                all_curves.append(curve)
                window_returns.append(window.backtest_result.total_return)
        
        if not all_curves:
            raise ValueError("No valid backtest results to aggregate")
        
        # Combine balance curves (simple concatenation)
        # In production, might want more sophisticated stitching
        combined_curve = pd.concat(all_curves)
        combined_curve = combined_curve.sort_index()
        
        # Calculate overall metrics
        total_return = (combined_curve.iloc[-1] / combined_curve.iloc[0]) - 1
        
        # Annualize return (assuming daily data)
        days = (combined_curve.index[-1] - combined_curve.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio
        returns = combined_curve.pct_change().fillna(0)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        peak = combined_curve.expanding().max()
        drawdown = (combined_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else float('inf')
        
        # Win rate and profit factor across all windows
        all_trades = []
        for window in self.windows:
            if window.backtest_result:
                all_trades.extend(window.backtest_result.trades)
        
        if all_trades:
            wins = [t for t in all_trades if t.pnl and t.pnl > 0]
            losses = [t for t in all_trades if t.pnl and t.pnl <= 0]
            win_rate = len(wins) / len(all_trades)
            profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        result = WalkForwardResult(
            windows=self.windows,
            overall_balance_curve=combined_curve,
            total_return=total_return,
            annualized_return=annualized_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            num_windows=len(self.windows),
            avg_window_return=np.mean(window_returns) if window_returns else 0,
            window_returns=window_returns,
            metrics={
                'total_windows': len(self.windows),
                'successful_windows': len([w for w in self.windows if w.backtest_result]),
                'avg_trades_per_window': len(all_trades) / len(self.windows) if self.windows else 0,
                'data_period_days': (combined_curve.index[-1] - combined_curve.index[0]).days,
                'avg_window_length_days': np.mean([
                    (w.test_end - w.test_start).days for w in self.windows
                ])
            }
        )
        
        logger.info(f"Walk-forward complete: {result.num_windows} windows, "
                   f"Total Return={result.total_return:.2%}, "
                   f"Annualized={result.annualized_return:.2%}, "
                   f"Win Rate={result.win_rate:.1%}, "
                   f"Max DD={result.max_drawdown:.2%}")
        
        return result

    def get_window_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all windows."""
        rows = []
        for i, window in enumerate(self.windows):
            row = {
                'window_id': i,
                'train_start': window.train_start,
                'train_end': window.train_end,
                'test_start': window.test_start,
                'test_end': window.test_end,
                'train_days': (window.train_end - window.train_start).days,
                'test_days': (window.test_end - window.test_start).days,
            }
            
            if window.backtest_result:
                result = window.backtest_result
                row.update({
                    'total_return': result.total_return,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'sharpe': result.sharpe,
                    'max_drawdown': result.max_drawdown,
                    'num_trades': len(result.trades),
                    'final_balance': result.balance_curve.iloc[-1] if result.balance_curve is not None else None
                })
            else:
                row.update({k: None for k in ['total_return', 'win_rate', 'profit_factor', 
                                             'sharpe', 'max_drawdown', 'num_trades', 'final_balance']})
            
            rows.append(row)
        
        return pd.DataFrame(rows)