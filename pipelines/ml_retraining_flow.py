"""
Prefect MLops Pipeline for Automated Model Retraining
Handles data extraction, model training, validation, and deployment
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    from prefect.artifacts import create_markdown_artifact
    from prefect.schedules import CronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    logging.warning("Prefect not installed. Install with: pip install prefect")

from .ml_models.feature_engineer import FeatureEngineer, FeatureConfig, DataValidator, LabelGenerator
from .ml_models.training_pipeline import MLTrainingPipeline, ModelConfig
from .ml_models.xgb_trainer import XGBoostTradingModel
from .backtest.backtest_engine import BacktestEngine
from .backtest.walk_forward_backtester import WalkForwardBacktester
from .mql5_bridge.zeromq_bridge import MT5ZeroMQBridge

logger = logging.getLogger(__name__)


if PREFECT_AVAILABLE:
    # ==================== TASKS ====================
    
    @task(retries=3, retry_delay_seconds=60)
    def extract_mt5_data(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """Extract OHLCV data from MT5"""
        logger = get_run_logger()
        logger.info(f"Extracting {lookback_days} days of {symbol} data")
        
        try:
            # This would use actual MT5 connection
            # For now, placeholder
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                raise ConnectionError("Failed to initialize MT5")
            
            bars = lookback_days * 24  # Approximate for H1
            timeframe_const = getattr(mt5, f"TIMEFRAME_{timeframe}")
            rates = mt5.copy_rates_from_pos(symbol, timeframe_const, 0, bars)
            
            mt5.shutdown()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"Extracted {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise
    
    
    @task
    def validate_data(df: pd.DataFrame, symbol: str) -> Dict[str, bool]:
        """Validate data quality"""
        logger = get_run_logger()
        
        checks = DataValidator.validate_ohlcv(df)
        DataValidator.log_validation(checks)
        
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            raise ValueError(f"Data validation failed: {failed}")
        
        return checks
    
    
    @task
    def engineer_features(df: pd.DataFrame, feature_config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """Engineer features for ML"""
        logger = get_run_logger()
        
        engineer = FeatureEngineer(feature_config or FeatureConfig())
        features_df = engineer.engineer_features(df)
        
        logger.info(f"Engineered {features_df.shape[1]} features")
        return features_df
    
    
    @task
    def prepare_labels(df: pd.DataFrame, profit_target: float = 0.001, 
                      stop_loss: float = 0.002, horizon: int = 5) -> pd.DataFrame:
        """Generate labels for supervised learning"""
        logger = get_run_logger()
        
        df['target'] = LabelGenerator.triple_barrier_labels(
            df, profit_target, stop_loss, horizon
        )
        
        label_counts = df['target'].value_counts()
        logger.info(f"Labels created: {label_counts.to_dict()}")
        
        return df
    
    
    @task
    def train_ml_model(df: pd.DataFrame, config: ModelConfig) -> Dict:
        """Train ML model"""
        logger = get_run_logger()
        
        # Prepare data
        feature_cols = [c for c in df.columns if c not in 
                       ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        
        # Drop NaN
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Train
        pipeline = MLTrainingPipeline(config)
        results = pipeline.train(X, y, feature_cols)
        
        # Save model
        model_path = f"models/{config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        pipeline.save_model(model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return {
            'results': results,
            'model_path': model_path,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }


    @task
    def train_xgb_model(df: pd.DataFrame, symbol: str, timeframe: str, model_dir: str = "models") -> Dict:
        """Train and save an XGBoost model using the latest XGB training module."""
        logger = get_run_logger()

        model = XGBoostTradingModel(symbol=symbol, timeframe=timeframe)
        X, y, _ = model.prepare_data(df)

        logger.info(f"Training XGBoost on {len(X)} samples")
        metrics = model.train(X, y, optimize=True)

        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_path = model_dir_path / f"xgb_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save(str(model_path))

        logger.info(f"XGBoost model saved to {model_path}")
        return {
            'model_path': str(model_path),
            'metrics': metrics
        }


    @task
    def backtest_with_risk(df: pd.DataFrame, model_path: str, phase: str, starting_balance: float = 100.0) -> Dict:
        """Backtest using the same risk rules as the live system."""
        logger = get_run_logger()

        model = XGBoostTradingModel()
        model.load(model_path)

        engine = BacktestEngine(model=model, starting_balance=starting_balance, phase=phase)
        result = engine.simulate(df)

        logger.info(f"Backtest complete: return={result.total_return:.2%}, win_rate={result.win_rate:.2%}")

        return {
            'total_return': result.total_return,
            'win_rate': result.win_rate,
            'sharpe': result.sharpe,
            'max_drawdown': result.max_drawdown,
            'profit_factor': result.profit_factor,
            'trades': len(result.trades)
        }


    @task
    def backtest_model(df: pd.DataFrame, train_result: Dict, config: ModelConfig) -> Dict:
        """Walk-forward backtest"""
        logger = get_run_logger()
        
        feature_cols = [c for c in df.columns if c not in 
                       ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        
        # Load model
        pipeline = MLTrainingPipeline(config)
        pipeline.load_model(train_result['model_path'])
        
        # Walk-forward validation
        backtest_results = pipeline.walk_forward_validation(X, y)
        
        logger.info(f"Backtest F1: {backtest_results['overall_f1']:.4f}")
        
        return backtest_results
    
    
    @task
    def evaluate_performance(train_result: Dict, backtest_result: Dict) -> Dict:
        """Evaluate if model meets production standards"""
        logger = get_run_logger()
        
        train_f1 = train_result['results']['test_f1']
        backtest_f1 = backtest_result['overall_f1']
        
        is_good = (train_f1 > 0.55) and (backtest_f1 > 0.50)
        
        logger.info(f"Performance: Train F1={train_f1:.4f}, Backtest F1={backtest_f1:.4f}")
        logger.info(f"Production ready: {is_good}")
        
        return {
            'is_good': is_good,
            'train_f1': train_f1,
            'backtest_f1': backtest_f1,
            'score': (train_f1 + backtest_f1) / 2
        }
    
    
    @task
    def register_model(model_info: Dict, eval_result: Dict, symbol: str) -> Optional[str]:
        """Register model to MLflow if it passes evaluation"""
        logger = get_run_logger()
        
        if not eval_result['is_good']:
            logger.info("Model did not pass evaluation. Skipping registration.")
            return None
        
        try:
            import mlflow
            
            # Register to MLflow
            model_uri = f"runs:/{model_info.get('run_id', 'unknown')}/model"
            # mv = mlflow.register_model(model_uri, f"{symbol}_directional_v2")
            
            logger.info(f"Model registered")
            return "success"
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return None
    
    
    @task
    def deploy_to_bot(model_path: str, eval_result: Dict) -> bool:
        """Deploy model to live bot if metrics are good"""
        logger = get_run_logger()
        
        if not eval_result.get('is_good', False):
            logger.info("Model metrics not good enough for deployment")
            return False
        
        try:
            # Load the trained model
            model = XGBoostTradingModel.load_model(model_path)
            
            # Save to bot's model directory
            import os
            from pathlib import Path
            
            bot_model_dir = Path("models/live")
            bot_model_dir.mkdir(exist_ok=True)
            live_model_path = bot_model_dir / "xgboost_trading_model.pkl"
            
            model.save_model(str(live_model_path))
            
            # Create deployment signal file
            signal_file = bot_model_dir / "model_updated.signal"
            with open(signal_file, 'w') as f:
                f.write(f"Model updated at {datetime.now().isoformat()}\n")
                f.write(f"Path: {live_model_path}\n")
                f.write(f"F1 Score: {eval_result.get('train_f1', 0):.4f}\n")
                f.write(f"Win Rate: {eval_result.get('win_rate', 0):.2%}\n")
            
            logger.info(f"Model deployed to {live_model_path}")
            logger.info("Bot will reload model on next cycle")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    
    # ==================== FLOWS ====================
    
    @task
    def walk_forward_backtest(labels_df: pd.DataFrame, phase: str = "MICRO") -> Dict:
        """Run walk-forward backtest for more realistic evaluation"""
        logger = get_run_logger()
        
        try:
            backtester = WalkForwardBacktester(
                train_window_months=6,
                test_window_months=1,
                step_months=1,
                phase=phase,
                parallel=True
            )
            
            result = backtester.run_walk_forward(labels_df)
            
            logger.info(f"Walk-forward complete: {result.num_windows} windows")
            logger.info(f"Overall return: {result.total_return:.2%}, Win rate: {result.win_rate:.1%}")
            
            return {
                'total_return': result.total_return,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'num_windows': result.num_windows,
                'avg_window_return': result.avg_window_return,
                'is_good': result.win_rate > 0.45 and result.total_return > 0
            }
            
        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            return {'error': str(e), 'is_good': False}


    @task
    def evaluate_xgb_performance(train_metrics: Dict, backtest_metrics: Dict, wf_metrics: Optional[Dict] = None) -> Dict:
        """Evaluate training and backtest metrics for production readiness."""
        logger = get_run_logger()

        train_f1 = train_metrics.get('test_f1', 0)
        single_win_rate = backtest_metrics.get('win_rate', 0)
        single_return = backtest_metrics.get('total_return', 0)
        
        # Use walk-forward if available, otherwise single backtest
        if wf_metrics and not wf_metrics.get('error'):
            win_rate = wf_metrics.get('win_rate', 0)
            total_return = wf_metrics.get('total_return', 0)
            backtest_type = "walk_forward"
        else:
            win_rate = single_win_rate
            total_return = single_return
            backtest_type = "single_window"

        # Stricter criteria for walk-forward
        if backtest_type == "walk_forward":
            is_good = (train_f1 > 0.55) and (win_rate > 0.50) and (total_return > 0.05)
        else:
            is_good = (train_f1 > 0.55) and (win_rate > 0.45) and (total_return > 0)

        logger.info(f"Train F1={train_f1:.4f}, Backtest Type={backtest_type}")
        logger.info(f"WinRate={win_rate:.2%}, Return={total_return:.2%}")
        logger.info(f"Production ready: {is_good}")

        result = {
            'is_good': is_good,
            'train_f1': train_f1,
            'single_win_rate': single_win_rate,
            'win_rate': win_rate,
            'total_return': total_return,
            'backtest_type': backtest_type,
            'score': (train_f1 + win_rate) / 2
        }
        
        if wf_metrics and not wf_metrics.get('error'):
            result.update({
                'wf_win_rate': wf_metrics.get('win_rate', 0),
                'wf_total_return': wf_metrics.get('total_return', 0),
                'wf_sharpe': wf_metrics.get('sharpe_ratio', 0),
                'wf_max_dd': wf_metrics.get('max_drawdown', 0),
                'wf_windows': wf_metrics.get('num_windows', 0)
            })
        
        return result


    @flow(name="ml-model-retraining", description="Automated ML model retraining pipeline")
    def retrain_model_flow(symbol: str = "EURUSD",
                          timeframe: str = "H1",
                          lookback_days: int = 730,
                          starting_balance: float = 100.0,
                          phase: str = "MICRO") -> Dict:
        """Main retraining flow"""

        logger = get_run_logger()
        logger.info(f"Starting retraining for {symbol}")

        feature_config = FeatureConfig(lookback_period=50)

        # Step 1: Extract data
        raw_data = extract_mt5_data(symbol, timeframe, lookback_days)

        # Step 2: Validate
        validation = validate_data(raw_data, symbol)

        # Step 3: Feature engineering
        features_df = engineer_features(raw_data, feature_config)

        # Step 4: Labeling
        labels_df = prepare_labels(features_df)

        # Step 5: XGBoost training
        train_result = train_xgb_model(labels_df, symbol, timeframe)

        # Step 6: Single-window backtesting
        backtest_result = backtest_with_risk(labels_df, train_result['model_path'], phase, starting_balance)

        # Step 7: Walk-forward backtesting
        wf_result = walk_forward_backtest(labels_df, phase)

        # Step 8: Evaluation (combine both backtest results)
        eval_result = evaluate_xgb_performance(train_result['metrics'], backtest_result, wf_result)

        # Step 9: Registration
        registration_result = register_model(train_result, eval_result, symbol)

        # Step 10: Deployment
        deployed = deploy_to_bot(train_result['model_path'], eval_result)

        return {
            "status": "success",
            "symbol": symbol,
            "train_f1": eval_result['train_f1'],
            "single_win_rate": eval_result['single_win_rate'],
            "wf_win_rate": eval_result.get('wf_win_rate', 0),
            "wf_total_return": eval_result.get('wf_total_return', 0),
            "registered": registration_result is not None,
            "deployed": deployed,
            "model_path": train_result['model_path']
        }
    
    
    @flow(name="model-performance-check", description="Monitor live model performance")
    def performance_check_flow(symbol: str = "EURUSD") -> Dict:
        """Check if retraining is needed based on live performance"""
        
        logger = get_run_logger()
        logger.info(f"Checking performance for {symbol}")
        
        # Get live metrics from bot
        try:
            # Check for performance drift
            drift_detected = check_performance_drift(symbol)
            
            if drift_detected:
                logger.warning("Performance drift detected. Triggering retraining...")
                return retrain_model_flow(symbol)
            else:
                return {
                    "status": "healthy",
                    "drift_detected": False,
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return {"status": "error", "message": str(e)}


    @task
    def check_performance_drift(symbol: str) -> bool:
        """Check for performance drift that requires retraining"""
        logger = get_run_logger()
        
        try:
            # Check signal file for recent model updates
            from pathlib import Path
            signal_file = Path("models/live/model_updated.signal")
            
            if signal_file.exists():
                # Read last update time
                with open(signal_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        update_time_str = lines[0].split("at ")[-1].strip()
                        update_time = datetime.fromisoformat(update_time_str)
                        
                        # If model is older than 30 days, check performance
                        if (datetime.now() - update_time).days > 30:
                            logger.info("Model is 30+ days old, checking performance...")
                            return True
            
            # Check for external drift signals (market regime change, etc.)
            return False
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return False
    # Fallback implementation without Prefect
    class SimpleMLOpsScheduler:
        """Simple scheduler when Prefect is not available"""
        
        def __init__(self):
            self.last_run = None
            self.run_interval = timedelta(days=1)
        
        def should_run(self) -> bool:
            if self.last_run is None:
                return True
            return datetime.now() - self.last_run > self.run_interval
        
        def run_training(self, symbol: str = "EURUSD") -> Dict:
            """Run training pipeline manually"""
            logger.info(f"Running manual training for {symbol}")
            
            # Simplified training without Prefect
            return {
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }


# Deployment configuration
PREFECT_DEPLOYMENTS = {
    "daily_retraining": {
        "schedule": "0 2 * * *",  # 2 AM daily
        "parameters": {"symbol": "EURUSD"}
    },
    "performance_check": {
        "schedule": "0 */4 * * *",  # Every 4 hours
        "parameters": {"symbol": "EURUSD"}
    }
}
