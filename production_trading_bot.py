"""Production-Ready Trading Bot with Enhanced Features

Demonstrates the three key production enhancements:
1. Automated backups with cloud storage
2. Sophisticated multi-channel alerting (Telegram, SMS, Email)
3. GPU acceleration for faster ML training

Usage:
    python production_trading_bot.py --gpu --backup --alerts
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Import our enhanced modules
from jaredis_backend.backup.backup_manager import BackupManager, ScheduledBackup, BackupConfig
from jaredis_backend.monitoring.alerts import (
    AlertManager, AlertRule, AlertLevel, TelegramAlertHandler,
    SMSAlertHandler, MultiChannelAlertHandler, AlertEscalationManager
)
from jaredis_backend.ml_models.xgb_trainer import XGBoostTradingModel, LSTMTradingModel
from jaredis_backend.ml_models.gpu_utils import enable_gpu_acceleration, get_optimal_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTradingBot:
    """Production-ready trading bot with all enhancements"""

    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "M15",
        use_gpu: bool = True,
        enable_backups: bool = True,
        enable_alerts: bool = True
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_gpu = use_gpu
        self.enable_backups = enable_backups
        self.enable_alerts = enable_alerts

        # Initialize components
        self.backup_manager: Optional[BackupManager] = None
        self.scheduled_backup: Optional[ScheduledBackup] = None
        self.alert_manager: Optional[AlertManager] = None
        self.alert_escalation: Optional[AlertEscalationManager] = None

        self.xgb_model = XGBoostTradingModel(symbol=symbol, timeframe=timeframe)
        self.lstm_model = LSTMTradingModel()
        self.lstm_model.symbol = symbol  # Set attributes manually

        logger.info(f"Initialized ProductionTradingBot for {symbol} {timeframe}")

    async def initialize_production_features(self):
        """Initialize all production features"""

        # 1. Enable GPU acceleration
        if self.use_gpu:
            logger.info("Enabling GPU acceleration...")
            gpu_manager = enable_gpu_acceleration()
            if gpu_manager.gpu_available:
                logger.info("✅ GPU acceleration enabled")
            else:
                logger.warning("❌ GPU not available, using CPU")

        # 2. Setup automated backups
        if self.enable_backups:
            logger.info("Setting up automated backups...")
            backup_config = BackupConfig(
                backup_dir="backups",
                retention_days=30,
                cloud_storage={
                    'aws': {
                        'bucket': 'jaredis-trading-backups',
                        'region': 'us-east-1'
                        # Add credentials in production
                    }
                }
            )
            self.backup_manager = BackupManager(backup_config)
            self.scheduled_backup = ScheduledBackup(self.backup_manager)

            # Create initial backup
            await self.backup_manager.create_backup("config")
            logger.info("✅ Automated backups configured")

        # 3. Setup sophisticated alerting
        if self.enable_alerts:
            logger.info("Setting up multi-channel alerting...")
            self.alert_manager = AlertManager()

            # Add alert handlers
            telegram_handler = TelegramAlertHandler(
                token="YOUR_TELEGRAM_BOT_TOKEN",
                chat_id="YOUR_TELEGRAM_CHAT_ID"
            )

            sms_handler = SMSAlertHandler(
                account_sid="YOUR_TWILIO_ACCOUNT_SID",
                auth_token="YOUR_TWILIO_AUTH_TOKEN",
                from_number="+1234567890",
                to_numbers=["+0987654321"]  # Add your phone numbers
            )

            # Multi-channel handler
            multi_handler = MultiChannelAlertHandler(
                telegram_handler=telegram_handler,
                sms_handler=sms_handler
            )

            self.alert_manager.add_handler(multi_handler)

            # Add alert rules
            self._setup_alert_rules()

            # Setup escalation
            self.alert_escalation = AlertEscalationManager(self.alert_manager)

            logger.info("✅ Multi-channel alerting configured")

    def _setup_alert_rules(self):
        """Setup comprehensive alert rules"""
        rules = [
            AlertRule(
                name="High Drawdown",
                metric="drawdown",
                condition=lambda x: x > 0.05,  # 5%
                level=AlertLevel.WARNING,
                message_template="⚠️ Drawdown alert: {value:.2%}"
            ),
            AlertRule(
                name="Critical Drawdown",
                metric="drawdown",
                condition=lambda x: x > 0.10,  # 10%
                level=AlertLevel.CRITICAL,
                message_template="🚨 CRITICAL drawdown: {value:.2%}"
            ),
            AlertRule(
                name="MT5 Connection Lost",
                metric="mt5_connected",
                condition=lambda x: x == 0,
                level=AlertLevel.CRITICAL,
                message_template="🔌 MT5 connection lost!"
            ),
            AlertRule(
                name="Model Accuracy Drop",
                metric="model_accuracy",
                condition=lambda x: x < 0.6,
                level=AlertLevel.ERROR,
                message_template="📉 Model accuracy dropped: {value:.2%}"
            ),
            AlertRule(
                name="High Latency",
                metric="execution_latency_ms",
                condition=lambda x: x > 1000,
                level=AlertLevel.WARNING,
                message_template="🐌 High execution latency: {value:.0f}ms"
            )
        ]

        for rule in rules:
            self.alert_manager.add_rule(rule)

    async def train_models_with_gpu(self, df) -> Dict:
        """Train ML models with GPU acceleration"""
        logger.info("Training models with GPU acceleration...")

        # Prepare data
        X, y, features = self.xgb_model.prepare_data(df)
        self.xgb_model.feature_names = features

        # Train XGBoost with GPU
        logger.info("Training XGBoost model...")
        xgb_metrics = self.xgb_model.train(X, y, use_gpu=self.use_gpu)

        # Save XGBoost model
        xgb_path = f"models/xgb_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.xgb_model.save(xgb_path)
        logger.info(f"XGBoost model saved to {xgb_path}")

        # Train LSTM with GPU (skip if TensorFlow not available)
        logger.info("Training LSTM model...")
        try:
            lstm_metrics = self.lstm_model.train(X, y)
            # Save LSTM model if available
            lstm_path = f"models/lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            # Assuming lstm_model has save method
            if hasattr(self.lstm_model, 'save'):
                self.lstm_model.save(lstm_path)
                logger.info(f"LSTM model saved to {lstm_path}")
        except RuntimeError as e:
            if "TensorFlow" in str(e):
                logger.warning("LSTM training skipped - TensorFlow not available")
                lstm_metrics = {'test_accuracy': 0, 'test_f1': 0}
            else:
                raise

        # Send training completion alert
        if self.alert_manager:
            await self.alert_manager._trigger_alert(
                AlertRule(
                    name="Model Training Complete",
                    metric="training_status",
                    condition=lambda x: x == 1,
                    level=AlertLevel.INFO,
                    message_template="✅ Model training completed successfully"
                ),
                1
            )

        return {
            'xgb_metrics': xgb_metrics,
            'lstm_metrics': lstm_metrics,
            'gpu_used': self.use_gpu
        }

    async def start_production_operations(self):
        """Start all production operations"""
        logger.info("Starting production operations...")

        # Start scheduled backups
        if self.scheduled_backup:
            asyncio.create_task(
                self.scheduled_backup.start_schedule(
                    full_backup_interval_hours=24,
                    model_backup_interval_hours=6,
                    config_backup_interval_hours=12
                )
            )

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

        logger.info("✅ All production operations started")

    async def _monitoring_loop(self):
        """Continuous monitoring and alerting"""
        while True:
            try:
                # Simulate metrics collection (replace with real MT5 data)
                metrics = {
                    'drawdown': 0.02,  # 2%
                    'mt5_connected': 1,
                    'model_accuracy': 0.85,
                    'execution_latency_ms': 150
                }

                # Evaluate alerts
                if self.alert_manager:
                    self.alert_manager.evaluate_rules(metrics)

                    # Check escalation
                    if self.alert_escalation:
                        # This would be called when alerts are triggered
                        pass

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def create_backup(self, backup_type: str = "full") -> Optional[str]:
        """Create a backup"""
        if self.backup_manager:
            return await self.backup_manager.create_backup(backup_type)
        return None

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down production bot...")

        if self.scheduled_backup:
            self.scheduled_backup.stop()

        logger.info("✅ Shutdown complete")


async def main():
    """Main production bot execution"""
    parser = argparse.ArgumentParser(description="Production Trading Bot")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--backup", action="store_true", help="Enable automated backups")
    parser.add_argument("--alerts", action="store_true", help="Enable multi-channel alerts")
    parser.add_argument("--symbol", default="EURUSD", help="Trading symbol")
    parser.add_argument("--timeframe", default="M15", help="Timeframe")

    args = parser.parse_args()

    # Initialize production bot
    bot = ProductionTradingBot(
        symbol=args.symbol,
        timeframe=args.timeframe,
        use_gpu=args.gpu,
        enable_backups=args.backup,
        enable_alerts=args.alerts
    )

    try:
        # Initialize production features
        await bot.initialize_production_features()

        # Load sample data for demonstration
        import pandas as pd
        df = pd.read_csv("data/sample_eurusd.csv")
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # Train models with GPU acceleration
        training_results = await bot.train_models_with_gpu(df)
        logger.info(f"Training results: {training_results}")

        # Start production operations
        await bot.start_production_operations()

        # Create a manual backup
        if args.backup:
            backup_path = await bot.create_backup("models")
            if backup_path:
                logger.info(f"Manual backup created: {backup_path}")

        # Keep running
        logger.info("Production bot running... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Production bot error: {e}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())