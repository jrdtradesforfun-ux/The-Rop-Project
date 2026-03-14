"""
Professional AI Trading Bot with JustMarkets Integration
Complete example using all system components
"""

import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd

from jaredis_backend.brokers import JustMarketsBroker
from jaredis_backend.trading_engine import TradingEngine, RiskManager
from jaredis_backend.execution import ExecutionEngine
from jaredis_backend.monitoring import PerformanceMonitor, SystemMonitor
from jaredis_backend.ml_models import ModelManager
from jaredis_backend.data_processing import DataLoader, FeatureEngineer, Preprocessor
from jaredis_backend.advanced_models import RandomForestPredictor, GradientBoostingPredictor
from jaredis_backend.ensemble import EnsemblePredictor, MarketRegimeDetector
from jaredis_backend.utils import setup_logging


class ProfessionalTradingBot:
    """
    Production-ready trading bot combining ML predictions,
    risk management, and real-time monitoring.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 5000,
        account_size: float = 10000,
        risk_per_trade: float = 0.02,
    ):
        """Initialize professional trading bot"""
        self.logger = setup_logging()
        self.logger.info("=" * 70)
        self.logger.info("Professional AI Trading Bot Starting")
        self.logger.info("=" * 70)

        # Brokers
        self.broker = JustMarketsBroker(host=broker_host, port=broker_port)

        # Core trading systems
        self.risk_manager = RiskManager(
            account_size=account_size,
            max_risk_per_trade=risk_per_trade,
        )
        self.trading_engine = TradingEngine(
            account_size=account_size,
            risk_per_trade=risk_per_trade,
        )
        self.execution_engine = ExecutionEngine(self.broker, self.risk_manager)

        # Data systems
        self.data_loader = DataLoader(data_dir="data")
        self.model_manager = ModelManager(models_dir="models")

        # ML predictions
        self.ensemble = EnsemblePredictor()
        self.regime_detector = MarketRegimeDetector()

        # Monitoring
        self.performance_monitor = PerformanceMonitor(alert_threshold_drawdown=0.05)
        self.system_monitor = SystemMonitor()

        # Trading state
        self.connected = False
        self.running = False
        self.trading_symbols = ["EURUSD", "GBPUSD"]

    def initialize(self) -> bool:
        """Initialize all components"""
        self.logger.info("Initializing trading bot components...")

        # Connect to broker
        if not self.broker.connect():
            self.system_monitor.record_connection_error("Failed to connect to broker")
            return False

        self.connected = True
        self.logger.info("Broker connected successfully")

        # Initialize ML models
        self._setup_models()

        return True

    def _setup_models(self) -> None:
        """Setup and train ML models"""
        self.logger.info("Setting up ML models...")

        # Create Random Forest model
        rf_model = RandomForestPredictor(n_trees=100, max_depth=10)
        self.ensemble.add_model("random_forest", rf_model, weight=1.0)

        # Try to add Gradient Boosting
        try:
            gb_model = GradientBoostingPredictor()
            self.ensemble.add_model("xgboost", gb_model, weight=1.2)
        except Exception as e:
            self.logger.warning(f"XGBoost not available: {e}")

        self.logger.info(f"Ensemble ready with {len(self.ensemble.models)} models")

    def check_for_model_updates(self) -> bool:
        """Check for model updates from the retraining pipeline"""
        try:
            from pathlib import Path
            signal_file = Path("models/live/model_updated.signal")
            
            if signal_file.exists():
                # Read signal file
                with open(signal_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        model_path_line = lines[1]
                        model_path = model_path_line.split("Path: ")[-1].strip()
                        
                        # Load the new model
                        from jaredis_backend.ml_models.xgb_trainer import XGBoostTradingModel
                        new_model = XGBoostTradingModel.load_model(model_path)
                        
                        # Replace in ensemble
                        success = self.ensemble.replace_model("xgboost", new_model)
                        
                        if success:
                            self.logger.info(f"Successfully updated XGBoost model from {model_path}")
                            # Remove signal file to avoid reprocessing
                            signal_file.unlink()
                            return True
                        else:
                            self.logger.warning("Failed to replace model in ensemble")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for model updates: {e}")
            return False

    def generate_features(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Generate ML features from market data"""
        try:
            # Get historical bars
            bars_data = self.broker.get_bars(symbol, "M5", count=bars)
            if not bars_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(bars_data)

            # Create technical features
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma50"] = df["close"].rolling(50).mean()
            df["rsi"] = FeatureEngineer.calculate_rsi(df["close"].values)
            df["atr"] = FeatureEngineer.calculate_atr(
                df["high"].values,
                df["low"].values,
                df["close"].values,
            )

            # Momentum features
            df["momentum"] = df["close"].pct_change()
            df["volume_ma"] = df["volume"].rolling(20).mean()

            # Target: price goes up in next candle
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

            return df.dropna()

        except Exception as e:
            self.logger.error(f"Error generating features for {symbol}: {e}")
            self.system_monitor.record_data_error(str(e))
            return pd.DataFrame()

    def predict_direction(self, symbol: str) -> dict:
        """Get ensemble prediction for symbol"""
        try:
            df = self.generate_features(symbol)
            if df.empty:
                return {"error": "No data"}

            # Prepare features for model
            feature_cols = ["ma20", "ma50", "rsi", "atr", "momentum", "volume_ma"]
            X = df[feature_cols].values

            # Get ensemble prediction
            prediction = self.ensemble.predict(X[-1:])  # Last candle

            return {
                "symbol": symbol,
                "prediction": prediction.get("prediction", 0),
                "confidence": prediction.get("confidence", 0),
                "consensus": prediction.get("consensus", 0),
                "volatility": df["atr"].iloc[-1],
            }

        except Exception as e:
            self.logger.error(f"Error predicting for {symbol}: {e}")
            self.system_monitor.record_data_error(str(e))
            return {"error": str(e)}

    def detect_regime(self, symbol: str) -> dict:
        """Detect current market regime"""
        try:
            bars_data = self.broker.get_bars(symbol, "M5", count=100)
            if not bars_data:
                return {"regime": "unknown"}

            closes = np.array([b["close"] for b in bars_data])
            volumes = np.array([b.get("volume", 0) for b in bars_data])

            regime = self.regime_detector.detect_regime(closes, volumes)
            return regime

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return {"regime": "unknown"}

    def generate_signal(self, symbol: str) -> dict:
        """Generate complete trading signal"""
        # Get prediction
        prediction = self.predict_direction(symbol)
        if "error" in prediction:
            return None

        # Get regime
        regime = self.detect_regime(symbol)
        regime_suggestion = self.regime_detector.get_strategy_suggestion(
            regime.get("regime")
        )

        # Get current price
        tick = self.broker.get_tick(symbol)
        if not tick:
            return None

        current_price = tick.get("bid", 0)
        if current_price <= 0:
            return None

        # Build signal
        is_bullish = prediction["prediction"] == 1
        confidence = prediction["confidence"]

        # Only trade if confidence is sufficient
        if confidence < 0.55:
            self.logger.debug(f"Low confidence ({confidence:.2%}), skipping {symbol}")
            return None

        # Calculate levels
        atr = prediction.get("volatility", current_price * 0.001)
        stop_loss_points = atr * 2
        take_profit_points = atr * 3

        signal = {
            "symbol": symbol,
            "direction": "long" if is_bullish else "short",
            "entry_price": current_price,
            "stop_loss": current_price - stop_loss_points if is_bullish else current_price + stop_loss_points,
            "take_profit": current_price + take_profit_points if is_bullish else current_price - take_profit_points,
            "confidence": confidence,
            "signal_type": "ensemble_ml",
            "regime": regime.get("regime"),
            "timestamp": datetime.now().isoformat(),
        }

        return signal

    def run(self, duration_minutes: int = 60):
        """Run the trading bot"""
        if not self.initialize():
            self.logger.error("Initialization failed")
            return

        self.running = True
        start_time = datetime.now()

        self.logger.info(f"Trading bot running for {duration_minutes} minutes")

        try:
            while self.running:
                # Check for model updates
                self.check_for_model_updates()
                
                # Check elapsed time
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                if elapsed > duration_minutes:
                    self.logger.info(f"Duration exceeded ({elapsed:.1f} min), stopping")
                    break

                # Process each symbol
                for symbol in self.trading_symbols:
                    try:
                        signal = self.generate_signal(symbol)

                        if signal:
                            self.logger.info(f"Signal generated: {signal}")

                            # Execute signal
                            result = self.execution_engine.execute_signal(signal)

                            if result:
                                self.logger.info(f"Trade executed: {result}")

                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        self.system_monitor.record_execution_error(str(e))

                # Update monitoring
                self._update_monitoring()

                # Wait before next cycle
                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        finally:
            self.shutdown()

    def _update_monitoring(self):
        """Update performance and system monitoring"""
        try:
            balance = self.broker.get_account_balance()
            equity = self.broker.get_account_equity()

            self.performance_monitor.update_equity(equity, 10000)

            status = self.system_monitor.get_system_status()
            self.logger.info(
                f"Balance: {balance:.2f} | Equity: {equity:.2f} | "
                f"System: {'Healthy' if status['healthy'] else 'ISSUES'}"
            )

        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

    def shutdown(self):
        """Shutdown bot gracefully"""
        self.running = False
        self.logger.info("Shutting down trading bot...")

        # Generate final report
        report = self.performance_monitor.generate_session_report()
        self.logger.info(f"Session Report: {report}")

        # Disconnect
        if self.connected:
            self.broker.disconnect()

        self.logger.info("Bot shutdown complete")


if __name__ == "__main__":
    # Create and run bot
    bot = ProfessionalTradingBot(
        broker_host="localhost",
        broker_port=5000,
        account_size=10000,
        risk_per_trade=0.02,
    )

    # Run for 2 hours
    bot.run(duration_minutes=120)
