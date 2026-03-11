"""
Main entry point for Jaredis Smart Trading Backend
"""

import argparse
import logging
from pathlib import Path

from jaredis_backend import TradingEngine, ModelManager
from jaredis_backend.mql5_bridge import MQL5Connector, SignalCommunicator
from jaredis_backend.data_processing import DataLoader, FeatureEngineer, Preprocessor
from jaredis_backend.utils import setup_logging


def main():
    """Main trading engine runner"""
    parser = argparse.ArgumentParser(description="Jaredis Smart Trading Backend")
    parser.add_argument("--account-size", type=float, default=10000,
                        help="Initial account size")
    parser.add_argument("--risk-per-trade", type=float, default=0.02,
                        help="Risk percentage per trade")
    parser.add_argument("--mql5-host", type=str, default="localhost",
                        help="MQL5 EA host address")
    parser.add_argument("--mql5-port", type=int, default=5000,
                        help="MQL5 EA port")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(log_level=getattr(logging, args.log_level))
    logger.info("=" * 60)
    logger.info("Jaredis Smart Trading Backend Starting")
    logger.info("=" * 60)

    # Initialize components
    logger.info(f"Account Size: {args.account_size}")
    logger.info(f"Risk per Trade: {args.risk_per_trade * 100}%")

    trading_engine = TradingEngine(
        account_size=args.account_size,
        risk_per_trade=args.risk_per_trade
    )

    model_manager = ModelManager(models_dir="models")
    data_loader = DataLoader(data_dir="data")

    # Initialize MQL5 connection
    mql5_connector = MQL5Connector(
        host=args.mql5_host,
        port=args.mql5_port
    )

    if mql5_connector.connect():
        logger.info("Successfully connected to MQL5 EA")
        signal_comm = SignalCommunicator(mql5_connector)

        # Main trading loop would go here
        # For now, just verify connection
        mql5_connector.heartbeat()
        mql5_connector.disconnect()
    else:
        logger.warning("Could not connect to MQL5 EA (normal if EA not running)")

    logger.info("Backend initialization complete")
    logger.info("Ready for trading signals...")


if __name__ == "__main__":
    main()
