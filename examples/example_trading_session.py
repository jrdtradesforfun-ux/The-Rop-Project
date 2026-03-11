"""
Example: Running the complete trading backend
"""

import time
from jaredis_backend.trading_engine import TradingEngine
from jaredis_backend.ml_models import ModelManager, TrendAnalyzer
from jaredis_backend.mql5_bridge import MQL5Connector, SignalCommunicator
from jaredis_backend.data_processing import DataLoader, FeatureEngineer
from jaredis_backend.utils import setup_logging
from examples.strategies import SimpleMomentumStrategy, MeanReversionStrategy


def run_trading_session():
    """Run a complete trading session"""
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting trading session...")
    
    # Initialize components
    engine = TradingEngine(account_size=10000, risk_per_trade=0.02)
    model_manager = ModelManager(models_dir="models")
    data_loader = DataLoader(data_dir="data")
    
    # Register strategies
    momentum_strategy = SimpleMomentumStrategy(lookback=20)
    reversion_strategy = MeanReversionStrategy(period=20, std_threshold=2.0)
    
    engine.register_strategy("momentum", momentum_strategy)
    engine.register_strategy("reversion", reversion_strategy)
    
    logger.info("Strategies registered")
    
    # Try to connect to MQL5
    mql5_connector = MQL5Connector(host="localhost", port=5000)
    
    if mql5_connector.connect():
        logger.info("Connected to MQL5")
        signal_comm = SignalCommunicator(mql5_connector)
    else:
        logger.warning("Could not connect to MQL5 (EA not running)")
        signal_comm = None
    
    # Simulation loop (in production, this would be real-time)
    logger.info("Starting main trading loop...")
    
    try:
        for i in range(10):  # 10 iterations for example
            # Simulate market data
            market_data = {
                "symbol": "EURUSD",
                "timestamp": f"2024-01-01 {i:02d}:00",
                "candles": [
                    {
                        "open": 1.0950 + i * 0.001,
                        "high": 1.0960 + i * 0.001,
                        "low": 1.0940 + i * 0.001,
                        "close": 1.0955 + i * 0.001,
                        "volume": 1500000 + i * 10000
                    }
                    for _ in range(20)
                ]
            }
            
            # Evaluate signals
            signals = engine.evaluate_signals(market_data)
            
            logger.info(f"Iteration {i}: Evaluated signals from {len(signals['strategy_signals'])} strategies")
            
            # Show portfolio status
            status = engine.get_portfolio_status()
            logger.info(f"Portfolio - Open positions: {len(status['open_positions'])}, "
                       f"Unrealized P&L: {status['unrealized_pnl']:.2f}")
            
            time.sleep(0.5)  # Small delay for readability
    
    except KeyboardInterrupt:
        logger.info("Trading session interrupted by user")
    finally:
        # Cleanup
        if signal_comm:
            mql5_connector.disconnect()
        
        # Show final statistics
        trades = engine.get_trade_history()
        logger.info(f"Session complete. Total trades: {len(trades)}")
        
        for trade in trades[-5:]:  # Show last 5 trades
            logger.info(f"Trade: {trade['symbol']} {trade.get('signal', {}).get('direction')} "
                       f"@ {trade.get('signal', {}).get('entry_price', 'N/A')}")


if __name__ == "__main__":
    run_trading_session()
