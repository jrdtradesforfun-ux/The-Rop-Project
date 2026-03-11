# Jaredis Smart Trading Backend

Core ML-powered trading engine with MetaTrader 5 integration.

## Features

- **ML Trading Models**: LSTM/GRU price predictors and trend analysis
- **Risk Management**: Position sizing, stop loss, take profit automation  
- **Position Management**: Track open positions and P&L in real-time
- **MQL5 Integration**: Socket-based communication with MT5 Expert Advisors
- **Data Processing**: Technical indicators, feature engineering, normalization
- **Logging**: Comprehensive logging with file rotation

## Architecture

```
jaredis_backend/
├── ml_models/           # ML models for predictions
│   ├── model_manager.py # Model persistence and loading
│   └── predictors.py    # Price and trend prediction models
├── trading_engine/      # Core trading logic
│   ├── engine.py        # Main trading coordinator
│   ├── position_manager.py  # Position tracking
│   └── risk_manager.py  # Risk controls
├── mql5_bridge/         # MT5 integration
│   ├── mql5_connector.py    # Socket connection handler
│   └── signal_communicator.py  # Signal formatting
├── data_processing/     # Market data prep
│   ├── data_loader.py   # Load OHLCV data
│   ├── feature_engineer.py  # Technical indicators
│   └── preprocessor.py  # Data cleaning
└── utils/              # Utilities
    ├── logger_config.py # Logging setup
    └── helpers.py       # Common functions
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend
python main.py --account-size=10000 --risk-per-trade=0.02

# Run tests
pytest tests/
```

## Configuration

See `config/` directory for configuration files.

## MQL5 Integration

The backend communicates with MT5 Expert Advisors via socket protocol. EA must be running and listening on configured port.

## Requirements

- Python 3.8+
- NumPy
- MetaTrader 5 (with compatible EA)

## License

Proprietary
