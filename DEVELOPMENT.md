# Developer Guide

## Setting Up Development Environment

1. Clone the repository and navigate to the project directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest  # For testing
   ```

## Project Structure

- **jaredis_backend/**: Main backend package
  - `ml_models/`: Machine learning models for prediction
  - `trading_engine/`: Core trading logic and position management
  - `mql5_bridge/`: Communication with MetaTrader 5
  - `data_processing/`: Market data handling and feature engineering
  - `utils/`: Logging and helper functions

- **MQL5/**: MetaTrader 5 Expert Advisor template
- **config/**: Configuration files
- **examples/**: Usage examples and tutorials
- **tests/**: Unit tests

## Running the Backend

Basic execution:
```bash
python main.py
```

With custom parameters:
```bash
python main.py --account-size=50000 --risk-per-trade=0.03 --mql5-host=192.168.1.100 --mql5-port=5001
```

## Running Tests

Execute test suite:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_trading_engine.py -v
```

Run with coverage:
```bash
pytest --cov=jaredis_backend tests/
```

## Key Components

### Trading Engine
Main coordinator for trading decisions and position management.
- Registers and evaluates multiple strategies
- Generates trade signals based on strategy outputs
- Manages positions and P&L tracking

### ML Models
Predictive models for market analysis.
- Price predictor (LSTM/GRU based)
- Trend analyzer (pattern recognition)
- Model manager for persistence

### MQL5 Bridge
Communicates trading signals to MetaTrader 5 Expert Advisors via socket protocol.

### Risk Management
Validates trades and enforces risk limits:
- Position sizing based on risk percentage
- Daily loss tracking
- Maximum position limits

## Writing Custom Strategies

Create a strategy by extending `BaseStrategy`:

```python
from examples.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def generate_signal(self, market_data):
        # Your signal generation logic
        # Return dict with: direction, entry_price, stop_loss, take_profit
        pass
```

Register with engine:
```python
engine.register_strategy("mystrategy", MyStrategy())
```

## Data Format

Market data structure for signals:
```json
{
    "symbol": "EURUSD",
    "timestamp": "2024-01-01T12:00:00",
    "candles": [
        {
            "open": 1.0950,
            "high": 1.0965,
            "low": 1.0945,
            "close": 1.0960,
            "volume": 1500000
        }
    ]
}
```

Signal response structure:
```json
{
    "direction": "long",
    "symbol": "EURUSD",
    "entry_price": 1.0960,
    "stop_loss": 1.0910,
    "take_profit": 1.1010,
    "confidence": 0.75,
    "signal_type": "momentum_buy"
}
```

## MT5 Integration

The backend communicates with MT5 Expert Advisors via JSON over sockets.

1. Ensure MT5 EA is running and listening on configured port
2. EA will receive buy/sell orders from the backend
3. Backend monitors order executions and updates positions

## Logging

Logs are written to `logs/jaredis.log` with daily rotation.
Console output shows real-time activity at configured log level.

Configure logging:
```python
from jaredis_backend.utils import setup_logging

logger = setup_logging(
    log_dir="logs",
    log_level="DEBUG",
    log_file="jaredis.log"
)
```

## Performance Metrics

Track strategy performance:
```python
from jaredis_backend.utils import calculate_metrics

metrics = calculate_metrics(trades)
print(f"Win rate: {metrics['win_rate']:.2%}")
print(f"Profit factor: {metrics['profit_factor']:.2f}")
```

## Deployment

For production deployment:

1. Use proper MT5 EA with full socket library support
2. Implement database for trade persistence
3. Add monitoring and alerting
4. Use configuration management for production settings
5. Implement secure communication with MT5
6. Set up proper error handling and recovery

## Troubleshooting

### Can't connect to MT5
- Ensure EA is running and listening on configured port
- Check firewall settings allowing the connection
- Verify host and port in configuration

### Data validation errors
- Ensure OHLCV data has valid relationship (L <= C <= H and L <= O <= H)
- Check for negative values or NaN
- Validate data source

### Model loading issues
- Ensure model files exist in `models/` directory
- Verify model was saved properly with metadata
- Check Python version compatibility for pickle

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Update relevant documentation
3. Ensure all tests pass
4. Follow existing code style

## License

Proprietary - All rights reserved
