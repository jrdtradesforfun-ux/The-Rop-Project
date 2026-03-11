# Changelog

## Version 1.0.0 (Initial Release)

### Features
- **ML Trading Engine**: Complete backend for ML-based trading decisions
  - LSTM/GRU price prediction models
  - Trend analysis and pattern recognition
  - Rule-based trading engine

- **Risk Management**: Comprehensive risk controls
  - Position sizing based on account risk
  - Daily loss tracking
  - Maximum position limits
  - Stop loss and take profit automation

- **Position Management**: Real-time position tracking
  - Open/close position management
  - P&L calculation
  - Position history logging

- **MQL5 Integration**: Socket-based communication with MetaTrader 5
  - Send trading signals to EA
  - Receive market updates
  - Order modification support

- **Data Processing**: Market data handling
  - CSV and MT5 data loading
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
  - Data normalization and preprocessing
  - Sequence creation for LSTM

- **Model Management**: ML model persistence
  - Model saving/loading
  - Metadata tracking
  - Version management

- **Example Strategies**: Baseline trading strategies
  - Simple momentum strategy
  - Mean reversion strategy
  - Extensible base class for custom strategies

- **Comprehensive Testing**: Unit test coverage
  - Trading engine tests
  - ML model tests
  - Data processing tests
  - Risk management tests

### Architecture
- Modular package structure
- Clean separation of concerns
- No external broker connections
- No AI chatbot integrations
- Python-focused implementation

### Dependencies
- PyNumPy
- pandas
- scikit-learn
- scipy

### Documentation
- README with quick start
- Developer guide with examples
- MQL5 integration guide
- Configuration reference

### Excluded Components
- ✗ AI chatbot integrations (DeepSeek, OpenAI, Kimi)
- ✗ Broker connections
- ✗ Frontend/UI components
- ✗ Web servers
- ✗ Database requirements

### Known Limitations
- Requires active MT5 EA for order execution
- Socket communication (no encrypted channels yet)
- Model training is simplified (placeholder implementation)
- No multi-broker support

### Future Enhancements
- Deep learning model implementations (TensorFlow/PyTorch)
- Database for trade persistence
- Web dashboard (optional, if requested)
- Additional technical indicators
- Monte Carlo simulation
- Walk-forward testing framework
- Portfolio optimization algorithms
