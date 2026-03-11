# Jaredis Smart Trading Bot

**Professional-Grade AI Trading System with JustMarkets Integration**

A complete, production-ready trading bot combining advanced machine learning, risk management, broker integration, execution pipeline, and real-time monitoring for forex trading via JustMarkets.

---

## 🚀 Features

### Core Trading Capabilities
- ✅ **Multi-Symbol Trading** - Trade multiple currency pairs simultaneously
- ✅ **Real-time Market Data** - Live tick and bar data from MetaTrader 5
- ✅ **Smart Order Execution** - Automated order placement with risk validation
- ✅ **Position Management** - Open, close, and modify positions programmatically
- ✅ **JustMarkets Integration** - Direct broker connectivity via MetaTrader 5

### Machine Learning System
- ✅ **Ensemble Predictions** - Multiple models (Random Forest, XGBoost, LSTM) with weighted voting
- ✅ **Market Regime Detection** - Automatically switches strategies based on trending/ranging/volatile markets
- ✅ **Confidence Scoring** - All predictions include confidence levels for trade filtering
- ✅ **Feature Engineering** - Technical indicators (MA, RSI, ATR, momentum)
- ✅ **Ensemble Voting** - Disagreement detection to avoid low-confidence trades

### Risk Management
- ✅ **Risk Per Trade** - Configurable percentage-based position sizing
- ✅ **Stop Loss & Take Profit** - Automatic calculation based on price volatility
- ✅ **Position Limits** - Maximum concurrent positions validation
- ✅ **Margin Monitoring** - Real-time account margin checking
- ✅ **Drawdown Alerts** - Automatic trading halt if equity drops > 5%

### Monitoring & Alerting
- ✅ **Real-time Metrics** - Win rate, profit factor, Sharpe ratio, drawdown
- ✅ **Performance Alerts** - Automatic alerts on trades, equity changes, losses
- ✅ **System Health Monitoring** - Connection, data, and execution error tracking
- ✅ **Session Reports** - Daily trading summaries and statistics
- ✅ **Equity Tracking** - Real-time account balance and unrealized P&L

---

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Trading Bot (Your Machine or VPS)                       │
│  ├── ProfessionalTradingBot (Main)                              │
│  ├── EnsemblePredictor (ML voting)                              │
│  ├── ExecutionEngine (Order routing)                            │
│  ├── PerformanceMonitor (Metrics & alerts)                      │
│  └── SystemMonitor (Health tracking)                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Socket (Port 5000)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ MetaTrader 5 + JaredisSmartEA                                    │
│  └── JustMarkets Account                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ JustMarkets Broker (Forex Market)                                │
└──────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
jaredis_backend/
├── brokers/                 # Broker connectors
│   └── justmarkets.py      # JustMarkets MT5 integration
├── execution/              # Order execution pipeline
│   └── engine.py           # ExecutionEngine with validation
├── monitoring/             # Real-time monitoring
│   └── metrics.py          # Performance & system monitoring
├── advanced_models/        # ML predictors
│   └── models.py           # RF, XGBoost, LSTM models
├── ensemble/               # ML ensemble & regimes
│   └── predictor.py        # EnsemblePredictor & MarketRegimeDetector
├── trading_engine/         # Core trading logic
├── mql5_bridge/           # MT5 socket connector
├── data_processing/       # Data pipeline
└── pytrader/              # PyTrader API wrapper

examples/
├── professional_trading_bot.py    # Complete working example
└── example_*.py                   # Other examples

mql5/
└── JaredisSmartEA.mq5     # MetaTrader 5 Expert Advisor
```

---

## 🎯 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/jrdtradesforfun-ux/The-Rop-Project
cd jaredis-smart
pip install -r requirements.txt
```

### 2. Setup JustMarkets
See [JUSTMARKETS_SETUP.md](JUSTMARKETS_SETUP.md) for complete instructions:
- Create JustMarkets demo account
- Install MetaTrader 5
- Deploy JaredisSmartEA.mq5 on chart
- Run Python bot

### 3. Run Bot
```bash
python examples/professional_trading_bot.py
```

---

## 🤖 Machine Learning Models

| Model | Type | Features |
|-------|------|----------|
| **Random Forest** | Ensemble Trees | Robust, handles noise well |
| **XGBoost** | Gradient Boosting | Fast, accurate predictions |
| **LSTM** | Deep Learning | Captures temporal patterns |

**Ensemble System**: Combines all 3 models with weighted voting for robust predictions.

---

## ⚙️ Core Components

### Broker Integration
```python
from jaredis_backend.brokers import JustMarketsBroker

broker = JustMarketsBroker(host="localhost", port=5000)
broker.connect()
balance = broker.get_account_balance()
```

### Execution Engine
```python
from jaredis_backend.execution import ExecutionEngine

engine = ExecutionEngine(broker, risk_manager)
signal = {"symbol": "EURUSD", "direction": "long", ...}
result = engine.execute_signal(signal)
```

### ML Ensemble
```python
from jaredis_backend.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor()
ensemble.add_model("rf", RandomForestPredictor())
ensemble.add_model("xgb", GradientBoostingPredictor())
prediction = ensemble.predict(features)
```

### Real-time Monitoring
```python
from jaredis_backend.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.record_trade(entry=1.0950, exit=1.1000, ...)
metrics = monitor.get_metrics()
```

---

## 📊 Configuration

### Risk Settings
```python
bot = ProfessionalTradingBot(
    account_size=10000,          # $
    risk_per_trade=0.02,         # 2% per trade
)
```

### Trading Symbols
```python
bot.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
```

### Duration
```python
bot.run(duration_minutes=120)  # 2 hours
```

---

## 📈 Trading Signals

Signal format:
```json
{
  "symbol": "EURUSD",
  "direction": "long",
  "entry_price": 1.0950,
  "stop_loss": 1.0900,
  "take_profit": 1.1000,
  "confidence": 0.75,
  "timestamp": "2024-01-01T12:30:45"
}
```

## 🚀 Deployment

### Local Testing
```bash
python examples/professional_trading_bot.py
```

### VPS (24/7 Automated)
```bash
# SSH into VPS, then:
python examples/professional_trading_bot.py &
# Or use screen/tmux for persistence
```

Recommended: [DigitalOcean](https://digitalocean.com) ($5-6/mo), [Contabo](https://contabo.com) ($4-10/mo)

---

## ⚠️ Risk Disclaimer

**IMPORTANT**: Forex trading with leverage is **HIGH RISK**.

⚠️ **START WITH DEMO ACCOUNT** ⚠️

Safe deployment:
- Test on demo 1-2 weeks minimum
- Backtest on historical data
- Start live with 0.01 lot positions
- Use stop losses on every trade
- Keep daily loss limit at 5%

---

## 🐛 Troubleshooting

See [JUSTMARKETS_SETUP.md](JUSTMARKETS_SETUP.md#troubleshooting) for common issues.

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [JUSTMARKETS_SETUP.md](JUSTMARKETS_SETUP.md) | Broker setup guide |
| [examples/professional_trading_bot.py](examples/professional_trading_bot.py) | Working example |

---

## 📝 License

Proprietary - See LICENSE

---

**Professional Trading Infrastructure Built for Quantitative Traders**
