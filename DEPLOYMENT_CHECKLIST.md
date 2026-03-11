# Deployment Checklist & Quick Reference

## ✅ Pre-Trading Setup Checklist

### Step 1: Install & Configure (Windows/VPS)
- [ ] Python 3.8+ installed
- [ ] Repository cloned: `git clone https://github.com/jrdtradesforfun-ux/The-Rop-Project`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] All core modules imported without errors

### Step 2: Broker Setup
- [ ] JustMarkets account created (demo for testing)
- [ ] MetaTrader 5 downloaded and installed
- [ ] MT5 logged in with JustMarkets credentials
- [ ] EURUSD and GBPUSD symbols in Market Watch

### Step 3: Expert Advisor Deployment
- [ ] `mql5/JaredisSmartEA.mq5` copied to MT5 Experts folder
- [ ] EA compiled without errors in MetaEditor
- [ ] EA dragged onto EURUSD M5 chart
- [ ] AutoTrading button enabled (checked)
- [ ] EA confirmed running (status bar shows "Expert running")

### Step 4: Python Bot Setup
- [ ] Port 5000 is allowed by firewall
- [ ] Example config reviewed: `examples/professional_trading_bot.py`
- [ ] Account size set correctly
- [ ] Risk per trade configured (recommend 2%)

### Step 5: Test & Verify
- [ ] Run ML training example: `python examples/train_ml_models.py`
- [ ] Bot starts without connection errors
- [ ] Check logs: `tail -f logs/jaredis.log`
- [ ] Verify EA receives at least one signal from bot

### Step 6: Demo Trading (1-2 weeks)
- [ ] Place 5-10 manual test trades to verify order execution
- [ ] Monitor equity, drawdown, win rate in logs
- [ ] Review performance metrics daily
- [ ] Verify stop losses and take profits work correctly

### Step 7: Go Live (if confident)
- [ ] Switch to live account (not demo)
- [ ] Start with smallest position sizes (0.01 lots)
- [ ] Use daily loss limit (5% maximum)
- [ ] Monitor first week closely

---

## 🔧 Quick Command Reference

### Train ML Models
```bash
python examples/train_ml_models.py
```
Output: Trains Random Forest, XGBoost, LSTM on synthetic data

### Run Professional Trading Bot
```bash
python examples/professional_trading_bot.py
```
Output: Begins trading based on ensemble predictions

### Run All Tests
```bash
pytest tests/ -v
```
Output: Validates all modules work correctly

### View Logs
```bash
tail -f logs/jaredis.log
```
Output: Real-time log stream (Ctrl+C to exit)

### Git Status
```bash
git status
git log --oneline -10
```
Output: Current changes and recent commits

---

## 🎯 Configuration Settings

### In `examples/professional_trading_bot.py`

#### Broker Connection
```python
bot = ProfessionalTradingBot(
    broker_host="localhost",        # Change to VPS IP for remote
    broker_port=5000,               # MetaTrader socket port
    account_size=10000,             # Initial account size
    risk_per_trade=0.02,            # 2% risk per trade
)
```

#### Trading Symbols
```python
bot.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
```

#### Run Duration
```python
bot.run(duration_minutes=120)  # 2 hours
```

#### Risk Manager Limits
```python
risk_manager = RiskManager(
    account_size=10000,
    max_risk_per_trade=0.02,        # Max loss per trade: 2%
    max_daily_loss=0.05,            # Stop trading if down 5% today
)
```

---

## 📊 Monitoring & Alerts

### Key Metrics (from logs)
```
Balance: $10,150 | Equity: $10,156 | System: Healthy
Win Rate: 65% | Profit Factor: 1.8 | Sharpe: 1.2
Drawdown: 4.5% | Total Trades: 20
```

### Alert Triggers
```
ALERT: Equity down 2.5% - Monitor positions
ALERT: Drawdown exceeds 5% - Trading HALTED
ALERT: Connection lost - Attempting reconnect
ALERT: Order failed - Check margin
```

### Log Levels
```
INFO    - Normal operations (signals, trades, metrics)
WARNING - Minor issues, bot continues
ERROR   - Connection/execution problems
CRITICAL - Fatal errors, bot stops
```

---

## ⚠️ Trading Rules

### Position Sizing
```
Position Size = (Account Size × Risk %) / (Stop Loss in pips)

Example:
Account: $10,000
Risk: 2% = $200
Stop Loss: 40 pips = 0.0040
Lot Size = $200 / 0.0040 = 0.1 lots
```

### Risk Management
```
✓ Every trade has a stop loss
✓ Take profit ≥ 2× stop loss size
✓ Max 5 concurrent positions
✓ Max 5% account loss per day
✓ Risk per trade: 1-3%
```

### Trading Hours (EURUSD)
```
Sydney:    21:00 - 06:00 GMT
Tokyo:     00:00 - 09:00 GMT
London:    08:00 - 17:00 GMT
NewYork:   13:00 - 22:00 GMT
```

---

## 🐛 Troubleshooting Quick Fixes

### Bot won't connect to EA
```bash
1. Ensure MT5 is running and EA enabled
2. Check firewall: netstat -an | find "5000"
3. Verify PythonHost in EA settings
4. Restart both MT5 and Python bot
```

### Orders not executing
```bash
1. Check account margin: free_margin > required_margin
2. Verify trading hours (not weekends, holidays)
3. Confirm stop loss > 1 pip and sound levels
4. Review EA logs (View → Experts → Journal)
```

### Poor trading performance
```bash
1. Backtest strategy on historical data
2. Increase training data for ML models
3. Check technical indicators are correct
4. Verify target variable (up/down) is correct
5. Test on demo before live trading
```

---

## 📈 Performance Targets (Demo Trading)

### Realistic Expectations
```
Win Rate:        50-65% (better models achieve 60%+)
Profit Factor:   1.5x+ (revenue / losses)
Monthly Profit:  2-5% of account (if lucky)
Max Drawdown:    3-5% (with good risk management)
Sharpe Ratio:    1.0+ (return per unit of risk)
```

### Red Flags
```
❌ Win Rate < 40%  (strategy not working)
❌ Profit Factor < 1.0  (losing money)
❌ Consistency varies wildly (not robust)
❌ Confident but losing  (overfitting likely)
❌ All winners/all losers  (check backtest logic)
```

---

## 🚀 Next Steps After Setup

### Week 1-2: Testing Phase
1. Run on demo account only
2. Trade 10-20 times per symbol
3. Verify order execution works
4. Check stop loss / take profit accuracy
5. Monitor drawdown and equity curves

### Week 3-4: Optimization
1. Analyze trading logs
2. Identify losing signal types
3. Retrain ML models with feedback
4. Experiment with different symbols
5. Adjust risk per trade based on results

### Week 5+: Production Deployment
1. If >55% win rate on demo, consider live
2. Start with 0.01 lot size (minimal risk)
3. Monitor daily for first month
4. Scale position sizes gradually
5. Document all changes for improvement

---

## 📞 Getting Help

### Check These Files First
1. **Setup Issues**: [JUSTMARKETS_SETUP.md](../JUSTMARKETS_SETUP.md)
2. **Code Examples**: [examples/professional_trading_bot.py](../examples/professional_trading_bot.py)
3. **Training**: [examples/train_ml_models.py](../examples/train_ml_models.py)
4. **API Reference**: [jaredis_backend/brokers/justmarkets.py](../jaredis_backend/brokers/justmarkets.py)

### Common Issues & Solutions

**Q: "Failed to connect to Python bot"**
- A: EA settings → check PythonHost is "localhost" (or VPS IP)
- A: Windows Firewall → allow Python.exe on port 5000

**Q: "Order failed. Error: 131"**
- A: Account doesn't have enough margin
- A: Symbol trading is closed (not trading hours)
- A: Order parameters invalid (volume, SL, TP)

**Q: "Bot trades but always loses"**
- A: ML model needs better training data
- A: Strategy parameters not optimized
- A: Random luck - test longer period
- A: Check if predictions are correct (manual verification)

**Q: "Connection keeps dropping"**
- A: VPS network unstable (restart VPS)
- A: MT5 crashed (need to restart)
- A: Python bot out of memory (reduce frequency)

---

## 📝 Important Documentation

| Document | Contents |
|----------|----------|
| [README.md](../README.md) | Project overview, features, architecture |
| [JUSTMARKETS_SETUP.md](../JUSTMARKETS_SETUP.md) | Broker setup guide, VPS deployment |
| [examples/](../examples/) | Working code examples |
| [jaredis_backend/](../jaredis_backend/) | Core module documentation |

---

**Last Updated**: January 2024
**Status**: Production Ready ✓
**Tested On**: Windows 10/11, Ubuntu 20.04, VPS servers
