# JustMarkets Broker Setup Guide

## Complete Setup for Production Trading

### Step 1: Create Demo Account at JustMarkets

1. Visit [JustMarkets.com](https://justmarkets.com)
2. Click "Open Account"
3. Select "Demo Account" for testing
4. Complete registration with email/phone
5. Note your login credentials (email + password)

### Step 2: Install MetaTrader 5

**Windows:**
- Download from [MetaTrader Official](https://www.metatrader5.com/en/download)
- Run installer
- Complete installation

**VPS Server (for 24/7 trading):**
- Rent VPS: [DigitalOcean](https://www.digitalocean.com), [AWS](https://aws.amazon.com), [Contabo](https://contabo.com)
- SSH into VPS
- Install Windows Server or Linux with Wine
- Install MetaTrader 5

### Step 3: Configure MetaTrader 5

1. **Launch MT5**
2. **Add JustMarkets Account:**
   - File → Login
   - Select "JustMarkets" from broker list
   - Enter email and password
   - Click "Login"
3. **Verify Connection:**
   - Check "Navigator" panel for account info
   - See available symbols in Market Watch

### Step 4: Deploy PyTrader EA

The system uses the PyTrader API connector via Expert Advisor.

1. **Access EA Files:**
   - Find `mql5/JaredisSmartEA.mq5` in this repo
   - Copy to MT5 Experts folder:
     - Windows: `C:\Users\[User]\AppData\Roaming\MetaQuotes\Terminal\[TerminalID]\MQL5\Experts\`
     - Linux: `~/.wine/drive_c/.../MQL5/Experts/`

2. **Compile EA:**
   - Open MetaEditor in MT5
   - File → Open → select `JaredisSmartEA.mq5`
   - Compile (F5)

3. **Deploy EA on Chart:**
   - Open any chart (e.g., EURUSD M5)
   - Drag EA from Navigator → Expert Advisors onto chart
   - Configure settings:
     - PythonHost: `localhost` (local) or `VPS_IP` (remote)
     - PythonPort: `5000`
     - MagicNumber: `12345`
   - Click OK

4. **Enable AutoTrading:**
   - Top right corner: Click "AutoTrading" button
   - Verify status changes to "enabled"

### Step 5: Run Python Trading Bot

**On Local Machine:**
```bash
cd jaredis-smart
python examples/professional_trading_bot.py
```

**On VPS Server:**
```bash
# SSH into VPS
ssh root@your_vps_ip

# Install Python and dependencies
apt-get update
apt-get install python3 python3-pip git

# Clone repo
git clone [your-repo-url]
cd jaredis-smart

# Install dependencies
pip install -r requirements.txt

# Run bot (in background with screen/tmux)
screen -S trading_bot
python examples/professional_trading_bot.py
# Press Ctrl+A then D to detach

# View logs
tail -f logs/jaredis.log
```

## System Architecture

```
Your Python Bot
    ↓ (Socket connection to port 5000)
    ↓
JaredisSmartEA (MetaTrader Expert Advisor)
    ↓ (Native MetaTrader API)
    ↓
MetaTrader 5 Terminal
    ↓ (Secure HTTPS/FIX Protocol)
    ↓
JustMarkets Broker Servers
    ↓ (Live Market Data)
    ↓
Forex Market → Execution
```

## API Reference

### Connect to Broker
```python
from jaredis_backend.brokers import JustMarketsBroker

broker = JustMarketsBroker(host="localhost", port=5000)
broker.connect()
```

### Get Market Data
```python
# Get available symbols
symbols = broker.get_available_symbols()

# Get last tick
tick = broker.get_tick("EURUSD")
print(tick["bid"], tick["ask"])

# Get bars (history)
bars = broker.get_bars("EURUSD", "M5", count=100)
```

### Place Trade
```python
result = broker.place_order(
    symbol="EURUSD",
    order_type="buy",      # or "sell"
    volume=0.1,            # Position size in lots
    entry_price=1.0950,    # 0 for market order
    stop_loss=1.0900,
    take_profit=1.1000,
    comment="ML Signal"
)

print(result["ticket"])  # Order ticket number
```

### Close Trade
```python
broker.close_position(ticket=123456)
```

### Check Balance
```python
balance = broker.get_account_balance()
equity = broker.get_account_equity()
print(f"Balance: {balance}, Equity: {equity}")
```

## Important Notes

### For Testing (Demo):
- Use the bot on **Demo Account** first
- Test strategies with fake money
- Monitor performance for 1-2 weeks
- Verify all orders execute correctly

### For Live Trading:
1. Start with **very small position sizes** (0.01 lots)
2. Use strict **stop losses** on every trade
3. Monitor **drawdown limits** (recommend max 5%)
4. Have a **kill switch** to stop trading if something goes wrong

### Risk Management Rules
```python
# In professional_trading_bot.py

# Risk per trade = 2% of account (customizable)
risk_per_trade = 0.02  

# Max concurrent positions = 5
max_positions = 5

# Daily loss limit = 5% (triggers auto shutdown)
daily_loss_limit = 0.05
```

## Troubleshooting

### EA Won't Connect to Python
1. Check firewall allows port 5000
2. Ensure Python bot is running first
3. Check EA logs (View → Experts)
4. Verify PythonHost is correct (localhost if local, VPS IP if remote)

### Bot Places Orders but They Fail
1. Check account has sufficient margin
2. Verify symbol is tradeable in your account
3. Check broker trading hours (EURUSD trades 23:00-21:00 UTC)
4. Review position restrictions (hedging vs netting mode)

### Performance is Poor
1. Validate trading strategy on historical data using backtesting
2. Check if machine learning models are trained
3. Monitor for overfitting (model works on training data, fails on new data)
4. Increase sample size for model training

### Connection Drops
1. Restart MetaTrader 5
2. Restart Python bot
3. Check if VPS is running (if using remote)
4. Implement reconnection logic (already in code)

## Next Steps

1. **Test on Demo:** Run bot for 1 week on demo
2. **Backtest Strategy:** Use historical data to verify performance
3. **Add More Models:** Include more ML models for better predictions
4. **Optimize Risk:** Adjust position sizing based on results
5. **Go to Live:** Once confident, switch to live account with small sizes

## VPS Deployment (24/7 Trading)

Recommended setup costs:
- **VPS Server**: $5-10/month
- **MetaTrader**: Free
- **Python Hosting**: Free
- **Total**: ~$6-11/month for automated trading platform

Example VPS providers:
- [Contabo](https://contabo.com): €3.99/month (Windows VPS)
- [DigitalOcean](https://www.digitalocean.com): $5-6/month
- [AWS EC2](https://aws.amazon.com): $10-50/month (varies by usage)

Once set up, your trading bot runs **24 hours a day, 5 days a week** automatically.
