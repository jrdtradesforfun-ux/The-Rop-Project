# Quick Start Guide - 5 Minutes to Production

## 🚀 Fast Track Setup

### Step 1: Prepare Environment (1 min)
```bash
cd jaredis-smart

# Copy environment template
cp .env.example .env

# Edit with your credentials (minimum required)
cat > .env << EOF
MT5_LOGIN=your_account_id
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo

DB_PASSWORD=secure_password
GRAFANA_PASSWORD=admin
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF
```

### Step 2: Start Docker Stack (2 min)
```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# Verify services started
docker-compose ps

# Should show all services as "Up"
```

### Step 3: Train Initial Model (1 min)
```bash
# Quick model training on sample data
python -c "
from jaredis_backend.ml_models.training_pipeline import MLTrainingPipeline, ModelConfig
from jaredis_backend.ml_models.feature_engineer import FeatureEngineer
import numpy as np
import pandas as pd

# Generate sample data for demo
X = np.random.randn(500, 50)
y = np.random.randint(0, 3, 500)
feature_names = [f'feature_{i}' for i in range(50)]

config = ModelConfig(symbol='EURUSD', model_type='random_forest')
pipeline = MLTrainingPipeline(config)
results = pipeline.train(X, y, feature_names)
pipeline.save_model('models/eurusd_initial.joblib')

print(f'✅ Model trained! F1 Score: {results[\"test_f1\"]:.4f}')
"
```

### Step 4: Access Dashboards (1 min)
```
MLflow Dashboard:  http://localhost:5000
Grafana Dashboard: http://localhost:3000 (admin/admin)
Prefect Flow:      http://localhost:4200
Bot API:           http://localhost:8080
```

---

## ✅ Verify Setup Works

### Test 1: Feature Engineering
```python
from jaredis_backend.ml_models.feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=100, freq='H')
df = pd.DataFrame({
    'open': 1.0750 + np.random.randn(100) * 0.0001,
    'high': 1.0760 + np.random.randn(100) * 0.0001,
    'low': 1.0740 + np.random.randn(100) * 0.0001,
    'close': 1.0750 + np.random.randn(100) * 0.0001,
    'volume': np.random.randint(1000, 5000, 100)
}, index=dates)

# Engineer features
engineer = FeatureEngineer()
features = engineer.engineer_features(df)

print(f"✅ Generated {features.shape[1]} features from {features.shape[0]} bars")
```

### Test 2: Risk Validation
```python
from jaredis_backend.trading_engine.advanced_risk_manager import AdvancedRiskManager
from unittest.mock import Mock

rm = AdvancedRiskManager()

# Mock account data
rm.account_metrics = Mock()
rm.account_metrics.balance = 10000
rm.account_metrics.equity = 10000
rm.account_metrics.open_positions = 1
rm.account_metrics.daily_pnl = 0
rm.account_metrics.margin_level = 5.0

# Validate trade
is_valid, error = rm.validate_trade(
    symbol='EURUSD', action='BUY', volume=0.1,
    entry_price=1.0750, sl=1.0700, tp=1.0800
)

print(f"✅ Trade validation: {is_valid}" if is_valid else f"❌ {error}")
```

### Test 3: Monitoring & Metrics
```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up

# Check Grafana data source
curl http://localhost:3000/api/datasources

# Check MLflow experiments
curl http://localhost:5000/api/2.0/experiments/list
```

---

## 📊 Configure for Your Broker

### Edit Trading Parameters
```yaml
# config/trading_config.yaml
symbols:
  - EURUSD    # Add your pairs
  - GBPUSD
  - USDJPY

risk_management:
  max_risk_per_trade: 0.02      # 2% per trade
  max_daily_loss: 0.05           # 5% daily limit
  max_drawdown: 0.10             # 10% max drawdown
  
ml_model:
  model_type: random_forest
  retrain_frequency: daily
```

### Update MT5 Connection
```bash
# Edit .env
MT5_LOGIN=12345678          # Your account ID
MT5_PASSWORD=YourPassword   # Your password
MT5_SERVER=JustMarkets-Live # Your broker server
```

---

## 🤖 Start Live Bot

### Option A: Docker (Recommended)
```bash
# Already running from docker-compose up
docker logs -f trading-bot

# To restart
docker restart trading-bot
```

### Option B: Direct Python
```bash
python main.py \
  --account-size 10000 \
  --risk-per-trade 0.02 \
  --mql5-host localhost \
  --mql5-port 5555 \
  --log-level INFO
```

---

## 📈 Monitor Performance

### View Grafana Dashboards
1. Open http://localhost:3000
2. Login: admin / admin
3. Navigate to "Trading Performance" dashboard
4. Watch in real-time:
   - Win rate
   - Profit factor
   - Drawdown
   - Sharpe ratio

### Check MLflow Models
```bash
# View model registry
curl http://localhost:5000/api/2.0/registered-models/list

# Query best model
curl "http://localhost:5000/api/2.0/registered-models/get?name=eurusd_directional_v2"
```

### View Trading Logs
```bash
# Real-time logs
tail -f logs/bot.log

# Filter by level
grep ERROR logs/bot.log
grep TRADE logs/bot.log
```

---

## 🔄 Configure Automated Retraining

### Setup Daily Retraining
```python
# Create Prefect schedule
python -c "
from pipelines.ml_retraining_flow import retrain_model_flow
from prefect.deployments import Deployment
from prefect.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=retrain_model_flow,
    name='daily-retraining',
    schedule=CronSchedule(cron='0 2 * * *'),  # 2 AM daily
    parameters={'symbol': 'EURUSD'},
    work_queue_name='ml-training'
)
deployment.apply()

print('✅ Daily retraining scheduled')
"
```

---

## 🚨 Common Issues & Quick Fixes

### Issue: "Connection refused" on port 5555
```bash
# Check if ZeroMQ bridge is running
docker ps | grep trading-bot

# If not running, restart
docker restart trading-bot

# Check logs
docker logs trading-bot
```

**Solution**: Ensure MT5 is running with ZeroMQ EA attached

### Issue: "No data" in Grafana
```bash
# Check Prometheus has data
curl http://localhost:9090/api/v1/query?query='trades_executed_total'

# If empty, data needs to be generated by trading activity
```

**Solution**: Start live trading to generate metrics

### Issue: Model F1 score too low (< 0.50)
```bash
# Retrain with more data
python -c "
from pipelines.ml_retraining_flow import retrain_model_flow

result = retrain_model_flow(
    symbol='EURUSD',
    timeframe='H1',
    lookback_days=730  # Use 2 years
)

print(result)
"
```

**Solution**: Use more historical data or optimize features

---

## 📋 Daily Checklist

Running the bot every day? Use this checklist:

```
[ ] 08:00 - Check bot is running: docker-compose ps
[ ] 08:05 - View Grafana dashboard for overnight trades
[ ] 09:00 - Review P&L and win rate
[ ] 12:00 - Check system health: curl http://localhost:8080/health
[ ] 17:00 - Verify margin level and open positions
[ ] 20:00 - Check for any alerts or errors
[ ] 22:00 - Monitor equity before overnight
```

---

## 🆘 Need Help?

### Check Logs
```bash
# Main bot logs
docker logs trading-bot

# Prefect logs
docker logs prefect-agent

# MLflow logs
docker logs mlflow

# Full stack logs
docker-compose logs
```

### System Status
```bash
# Check all services
docker-compose ps

# View resource usage
docker stats

# Check database
docker exec postgres-mlflow psql -U mlflow -d mlflow -c "\d+"
```

### Read Documentation
- Detailed setup: [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Architecture overview: [README.md](README.md)

---

## 🎉 You're Ready!

Your trading bot is now:
✅ Monitoring market 24/7  
✅ Executing ML-powered trades  
✅ Managing risk automatically  
✅ Retraining models daily  
✅ Tracking performance in real-time  
✅ Alerting you to problems  

**Happy trading! 📈**

---

**Quick Links:**
- 🌐 Dashboard: http://localhost:3000
- 📊 MLflow: http://localhost:5000
- ⚙️ API: http://localhost:8080
- 🔄 Prefect: http://localhost:4200
