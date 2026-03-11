"""
Default configuration for Jaredis Smart Trading Backend
"""

# Trading Parameters
ACCOUNT_SIZE = 10000
RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_POSITIONS = 5
MAX_DAILY_LOSS_PERCENT = 0.05  # 5% daily loss limit

# MQL5 Connection
MQL5_HOST = "localhost"
MQL5_PORT = 5000
MQL5_TIMEOUT = 5.0
MQL5_RECONNECT_TRIES = 3

# Data Processing
LOOKBACK_PERIOD = 60  # Candles for ML input
FORECAST_PERIOD = 5   # Candles to forecast
DATA_DIR = "data"
MODELS_DIR = "models"

# Logging
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
LOG_FILE = "jaredis.log"

# Feature Engineering
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD_DEV = 2.0
ATR_PERIOD = 14

# Model Training
TRAIN_VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Trading Symbols
TRADING_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
]

# Timeframes (MT5 compatible)
TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "D1": 1440,
}
