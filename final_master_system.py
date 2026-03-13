"""Final Master System: Hybrid Micro-to-Prop Trading Bot

This is the "master" orchestration script that stitches together:
- micro account survival (VWAP mean-reversion)
- hybrid activation (momentum scalp + trend)
- full automation (ensemble ML + retraining)
- prop firm readiness (scaling, risk controls)

It is intended as a reference implementation. The real system should
use robust execution engines (ZeroMQ bridge, API connectors, order tracking)
and should be run in a containerized production environment.

Requirements:
- Python 3.11+
- pandas, numpy, scikit-learn, xgboost
- aiomql (for MT5 integration)
- tensorflow (optional, for LSTM training)

Usage:
    python final_master_system.py train
    python final_master_system.py backtest
    python final_master_system.py live
    python final_master_system.py report
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# TensorFlow is optional; training LSTM models will be skipped if unavailable
try:
    import tensorflow as tf
    from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout, LSTM)
    from tensorflow.keras.models import Sequential
    _HAS_TF = True
except ImportError:  # pragma: no cover
    _HAS_TF = False

# aiomql is used for MetaTrader connectivity (MT5)
try:
    from aiomql import Bot, ForexSymbol, OrderType, Strategy, TimeFrame
except ImportError:  # pragma: no cover
    Bot = None  # type: ignore
    ForexSymbol = None  # type: ignore
    OrderType = None  # type: ignore
    Strategy = object  # type: ignore
    TimeFrame = Enum('TimeFrame', {'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440})


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION LAYERS
# ═══════════════════════════════════════════════════════════════════════════


class AccountPhase(Enum):
    MICRO = 1
    BUILDING = 2
    GROWING = 3
    PROP = 4


@dataclass
class MasterConfig:
    """Master config: adaptive parameters per balance phase."""

    def get_phase(self, balance: float) -> AccountPhase:
        if balance < 500:
            return AccountPhase.MICRO
        if balance < 2000:
            return AccountPhase.BUILDING
        if balance < 10000:
            return AccountPhase.GROWING
        return AccountPhase.PROP

    def get_params(self, balance: float) -> Dict:
        phase = self.get_phase(balance)
        configs = {
            AccountPhase.MICRO: {
                'risk_per_trade': 0.02,
                'max_positions': 1,
                'max_daily_trades': 1,
                'timeframes': ['M15'],
                'strategies': ['vwap_micro'],
                'min_risk_reward': 2.0,
                'kill_switch_drawdown': -0.20,
                'position_sizes': {'micro_lots_max': 5}
            },
            AccountPhase.BUILDING: {
                'risk_per_trade': 0.015,
                'max_positions': 3,
                'max_daily_trades': 5,
                'timeframes': ['M5', 'M15'],
                'strategies': ['vwap_micro', 'momentum_scalp', 'trend_day'],
                'min_risk_reward': 1.5,
                'kill_switch_drawdown': -0.15,
                'position_sizes': {'micro_lots_max': 15}
            },
            AccountPhase.GROWING: {
                'risk_per_trade': 0.01,
                'max_positions': 6,
                'max_daily_trades': 10,
                'timeframes': ['M5', 'M15', 'H4'],
                'strategies': ['vwap_micro', 'momentum_scalp', 'trend_day', 'swing_trend'],
                'min_risk_reward': 1.5,
                'kill_switch_drawdown': -0.10,
                'position_sizes': {'micro_lots_max': 50}
            },
            AccountPhase.PROP: {
                'risk_per_trade': 0.01,
                'max_positions': 10,
                'max_daily_trades': 15,
                'timeframes': ['M1', 'M5', 'M15', 'H4', 'D1'],
                'strategies': ['full_hybrid'],
                'min_risk_reward': 1.3,
                'kill_switch_drawdown': -0.05,
                'position_sizes': {'standard_lots_max': 5.0}
            }
        }
        return configs[phase]


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY LAYER: ALL TIMEFRAMES
# ═══════════════════════════════════════════════════════════════════════════


class StrategyVWAPMicro:
    """VWAP mean-reversion strategy used in all phases."""

    def __init__(self, timeframe: TimeFrame = TimeFrame.M15):
        self.timeframe = timeframe
        self.name = "vwap_micro"

    def generate(self, df: pd.DataFrame) -> Optional[Dict]:
        df = df.copy()
        typical = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical * df['tick_volume']).groupby(df.index.date).cumsum() / (
            df['tick_volume'].groupby(df.index.date).cumsum()
        )
        df['std'] = df['close'].rolling(20).std()
        df['upper'] = df['vwap'] + (2 * df['std'])
        df['lower'] = df['vwap'] - (2 * df['std'])
        df['dist'] = (df['close'] - df['vwap']) / df['vwap']
        df['rsi'] = self._rsi(df['close'], 14)

        latest = df.iloc[-1]

        if latest['close'] < latest['lower'] and latest['dist'] < -0.002 and latest['rsi'] > 25:
            return {
                'direction': 'long',
                'entry': latest['close'],
                'stop': latest['close'] * 0.995,
                'target': latest['vwap'],
                'risk_reward': (latest['vwap'] - latest['close']) / (latest['close'] * 0.005),
                'confidence': min(abs(latest['dist']) * 500, 0.9),
                'setup': 'vwap_bounce'
            }

        if latest['close'] > latest['upper'] and latest['dist'] > 0.002 and latest['rsi'] < 75:
            return {
                'direction': 'short',
                'entry': latest['close'],
                'stop': latest['close'] * 1.005,
                'target': latest['vwap'],
                'risk_reward': (latest['close'] - latest['vwap']) / (latest['close'] * 0.005),
                'confidence': min(abs(latest['dist']) * 500, 0.9),
                'setup': 'vwap_reject'
            }

        return None

    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))


class StrategyMomentumScalp:
    """Momentum scalp strategy for phases 2+."""

    def __init__(self, timeframe: TimeFrame = TimeFrame.M5):
        self.timeframe = timeframe
        self.name = "momentum_scalp"

    def generate(self, df: pd.DataFrame) -> Optional[Dict]:
        df = df.copy()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['vol_sma'] = df['tick_volume'].rolling(20).mean()
        df['vol_ratio'] = df['tick_volume'] / df['vol_sma']
        df['rsi_7'] = self._rsi(df['close'], 7)
        df['mom'] = df['close'] - df['close'].shift(3)

        latest = df.iloc[-1]

        if (
            latest['ema_9'] > latest['ema_21'] and
            df['ema_9'].iloc[-2] <= df['ema_21'].iloc[-2] and
            latest['vol_ratio'] > 1.5 and
            50 < latest['rsi_7'] < 70 and
            latest['mom'] > 0
        ):
            return {
                'direction': 'long',
                'entry': latest['close'],
                'stop': latest['ema_21'],
                'target': latest['close'] + (latest['close'] - latest['ema_21']) * 1.5,
                'risk_reward': 1.5,
                'confidence': 0.75 if latest['vol_ratio'] > 2.0 else 0.65,
                'setup': 'momentum_break'
            }

        if (
            latest['ema_9'] < latest['ema_21'] and
            df['ema_9'].iloc[-2] >= df['ema_21'].iloc[-2] and
            latest['vol_ratio'] > 1.5 and
            30 < latest['rsi_7'] < 50 and
            latest['mom'] < 0
        ):
            return {
                'direction': 'short',
                'entry': latest['close'],
                'stop': latest['ema_21'],
                'target': latest['close'] - (latest['ema_21'] - latest['close']) * 1.5,
                'risk_reward': 1.5,
                'confidence': 0.75 if latest['vol_ratio'] > 2.0 else 0.65,
                'setup': 'momentum_fade'
            }

        return None

    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))


class StrategySwingTrend:
    """Swing trend strategy for phases 3+."""

    def __init__(self, timeframe: TimeFrame = TimeFrame.H4):
        self.timeframe = timeframe
        self.name = "swing_trend"

    def generate(self, df: pd.DataFrame) -> Optional[Dict]:
        df = df.copy()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['trend_up'] = df['sma_50'] > df['sma_200']
        df['adx'] = self._adx(df, 14)
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['dist_ema'] = (df['close'] - df['ema_20']) / df['ema_20']
        df['rsi'] = self._rsi(df['close'], 14)

        latest = df.iloc[-1]

        if (
            latest['trend_up'] and
            latest['adx'] > 25 and
            latest['dist_ema'] < -0.01 and
            latest['rsi'] > 40
        ):
            return {
                'direction': 'long',
                'entry': latest['close'],
                'stop': latest['sma_50'],
                'target': latest['close'] * 1.06,
                'risk_reward': 3.0,
                'confidence': 0.70,
                'setup': 'trend_pullback',
                'hold_days': 5
            }

        if (
            not latest['trend_up'] and
            latest['adx'] > 25 and
            latest['dist_ema'] > 0.01 and
            latest['rsi'] < 60
        ):
            return {
                'direction': 'short',
                'entry': latest['close'],
                'stop': latest['sma_50'],
                'target': latest['close'] * 0.94,
                'risk_reward': 3.0,
                'confidence': 0.70,
                'setup': 'trend_pullback_short',
                'hold_days': 5
            }

        return None

    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION & CAPITAL ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════


class RegimeEngine:
    """Detects market regime and allocates capital."""

    def __init__(self):
        self.regime_history: List[str] = []

    def analyze(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame, df_h4: pd.DataFrame) -> Dict:
        atr_m5 = self._atr(df_m5, 14)
        vol_percentile = atr_m5.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        adx_h4 = self._adx(df_h4, 14)
        trend_strength = adx_h4.iloc[-1] / 100
        hour = datetime.utcnow().hour

        if vol_percentile.iloc[-1] > 0.8:
            regime = 'high_volatility'
        elif trend_strength > 0.5:
            regime = 'strong_trend'
        elif trend_strength < 0.25:
            regime = 'range_bound'
        elif 8 <= hour <= 10:
            regime = 'opening_drive'
        elif 14 <= hour <= 16:
            regime = 'closing_session'
        else:
            regime = 'normal'

        self.regime_history.append(regime)

        allocations = {
            'high_volatility': {
                'vwap_micro': 0.2,
                'momentum_scalp': 0.6,
                'trend_day': 0.2,
                'swing_trend': 0.0
            },
            'strong_trend': {
                'vwap_micro': 0.1,
                'momentum_scalp': 0.2,
                'trend_day': 0.3,
                'swing_trend': 0.4
            },
            'range_bound': {
                'vwap_micro': 0.5,
                'momentum_scalp': 0.3,
                'trend_day': 0.2,
                'swing_trend': 0.0
            },
            'opening_drive': {
                'vwap_micro': 0.2,
                'momentum_scalp': 0.7,
                'trend_day': 0.1,
                'swing_trend': 0.0
            },
            'closing_session': {
                'vwap_micro': 0.3,
                'momentum_scalp': 0.2,
                'trend_day': 0.3,
                'swing_trend': 0.2
            },
            'normal': {
                'vwap_micro': 0.3,
                'momentum_scalp': 0.3,
                'trend_day': 0.3,
                'swing_trend': 0.1
            }
        }

        return {
            'regime': regime,
            'allocation': allocations.get(regime, allocations['normal']),
            'trend_strength': trend_strength,
            'volatility_percentile': float(vol_percentile.iloc[-1])
        }

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(period).mean()

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
# ML TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


class MLTrainingPipeline:
    """End-to-end ML training for multi-timeframe ensemble."""

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def fetch_data(self, timeframe: str, bars: int = 5000) -> pd.DataFrame:
        """Fetch historical OHLCV data from MT5 via aiomql."""
        from aiomql.core.sync import MetaTrader as SyncMT5

        mt5 = SyncMT5()
        mt5.initialize_sync()

        tf_map = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }

        rates = mt5._copy_rates_from_pos(self.symbol, tf_map[timeframe], 0, bars)
        mt5.shutdown()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def engineer_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Feature engineering across all timeframes."""
        df = df.copy()
        for period in [1, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()

        df['atr'] = self._atr(df, 14)
        df['bb_width'] = self._bb_width(df, 20)

        df['rsi'] = self._rsi(df['close'], 14)
        df['macd'], df['macd_signal'], _ = self._macd(df['close'])

        df['volume_sma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']

        if timeframe in ['M5', 'M15']:
            df['hour'] = df.index.hour
            df['vwap'] = self._vwap(df)
            df['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        else:
            df['sma_200'] = df['close'].rolling(200).mean()
            df['adx'] = self._adx(df, 14)

        return df

    def create_labels(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """Create triple-barrier target labels."""
        params = {
            'M5': {'horizon': 3, 'profit': 0.003, 'stop': 0.002},
            'M15': {'horizon': 4, 'profit': 0.005, 'stop': 0.003},
            'H4': {'horizon': 10, 'profit': 0.015, 'stop': 0.010},
            'D1': {'horizon': 5, 'profit': 0.03, 'stop': 0.02}
        }
        p = params.get(timeframe, params['M15'])

        labels = pd.Series(0, index=df.index)
        for i in range(len(df) - p['horizon']):
            price = df['close'].iloc[i]
            future = df.iloc[i + 1:i + p['horizon'] + 1]
            max_up = (future['high'].max() - price) / price
            max_down = (future['low'].min() - price) / price

            if max_up >= p['profit'] and max_down > -p['stop']:
                labels.iloc[i] = 1
            elif max_down <= -p['profit'] and max_up < p['stop']:
                labels.iloc[i] = -1

        return labels

    def train_ensemble(self, X: np.ndarray, y: np.ndarray, name: str) -> Dict:
        """Train RandomForest + XGBoost ensemble."""
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        self.scalers[name] = scaler

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train_s, y_train)

        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise RuntimeError("xgboost is required for the ensemble training")

        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb.fit(X_train_s, y_train)

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
        ensemble.fit(X_train_s, y_train)

        preds = ensemble.predict(X_test_s)
        f1 = f1_score(y_test, preds, average='weighted')

        self.models[name] = ensemble
        return {
            'model': ensemble,
            'f1_score': f1,
            'scaler': scaler,
            'feature_importance': dict(zip(range(X.shape[1]), rf.feature_importances_))
        }

    def train_lstm(self, X: np.ndarray, y: np.ndarray, name: str, seq_len: int = 20):
        """Train an LSTM sequence model."""
        if not _HAS_TF:
            logger.warning("TensorFlow not installed; skipping LSTM training")
            return None

        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len] + 1)

        X_seq = np.array(X_seq)
        y_seq = tf.keras.utils.to_categorical(y_seq, 3)

        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        for i in range(X_train.shape[2]):
            mean = X_train[:, :, i].mean()
            std = X_train[:, :, i].std()
            X_train[:, :, i] = (X_train[:, :, i] - mean) / (std + 1e-8)
            X_test[:, :, i] = (X_test[:, :, i] - mean) / (std + 1e-8)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[1])),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )

        self.models[f"{name}_lstm"] = model
        return model

    def train_all_timeframes(self):
        timeframes = ['M5', 'M15', 'H4']
        for tf in timeframes:
            logger.info(f"Training {tf} model...")
            df = self.fetch_data(tf, 5000)
            df = self.engineer_features(df, tf)
            df['target'] = self.create_labels(df, tf)
            df = df.dropna()

            exclude = ['open', 'high', 'low', 'close', 'tick_volume', 'target', 'spread']
            features = [c for c in df.columns if c not in exclude]

            X = df[features].values
            y = df['target'].values

            result = self.train_ensemble(X, y, f"{tf}_ensemble")
            logger.info(f"{tf} F1: {result['f1_score']:.4f}")

            if len(X) > 2000:
                self.train_lstm(X, y, tf, seq_len=20)

        self.save_models()

    def save_models(self):
        package = {
            'models': self.models,
            'scalers': self.scalers,
            'timestamp': datetime.now().isoformat()
        }
        path = Path('models')
        path.mkdir(exist_ok=True)
        dst = path / f"hybrid_master_{datetime.now().strftime('%Y%m%d')}.joblib"
        joblib.dump(package, dst)
        logger.info(f"Models saved to {dst}")

    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))

    def _macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(period).mean()

    def _bb_width(self, df: pd.DataFrame, period: int) -> pd.Series:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (upper - lower) / sma

    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        typical = (df['high'] + df['low'] + df['close']) / 3
        return (typical * df['tick_volume']).groupby(df.index.date).cumsum() / (
            df['tick_volume'].groupby(df.index.date).cumsum()
        )

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
# MASTER BOT: INTEGRATION LAYER
# ═══════════════════════════════════════════════════════════════════════════


class MasterHybridBot(Strategy):
    """Hybrid master bot integrating strategies, ML, and risk controls."""

    parameters = {
        'symbol': 'EURUSD',
        'auto_phase': True,
        'ml_confidence_threshold': 0.6,
    }

    def __init__(self, symbol: ForexSymbol, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self.config = MasterConfig()
        self.regime_engine = RegimeEngine()
        self.strategies = {
            'vwap_micro': StrategyVWAPMicro(TimeFrame.M15),
            'momentum_scalp': StrategyMomentumScalp(TimeFrame.M5),
            'trend_day': StrategyVWAPMicro(TimeFrame.M15),
            'swing_trend': StrategySwingTrend(TimeFrame.H4)
        }
        self.active_signals: List[Dict] = []
        self.daily_stats = {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        self.phase_history: List[AccountPhase] = []
        self.consecutive_losses = 0
        self.ml_models = self._load_ml_models()

    def _load_ml_models(self) -> Dict:
        latest = None
        for path in sorted(Path('models').glob('hybrid_master_*.joblib'), reverse=True):
            latest = path
            break

        if not latest:
            logger.warning('No ML models found, using rule-based signals only')
            return {}

        package = joblib.load(latest)
        logger.info(f"Loaded ML models from {latest}")
        return package.get('models', {})

    async def trade(self):
        account = await self.account.info()
        balance = account.balance
        equity = account.equity
        phase = self.config.get_phase(balance)
        params = self.config.get_params(balance)

        today = datetime.utcnow().date()
        if not hasattr(self, 'last_trade_date') or self.last_trade_date != today:
            self.daily_stats = {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
            self.last_trade_date = today
            self.consecutive_losses = 0

        if self._check_kill_switches(balance, equity, params):
            return

        data = await self._fetch_all_data()
        regime = self.regime_engine.analyze(data['M5'], data['M15'], data['H4'])

        signals = []
        for name, strategy in self.strategies.items():
            if name not in params['strategies']:
                continue
            allocation = regime['allocation'].get(name, 0.1)
            if allocation < 0.1:
                continue
            df = data.get(strategy.timeframe.name)
            if df is None or len(df) < 50:
                continue
            signal = strategy.generate(df)
            if signal:
                signal['strategy'] = name
                signal['allocation'] = allocation
                signal['regime'] = regime['regime']
                signal['timeframe'] = strategy.timeframe.name
                signals.append(signal)

        if self.ml_models:
            signals = self._enhance_with_ml(signals, data)

        signals = sorted(signals, key=lambda x: (x['confidence'] * x['risk_reward']), reverse=True)

        max_positions = params['max_positions']
        current_positions = len(self.active_signals)

        for signal in signals[: max_positions - current_positions]:
            if self.daily_stats['trades'] >= params['max_daily_trades']:
                break
            if signal['risk_reward'] < params['min_risk_reward']:
                continue
            await self._execute_signal(signal, balance, params)

        await self._manage_positions()
        self._log_status(balance, phase, regime, signals)

    def _check_kill_switches(self, balance: float, equity: float, params: Dict) -> bool:
        daily_loss_pct = self.daily_stats['pnl'] / max(balance, 1)
        if daily_loss_pct <= -0.06:
            logger.error(f"Daily loss limit hit: {daily_loss_pct:.2%}")
            return True

        if self.consecutive_losses >= 3:
            logger.error(f"Consecutive loss limit hit: {self.consecutive_losses}")
            return True

        if balance < 100 * (1 + params['kill_switch_drawdown']):
            logger.error(f"Phase kill switch triggered: {params['kill_switch_drawdown']}")
            return True

        return False

    async def _fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for tf in [TimeFrame.M5, TimeFrame.M15, TimeFrame.H4]:
            df = await self.symbol.copy_rates_from_pos(timeframe=tf, count=200)
            data[tf.name] = df
        return data

    def _enhance_with_ml(self, signals: List[Dict], data: Dict) -> List[Dict]:
        enhanced: List[Dict] = []
        for signal in signals:
            model_name = f"{signal['strategy'].split('_')[0]}_ensemble"
            model = self.ml_models.get(model_name)
            scaler = self.ml_models.get(f"{model_name}_scaler")
            if not model:
                enhanced.append(signal)
                continue

            df = data.get(signal.get('timeframe', 'M15'))
            if df is None or len(df) < 50:
                enhanced.append(signal)
                continue

            features = self._extract_features(df)
            if scaler is not None:
                features = scaler.transform([features])
            proba = model.predict_proba([features])[0]
            ml_confidence = max(proba)
            ml_direction = int(np.argmax(proba) - 1)

            if (signal['direction'] == 'long' and ml_direction == 1) or (
                signal['direction'] == 'short' and ml_direction == -1
            ):
                signal['confidence'] = (signal['confidence'] + ml_confidence) / 2
                signal['ml_agreement'] = True
                enhanced.append(signal)
            elif ml_direction == 0:
                signal['confidence'] *= 0.7
                signal['ml_agreement'] = False
                enhanced.append(signal)

        return enhanced

    def _extract_features(self, df: pd.DataFrame) -> List[float]:
        latest = df.iloc[-1]
        ema_diff = latest.get('ema_9', 0) - latest.get('ema_21', 0)
        return [
            float(latest.get('rsi', 50)),
            float(latest.get('atr', 0) / max(latest.get('close', 1), 1)),
            float(latest.get('volume_ratio', 1)),
            float(latest.get('dist_vwap', 0)),
            float(ema_diff)
        ]

    async def _execute_signal(self, signal: Dict, balance: float, params: Dict):
        risk_amount = balance * params['risk_per_trade'] * signal.get('allocation', 1.0)
        if 'micro_lots_max' in params['position_sizes']:
            pip_value = 0.10
            stop_pips = abs(signal['entry'] - signal['stop']) / 0.0001
            micro_lots = risk_amount / (stop_pips * pip_value)
            micro_lots = min(micro_lots, params['position_sizes']['micro_lots_max'])
            micro_lots = max(micro_lots, 1)
            volume = micro_lots * 0.01
        else:
            volume = risk_amount / max(abs(signal['entry'] - signal['stop']), 1e-8)

        order_type = OrderType.BUY if signal['direction'] == 'long' else OrderType.SELL
        result = await self.trade(order_type)

        if result:
            self.active_signals.append({
                'ticket': result.order,
                'signal': signal,
                'volume': volume,
                'risk_amount': risk_amount,
                'entry_time': datetime.utcnow()
            })
            self.daily_stats['trades'] += 1
            logger.info(
                f"EXECUTED: {signal['strategy']} {signal['direction']} | "
                f"Risk: ${risk_amount:.2f} | R/R: {signal['risk_reward']:.1f}"
            )

    async def _manage_positions(self):
        for pos in list(self.active_signals):
            position = await self.get_position(pos['ticket'])
            if not position:
                pnl = getattr(position, 'profit', 0) if position else 0
                self.daily_stats['pnl'] += pnl
                if pnl > 0:
                    self.daily_stats['wins'] += 1
                    self.consecutive_losses = 0
                else:
                    self.daily_stats['losses'] += 1
                    self.consecutive_losses += 1
                self.active_signals.remove(pos)
                continue

            current_price = position.price_current
            entry = pos['signal']['entry']
            stop = pos['signal']['stop']

            if pos['signal']['direction'] == 'long':
                r_multiple = (current_price - entry) / max(entry - stop, 1e-8)
                if r_multiple >= 1.5 and stop < entry:
                    new_stop = entry
                    await self.modify_stop(pos['ticket'], new_stop)
                    pos['signal']['stop'] = new_stop
            else:
                r_multiple = (entry - current_price) / max(stop - entry, 1e-8)
                if r_multiple >= 1.5 and stop > entry:
                    new_stop = entry
                    await self.modify_stop(pos['ticket'], new_stop)
                    pos['signal']['stop'] = new_stop

    def _log_status(self, balance: float, phase: AccountPhase, regime: Dict, signals: List[Dict]):
        logger.info(
            f"\n"  
            "═══════════════════════════════════════════\n"
            f"MASTER BOT STATUS\n"
            f"═══════════════════════════════════════════\n"
            f"Balance: ${balance:.2f} | Phase: {phase.name}\n"
            f"Regime: {regime['regime']}\n"
            f"Daily P&L: ${self.daily_stats['pnl']:.2f}\n"
            f"Trades Today: {self.daily_stats['trades']}\n"
            f"Active Positions: {len(self.active_signals)}\n"
            f"Signals Queued: {len(signals)}\n"
            f"Consecutive Losses: {self.consecutive_losses}\n"
            f"═══════════════════════════════════════════\n"
        )

    async def get_position(self, ticket: int):
        positions = await self.symbol.positions_get()
        for pos in positions:
            if getattr(pos, 'ticket', None) == ticket:
                return pos
        return None

    async def modify_stop(self, ticket: int, new_stop: float):
        # The actual modification depends on broker API. If using aiomql, implement accordingly.
        try:
            await self.modify_order(ticket, stop=new_stop)  # type: ignore
        except Exception:
            logger.warning("modify_stop is not implemented for this broker interface")


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION & TRAINING WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════


def run_training():
    trainer = MLTrainingPipeline("EURUSD")
    trainer.train_all_timeframes()


def run_backtest():
    logger.warning("Backtesting logic is not implemented in this reference file.")


def run_live():
    if Bot is None or ForexSymbol is None:
        raise RuntimeError("aiomql is not installed; live trading is disabled.")

    bot = Bot()
    master = MasterHybridBot(symbol=ForexSymbol("EURUSD"), name="MasterHybrid_v1")
    bot.add_strategy(master)
    bot.execute()


def generate_report():
    logger.warning("Report generation is not implemented in this reference file.")


COMMANDS = {
    'train': run_training,
    'backtest': run_backtest,
    'live': run_live,
    'report': generate_report
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        COMMANDS[sys.argv[1]]()
    else:
        print("Usage: python final_master_system.py [train|backtest|live|report]")
        print("\nRecommended workflow:")
        print("1. python final_master_system.py train    # Train all models")
        print("2. python final_master_system.py backtest # Validate for 6 months")
        print("3. python final_master_system.py live     # Start with $100")
        print("4. python final_master_system.py report   # Weekly analytics")
