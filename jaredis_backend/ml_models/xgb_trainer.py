"""Production-grade ML training components for trading.

This module implements the best-practice ML training methods for trading bots,
including XGBoost, LSTM, and an ensemble that combines both.

It is designed to be compatible with the existing system structure and can be
used directly from the retraining pipeline or from standalone scripts.

Usage:
    from jaredis_backend.ml_models.xgb_trainer import XGBoostTradingModel

    model = XGBoostTradingModel(symbol='EURUSD', timeframe='M15')
    df = ...  # load OHLCV data
    X, y, features = model.prepare_data(df)
    model.train(X, y)
    model.save('models/xgb_eurusd.joblib')

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout, LSTM)
    from tensorflow.keras.models import Sequential
    _HAS_TF = True
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    Sequential = None  # type: ignore
    _HAS_TF = False

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ═══════════════════════════════════════════════════════════════════════════
# XGBOOST TRAINING MODEL
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class XGBoostTradingModel:
    """Production-grade XGBoost for trading."""

    symbol: str = "EURUSD"
    timeframe: str = "M15"
    prediction_horizon: int = 4
    target_type: str = "directional"  # directional, returns, triple_barrier, meta

    model: Optional["xgb.XGBClassifier"] = None  # type: ignore
    scaler: StandardScaler = StandardScaler()
    feature_names: List[str] = None  # type: ignore
    best_params: Dict = None  # type: ignore
    training_metrics: Dict = None  # type: ignore

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a robust feature set for XGBoost."""
        df = df.copy()

        # Price action
        for period in [1, 2, 3, 5, 8, 13, 21]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['volatility_regime'] = pd.qcut(df['volatility'].fillna(0), q=5, labels=[1, 2, 3, 4, 5])

        df['position_in_range'] = (
            (df['close'] - df['low'].rolling(20).min()) /
            (df['high'].rolling(20).max() - df['low'].rolling(20).min()).replace(0, np.nan)
        )

        # Technical indicators
        for period in [9, 21, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'].replace(0, np.nan)
            df[f'slope_ema_{period}'] = df[f'ema_{period}'].diff(5)

        df['rsi_14'] = self._rsi(df['close'], 14)
        df['rsi_7'] = self._rsi(df['close'], 7)
        df['rsi_slope'] = df['rsi_14'].diff(5)
        df['rsi_divergence'] = df['rsi_14'] - df['close'].pct_change(14) * 100

        df['macd'], df['macd_signal'], df['macd_hist'] = self._macd(df['close'])
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)

        df['atr_14'] = self._atr(df, 14)
        df['atr_normalized'] = df['atr_14'] / df['close'].replace(0, np.nan)

        df['adx'] = self._adx(df, 14)
        df['trending'] = (df['adx'] > 25).astype(int)

        # Volume
        df['volume_sma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma'].replace(0, np.nan)
        df['volume_trend'] = df['volume_ratio'].rolling(5).mean()

        df['obv'] = (np.sign(df['close'].diff()) * df['tick_volume']).cumsum()
        df['obv_slope'] = df['obv'].diff(10)
        df['obv_divergence'] = df['obv'] - df['obv'].rolling(20).mean()

        df['volume_at_high'] = (
            ((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan) > 0.7).astype(int)
            * df['tick_volume']
        )
        df['volume_at_low'] = (
            ((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan) < 0.3).astype(int)
            * df['tick_volume']
        )

        # Market microstructure
        df['spread'] = df.get('spread', 0)
        df['spread_percentile'] = df['spread'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0
        )

        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, np.nan)
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        df['gap_filled'] = (
            ((df['gap'] > 0) & (df['low'] <= df['close'].shift(1))).astype(int)
            | ((df['gap'] < 0) & (df['high'] >= df['close'].shift(1))).astype(int)
        )

        # Temporal
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
        df['is_asia'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Lag features
        for lag in [1, 2, 3, 5, 8]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume_ratio'].shift(lag)

        # Interaction
        df['rsi_volume'] = df['rsi_14'] * df['volume_ratio']
        df['trend_volatility'] = df['trending'] * df['volatility']
        df['ema_distance_ratio'] = df['dist_ema_9'] / df['dist_ema_50'].replace(0, np.nan)

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create label series based on configured target_type."""
        if self.target_type == "directional":
            future_return = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
            labels = pd.Series(0, index=df.index)
            labels[future_return > 0.001] = 1
            labels[future_return < -0.001] = -1
        elif self.target_type == "triple_barrier":
            labels = self._triple_barrier_labels(df)
        elif self.target_type == "returns":
            labels = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        elif self.target_type == "meta":
            labels = self._create_meta_labels(df)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
        return labels

    def _triple_barrier_labels(self, df: pd.DataFrame) -> pd.Series:
        labels = pd.Series(0, index=df.index)
        for i in range(len(df) - self.prediction_horizon):
            price = df['close'].iloc[i]
            future = df.iloc[i+1:i+self.prediction_horizon+1]
            volatility = df['atr_14'].iloc[i] / price
            profit = volatility * 2
            stop = volatility * 1
            max_up = (future['high'].max() - price) / price
            max_down = (future['low'].min() - price) / price

            up_idx = future[future['high'] >= price * (1 + profit)].index.min() if (future['high'] >= price * (1 + profit)).any() else None
            down_idx = future[future['low'] <= price * (1 - stop)].index.min() if (future['low'] <= price * (1 - stop)).any() else None

            if up_idx is not None and (down_idx is None or up_idx < down_idx):
                labels.iloc[i] = 1
            elif down_idx is not None and (up_idx is None or down_idx < up_idx):
                labels.iloc[i] = -1
        return labels

    def _create_meta_labels(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        df = self.create_advanced_features(df)
        df['target'] = self.create_labels(df)

        exclude = [
            'open', 'high', 'low', 'close', 'tick_volume', 'target', 'spread',
            'real_volume', 'volatility_regime'
        ]

        self.feature_names = [
            c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32']
        ]

        df_clean = df[self.feature_names + ['target']].dropna()
        X = df_clean[self.feature_names]
        y = df_clean['target']
        return X, y, self.feature_names

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict:
        if optuna is None:
            raise RuntimeError("optuna is required for hyperparameter optimization")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0)
            }
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
                model.fit(X_train_s, y_train)
                preds = model.predict(X_val_s)
                f1 = f1_score(y_val, preds, average='weighted')
                scores.append(f1)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best F1: {study.best_value:.4f}")
        return self.best_params

    def train(self, X: pd.DataFrame, y: pd.Series, optimize: bool = True, use_gpu: bool = False) -> Dict:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))

        if optimize or not self.best_params:
            self.optimize_hyperparameters(X_train, y_train, n_trials=50)

        params: Dict = {
            **self.best_params,
            'objective': 'multi:softprob',
            'num_class': len(classes),
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }

        if use_gpu:
            params['tree_method'] = 'gpu_hist'

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )

        train_pred = self.model.predict(X_train_s)
        test_pred = self.model.predict(X_test_s)
        test_proba = self.model.predict_proba(X_test_s)

        self.training_metrics = {
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'test_precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, test_pred, average='weighted', zero_division=0),
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }

        logger.info(f"Training complete: {self.training_metrics}")

        return self.training_metrics

    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained")
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        if shap is not None:
            explainer = shap.TreeExplainer(self.model)
            return importance.head(top_n)
        return importance.head(top_n)

    def predict_with_confidence(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained")
        X_s = self.scaler.transform(X)
        proba = self.model.predict_proba(X_s)
        results = pd.DataFrame(index=X.index)
        results['prediction'] = self.model.predict(X_s)
        results['confidence'] = np.max(proba, axis=1)
        results['long_prob'] = proba[:, 2] if proba.shape[1] > 2 else proba[:, 1]
        results['short_prob'] = proba[:, 0]
        results['neutral_prob'] = proba[:, 1] if proba.shape[1] > 2 else 0
        results['valid_signal'] = results['confidence'] >= confidence_threshold
        results['position_size_hint'] = results.apply(
            lambda row: (row['confidence'] - 0.5) * 2 if row['valid_signal'] else 0,
            axis=1
        )
        return results

    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series,
                                 train_size: int = 1000,
                                 test_size: int = 200,
                                 step: int = 50) -> Dict:
        returns = []
        predictions = []
        actuals = []
        for i in range(train_size, len(X) - test_size, step):
            X_train = X.iloc[i-train_size:i]
            y_train = y.iloc[i-train_size:i]
            X_test = X.iloc[i:i+test_size]
            y_test = y.iloc[i:i+test_size]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)
            for j, (pred, actual) in enumerate(zip(preds, y_test)):
                confidence = np.max(proba[j])
                if confidence > 0.6:
                    if pred == 1 and actual == 1:
                        returns.append(0.005)
                    elif pred == 1 and actual == -1:
                        returns.append(-0.003)
                    elif pred == -1 and actual == -1:
                        returns.append(0.005)
                    elif pred == -1 and actual == 1:
                        returns.append(-0.003)
                    else:
                        returns.append(0)
                    predictions.append(pred)
                    actuals.append(actual)
        returns = np.array(returns)
        win_rate = np.mean(np.array(predictions) == np.array(actuals)) if predictions else 0
        total_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 26)) if np.std(returns) > 0 else 0
        max_drawdown = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min() if len(returns) > 0 else 0
        profit_factor = abs(sum(returns[returns > 0])) / abs(sum(returns[returns < 0])) if sum(returns[returns < 0]) != 0 else float('inf')
        return {
            'total_trades': len([r for r in returns if r != 0]),
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }

    def save(self, path: str):
        package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'training_metrics': self.training_metrics,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'horizon': self.prediction_horizon,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(package, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        package = joblib.load(path)
        self.model = package['model']
        self.scaler = package['scaler']
        self.feature_names = package['feature_names']
        self.best_params = package['best_params']
        self.training_metrics = package['training_metrics']
        logger.info(f"Model loaded from {path}")
        return self

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

    def _bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0):
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

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
# LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════

dataclass
class LSTMTradingModel:
    sequence_length: int = 20
    prediction_horizon: int = 4
    symbol: str = "EURUSD"
    model: Optional["tf.keras.Model"] = None  # type: ignore
    scaler: StandardScaler = StandardScaler()

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, n_features: int):
        if not _HAS_TF:
            raise RuntimeError("TensorFlow is required for LSTM training")
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, n_features),
                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.3),
            BatchNormalization(),
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score(average='weighted')]
        )
        return model

    def train(self, X: pd.DataFrame, y: pd.Series, epochs: int = 100, batch_size: int = 32) -> Dict:
        if not _HAS_TF:
            raise RuntimeError("TensorFlow is required for LSTM training")

        X_seq, y_seq = self.create_sequences(X.values, y.values)
        y_seq = y_seq + 1
        y_cat = tf.keras.utils.to_categorical(y_seq, num_classes=3)
        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_cat[:split], y_cat[split:]
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.scaler.transform(X_test_reshaped).reshape(-1, n_timesteps, n_features)
        self.model = self.build_model(n_features)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=15, restore_best_weights=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            tf.keras.callbacks.ModelCheckpoint(f'models/lstm_{self.symbol}_best.h5', monitor='val_f1_score', save_best_only=True, mode='max')
        ]
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        test_loss, test_acc, test_f1 = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        return {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'final_epoch': len(history.history['loss']),
            'history': history.history
        }

    def predict(self, X_recent: pd.DataFrame) -> pd.DataFrame:
        if len(X_recent) < self.sequence_length:
            raise ValueError(f"Need {self.sequence_length} rows, got {len(X_recent)}")
        X_seq = X_recent.values[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped).reshape(1, self.sequence_length, -1)
        proba = self.model.predict(X_scaled, verbose=0)[0]
        return pd.DataFrame([{
            'prediction': np.argmax(proba) - 1,
            'confidence': np.max(proba),
            'long_prob': proba[2],
            'neutral_prob': proba[1],
            'short_prob': proba[0]
        }])


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE MODEL
# ═══════════════════════════════════════════════════════════════════════════


class EnsembleTradingModel:
    """Combine XGBoost and LSTM for hybrid predictions."""

    def __init__(self, symbol: str = "EURUSD", timeframe: str = "M15"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.xgb_model = XGBoostTradingModel(symbol, timeframe)
        self.lstm_model = LSTMTradingModel(symbol=symbol)
        self.weights = {'xgb': 0.6, 'lstm': 0.4}

    def train_both(self, df: pd.DataFrame) -> Dict:
        X_xgb, y_xgb, _ = self.xgb_model.prepare_data(df)
        xgb_results = self.xgb_model.train(X_xgb, y_xgb, optimize=True)
        lstm_results = self.lstm_model.train(X_xgb, y_xgb)
        return {'xgb': xgb_results, 'lstm': lstm_results}

    def predict_ensemble(self, X_xgb: pd.DataFrame, X_lstm: pd.DataFrame) -> Dict:
        xgb_pred = self.xgb_model.predict_with_confidence(X_xgb.iloc[-1:])
        lstm_pred = self.lstm_model.predict(X_lstm)
        combined_long = xgb_pred['long_prob'].iloc[0] * self.weights['xgb'] + lstm_pred['long_prob'].iloc[0] * self.weights['lstm']
        combined_short = xgb_pred['short_prob'].iloc[0] * self.weights['xgb'] + lstm_pred['short_prob'].iloc[0] * self.weights['lstm']
        combined_neutral = xgb_pred['neutral_prob'].iloc[0] * self.weights['xgb'] + lstm_pred['neutral_prob'].iloc[0] * self.weights['lstm']
        probs = [combined_short, combined_neutral, combined_long]
        prediction = np.argmax(probs) - 1
        confidence = max(probs)
        return {
            'prediction': prediction,
            'confidence': confidence,
            'long_prob': combined_long,
            'short_prob': combined_short,
            'neutral_prob': combined_neutral,
            'xgb_confidence': xgb_pred['confidence'].iloc[0],
            'lstm_confidence': lstm_pred['confidence'].iloc[0]
        }


# ═══════════════════════════════════════════════════════════════════════════
# RETRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


class RetrainingPipeline:
    """Automated retraining logic with concept drift detection."""

    def __init__(
        self,
        model_path: str = "models/",
        performance_threshold: float = 0.55,
        drift_threshold: float = 0.1
    ):
        self.model_path = Path(model_path)
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.performance_history: List[Dict] = []

    def check_performance_degradation(self, recent_trades: List[Dict]) -> bool:
        if len(recent_trades) < 20:
            return False
        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        win_rate = wins / len(recent_trades)
        returns = [t['pnl'] for t in recent_trades]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        self.performance_history.append({'win_rate': win_rate, 'sharpe': sharpe, 'timestamp': datetime.now()})
        if win_rate < 0.45 or sharpe < 0.5:
            logger.warning(f"Performance degraded: WR={win_rate:.2%}, Sharpe={sharpe:.2f}")
            return True
        return False

    def check_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
        from scipy.stats import ks_2samp
        drift_detected = False
        features_to_check = ['volatility', 'volume_ratio', 'rsi_14', 'atr_14']
        for feature in features_to_check:
            if feature in reference_data.columns and feature in current_data.columns:
                statistic, p_value = ks_2samp(reference_data[feature].dropna(), current_data[feature].dropna())
                if p_value < 0.05:
                    logger.warning(f"Drift detected in {feature}: p={p_value:.4f}")
                    drift_detected = True
        return drift_detected

    def retrain(self, new_data: pd.DataFrame, model_type: str = "ensemble") -> str:
        logger.info("Starting retraining...")
        if model_type == "xgb":
            model = XGBoostTradingModel()
            X, y, _ = model.prepare_data(new_data)
            model.train(X, y, optimize=True)
            path = self.model_path / f"xgb_retrained_{datetime.now().strftime('%Y%m%d')}.joblib"
            model.save(str(path))
        elif model_type == "lstm":
            model = LSTMTradingModel()
            X, y, _ = XGBoostTradingModel().prepare_data(new_data)
            model.train(X, y)
            path = self.model_path / f"lstm_retrained_{datetime.now().strftime('%Y%m%d')}.h5"
            model.model.save(str(path))
        else:
            model = EnsembleTradingModel()
            model.train_both(new_data)
            path = self.model_path / f"ensemble_retrained_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Retraining complete: {path}")
        return str(path)

    def run_scheduled(self, data_fetcher, check_interval_hours: int = 24):
        import time
        while True:
            current_data = data_fetcher()
            try:
                reference = pd.read_parquet(self.model_path / "reference_data.parquet")
                if self.check_data_drift(reference, current_data):
                    self.retrain(current_data)
                    current_data.to_parquet(self.model_path / "reference_data.parquet")
            except FileNotFoundError:
                current_data.to_parquet(self.model_path / "reference_data.parquet")
            time.sleep(check_interval_hours * 3600)


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════


def main():
    from aiomql.core.sync import MetaTrader as SyncMT5

    mt5 = SyncMT5()
    mt5.initialize_sync()

    rates = mt5._copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 5000)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    mt5.shutdown()

    xgb_model = XGBoostTradingModel("EURUSD", "M15", prediction_horizon=4)
    X, y, features = xgb_model.prepare_data(df)

    logger.info(f"Training XGBoost with {len(X)} samples, {len(features)} features")
    metrics = xgb_model.train(X, y, optimize=True, use_gpu=False)
    logger.info(f"Test F1: {metrics['test_f1']:.4f}")

    importance = xgb_model.analyze_feature_importance(20)
    logger.info("Top 10 Features:\n%s", importance.head(10))

    wf_results = xgb_model.walk_forward_validation(X, y)
    logger.info("Walk-Forward Results: %s", wf_results)

    xgb_model.save("models/xgb_eurusd_m15_v1.joblib")

    if _HAS_TF:
        logger.info("Training LSTM...")
        lstm_model = LSTMTradingModel(sequence_length=20)
        lstm_results = lstm_model.train(X, y, epochs=50)
        logger.info(f"LSTM Test F1: {lstm_results['test_f1']:.4f}")

        logger.info("Training Ensemble...")
        ensemble = EnsembleTradingModel("EURUSD", "M15")
        ensemble.train_both(df)
        logger.info("Training complete!")


if __name__ == "__main__":
    main()
