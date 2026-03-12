"""
Comprehensive Test Suite for Jaredis Smart Trading Bot
Tests for ML pipeline, risk management, execution, and integration
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Import modules to test
from jaredis_backend.ml_models.feature_engineer import (
    FeatureEngineer, FeatureConfig, DataValidator, LabelGenerator
)
from jaredis_backend.ml_models.training_pipeline import (
    MLTrainingPipeline, ModelConfig
)
from jaredis_backend.trading_engine.advanced_risk_manager import (
    AdvancedRiskManager, RiskLimits, CorrelationMonitor
)
from jaredis_backend.mql5_bridge.zeromq_bridge import (
    MT5ZeroMQBridge, TradeAction, Tick, TradeResult
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data"""
    dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
    close = 1.0750 + np.random.randn(500).cumsum() * 0.0001
    
    df = pd.DataFrame({
        'open': close + np.random.randn(500) * 0.00005,
        'high': close + np.abs(np.random.randn(500) * 0.0001),
        'low': close - np.abs(np.random.randn(500) * 0.0001),
        'close': close,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    return df


@pytest.fixture
def feature_engineer():
    """Initialize feature engineer"""
    return FeatureEngineer(FeatureConfig(lookback_period=50))


@pytest.fixture
def risk_manager():
    """Initialize risk manager"""
    return AdvancedRiskManager(
        limits=RiskLimits(
            max_risk_per_trade=0.02,
            max_open_positions=5,
            max_daily_loss=0.05,
            max_drawdown=0.10
        )
    )


# ==================== FEATURE ENGINEERING TESTS ====================

class TestFeatureEngineering:
    """Test feature engineering pipeline"""
    
    def test_feature_engineer_creates_features(self, sample_ohlcv_data, feature_engineer):
        """Test that features are created"""
        result = feature_engineer.engineer_features(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert result.shape[1] > sample_ohlcv_data.shape[1]  # More columns
    
    def test_feature_engineer_includes_technical_indicators(self, sample_ohlcv_data, feature_engineer):
        """Test that technical indicators are included"""
        result = feature_engineer.engineer_features(sample_ohlcv_data)
        
        required_features = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'atr_14']
        for feat in required_features:
            assert feat in result.columns, f"Missing feature: {feat}"
    
    def test_data_validation_passes_good_data(self, sample_ohlcv_data):
        """Test data validation with good data"""
        checks = DataValidator.validate_ohlcv(sample_ohlcv_data)
        
        assert all(checks.values()), "All checks should pass"
    
    def test_label_generation_creates_labels(self, sample_ohlcv_data):
        """Test label generation"""
        labels = LabelGenerator.triple_barrier_labels(
            sample_ohlcv_data,
            profit_target=0.001,
            stop_loss=0.002,
            horizon=5
        )
        
        assert isinstance(labels, pd.Series)
        assert labels.dtype in [np.int64, int]
        assert set(labels.unique()).issubset({-1, 0, 1})


# ==================== ML TRAINING TESTS ====================

class TestMLTraining:
    """Test ML training pipeline"""
    
    def test_model_training(self):
        """Test model training"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)  # 3-class problem
        feature_names = [f"feature_{i}" for i in range(10)]
        
        config = ModelConfig(symbol="EURUSD", model_type="random_forest")
        pipeline = MLTrainingPipeline(config)
        
        results = pipeline.train(X, y, feature_names)
        
        assert 'test_f1' in results
        assert 'test_accuracy' in results
        assert results['test_f1'] >= 0 and results['test_f1'] <= 1
    
    def test_model_prediction(self):
        """Test model prediction"""
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 2, 50)
        feature_names = [f"f{i}" for i in range(5)]
        
        X_test = np.random.randn(10, 5)
        
        config = ModelConfig(model_type="random_forest")
        pipeline = MLTrainingPipeline(config)
        pipeline.train(X_train, y_train, feature_names)
        
        preds, confidence = pipeline.predict(X_test)
        
        assert len(preds) == len(X_test)
        assert len(confidence) == len(X_test)
        assert all(0 <= c <= 1 for c in confidence)
    
    def test_walk_forward_validation(self):
        """Test walk-forward validation"""
        X = np.random.randn(300, 5)
        y = np.random.randint(0, 2, 300)
        
        config = ModelConfig(model_type="random_forest")
        pipeline = MLTrainingPipeline(config)
        
        results = pipeline.walk_forward_validation(X, y, train_window=100, test_window=50, step=25)
        
        assert 'overall_f1' in results
        assert 'mean_window_f1' in results
        assert len(results['window_results']) > 0


# ==================== RISK MANAGEMENT TESTS ====================

class TestRiskManagement:
    """Test risk management system"""
    
    def test_trade_validation_accepts_good_trade(self, risk_manager):
        """Test that valid trades are accepted"""
        risk_manager.account_metrics = Mock()
        risk_manager.account_metrics.balance = 10000
        risk_manager.account_metrics.equity = 10000
        risk_manager.account_metrics.open_positions = 2
        risk_manager.account_metrics.daily_pnl = 0
        risk_manager.account_metrics.margin_level = 5.0
        
        is_valid, error = risk_manager.validate_trade(
            symbol='EURUSD',
            action='BUY',
            volume=0.1,
            entry_price=1.0750,
            sl=1.0700,
            tp=1.0800
        )
        
        assert is_valid, f"Trade should be valid: {error}"
    
    def test_trade_validation_rejects_oversized_position(self, risk_manager):
        """Test that oversized positions are rejected"""
        risk_manager.account_metrics = Mock()
        risk_manager.account_metrics.balance = 10000
        risk_manager.account_metrics.open_positions = 2
        risk_manager.account_metrics.daily_pnl = 0
        risk_manager.account_metrics.margin_level = 5.0
        
        is_valid, error = risk_manager.validate_trade(
            symbol='EURUSD',
            action='BUY',
            volume=10.0,  # Too large
            entry_price=1.0750,
            sl=1.0700,
            tp=1.0800
        )
        
        assert not is_valid, "Oversized position should be rejected"
    
    def test_position_sizing(self, risk_manager):
        """Test position size calculation"""
        risk_manager.account_metrics = Mock()
        risk_manager.account_metrics.balance = 10000
        
        size = risk_manager.calculate_position_size(
            entry_price=1.0750,
            sl=1.0700,
            risk_percent=0.01
        )
        
        assert 0 < size <= risk_manager.limits.max_position_size
    
    def test_kelly_criterion(self, risk_manager):
        """Test Kelly Criterion calculation"""
        kelly = risk_manager.calculate_kelly_criterion(
            win_rate=0.55,
            profit_factor=1.5
        )
        
        assert 0 <= kelly <= risk_manager.limits.max_risk_per_trade
    
    def test_correlation_monitoring(self):
        """Test correlation monitoring"""
        monitor = CorrelationMonitor(correlation_limit=0.8)
        
        prices1 = np.random.randn(100).cumsum()
        prices2 = prices1 + np.random.randn(100) * 0.1  # Highly correlated
        prices3 = np.random.randn(100).cumsum()  # Not correlated
        
        monitor.add_symbol_data('EURUSD', prices1)
        monitor.add_symbol_data('GBPUSD', prices2)
        monitor.add_symbol_data('AUDUSD', prices3)
        
        # Correlated pair should be rejected
        allowed, corr = monitor.check_correlation('GBPUSD', ['EURUSD'])
        assert not allowed, "Correlated pair should not be allowed"
        
        # Non-correlated pair should be accepted
        allowed, corr = monitor.check_correlation('AUDUSD', ['EURUSD'])
        assert allowed, "Non-correlated pair should be allowed"


# ==================== ZEROMQ BRIDGE TESTS ====================

class TestZeroMQBridge:
    """Test ZeroMQ communication bridge"""
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self):
        """Test bridge initialization"""
        try:
            bridge = MT5ZeroMQBridge()
            assert bridge.connected == False
            assert bridge.request_count == 0
        except ImportError:
            pytest.skip("ZeroMQ not available")
    
    @pytest.mark.asyncio
    async def test_trade_result_creation(self):
        """Test trade result object"""
        result = TradeResult(success=True, order_id=12345)
        
        assert result.success
        assert result.order_id == 12345
        assert result.error is None


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_end_to_end_ml_pipeline(self, sample_ohlcv_data):
        """Test complete ML pipeline from data to prediction"""
        # Feature engineering
        engineer = FeatureEngineer(FeatureConfig())
        features_df = engineer.engineer_features(sample_ohlcv_data)
        
        # Labeling
        features_df['target'] = LabelGenerator.triple_barrier_labels(features_df)
        features_df = features_df.dropna()
        
        # Training
        feature_cols = [c for c in features_df.columns if c not in 
                       ['open', 'high', 'low', 'close', 'volume', 'target']]
        X = features_df[feature_cols].values
        y = features_df['target'].values
        
        config = ModelConfig(symbol="EURUSD")
        pipeline = MLTrainingPipeline(config)
        results = pipeline.train(X, y, feature_cols)
        
        # Verify results
        assert 'test_f1' in results
        assert results['test_f1'] >= 0
        
        # Prediction
        X_test = features_df[feature_cols].iloc[-10:].values
        preds, confidence = pipeline.predict(X_test)
        
        assert len(preds) == 10
        assert all(0 <= c <= 1 for c in confidence)
    
    def test_risk_and_execution_flow(self, risk_manager):
        """Test risk management with trade execution"""
        # Setup account
        risk_manager.account_metrics = Mock()
        risk_manager.account_metrics.balance = 10000
        risk_manager.account_metrics.equity = 9500
        risk_manager.account_metrics.open_positions = 1
        risk_manager.account_metrics.daily_pnl = -200
        risk_manager.account_metrics.margin_level = 3.0
        
        # Validate first trade
        is_valid1, _ = risk_manager.validate_trade(
            symbol='EURUSD',
            action='BUY',
            volume=0.1,
            entry_price=1.0750,
            sl=1.0700,
            tp=1.0800
        )
        
        # Add trade
        if is_valid1:
            risk_manager.add_trade(
                ticket=1,
                symbol='EURUSD',
                action='BUY',
                volume=0.1,
                entry_price=1.0750,
                sl=1.0700,
                tp=1.0800
            )
        
        # Verify trade was recorded
        assert 1 in risk_manager.position_tracking


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_feature_engineering_performance(self, sample_ohlcv_data):
        """Test feature engineering speed"""
        import time
        
        engineer = FeatureEngineer()
        
        start = time.time()
        result = engineer.engineer_features(sample_ohlcv_data)
        elapsed = time.time() - start
        
        # Should complete in < 1 second for 500 bars
        assert elapsed < 1.0, f"Feature engineering too slow: {elapsed:.2f}s"
    
    def test_model_prediction_latency(self):
        """Test model prediction latency"""
        import time
        
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        
        config = ModelConfig()
        pipeline = MLTrainingPipeline(config)
        pipeline.train(X_train, y_train, [f"f{i}" for i in range(20)])
        
        X_test = np.random.randn(100, 20)
        
        start = time.time()
        preds, _ = pipeline.predict(X_test)
        elapsed = time.time() - start
        
        per_prediction = (elapsed / len(X_test)) * 1000  # ms
        
        # Each prediction should be < 5ms
        assert per_prediction < 5.0, f"Prediction too slow: {per_prediction:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
