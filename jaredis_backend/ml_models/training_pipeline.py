"""
Machine Learning Training Pipeline
Handles model training, validation, and performance evaluation
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML model training"""
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    lookback_days: int = 730  # 2 years
    model_type: str = "random_forest"  # random_forest, xgboost, gradient_boosting
    test_size: float = 0.2
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 50
    learning_rate: float = 0.1
    
    # Training parameters
    use_class_weights: bool = True
    random_state: int = 42
    
    # MLflow
    experiment_name: str = "forex_trading"
    model_registry_name: Optional[str] = None


class MLTrainingPipeline:
    """Complete ML training pipeline with validation"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_results = {}
        self.run_id = None
        
        if MLFLOW_AVAILABLE and self.config.experiment_name:
            mlflow.set_experiment(self.config.experiment_name)
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train model with MLflow tracking
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Training results dictionary
        """
        self.feature_names = feature_names
        logger.info(f"Starting training: {self.config.model_type} on {X.shape[0]} samples")
        
        # Train/test split (preserve temporal order)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run() as run:
                self.run_id = run.info.run_id
                results = self._train_and_log(X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            results = self._train_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        self.training_results = results
        logger.info(f"Training complete. Test F1: {results['test_f1']:.4f}")
        
        return results
    
    def _train_and_log(self, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train model with MLflow logging"""
        # Log parameters
        mlflow.log_params({
            "symbol": self.config.symbol,
            "model_type": self.config.model_type,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "test_size": self.config.test_size,
            "n_features": len(self.feature_names)
        })
        
        results = self._train_model(X_train, X_test, y_train, y_test)
        
        # Log metrics
        mlflow.log_metrics({
            "train_f1": results['train_f1'],
            "test_f1": results['test_f1'],
            "test_accuracy": results['test_accuracy'],
            "test_precision": results['test_precision'],
            "test_recall": results['test_recall']
        })
        
        # Log feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, self.model.feature_importances_)
            }
            mlflow.log_dict(importance_dict, "feature_importance.json")
        
        # Log model
        mlflow.sklearn.log_model(self.model, "model")
        
        return results
    
    def _train_model(self, X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train and evaluate model"""
        # Build model
        if self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                class_weight='balanced' if self.config.use_class_weights else None,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_proba = self.model.predict_proba(X_test)
        
        results = {
            'train_f1': f1_score(y_train, train_pred, average='weighted', zero_division=0),
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_f1': f1_score(y_test, test_pred, average='weighted', zero_division=0),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, test_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist(),
            'classification_report': classification_report(y_test, test_pred)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            results['test_roc_auc'] = roc_auc_score(y_test, test_proba[:, 1])
        
        return results
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray,
                               train_window: int = 500,
                               test_window: int = 100,
                               step: int = 50) -> Dict:
        """
        Walk-forward validation (out-of-sample testing)
        
        Args:
            X: Feature matrix
            y: Target labels
            train_window: Training window size
            test_window: Test window size
            step: Step size for rolling windows
            
        Returns:
            Validation results
        """
        logger.info(f"Starting walk-forward validation with {train_window} train, {test_window} test")
        
        results = {
            'window_results': [],
            'predictions': [],
            'actuals': [],
            'dates': []
        }
        
        for i in range(train_window, len(X) - test_window, step):
            X_train = X[i-train_window:i]
            X_test = X[i:i+test_window]
            y_train = y[i-train_window:i]
            y_test = y[i:i+test_window]
            
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Train
            model = self._create_model()
            model.fit(X_train_s, y_train)
            
            # Test
            preds = model.predict(X_test_s)
            window_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
            
            results['window_results'].append({
                'window': i,
                'f1_score': window_f1,
                'n_samples': len(y_test)
            })
            
            results['predictions'].extend(preds.tolist())
            results['actuals'].extend(y_test.tolist())
        
        # Calculate overall metrics
        overall_f1 = f1_score(results['actuals'], results['predictions'], average='weighted', zero_division=0)
        overall_accuracy = accuracy_score(results['actuals'], results['predictions'])
        
        results['overall_f1'] = overall_f1
        results['overall_accuracy'] = overall_accuracy
        results['mean_window_f1'] = np.mean([r['f1_score'] for r in results['window_results']])
        
        logger.info(f"Walk-forward validation complete. Overall F1: {overall_f1:.4f}")
        
        return results
    
    def _create_model(self):
        """Create fresh model instance"""
        if self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                class_weight='balanced' if self.config.use_class_weights else None,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
    
    def save_model(self, filepath: str) -> str:
        """Save trained model and scaler"""
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': asdict(self.config),
            'training_results': self.training_results,
            'run_id': self.run_id,
            'saved_at': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> Dict:
        """Load saved model"""
        artifacts = joblib.load(filepath)
        
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.feature_names = artifacts['feature_names']
        self.config = ModelConfig(**artifacts['config'])
        self.training_results = artifacts['training_results']
        self.run_id = artifacts.get('run_id')
        
        logger.info(f"Model loaded from {filepath}")
        
        return artifacts
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence


class ModelComparator:
    """Compare models and select best performer"""
    
    @staticmethod
    def compare_models(results_list: List[Dict[str, float]], 
                       metric: str = 'test_f1') -> Dict:
        """
        Compare multiple model results
        
        Args:
            results_list: List of training results
            metric: Metric to compare by
            
        Returns:
            Comparison results with best model
        """
        comparison = {
            'metric': metric,
            'results': results_list,
            'best_index': np.argmax([r.get(metric, 0) for r in results_list]),
            'best_value': max([r.get(metric, 0) for r in results_list]),
            'mean_value': np.mean([r.get(metric, 0) for r in results_list]),
            'std_value': np.std([r.get(metric, 0) for r in results_list])
        }
        
        logger.info(f"Model comparison: Best {metric}={comparison['best_value']:.4f}")
        
        return comparison
    
    @staticmethod
    def register_model_to_mlflow(model_name: str, run_id: str, 
                                 metrics: Dict, stage: str = "Staging") -> Optional[str]:
        """Register trained model to MLflow Model Registry"""
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)
            
            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage
            )
            
            logger.info(f"Model {model_name} v{mv.version} registered to {stage}")
            return f"{model_name}/v{mv.version}"
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
