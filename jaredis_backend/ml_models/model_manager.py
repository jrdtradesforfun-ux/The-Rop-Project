"""
Model Manager: Manages training, loading, and inference of ML models
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Handles loading, saving, and managing ML models for trading decisions
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager
        
        Args:
            models_dir: Directory to store/load models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}

    def register_model(self, name: str, model: Any, metadata: Dict = None) -> None:
        """
        Register a model in the manager
        
        Args:
            name: Model name/identifier
            model: The model object
            metadata: Model metadata (performance metrics, hyperparameters, etc.)
        """
        self.models[name] = model
        self.model_metadata[name] = metadata or {}
        logger.info(f"Model '{name}' registered")

    def save_model(self, name: str, filepath: Optional[str] = None) -> str:
        """
        Save a model to disk
        
        Args:
            name: Model identifier
            filepath: Optional custom filepath
            
        Returns:
            Path where model was saved
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")

        if filepath is None:
            filepath = str(self.models_dir / f"{name}.pkl")

        with open(filepath, "wb") as f:
            pickle.dump(self.models[name], f)

        # Save metadata
        metadata_path = filepath.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.model_metadata[name], f, indent=2)

        logger.info(f"Model '{name}' saved to {filepath}")
        return filepath

    def load_model(self, name: str, filepath: Optional[str] = None) -> Any:
        """
        Load a model from disk
        
        Args:
            name: Model identifier
            filepath: Optional custom filepath
            
        Returns:
            The loaded model
        """
        if filepath is None:
            filepath = str(self.models_dir / f"{name}.pkl")

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        # Load metadata if exists
        metadata_path = filepath.replace(".pkl", "_metadata.json")
        if Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        self.models[name] = model
        self.model_metadata[name] = metadata
        logger.info(f"Model '{name}' loaded from {filepath}")
        return model

    def get_model(self, name: str) -> Optional[Any]:
        """Get a registered model"""
        return self.models.get(name)

    def get_metadata(self, name: str) -> Dict:
        """Get model metadata"""
        return self.model_metadata.get(name, {})

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())

    def update_metadata(self, name: str, metadata: Dict) -> None:
        """Update model metadata"""
        if name in self.models:
            self.model_metadata[name].update(metadata)
            logger.info(f"Metadata updated for model '{name}'")
