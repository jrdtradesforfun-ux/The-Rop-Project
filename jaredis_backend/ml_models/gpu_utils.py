"""GPU Acceleration Utilities for ML Training

Provides GPU detection, memory management, and optimized training configurations
for XGBoost, TensorFlow, and PyTorch models.
"""

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and configurations for ML training"""

    def __init__(self):
        self.gpu_available = self._detect_gpu()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        self.mixed_precision = self._setup_mixed_precision()

    def _detect_gpu(self) -> bool:
        """Detect available GPU hardware"""
        gpu_found = False

        # Check CUDA (NVIDIA)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"CUDA GPUs detected: {gpu_count}")
                for i in range(gpu_count):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                gpu_found = True
        except ImportError:
            pass

        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow GPUs detected: {len(gpus)}")
                for gpu in gpus:
                    logger.info(f"GPU: {gpu}")
                gpu_found = True
        except ImportError:
            pass

        # Check XGBoost GPU
        try:
            import xgboost as xgb
            # XGBoost GPU detection is done during training
            logger.info("XGBoost GPU support available")
        except ImportError:
            pass

        return gpu_found

    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                logger.info(f"GPU memory: {memory_gb:.1f} GB")
                return memory_gb
        except ImportError:
            pass

        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Estimate based on common GPU types
                gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', '')
                memory_map = {
                    'rtx': 8, 'gtx': 8, 'titan': 24, 'a100': 40, 'v100': 32, 'p100': 16
                }
                for key, mem in memory_map.items():
                    if key in gpu_name.lower():
                        logger.info(f"Estimated GPU memory: {mem} GB")
                        return mem
        except:
            pass

        logger.info("GPU memory detection failed, using conservative defaults")
        return 8.0  # Conservative default

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU memory"""
        if not self.gpu_available:
            return 32  # CPU default

        # Base batch size on GPU memory
        memory_gb = self.gpu_memory_gb
        if memory_gb >= 24:  # High-end GPUs
            return 512
        elif memory_gb >= 12:  # Mid-range GPUs
            return 256
        elif memory_gb >= 8:  # Entry-level GPUs
            return 128
        else:
            return 64

    def _setup_mixed_precision(self) -> bool:
        """Setup mixed precision training if supported"""
        try:
            import tensorflow as tf
            from tensorflow.keras.mixed_precision import experimental as mixed_precision

            # Enable mixed precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
            logger.info("TensorFlow mixed precision enabled")
            return True
        except ImportError:
            logger.info("TensorFlow mixed precision not available")
            return False
        except Exception as e:
            logger.warning(f"Mixed precision setup failed: {e}")
            return False

    def get_xgboost_config(self) -> Dict:
        """Get optimal XGBoost configuration for GPU training"""
        if not self.gpu_available:
            return {
                'tree_method': 'hist',
                'n_jobs': -1
            }

        return {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'n_jobs': 1,  # GPU handles parallelism
            'max_bin': 256,  # Optimize for GPU
            'grow_policy': 'lossguide'
        }

    def get_tensorflow_config(self) -> Dict:
        """Get optimal TensorFlow configuration for GPU training"""
        config = {}

        if self.gpu_available:
            try:
                import tensorflow as tf

                # Enable GPU memory growth
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                # Multi-GPU strategy
                if len(gpus) > 1:
                    strategy = tf.distribute.MirroredStrategy()
                    config['strategy'] = strategy
                    logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
                else:
                    config['strategy'] = tf.distribute.OneDeviceStrategy("/gpu:0")

            except Exception as e:
                logger.warning(f"TensorFlow GPU config failed: {e}")

        return config

    def get_pytorch_config(self) -> Dict:
        """Get optimal PyTorch configuration for GPU training"""
        config = {}

        if self.gpu_available:
            try:
                import torch

                if torch.cuda.is_available():
                    config['device'] = torch.device('cuda')
                    config['gpu_count'] = torch.cuda.device_count()

                    # Set optimal CUDA settings
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False

                    logger.info(f"PyTorch GPU config: {config}")

            except Exception as e:
                logger.warning(f"PyTorch GPU config failed: {e}")

        return config

    def optimize_memory_usage(self):
        """Apply memory optimization techniques"""
        if not self.gpu_available:
            return

        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except ImportError:
            pass

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            logger.info("TensorFlow session cleared")
        except ImportError:
            pass

    def get_training_config(self, model_type: str) -> Dict:
        """Get comprehensive training configuration for specific model type"""
        base_config = {
            'gpu_available': self.gpu_available,
            'gpu_memory_gb': self.gpu_memory_gb,
            'optimal_batch_size': self.optimal_batch_size,
            'mixed_precision': self.mixed_precision
        }

        if model_type.lower() == 'xgboost':
            base_config.update(self.get_xgboost_config())
        elif model_type.lower() in ['tensorflow', 'keras', 'lstm']:
            base_config.update(self.get_tensorflow_config())
        elif model_type.lower() in ['pytorch', 'torch']:
            base_config.update(self.get_pytorch_config())

        return base_config


class GPUMemoryMonitor:
    """Monitor GPU memory usage during training"""

    def __init__(self):
        self.monitoring = False

    def start_monitoring(self, interval_seconds: int = 10):
        """Start GPU memory monitoring"""
        if not self._gpu_available():
            logger.warning("GPU monitoring not available")
            return

        self.monitoring = True
        import threading
        import time

        def monitor():
            while self.monitoring:
                self._log_memory_usage()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("GPU memory monitoring started")

    def stop_monitoring(self):
        """Stop GPU memory monitoring"""
        self.monitoring = False
        logger.info("GPU memory monitoring stopped")

    def _gpu_available(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _log_memory_usage(self):
        """Log current GPU memory usage"""
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        except Exception as e:
            logger.warning(f"GPU memory logging failed: {e}")


# Global GPU manager instance
gpu_manager = GPUManager()
gpu_monitor = GPUMemoryMonitor()

def enable_gpu_acceleration():
    """Enable GPU acceleration for all supported frameworks"""
    global gpu_manager

    logger.info("Enabling GPU acceleration...")

    # Set environment variables for GPU optimization
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    # Initialize GPU manager
    gpu_manager = GPUManager()

    if gpu_manager.gpu_available:
        logger.info("GPU acceleration enabled successfully")
        gpu_monitor.start_monitoring()
    else:
        logger.warning("No GPU detected, using CPU acceleration")

    return gpu_manager

def get_optimal_config(model_type: str) -> Dict:
    """Get optimal configuration for model training"""
    return gpu_manager.get_training_config(model_type)

def optimize_memory():
    """Optimize GPU memory usage"""
    gpu_manager.optimize_memory_usage()