"""
Logger Configuration: Centralized logging setup
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_file: str = "jaredis.log"
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_file: Log file name
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger("jaredis")
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
