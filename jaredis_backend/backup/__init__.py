"""Automated Backup System for Trading Bot

Provides comprehensive backup functionality for:
- ML models and training data
- Configuration files
- Trade history and performance metrics
- System logs and alerts
- Database backups (if applicable)

Supports multiple storage backends:
- Local filesystem
- Cloud storage (AWS S3, Google Cloud, Azure)
- Network drives
"""

from .backup_manager import (
    BackupConfig,
    BackupManager,
    ScheduledBackup,
    create_backup,
    restore_backup
)

__all__ = [
    'BackupConfig',
    'BackupManager',
    'ScheduledBackup',
    'create_backup',
    'restore_backup'
]