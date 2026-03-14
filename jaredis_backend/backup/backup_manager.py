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

import asyncio
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import gzip

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

logger = logging.getLogger(__name__)


class BackupConfig:
    """Configuration for backup operations"""

    def __init__(
        self,
        backup_dir: str = "backups",
        retention_days: int = 30,
        compression_level: int = 9,
        max_backup_size_mb: int = 1000,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        cloud_storage: Optional[Dict] = None
    ):
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.compression_level = compression_level
        self.max_backup_size_mb = max_backup_size_mb
        self.include_patterns = include_patterns or [
            "models/**/*.pkl",
            "models/**/*.joblib",
            "models/**/*.h5",
            "models/**/*.pb",
            "config/**/*.json",
            "config/**/*.yaml",
            "config/**/*.py",
            "logs/**/*.log",
            "data/**/*.csv",
            "data/**/*.parquet",
            "pipelines/**/*.py",
            "jaredis_backend/**/*.py",
            "*.py",
            "requirements*.txt"
        ]
        self.exclude_patterns = exclude_patterns or [
            "**/__pycache__/**",
            "**/*.pyc",
            "**/node_modules/**",
            "**/.git/**",
            "**/backups/**"
        ]
        self.cloud_storage = cloud_storage or {}


class BackupManager:
    """Manages automated backups of critical system components"""

    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_dir = config.backup_dir
        self.backup_dir.mkdir(exist_ok=True)

        # Initialize cloud clients
        self.s3_client = None
        self.gcs_client = None

        if config.cloud_storage:
            self._init_cloud_clients()

    def _init_cloud_clients(self):
        """Initialize cloud storage clients"""
        cloud_config = self.config.cloud_storage

        # AWS S3
        if AWS_AVAILABLE and 'aws' in cloud_config:
            aws_config = cloud_config['aws']
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config.get('access_key'),
                aws_secret_access_key=aws_config.get('secret_key'),
                region_name=aws_config.get('region', 'us-east-1')
            )

        # Google Cloud Storage
        if GOOGLE_CLOUD_AVAILABLE and 'gcp' in cloud_config:
            gcp_config = cloud_config['gcp']
            self.gcs_client = storage.Client.from_service_account_json(
                gcp_config.get('credentials_file')
            )

    async def create_backup(self, backup_type: str = "full") -> Optional[str]:
        """Create a new backup archive

        Args:
            backup_type: Type of backup ('full', 'models', 'config', 'data')

        Returns:
            Path to created backup file, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"jaredis_backup_{backup_type}_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_name

            logger.info(f"Creating {backup_type} backup: {backup_name}")

            # Create temporary directory for staging
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Collect files based on backup type
                files_to_backup = self._collect_files(backup_type)

                if not files_to_backup:
                    logger.warning(f"No files found for {backup_type} backup")
                    return None

                # Copy files to staging area
                for src_file in files_to_backup:
                    if src_file.exists():
                        # Create relative path structure
                        rel_path = src_file.relative_to(Path.cwd())
                        dest_file = temp_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        if src_file.is_file():
                            shutil.copy2(src_file, dest_file)
                        elif src_file.is_dir():
                            shutil.copytree(src_file, dest_file, dirs_exist_ok=True)

                # Create compressed archive
                with tarfile.open(backup_path, "w:gz", compresslevel=self.config.compression_level) as tar:
                    for file_path in temp_path.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(temp_path)
                            tar.add(file_path, arcname=arcname)

                # Check backup size
                backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
                if backup_size_mb > self.config.max_backup_size_mb:
                    logger.error(f"Backup too large: {backup_size_mb:.1f}MB > {self.config.max_backup_size_mb}MB")
                    backup_path.unlink()
                    return None

                logger.info(f"Backup created: {backup_path} ({backup_size_mb:.1f}MB)")

                # Upload to cloud if configured
                await self._upload_to_cloud(backup_path, backup_name)

                return str(backup_path)

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None

    def _collect_files(self, backup_type: str) -> List[Path]:
        """Collect files to include in backup"""
        files = []

        if backup_type == "full":
            patterns = self.config.include_patterns
        elif backup_type == "models":
            patterns = [p for p in self.config.include_patterns if "models" in p]
        elif backup_type == "config":
            patterns = [p for p in self.config.include_patterns if "config" in p]
        elif backup_type == "data":
            patterns = [p for p in self.config.include_patterns if "data" in p or "logs" in p]
        else:
            patterns = self.config.include_patterns

        for pattern in patterns:
            try:
                for path in Path.cwd().glob(pattern):
                    if path.exists() and not self._should_exclude(path):
                        files.append(path)
            except Exception as e:
                logger.warning(f"Error processing pattern {pattern}: {e}")

        return files

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from backup"""
        path_str = str(path)

        for pattern in self.config.exclude_patterns:
            try:
                if path.match(pattern):
                    return True
            except Exception:
                continue

        return False

    async def _upload_to_cloud(self, backup_path: Path, backup_name: str):
        """Upload backup to cloud storage"""
        try:
            # AWS S3
            if self.s3_client and 'aws' in self.config.cloud_storage:
                aws_config = self.config.cloud_storage['aws']
                bucket = aws_config.get('bucket')

                if bucket:
                    self.s3_client.upload_file(
                        str(backup_path),
                        bucket,
                        f"backups/{backup_name}"
                    )
                    logger.info(f"Uploaded to S3: s3://{bucket}/backups/{backup_name}")

            # Google Cloud Storage
            if self.gcs_client and 'gcp' in self.config.cloud_storage:
                gcp_config = self.config.cloud_storage['gcp']
                bucket_name = gcp_config.get('bucket')

                if bucket_name:
                    bucket = self.gcs_client.bucket(bucket_name)
                    blob = bucket.blob(f"backups/{backup_name}")
                    blob.upload_from_filename(str(backup_path))
                    logger.info(f"Uploaded to GCS: gs://{bucket_name}/backups/{backup_name}")

        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")

    async def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

            for backup_file in self.backup_dir.glob("*.tar.gz"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    logger.info(f"Removed old backup: {backup_file.name}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def restore_backup(self, backup_path: str, restore_dir: Optional[str] = None) -> bool:
        """Restore from backup archive

        Args:
            backup_path: Path to backup file
            restore_dir: Directory to restore to (default: current directory)

        Returns:
            True if successful
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            restore_to = Path(restore_dir) if restore_dir else Path.cwd()

            logger.info(f"Restoring backup {backup_path} to {restore_to}")

            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(restore_to)

            logger.info("Backup restored successfully")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []

        for backup_file in self.backup_dir.glob("*.tar.gz"):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'path': str(backup_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'type': self._get_backup_type(backup_file.name)
            })

        return sorted(backups, key=lambda x: x['created'], reverse=True)

    def _get_backup_type(self, filename: str) -> str:
        """Extract backup type from filename"""
        parts = filename.split('_')
        if len(parts) >= 3:
            return parts[2]  # jaredis_backup_{type}_{timestamp}.tar.gz
        return "unknown"


class ScheduledBackup:
    """Handle scheduled backup operations"""

    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.running = False

    async def start_schedule(
        self,
        full_backup_interval_hours: int = 24,
        model_backup_interval_hours: int = 6,
        config_backup_interval_hours: int = 12
    ):
        """Start scheduled backup operations"""
        self.running = True

        logger.info("Starting scheduled backups")

        tasks = [
            self._schedule_backups("full", full_backup_interval_hours),
            self._schedule_backups("models", model_backup_interval_hours),
            self._schedule_backups("config", config_backup_interval_hours),
            self._cleanup_schedule()
        ]

        await asyncio.gather(*tasks)

    async def _schedule_backups(self, backup_type: str, interval_hours: int):
        """Schedule periodic backups of specific type"""
        interval_seconds = interval_hours * 3600

        while self.running:
            try:
                await self.backup_manager.create_backup(backup_type)
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Scheduled {backup_type} backup failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _cleanup_schedule(self):
        """Schedule cleanup of old backups"""
        while self.running:
            try:
                await self.backup_manager.cleanup_old_backups()
                await asyncio.sleep(24 * 3600)  # Daily cleanup
            except Exception as e:
                logger.error(f"Scheduled cleanup failed: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    def stop(self):
        """Stop scheduled backups"""
        self.running = False
        logger.info("Scheduled backups stopped")


# Convenience functions for easy integration
async def create_backup(backup_type: str = "full") -> Optional[str]:
    """Create a backup with default configuration"""
    config = BackupConfig()
    manager = BackupManager(config)
    return await manager.create_backup(backup_type)

async def restore_backup(backup_path: str, restore_dir: Optional[str] = None) -> bool:
    """Restore from backup with default configuration"""
    config = BackupConfig()
    manager = BackupManager(config)
    return await manager.restore_backup(backup_path, restore_dir)