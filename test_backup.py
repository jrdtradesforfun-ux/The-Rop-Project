#!/usr/bin/env python3
"""Test script for backup system"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaredis_backend.backup.backup_manager import BackupManager, BackupConfig

async def test_backup():
    """Test backup functionality"""
    print("Testing backup system...")

    # Create backup configuration
    config = BackupConfig(
        backup_dir="test_backups",
        retention_days=7
    )

    # Initialize backup manager
    manager = BackupManager(config)

    # Create a config backup
    print("Creating full backup...")
    backup_path = await manager.create_backup("full")

    if backup_path:
        print(f"✅ Backup created successfully: {backup_path}")

        # List backups
        backups = manager.list_backups()
        print(f"📋 Available backups: {len(backups)}")
        for backup in backups:
            print(f"  - {backup['name']} ({backup['size_mb']:.1f}MB)")

        # Test cleanup
        print("Testing cleanup...")
        await manager.cleanup_old_backups()
        print("✅ Cleanup completed")

    else:
        print("❌ Backup creation failed")

if __name__ == "__main__":
    asyncio.run(test_backup())