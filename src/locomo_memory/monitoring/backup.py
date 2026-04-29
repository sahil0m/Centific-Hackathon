"""Automated backup utilities for SQLite database."""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from loguru import logger


@dataclass
class BackupConfig:
    """Backup configuration."""
    backup_dir: Path
    max_backups: int = 10
    backup_interval_hours: int = 24


class BackupManager:
    """Manages automated backups of SQLite database."""
    
    BACKUP_SUFFIX: Final[str] = ".backup"
    
    def __init__(self, config: BackupConfig):
        """Initialize backup manager.
        
        Args:
            config: Backup configuration
        """
        self.config = config
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, db_path: Path) -> Path:
        """Create a backup of the database.
        
        Args:
            db_path: Path to source database
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{db_path.stem}_{timestamp}{self.BACKUP_SUFFIX}"
        backup_path = self.config.backup_dir / backup_name
        
        try:
            # Use SQLite's backup API for safe online backup
            source_conn = sqlite3.connect(str(db_path))
            backup_conn = sqlite3.connect(str(backup_path))
            
            with backup_conn:
                source_conn.backup(backup_conn)
            
            source_conn.close()
            backup_conn.close()
            
            logger.info("Created backup: {}", backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            logger.error("Backup failed: {}", e)
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    def restore_backup(self, backup_path: Path, target_path: Path) -> None:
        """Restore a backup to target location.
        
        Args:
            backup_path: Path to backup file
            target_path: Path to restore to
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Create a backup of current database before restoring
        if target_path.exists():
            safety_backup = target_path.with_suffix(".pre_restore")
            shutil.copy2(target_path, safety_backup)
            logger.info("Created safety backup: {}", safety_backup)
        
        try:
            shutil.copy2(backup_path, target_path)
            logger.info("Restored backup from {} to {}", backup_path, target_path)
        except Exception as e:
            logger.error("Restore failed: {}", e)
            raise
    
    def list_backups(self) -> list[Path]:
        """List all available backups, sorted by date (newest first)."""
        backups = list(self.config.backup_dir.glob(f"*{self.BACKUP_SUFFIX}"))
        return sorted(backups, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backups beyond max_backups limit."""
        backups = self.list_backups()
        
        if len(backups) > self.config.max_backups:
            for old_backup in backups[self.config.max_backups:]:
                try:
                    old_backup.unlink()
                    logger.info("Removed old backup: {}", old_backup)
                except Exception as e:
                    logger.warning("Could not remove old backup {}: {}", old_backup, e)
    
    def should_backup(self, db_path: Path) -> bool:
        """Check if a backup is needed based on interval.
        
        Args:
            db_path: Path to database
            
        Returns:
            True if backup is needed
        """
        backups = self.list_backups()
        
        if not backups:
            return True
        
        latest_backup = backups[0]
        age_hours = (time.time() - latest_backup.stat().st_mtime) / 3600
        
        return age_hours >= self.config.backup_interval_hours


__all__ = ["BackupManager", "BackupConfig"]
