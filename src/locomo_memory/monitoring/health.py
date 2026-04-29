"""Health check utilities for production monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self, store=None, faiss_index=None, bm25_index=None):
        """Initialize health checker.
        
        Args:
            store: MemoryStore instance
            faiss_index: MemoryFAISSIndex instance
            bm25_index: MemoryBM25Index instance
        """
        self.store = store
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
    
    def check_all(self) -> HealthCheckResult:
        """Run all health checks.
        
        Returns:
            HealthCheckResult with overall status
        """
        checks = {}
        
        # Check database
        checks["database"] = self._check_database()
        
        # Check indexes
        checks["faiss_index"] = self._check_faiss_index()
        checks["bm25_index"] = self._check_bm25_index()
        
        # Check disk space
        checks["disk_space"] = self._check_disk_space()
        
        # Determine overall status
        statuses = [c["status"] for c in checks.values()]
        if all(s == "healthy" for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == "unhealthy" for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED
        
        return HealthCheckResult(status=overall, checks=checks)
    
    def _check_database(self) -> dict[str, Any]:
        """Check database connectivity and integrity."""
        if self.store is None:
            return {"status": "unhealthy", "message": "Store not initialized"}
        
        try:
            # Try a simple query
            with self.store.reader() as conn:
                result = conn.execute("SELECT COUNT(*) FROM memory_units").fetchone()
                count = result[0] if result else 0
            
            return {
                "status": "healthy",
                "memory_units": count,
                "message": "Database accessible"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Database error: {str(e)}"
            }
    
    def _check_faiss_index(self) -> dict[str, Any]:
        """Check FAISS index status."""
        if self.faiss_index is None:
            return {"status": "degraded", "message": "FAISS index not initialized"}
        
        try:
            size = self.faiss_index.size()
            return {
                "status": "healthy",
                "indexed_items": size,
                "message": "FAISS index operational"
            }
        except Exception as e:
            return {
                "status": "degraded",
                "message": f"FAISS index error: {str(e)}"
            }
    
    def _check_bm25_index(self) -> dict[str, Any]:
        """Check BM25 index status."""
        if self.bm25_index is None:
            return {"status": "degraded", "message": "BM25 index not initialized"}
        
        try:
            size = self.bm25_index.size()
            return {
                "status": "healthy",
                "indexed_items": size,
                "message": "BM25 index operational"
            }
        except Exception as e:
            return {
                "status": "degraded",
                "message": f"BM25 index error: {str(e)}"
            }
    
    def _check_disk_space(self) -> dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            # Check workspace disk space
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            percent_free = (usage.free / usage.total) * 100
            
            if percent_free < 5:
                status = "unhealthy"
                message = f"Critical: Only {free_gb:.1f}GB free"
            elif percent_free < 10:
                status = "degraded"
                message = f"Warning: Only {free_gb:.1f}GB free"
            else:
                status = "healthy"
                message = f"{free_gb:.1f}GB free of {total_gb:.1f}GB"
            
            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_free": round(percent_free, 2),
                "message": message
            }
        except Exception as e:
            return {
                "status": "degraded",
                "message": f"Could not check disk space: {str(e)}"
            }


__all__ = ["HealthChecker", "HealthStatus", "HealthCheckResult"]
