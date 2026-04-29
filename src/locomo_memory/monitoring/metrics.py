"""Metrics collection for observability."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class MetricType(str, Enum):
    """Type of metric."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric value."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collector."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical values to keep per metric
        """
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = Lock()
    
    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter.
        
        Args:
            name: Metric name
            value: Amount to increment
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
    
    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        """Get gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        """Get histogram statistics.
        
        Returns:
            Dict with min, max, mean, p50, p95, p99
        """
        key = self._make_key(name, labels)
        with self._lock:
            values = list(self._histograms.get(key, []))
        
        if not values:
            return {}
        
        values.sort()
        n = len(values)
        
        return {
            "count": n,
            "min": values[0],
            "max": values[-1],
            "mean": sum(values) / n,
            "p50": values[int(n * 0.50)],
            "p95": values[int(n * 0.95)],
            "p99": values[int(n * 0.99)],
        }
    
    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name)
                    for name in self._histograms.keys()
                },
            }
    
    @staticmethod
    def _make_key(name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _global_metrics


__all__ = ["MetricsCollector", "MetricType", "Metric", "get_metrics_collector"]
