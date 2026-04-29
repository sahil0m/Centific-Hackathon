"""Rate limiting utilities."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Final


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests: int = 100
    window_seconds: float = 60.0
    burst_size: int = 10


class TokenBucketLimiter:
    """Token bucket rate limiter (thread-safe).
    
    Allows burst traffic while maintaining average rate limit.
    """
    
    def __init__(
        self,
        rate: float = 100.0,  # tokens per second
        capacity: int = 100,  # max tokens
    ):
        """Initialize token bucket.
        
        Args:
            rate: Token refill rate (per second)
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = Lock()
    
    def acquire(self, tokens: int = 1, block: bool = True, timeout: float | None = None) -> bool:
        """Acquire tokens from bucket.
        
        Args:
            tokens: Number of tokens to acquire
            block: If True, wait for tokens to be available
            timeout: Maximum time to wait (None = infinite)
            
        Returns:
            True if tokens acquired, False otherwise
            
        Raises:
            RateLimitExceeded: If tokens cannot be acquired and block=False
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                if not block:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded: need {tokens} tokens, have {self.tokens:.2f}"
                    )
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed + wait_time > timeout:
                        return False
            
            # Sleep outside lock
            time.sleep(min(wait_time, 0.1))
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill()
            return self.tokens


class RateLimiter:
    """Sliding window rate limiter (thread-safe)."""
    
    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._requests: deque[float] = deque()
        self._lock = Lock()
    
    def check_limit(self, identifier: str = "default") -> bool:
        """Check if request is within rate limit.
        
        Args:
            identifier: Identifier for rate limiting (e.g., user_id, ip)
            
        Returns:
            True if within limit, False otherwise
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        now = time.monotonic()
        window_start = now - self.config.window_seconds
        
        with self._lock:
            # Remove old requests outside window
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()
            
            # Check limit
            if len(self._requests) >= self.config.max_requests:
                oldest = self._requests[0]
                wait_time = oldest + self.config.window_seconds - now
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {self.config.max_requests} requests per "
                    f"{self.config.window_seconds}s. Retry in {wait_time:.1f}s"
                )
            
            # Add current request
            self._requests.append(now)
            return True
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        now = time.monotonic()
        window_start = now - self.config.window_seconds
        
        with self._lock:
            # Remove old requests
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()
            
            return max(0, self.config.max_requests - len(self._requests))


__all__ = [
    "RateLimiter",
    "TokenBucketLimiter",
    "RateLimitExceeded",
    "RateLimitConfig",
]
