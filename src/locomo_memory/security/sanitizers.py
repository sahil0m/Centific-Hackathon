"""Data sanitization utilities."""

from __future__ import annotations

import re
from typing import Any


class InputSanitizer:
    """Sanitizes user inputs."""
    
    @staticmethod
    def sanitize_for_display(text: str, max_length: int = 200) -> str:
        """Sanitize text for safe display in UI.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length for display
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove control characters except newline and tab
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    @staticmethod
    def sanitize_for_logging(text: str, max_length: int = 100) -> str:
        """Sanitize text for safe logging.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length for logs
            
        Returns:
            Sanitized text
        """
        return InputSanitizer.sanitize_for_display(text, max_length)


class LogSanitizer:
    """Sanitizes sensitive data from logs."""
    
    # Patterns to redact
    API_KEY_PATTERN = re.compile(r'(sk-[a-zA-Z0-9-]{10,})', re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    
    @staticmethod
    def sanitize(data: Any) -> Any:
        """Recursively sanitize sensitive data.
        
        Args:
            data: Data to sanitize (str, dict, list, etc.)
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            return LogSanitizer._sanitize_string(data)
        elif isinstance(data, dict):
            return {k: LogSanitizer.sanitize(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(LogSanitizer.sanitize(item) for item in data)
        else:
            return data
    
    @staticmethod
    def _sanitize_string(text: str) -> str:
        """Sanitize a single string."""
        # Redact API keys
        text = LogSanitizer.API_KEY_PATTERN.sub(r'\1[:4]***', text)
        
        # Redact emails
        text = LogSanitizer.EMAIL_PATTERN.sub('[EMAIL]', text)
        
        # Redact phone numbers
        text = LogSanitizer.PHONE_PATTERN.sub('[PHONE]', text)
        
        # Redact SSNs
        text = LogSanitizer.SSN_PATTERN.sub('[SSN]', text)
        
        return text


__all__ = ["InputSanitizer", "LogSanitizer"]
