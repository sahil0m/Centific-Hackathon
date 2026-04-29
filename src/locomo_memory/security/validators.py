"""Input validation utilities - enterprise grade."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

# Constants
MAX_INPUT_LENGTH: Final[int] = 2000
MAX_CHUNK_LENGTH: Final[int] = 4000
MAX_CONVERSATION_ID_LENGTH: Final[int] = 100
CONVERSATION_ID_PATTERN: Final[re.Pattern] = re.compile(r'^[a-zA-Z0-9_-]+$')
API_KEY_PATTERN: Final[re.Pattern] = re.compile(r'^sk-or-v1-[a-f0-9]{64}$')


class ValidationError(ValueError):
    """Raised when validation fails."""
    pass


class InputValidator:
    """Validates user inputs with configurable limits."""
    
    @staticmethod
    def validate_text_input(
        text: str | None,
        max_length: int = MAX_INPUT_LENGTH,
        field_name: str = "input",
    ) -> str:
        """Validate and sanitize text input.
        
        Args:
            text: Input text to validate
            max_length: Maximum allowed length
            field_name: Name for error messages
            
        Returns:
            Validated and trimmed text
            
        Raises:
            ValidationError: If validation fails
        """
        if text is None:
            raise ValidationError(f"{field_name} cannot be None")
        
        text = text.strip()
        
        if not text:
            raise ValidationError(f"{field_name} cannot be empty")
        
        if len(text) > max_length:
            # Truncate instead of rejecting to preserve functionality
            text = text[:max_length]
        
        # Check for null bytes (security)
        if '\x00' in text:
            raise ValidationError(f"{field_name} contains invalid characters")
        
        return text
    
    @staticmethod
    def validate_chunk_text(text: str) -> str:
        """Validate chunk text for LLM processing."""
        return InputValidator.validate_text_input(
            text,
            max_length=MAX_CHUNK_LENGTH,
            field_name="chunk_text"
        )
    
    @staticmethod
    def is_safe_for_llm(text: str) -> bool:
        """Check if text is safe to send to LLM (basic prompt injection filter)."""
        # Check for common prompt injection patterns
        dangerous_patterns = [
            r'ignore\s+(previous|all)\s+instructions',
            r'system\s*:\s*you\s+are',
            r'<\s*script\s*>',
            r'javascript\s*:',
            r'\x00',  # null bytes
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False
        
        return True


class APIKeyValidator:
    """Validates API keys."""
    
    @staticmethod
    def validate_openrouter_key(key: str | None) -> str:
        """Validate OpenRouter API key format.
        
        Args:
            key: API key to validate
            
        Returns:
            Validated key
            
        Raises:
            ValidationError: If key is invalid
        """
        if not key:
            raise ValidationError("API key is required")
        
        key = key.strip()
        
        # OpenRouter keys start with sk-or-
        if not key.startswith('sk-or-'):
            raise ValidationError("Invalid OpenRouter API key format (must start with 'sk-or-')")
        
        # Basic length check (OpenRouter keys are typically 70+ chars)
        if len(key) < 20:
            raise ValidationError("API key is too short")
        
        return key
    
    @staticmethod
    def mask_key(key: str, visible_chars: int = 4) -> str:
        """Mask API key for safe display.
        
        Args:
            key: API key to mask
            visible_chars: Number of characters to show
            
        Returns:
            Masked key (e.g., "sk-o...xyz")
        """
        if not key or len(key) <= visible_chars * 2:
            return "***"
        
        return f"{key[:visible_chars]}...{key[-3:]}"


class ConversationIDValidator:
    """Validates conversation IDs to prevent injection."""
    
    @staticmethod
    def validate(conv_id: str | None) -> str:
        """Validate conversation ID.
        
        Args:
            conv_id: Conversation ID to validate
            
        Returns:
            Validated conversation ID
            
        Raises:
            ValidationError: If ID is invalid
        """
        if not conv_id:
            raise ValidationError("Conversation ID is required")
        
        conv_id = conv_id.strip()
        
        if len(conv_id) > MAX_CONVERSATION_ID_LENGTH:
            raise ValidationError(
                f"Conversation ID too long (max {MAX_CONVERSATION_ID_LENGTH} chars)"
            )
        
        # Only allow alphanumeric, underscore, hyphen
        if not CONVERSATION_ID_PATTERN.match(conv_id):
            raise ValidationError(
                "Conversation ID must contain only letters, numbers, underscores, and hyphens"
            )
        
        return conv_id


class PathValidator:
    """Validates file paths to prevent traversal attacks."""
    
    @staticmethod
    def validate_safe_path(
        path: str | Path,
        base_dir: str | Path,
        must_exist: bool = False,
    ) -> Path:
        """Validate that path is within base_dir.
        
        Args:
            path: Path to validate
            base_dir: Base directory that path must be within
            must_exist: If True, path must exist
            
        Returns:
            Resolved safe path
            
        Raises:
            ValidationError: If path is unsafe
        """
        try:
            path = Path(path).resolve()
            base_dir = Path(base_dir).resolve()
            
            # Check if path is within base_dir
            try:
                path.relative_to(base_dir)
            except ValueError:
                raise ValidationError(
                    f"Path {path} is outside allowed directory {base_dir}"
                )
            
            if must_exist and not path.exists():
                raise ValidationError(f"Path {path} does not exist")
            
            return path
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid path: {e}")


__all__ = [
    "InputValidator",
    "APIKeyValidator",
    "ConversationIDValidator",
    "PathValidator",
    "ValidationError",
]
