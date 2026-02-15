"""Base AI provider interface."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class AIProvider(ABC):
    """Base class for AI providers."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        context: Dict,
        guidelines: str = "",
        max_length: int = 500
    ) -> Dict:
        """
        Generate a response based on prompt and context.

        Args:
            prompt: The main prompt for the AI
            context: Context dictionary with post/comment info
            guidelines: Specific guidelines for this response
            max_length: Maximum response length

        Returns:
            Dict with 'response' and 'confidence' keys
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        pass
