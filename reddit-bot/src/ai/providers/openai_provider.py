"""OpenAI provider implementation."""

import os
from typing import Dict
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import AIProvider


class OpenAIProvider(AIProvider):
    """OpenAI provider using GPT models."""

    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self.client is not None and OPENAI_AVAILABLE

    def generate_response(
        self,
        prompt: str,
        context: Dict,
        guidelines: str = "",
        max_length: int = 500
    ) -> Dict:
        """Generate a response using OpenAI."""
        if not self.is_available():
            raise ValueError("OpenAI provider is not available")

        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context, guidelines, max_length)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful Reddit user having natural conversations."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            response_text = response.choices[0].message.content

            # Calculate confidence
            confidence = self._calculate_confidence(response_text, context)

            return {
                "response": response_text.strip(),
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            raise

    def _build_prompt(self, prompt: str, context: Dict, guidelines: str, max_length: int) -> str:
        """Build the full prompt for OpenAI."""
        parts = []

        # Add context
        parts.append(f"Context:")
        parts.append(f"Subreddit: r/{context.get('subreddit', 'unknown')}")
        parts.append(f"Title: {context.get('title', 'N/A')}")
        parts.append(f"Content: {context.get('content', '')}")

        if context.get('parent_comment'):
            parts.append(f"\nParent Comment: {context['parent_comment']}")

        # Add writing style
        if context.get('writing_style'):
            parts.append(f"\nYour typical writing style:")
            parts.append(context['writing_style'])

        # Add guidelines
        if guidelines:
            parts.append(f"\nGuidelines:")
            parts.append(guidelines)

        # Main instruction
        parts.append(f"\n{prompt}")

        # Constraints
        parts.append(f"\nKeep response under {max_length} characters. Sound natural and human.")

        return "\n".join(parts)

    def _calculate_confidence(self, response: str, context: Dict) -> float:
        """Calculate confidence score."""
        confidence = 0.7

        if len(response) < 20:
            confidence -= 0.2
        if len(response) > 500:
            confidence -= 0.1

        content = context.get('content', '').lower()
        response_lower = response.lower()
        common_words = set(content.split()) & set(response_lower.split())
        if len(common_words) > 3:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))
