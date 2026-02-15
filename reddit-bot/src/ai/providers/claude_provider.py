"""Claude AI provider implementation."""

import os
from typing import Dict
from loguru import logger

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import AIProvider


class ClaudeProvider(AIProvider):
    """Claude AI provider using Anthropic API."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4.5"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model
        self.client = None

        if self.api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Claude AI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Claude provider: {e}")

    def is_available(self) -> bool:
        """Check if Claude is available."""
        return self.client is not None and ANTHROPIC_AVAILABLE

    def generate_response(
        self,
        prompt: str,
        context: Dict,
        guidelines: str = "",
        max_length: int = 500
    ) -> Dict:
        """Generate a response using Claude."""
        if not self.is_available():
            raise ValueError("Claude provider is not available")

        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context, guidelines, max_length)

            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )

            response_text = message.content[0].text

            # Calculate confidence (simple heuristic for now)
            confidence = self._calculate_confidence(response_text, context)

            return {
                "response": response_text.strip(),
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating response with Claude: {e}")
            raise

    def _build_prompt(self, prompt: str, context: Dict, guidelines: str, max_length: int) -> str:
        """Build the full prompt for Claude."""
        parts = []

        # Add context
        parts.append("You are responding to a Reddit post/comment as a helpful human user.")
        parts.append(f"\nPost/Comment Context:")
        parts.append(f"Subreddit: r/{context.get('subreddit', 'unknown')}")
        parts.append(f"Title: {context.get('title', 'N/A')}")
        parts.append(f"Content: {context.get('content', '')}")

        if context.get('parent_comment'):
            parts.append(f"\nParent Comment: {context['parent_comment']}")

        # Add writing style instructions
        if context.get('writing_style'):
            parts.append(f"\nWriting Style Guidelines:")
            parts.append(context['writing_style'])

        # Add specific guidelines
        if guidelines:
            parts.append(f"\nResponse Guidelines:")
            parts.append(guidelines)

        # Add the main instruction
        parts.append(f"\n{prompt}")

        # Add constraints
        parts.append(f"\nIMPORTANT:")
        parts.append(f"- Keep response under {max_length} characters")
        parts.append(f"- Sound natural and human-like")
        parts.append(f"- Don't mention you're an AI")
        parts.append(f"- Be conversational and authentic")
        parts.append(f"- Match the tone of the subreddit")
        parts.append(f"- If you're not confident you can help, say so naturally")

        return "\n".join(parts)

    def _calculate_confidence(self, response: str, context: Dict) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.7  # Base confidence

        # Adjust based on response characteristics
        if len(response) < 20:
            confidence -= 0.2  # Too short
        if len(response) > 500:
            confidence -= 0.1  # Too long

        # Check if response seems relevant to context
        content = context.get('content', '').lower()
        response_lower = response.lower()

        # Simple relevance check
        common_words = set(content.split()) & set(response_lower.split())
        if len(common_words) > 3:
            confidence += 0.1

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
