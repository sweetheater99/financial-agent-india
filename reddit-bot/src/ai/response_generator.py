"""Main response generator that coordinates AI providers and style learning."""

from typing import Dict, Optional
from loguru import logger

from .providers import ClaudeProvider, OpenAIProvider
from .context_analyzer import ContextAnalyzer
from .style_learner import StyleLearner


class ResponseGenerator:
    """Generates natural Reddit responses using AI."""

    def __init__(self, config: Dict, database):
        self.config = config
        self.database = database
        self.context_analyzer = ContextAnalyzer()
        self.style_learner = StyleLearner(database)

        # Initialize AI provider
        provider_name = config.get('ai', {}).get('provider', 'claude')
        model = config.get('ai', {}).get('model', 'claude-sonnet-4.5')

        if provider_name == 'claude':
            self.provider = ClaudeProvider(model=model)
        elif provider_name == 'openai':
            self.provider = OpenAIProvider(model=model)
        else:
            raise ValueError(f"Unknown AI provider: {provider_name}")

        if not self.provider.is_available():
            logger.error(f"AI provider {provider_name} is not available!")
            raise ValueError(f"AI provider {provider_name} is not configured")

        logger.info(f"Response generator initialized with {provider_name}")

    def generate_reply(
        self,
        post_or_comment,
        context_type: str,
        guidelines: str = "",
        subreddit_config: Dict = None
    ) -> Optional[Dict]:
        """
        Generate a reply for a Reddit post or comment.

        Args:
            post_or_comment: PRAW post or comment object
            context_type: 'post' or 'comment'
            guidelines: Specific guidelines for this reply
            subreddit_config: Subreddit-specific configuration

        Returns:
            Dict with 'response' and 'confidence' or None if generation fails
        """
        try:
            # Analyze context
            if context_type == 'post':
                context = self.context_analyzer.analyze_post(post_or_comment)
            else:
                context = self.context_analyzer.analyze_comment(post_or_comment)

            if not context:
                logger.warning("Failed to analyze context")
                return None

            # Get writing style instructions
            if self.config.get('ai', {}).get('style', {}).get('learn_from_history', False):
                style_instructions = self.style_learner.get_style_instructions()
                context['writing_style'] = style_instructions

            # Build prompt
            prompt = self._build_prompt(context, context_type)

            # Generate response
            max_length = self.config.get('ai', {}).get('max_response_length', 500)

            result = self.provider.generate_response(
                prompt=prompt,
                context=context,
                guidelines=guidelines,
                max_length=max_length
            )

            logger.info(f"Generated response with confidence: {result['confidence']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error generating reply: {e}")
            return None

    def _build_prompt(self, context: Dict, context_type: str) -> str:
        """Build the main prompt for response generation."""
        if context_type == 'post':
            if context.get('is_question'):
                return (
                    "Generate a helpful, natural response to this question. "
                    "Provide useful information or ask clarifying questions if needed. "
                    "Write as a knowledgeable Reddit user would."
                )
            else:
                return (
                    "Generate a thoughtful, conversational response to this post. "
                    "Share relevant thoughts, experiences, or insights. "
                    "Be authentic and engaging."
                )
        else:  # comment
            return (
                "Generate a natural reply to this comment. "
                "Continue the conversation authentically. "
                "Be helpful, friendly, and genuine."
            )

    def learn_user_style(self, reddit_client, username: str):
        """
        Learn the user's writing style from their Reddit history.

        Args:
            reddit_client: PRAW Reddit instance
            username: Reddit username to analyze
        """
        max_samples = self.config.get('ai', {}).get('style', {}).get('comment_samples', 50)
        self.style_learner.learn_from_reddit(reddit_client, username, max_samples)
        logger.info(f"Learned writing style from u/{username}")

    def should_reply_based_on_confidence(self, confidence: float) -> bool:
        """
        Determine if we should post a reply based on confidence threshold.

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            bool: True if confidence is high enough
        """
        threshold = self.config.get('ai', {}).get('confidence_threshold', 0.8)
        return confidence >= threshold

    def validate_response(self, response: str) -> bool:
        """
        Validate that the response is appropriate.

        Args:
            response: Generated response text

        Returns:
            bool: True if response is valid
        """
        # Check minimum length
        if len(response) < 10:
            logger.warning("Response too short")
            return False

        # Check maximum length
        max_length = self.config.get('ai', {}).get('max_response_length', 500)
        if len(response) > max_length * 1.5:  # Allow some buffer
            logger.warning("Response too long")
            return False

        # Check for AI indicators (we don't want to sound like a bot)
        ai_phrases = [
            "as an ai",
            "i'm an ai",
            "i am an ai",
            "as a language model",
            "i don't have personal",
            "i cannot feel"
        ]

        response_lower = response.lower()
        if any(phrase in response_lower for phrase in ai_phrases):
            logger.warning("Response contains AI indicator phrases")
            return False

        return True
