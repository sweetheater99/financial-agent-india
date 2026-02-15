"""Learning and mimicking user's writing style from their comment history."""

from typing import List, Dict, Optional
from loguru import logger
import statistics


class StyleLearner:
    """Learns writing style from user's comment history."""

    def __init__(self, database):
        self.database = database

    def learn_from_reddit(self, reddit_client, username: str, max_comments: int = 50):
        """
        Analyze user's Reddit comment history to learn their style.

        Args:
            reddit_client: PRAW Reddit instance
            username: Reddit username to analyze
            max_comments: Number of recent comments to analyze
        """
        try:
            user = reddit_client.redditor(username)
            comments = list(user.comments.new(limit=max_comments))

            logger.info(f"Analyzing {len(comments)} comments from u/{username}")

            for comment in comments:
                try:
                    self.database.add_writing_sample(
                        comment_id=comment.id,
                        subreddit=comment.subreddit.display_name,
                        text=comment.body,
                        created_utc=int(comment.created_utc)
                    )
                except Exception as e:
                    logger.warning(f"Error storing comment {comment.id}: {e}")

            logger.info(f"Stored {len(comments)} writing samples")

        except Exception as e:
            logger.error(f"Error learning from Reddit history: {e}")

    def analyze_style(self) -> Dict:
        """
        Analyze stored writing samples to determine user's style.

        Returns:
            Dict with style characteristics
        """
        samples = self.database.get_writing_samples(limit=100)

        if not samples:
            logger.warning("No writing samples available for analysis")
            return self._default_style()

        # Analyze characteristics
        lengths = [s['length'] for s in samples]
        texts = [s['text'] for s in samples]

        style = {
            "avg_length": statistics.mean(lengths) if lengths else 100,
            "median_length": statistics.median(lengths) if lengths else 100,
            "length_variance": statistics.stdev(lengths) if len(lengths) > 1 else 50,
            "formality": self._analyze_formality(texts),
            "common_phrases": self._extract_common_phrases(texts),
            "punctuation_style": self._analyze_punctuation(texts),
            "emoji_usage": self._analyze_emoji_usage(texts),
            "greeting_style": self._analyze_greetings(texts),
            "sample_count": len(samples)
        }

        return style

    def get_style_instructions(self) -> str:
        """
        Generate instructions for AI to mimic the user's style.

        Returns:
            String with style instructions
        """
        style = self.analyze_style()

        instructions = []
        instructions.append("Match this writing style:")

        # Length
        avg_len = int(style["avg_length"])
        if avg_len < 50:
            instructions.append("- Keep responses very brief and concise (under 50 words)")
        elif avg_len < 150:
            instructions.append("- Keep responses moderately short (1-2 paragraphs)")
        elif avg_len < 300:
            instructions.append("- Write medium-length responses (2-3 paragraphs)")
        else:
            instructions.append("- Write detailed, comprehensive responses")

        # Formality
        formality = style["formality"]
        if formality < 0.3:
            instructions.append("- Use casual, informal language")
            instructions.append("- Use contractions (don't, can't, etc.)")
        elif formality < 0.7:
            instructions.append("- Use a balanced, conversational tone")
        else:
            instructions.append("- Use more formal, professional language")

        # Emoji usage
        if style["emoji_usage"] > 0.3:
            instructions.append("- Feel free to use emojis occasionally")
        else:
            instructions.append("- Avoid using emojis")

        # Common phrases
        if style["common_phrases"]:
            phrases = ", ".join(style["common_phrases"][:3])
            instructions.append(f"- Common phrases you use: {phrases}")

        # Greeting style
        if style["greeting_style"]:
            instructions.append(f"- Greeting style: {style['greeting_style']}")

        return "\n".join(instructions)

    def _default_style(self) -> Dict:
        """Return default style when no samples are available."""
        return {
            "avg_length": 150,
            "median_length": 120,
            "length_variance": 50,
            "formality": 0.5,
            "common_phrases": [],
            "punctuation_style": "moderate",
            "emoji_usage": 0.0,
            "greeting_style": "casual",
            "sample_count": 0
        }

    def _analyze_formality(self, texts: List[str]) -> float:
        """
        Analyze formality level (0.0 = very casual, 1.0 = very formal).
        """
        formal_indicators = ["however", "therefore", "furthermore", "regarding", "thus"]
        casual_indicators = ["lol", "btw", "gonna", "wanna", "yeah", "nah", "omg"]
        contractions = ["don't", "can't", "won't", "isn't", "aren't", "didn't"]

        formal_count = 0
        casual_count = 0

        for text in texts:
            text_lower = text.lower()
            formal_count += sum(1 for word in formal_indicators if word in text_lower)
            casual_count += sum(1 for word in casual_indicators if word in text_lower)
            casual_count += sum(1 for word in contractions if word in text_lower)

        total = formal_count + casual_count
        if total == 0:
            return 0.5  # Neutral

        return formal_count / total

    def _extract_common_phrases(self, texts: List[str], min_occurrences: int = 3) -> List[str]:
        """Extract commonly used phrases."""
        phrase_counts = {}

        common_starters = [
            "i think", "in my opinion", "i would", "you should",
            "you can", "try to", "make sure", "don't forget"
        ]

        for text in texts:
            text_lower = text.lower()
            for phrase in common_starters:
                if phrase in text_lower:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Return phrases that occur at least min_occurrences times
        common = [phrase for phrase, count in phrase_counts.items()
                 if count >= min_occurrences]
        return sorted(common, key=lambda p: phrase_counts[p], reverse=True)[:5]

    def _analyze_punctuation(self, texts: List[str]) -> str:
        """Analyze punctuation style."""
        exclamation_count = sum(text.count('!') for text in texts)
        question_count = sum(text.count('?') for text in texts)
        ellipsis_count = sum(text.count('...') for text in texts)

        total_chars = sum(len(text) for text in texts)
        if total_chars == 0:
            return "moderate"

        exclamation_ratio = exclamation_count / (total_chars / 1000)

        if exclamation_ratio > 5:
            return "enthusiastic"
        elif ellipsis_count > len(texts) * 0.3:
            return "thoughtful"
        else:
            return "moderate"

    def _analyze_emoji_usage(self, texts: List[str]) -> float:
        """Calculate emoji usage ratio."""
        emoji_count = 0
        common_emojis = ["😊", "😂", "👍", "❤️", "🙏", "😅", "🤔", "👌", ":)", ":(", ":D"]

        for text in texts:
            emoji_count += sum(text.count(emoji) for emoji in common_emojis)

        if not texts:
            return 0.0

        return emoji_count / len(texts)

    def _analyze_greetings(self, texts: List[str]) -> str:
        """Analyze how user greets/opens comments."""
        greeting_styles = {
            "hey": ["hey", "hi ", "hello"],
            "thanks": ["thanks", "thank you"],
            "direct": []  # No greeting, gets straight to point
        }

        greeting_counts = {style: 0 for style in greeting_styles}

        for text in texts[:20]:  # Check first 20 comments
            text_lower = text.lower()
            first_30_chars = text_lower[:30]

            matched = False
            for style, keywords in greeting_styles.items():
                if style == "direct":
                    continue
                if any(keyword in first_30_chars for keyword in keywords):
                    greeting_counts[style] += 1
                    matched = True
                    break

            if not matched:
                greeting_counts["direct"] += 1

        # Return most common style
        if not greeting_counts:
            return "casual"

        most_common = max(greeting_counts.items(), key=lambda x: x[1])
        return most_common[0]
