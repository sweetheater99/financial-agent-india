"""Context analysis for understanding Reddit posts and threads."""

from typing import Dict, List, Optional
from loguru import logger


class ContextAnalyzer:
    """Analyzes Reddit posts/comments to extract context for AI responses."""

    def analyze_post(self, post, max_comments: int = 5) -> Dict:
        """
        Analyze a Reddit post and extract relevant context.

        Args:
            post: PRAW post object
            max_comments: Maximum number of top comments to include

        Returns:
            Dict with analyzed context
        """
        try:
            context = {
                "type": "post",
                "subreddit": post.subreddit.display_name,
                "title": post.title,
                "content": post.selftext if hasattr(post, 'selftext') else "",
                "url": f"https://reddit.com{post.permalink}",
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "author": str(post.author) if post.author else "[deleted]",
                "flair": post.link_flair_text if hasattr(post, 'link_flair_text') else None,
                "is_question": self._is_question(post.title, post.selftext if hasattr(post, 'selftext') else ""),
                "sentiment": self._analyze_sentiment(post.title),
                "top_comments": []
            }

            # Get top comments for additional context
            try:
                post.comments.replace_more(limit=0)  # Remove MoreComments objects
                top_comments = list(post.comments)[:max_comments]

                for comment in top_comments:
                    if hasattr(comment, 'body'):
                        context["top_comments"].append({
                            "author": str(comment.author) if comment.author else "[deleted]",
                            "body": comment.body[:200],  # First 200 chars
                            "score": comment.score
                        })
            except Exception as e:
                logger.warning(f"Could not fetch comments: {e}")

            return context

        except Exception as e:
            logger.error(f"Error analyzing post: {e}")
            return {}

    def analyze_comment(self, comment) -> Dict:
        """
        Analyze a Reddit comment and extract context.

        Args:
            comment: PRAW comment object

        Returns:
            Dict with analyzed context
        """
        try:
            context = {
                "type": "comment",
                "subreddit": comment.subreddit.display_name,
                "content": comment.body,
                "url": f"https://reddit.com{comment.permalink}",
                "score": comment.score,
                "created_utc": comment.created_utc,
                "author": str(comment.author) if comment.author else "[deleted]",
                "is_question": self._is_question("", comment.body),
                "sentiment": self._analyze_sentiment(comment.body),
            }

            # Get parent context
            try:
                if hasattr(comment, 'parent') and callable(comment.parent):
                    parent = comment.parent()
                    if hasattr(parent, 'body'):
                        context["parent_comment"] = parent.body[:300]
                    elif hasattr(parent, 'title'):
                        context["title"] = parent.title
                        context["parent_post"] = parent.selftext[:300] if hasattr(parent, 'selftext') else ""
            except Exception as e:
                logger.warning(f"Could not fetch parent: {e}")

            return context

        except Exception as e:
            logger.error(f"Error analyzing comment: {e}")
            return {}

    def _is_question(self, title: str, content: str) -> bool:
        """Determine if the post/comment is a question."""
        text = f"{title} {content}".lower()

        question_indicators = [
            "?",
            "how do i",
            "how to",
            "what is",
            "what are",
            "why does",
            "why is",
            "can someone",
            "can anyone",
            "help me",
            "help with",
            "question",
            "asking"
        ]

        return any(indicator in text for indicator in question_indicators)

    def _analyze_sentiment(self, text: str) -> str:
        """
        Simple sentiment analysis.

        Returns:
            'positive', 'negative', or 'neutral'
        """
        text_lower = text.lower()

        positive_words = [
            "great", "awesome", "excellent", "good", "love", "thanks",
            "helpful", "amazing", "wonderful", "best", "appreciate"
        ]
        negative_words = [
            "bad", "terrible", "awful", "hate", "worst", "problem",
            "issue", "error", "broken", "help", "frustrated"
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def should_reply(self, context: Dict, keywords: List[str],
                    min_score: int = 0, max_age_hours: int = 24) -> bool:
        """
        Determine if we should reply based on context and rules.

        Args:
            context: Context dictionary from analyze_post/analyze_comment
            keywords: List of keywords to match
            min_score: Minimum score required
            max_age_hours: Maximum age in hours

        Returns:
            bool: True if we should consider replying
        """
        # Check score
        if context.get("score", 0) < min_score:
            return False

        # Check age
        import time
        age_hours = (time.time() - context.get("created_utc", 0)) / 3600
        if age_hours > max_age_hours:
            return False

        # Check keywords
        title = context.get("title", "").lower()
        content = context.get("content", "").lower()
        text = f"{title} {content}"

        if keywords:
            return any(keyword.lower() in text for keyword in keywords)

        return True

    def extract_key_topics(self, context: Dict) -> List[str]:
        """Extract key topics/themes from the context."""
        text = f"{context.get('title', '')} {context.get('content', '')}".lower()

        # Simple keyword extraction (can be enhanced with NLP)
        common_topics = {
            "python": ["python", "py", "pip", "django", "flask"],
            "javascript": ["javascript", "js", "node", "react", "vue"],
            "web": ["html", "css", "website", "web", "frontend", "backend"],
            "data": ["data", "dataset", "csv", "pandas", "analysis"],
            "ml": ["machine learning", "ml", "ai", "model", "neural"],
            "error": ["error", "exception", "traceback", "bug", "issue"],
            "beginner": ["beginner", "newbie", "new to", "learning", "start"],
        }

        topics = []
        for topic, keywords in common_topics.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)

        return topics
