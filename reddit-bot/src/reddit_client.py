"""Reddit API client wrapper using PRAW."""

import os
from typing import Optional
import praw
from loguru import logger


class RedditClient:
    """Wrapper for Reddit API operations using PRAW."""

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        username: str = None,
        password: str = None,
        user_agent: str = None
    ):
        """
        Initialize Reddit client.

        Args:
            client_id: Reddit app client ID (or set REDDIT_CLIENT_ID env var)
            client_secret: Reddit app client secret (or set REDDIT_CLIENT_SECRET env var)
            username: Reddit username (or set REDDIT_USERNAME env var)
            password: Reddit password (or set REDDIT_PASSWORD env var)
            user_agent: User agent string (or set REDDIT_USER_AGENT env var)
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.username = username or os.getenv('REDDIT_USERNAME')
        self.password = password or os.getenv('REDDIT_PASSWORD')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'RedditBot/1.0')

        if not all([self.client_id, self.client_secret, self.username, self.password]):
            raise ValueError(
                "Missing Reddit credentials. Set environment variables or pass to constructor."
            )

        self.reddit = self._initialize_reddit()
        logger.info(f"Reddit client initialized for user: {self.username}")

    def _initialize_reddit(self) -> praw.Reddit:
        """Initialize PRAW Reddit instance."""
        try:
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.username,
                password=self.password,
                user_agent=self.user_agent
            )

            # Test authentication
            reddit.user.me()
            logger.info("Successfully authenticated with Reddit")

            return reddit

        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise

    def get_subreddit(self, subreddit_name: str):
        """Get subreddit object."""
        return self.reddit.subreddit(subreddit_name)

    def get_user(self, username: str):
        """Get user/redditor object."""
        return self.reddit.redditor(username)

    def get_new_posts(self, subreddit_name: str, limit: int = 10):
        """Get new posts from a subreddit."""
        subreddit = self.get_subreddit(subreddit_name)
        return subreddit.new(limit=limit)

    def get_hot_posts(self, subreddit_name: str, limit: int = 10):
        """Get hot posts from a subreddit."""
        subreddit = self.get_subreddit(subreddit_name)
        return subreddit.hot(limit=limit)

    def get_rising_posts(self, subreddit_name: str, limit: int = 10):
        """Get rising posts from a subreddit."""
        subreddit = self.get_subreddit(subreddit_name)
        return subreddit.rising(limit=limit)

    def get_user_posts(self, username: str, limit: int = 10):
        """Get recent posts from a user."""
        user = self.get_user(username)
        return user.submissions.new(limit=limit)

    def get_user_comments(self, username: str, limit: int = 10):
        """Get recent comments from a user."""
        user = self.get_user(username)
        return user.comments.new(limit=limit)

    def reply_to_post(self, post, text: str) -> Optional[str]:
        """
        Reply to a post.

        Args:
            post: PRAW post object
            text: Reply text

        Returns:
            Comment ID if successful, None otherwise
        """
        try:
            comment = post.reply(text)
            logger.info(f"Successfully replied to post {post.id}")
            return comment.id

        except Exception as e:
            logger.error(f"Failed to reply to post {post.id}: {e}")
            return None

    def reply_to_comment(self, comment, text: str) -> Optional[str]:
        """
        Reply to a comment.

        Args:
            comment: PRAW comment object
            text: Reply text

        Returns:
            Comment ID if successful, None otherwise
        """
        try:
            reply = comment.reply(text)
            logger.info(f"Successfully replied to comment {comment.id}")
            return reply.id

        except Exception as e:
            logger.error(f"Failed to reply to comment {comment.id}: {e}")
            return None

    def create_post(
        self,
        subreddit_name: str,
        title: str,
        selftext: str = None,
        url: str = None,
        flair_id: str = None
    ) -> Optional[str]:
        """
        Create a new post in a subreddit.

        Args:
            subreddit_name: Name of subreddit
            title: Post title
            selftext: Text content (for text posts)
            url: URL (for link posts)
            flair_id: Flair ID (optional)

        Returns:
            Post ID if successful, None otherwise
        """
        try:
            subreddit = self.get_subreddit(subreddit_name)

            if selftext:
                post = subreddit.submit(title=title, selftext=selftext, flair_id=flair_id)
            elif url:
                post = subreddit.submit(title=title, url=url, flair_id=flair_id)
            else:
                raise ValueError("Must provide either selftext or url")

            logger.info(f"Successfully created post in r/{subreddit_name}: {post.id}")
            return post.id

        except Exception as e:
            logger.error(f"Failed to create post in r/{subreddit_name}: {e}")
            return None

    def get_my_recent_comments(self, limit: int = 50):
        """Get my own recent comments for style learning."""
        try:
            user = self.reddit.user.me()
            return list(user.comments.new(limit=limit))

        except Exception as e:
            logger.error(f"Failed to get my comments: {e}")
            return []

    def is_authenticated(self) -> bool:
        """Check if we're authenticated."""
        try:
            self.reddit.user.me()
            return True
        except Exception:
            return False
