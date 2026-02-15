"""Create Reddit posts based on context."""

from typing import Dict, Optional
from loguru import logger


class PostCreator:
    """Handles creating Reddit posts."""

    def __init__(self, reddit_client, database, config: Dict):
        self.reddit_client = reddit_client
        self.database = database
        self.config = config

    def create_text_post(
        self,
        subreddit: str,
        title: str,
        text: str,
        flair_id: str = None
    ) -> Optional[str]:
        """
        Create a text post.

        Args:
            subreddit: Subreddit name
            title: Post title
            text: Post body text
            flair_id: Optional flair ID

        Returns:
            Post ID if successful, None otherwise
        """
        try:
            logger.info(f"Creating text post in r/{subreddit}: {title[:50]}...")

            post_id = self.reddit_client.create_post(
                subreddit_name=subreddit,
                title=title,
                selftext=text,
                flair_id=flair_id
            )

            if post_id:
                logger.info(f"Successfully created post: {post_id}")
                return post_id
            else:
                logger.error("Failed to create post")
                return None

        except Exception as e:
            logger.error(f"Error creating text post: {e}")
            return None

    def create_link_post(
        self,
        subreddit: str,
        title: str,
        url: str,
        flair_id: str = None
    ) -> Optional[str]:
        """
        Create a link post.

        Args:
            subreddit: Subreddit name
            title: Post title
            url: Link URL
            flair_id: Optional flair ID

        Returns:
            Post ID if successful, None otherwise
        """
        try:
            logger.info(f"Creating link post in r/{subreddit}: {title[:50]}...")

            post_id = self.reddit_client.create_post(
                subreddit_name=subreddit,
                title=title,
                url=url,
                flair_id=flair_id
            )

            if post_id:
                logger.info(f"Successfully created link post: {post_id}")
                return post_id
            else:
                logger.error("Failed to create link post")
                return None

        except Exception as e:
            logger.error(f"Error creating link post: {e}")
            return None

    def create_post_from_context(
        self,
        subreddit: str,
        context: Dict,
        ai_generator = None
    ) -> Optional[str]:
        """
        Create a post based on provided context using AI.

        Args:
            subreddit: Subreddit to post in
            context: Context dictionary with information for the post
            ai_generator: Optional AI generator for creating content

        Returns:
            Post ID if successful, None otherwise
        """
        try:
            # This would use AI to generate post content based on context
            # For now, just a placeholder

            title = context.get('title', 'Untitled Post')
            content = context.get('content', '')

            if ai_generator:
                # Use AI to generate/enhance content
                # This could be implemented similar to response_generator
                pass

            return self.create_text_post(
                subreddit=subreddit,
                title=title,
                text=content
            )

        except Exception as e:
            logger.error(f"Error creating post from context: {e}")
            return None
