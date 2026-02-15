"""Main Reddit bot orchestrator."""

import time
import yaml
from pathlib import Path
from typing import Dict, List
from loguru import logger
from dotenv import load_dotenv

from .reddit_client import RedditClient
from .storage import Database
from .ai import ResponseGenerator
from .monitors import SubredditMonitor, UserMonitor
from .actions import SmartReply, PostCreator
from .notifications import Notifier


class RedditBot:
    """Main Reddit bot that coordinates all components."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the Reddit bot.

        Args:
            config_dir: Directory containing configuration files
        """
        # Load environment variables
        load_dotenv()

        # Load configurations
        self.config = self._load_config(config_dir)
        self.subreddit_configs = self._load_yaml(f"{config_dir}/subreddits.yaml").get('subreddits', [])
        self.user_configs = self._load_yaml(f"{config_dir}/users.yaml").get('tracked_users', [])

        # Initialize components
        self._initialize_components()

        logger.info("Reddit bot initialized successfully")

    def _load_config(self, config_dir: str) -> Dict:
        """Load main configuration."""
        config_path = Path(config_dir) / "config.yaml"
        return self._load_yaml(config_path)

    def _load_yaml(self, file_path) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}

    def _initialize_components(self):
        """Initialize all bot components."""
        # Database
        db_path = self.config.get('database', {}).get('path', 'data/bot.db')
        self.database = Database(db_path)

        # Reddit client
        self.reddit_client = RedditClient()

        # AI response generator
        self.response_generator = ResponseGenerator(self.config, self.database)

        # Monitors
        self.subreddit_monitor = SubredditMonitor(
            self.reddit_client,
            self.database,
            self.config
        )
        self.user_monitor = UserMonitor(
            self.reddit_client,
            self.database,
            self.config
        )

        # Actions
        self.smart_reply = SmartReply(
            self.reddit_client,
            self.database,
            self.response_generator,
            self.config
        )
        self.post_creator = PostCreator(
            self.reddit_client,
            self.database,
            self.config
        )

        # Notifications
        self.notifier = Notifier(self.config)

        logger.info("All components initialized")

    def learn_writing_style(self):
        """Learn user's writing style from their Reddit history."""
        logger.info("Learning writing style from Reddit history...")

        username = self.reddit_client.username
        self.response_generator.learn_user_style(
            self.reddit_client.reddit,
            username
        )

        logger.info("Writing style learning complete")

    def run_once(self):
        """Run one iteration of monitoring and processing."""
        logger.info("Running bot iteration...")

        # Monitor subreddits
        if self.config.get('subreddit_monitoring', {}).get('enabled', True):
            self._check_subreddits()

        # Monitor users
        if self.config.get('user_tracking', {}).get('enabled', True):
            self._check_users()

        logger.info("Bot iteration complete")

    def run_continuous(self):
        """Run bot continuously with scheduled checks."""
        logger.info("Starting continuous bot operation...")

        check_interval = self.config.get('monitoring', {}).get('check_interval', 300)

        try:
            while True:
                try:
                    self.run_once()

                except Exception as e:
                    logger.error(f"Error in bot iteration: {e}")
                    self.notifier.notify('errors', {'error': str(e)})

                logger.info(f"Sleeping for {check_interval} seconds...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")

    def _check_subreddits(self):
        """Check configured subreddits for new posts."""
        logger.info("Checking subreddits...")

        matching_posts = self.subreddit_monitor.check_subreddits(self.subreddit_configs)

        # Notify about new posts
        for match in matching_posts:
            post = match['post']
            self.notifier.notify('new_subreddit_posts', {
                'subreddit': match['subreddit_name'],
                'title': post.title,
                'author': str(post.author) if post.author else "[deleted]",
                'url': f"https://reddit.com{post.permalink}"
            })

            # Process for auto-reply if enabled
            if self.config.get('reply', {}).get('enabled', True):
                result = self.smart_reply.process_post(post, match['subreddit_config'])

                if result and result.get('status') == 'posted':
                    self.notifier.notify('auto_reply_posted', result)

        logger.info(f"Processed {len(matching_posts)} matching posts")

    def _check_users(self):
        """Check tracked users for new activity."""
        logger.info("Checking tracked users...")

        new_activities = self.user_monitor.check_users(self.user_configs)

        # Notify about each activity
        for activity in new_activities:
            self.notifier.notify('tracked_user_activity', activity)

        if new_activities:
            logger.info(f"Found {len(new_activities)} new user activities")

    def get_pending_replies(self) -> List[Dict]:
        """Get replies pending approval."""
        return self.smart_reply.get_pending_replies()

    def approve_reply(self, index: int) -> Dict:
        """Approve a pending reply."""
        result = self.smart_reply.approve_reply(index)

        if result.get('status') == 'posted':
            self.notifier.notify('auto_reply_posted', result)

        return result

    def reject_reply(self, index: int):
        """Reject a pending reply."""
        self.smart_reply.reject_reply(index)

    def edit_and_approve_reply(self, index: int, edited_text: str) -> Dict:
        """Edit and approve a pending reply."""
        result = self.smart_reply.edit_and_approve_reply(index, edited_text)

        if result.get('status') == 'posted':
            self.notifier.notify('auto_reply_posted', result)

        return result

    def create_post(self, subreddit: str, title: str, text: str = None,
                   url: str = None) -> str:
        """
        Create a Reddit post.

        Args:
            subreddit: Subreddit name
            title: Post title
            text: Post text (for text posts)
            url: URL (for link posts)

        Returns:
            Post ID if successful
        """
        if text:
            return self.post_creator.create_text_post(subreddit, title, text)
        elif url:
            return self.post_creator.create_link_post(subreddit, title, url)
        else:
            raise ValueError("Must provide either text or url")

    def get_stats(self) -> Dict:
        """Get bot statistics."""
        return self.database.get_stats()

    def shutdown(self):
        """Gracefully shutdown the bot."""
        logger.info("Shutting down bot...")
        # Clean up resources if needed
        logger.info("Bot shutdown complete")
