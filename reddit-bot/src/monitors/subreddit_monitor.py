"""Monitor subreddits for new posts matching criteria."""

import random
from typing import List, Dict, Optional
from loguru import logger


class SubredditMonitor:
    """Monitors subreddits for posts to reply to."""

    def __init__(self, reddit_client, database, config: Dict):
        self.reddit_client = reddit_client
        self.database = database
        self.config = config

    def check_subreddits(self, subreddit_configs: List[Dict]) -> List[Dict]:
        """
        Check configured subreddits for posts to reply to.

        Args:
            subreddit_configs: List of subreddit configurations

        Returns:
            List of posts that match reply criteria
        """
        matching_posts = []

        for sub_config in subreddit_configs:
            if not sub_config.get('enabled', False):
                continue

            subreddit_name = sub_config['name']
            logger.info(f"Checking r/{subreddit_name}")

            try:
                posts = self._get_posts_to_check(subreddit_name, sub_config)

                for post in posts:
                    if self._should_process_post(post, sub_config):
                        matching_posts.append({
                            'post': post,
                            'subreddit_config': sub_config,
                            'subreddit_name': subreddit_name
                        })

            except Exception as e:
                logger.error(f"Error checking r/{subreddit_name}: {e}")

        logger.info(f"Found {len(matching_posts)} matching posts")
        return matching_posts

    def _get_posts_to_check(self, subreddit_name: str, sub_config: Dict):
        """Get posts to check based on monitor settings."""
        posts = []
        monitor_config = sub_config.get('monitor', {})

        max_posts = self.config.get('monitoring', {}).get('max_posts_per_check', 50)

        if monitor_config.get('new_posts', True):
            new_posts = self.reddit_client.get_new_posts(subreddit_name, limit=max_posts)
            posts.extend(new_posts)

        if monitor_config.get('hot_posts', False):
            hot_posts = self.reddit_client.get_hot_posts(subreddit_name, limit=max_posts)
            posts.extend(hot_posts)

        if monitor_config.get('rising_posts', False):
            rising_posts = self.reddit_client.get_rising_posts(subreddit_name, limit=max_posts)
            posts.extend(rising_posts)

        return posts

    def _should_process_post(self, post, sub_config: Dict) -> bool:
        """Determine if we should process this post."""
        try:
            # Check if we've already seen this post
            if self.database.has_seen_post(post.id):
                return False

            # Mark as seen
            self.database.mark_post_seen(
                post_id=post.id,
                subreddit=post.subreddit.display_name,
                author=str(post.author) if post.author else "[deleted]",
                title=post.title,
                url=f"https://reddit.com{post.permalink}",
                created_utc=int(post.created_utc)
            )

            # Check if auto-reply is enabled
            auto_reply_config = sub_config.get('auto_reply', {})
            if not auto_reply_config.get('enabled', False):
                return False

            # Check if we've already replied
            if self.database.has_replied_to(post.id):
                return False

            # Check keywords
            keywords = auto_reply_config.get('keywords', [])
            if keywords and not self._matches_keywords(post, keywords):
                return False

            # Check filters
            filters = auto_reply_config.get('filters', {})
            if not self._passes_filters(post, filters):
                return False

            # Check reply probability (randomize to appear more natural)
            reply_probability = sub_config.get('reply_probability', 1.0)
            if random.random() > reply_probability:
                logger.debug(f"Skipping post {post.id} due to probability")
                return False

            # Check anti-detection settings
            if not self._passes_anti_detection(post):
                return False

            return True

        except Exception as e:
            logger.error(f"Error processing post {post.id}: {e}")
            return False

    def _matches_keywords(self, post, keywords: List[str]) -> bool:
        """Check if post contains any of the keywords."""
        title = post.title.lower()
        selftext = post.selftext.lower() if hasattr(post, 'selftext') else ""
        text = f"{title} {selftext}"

        return any(keyword.lower() in text for keyword in keywords)

    def _passes_filters(self, post, filters: Dict) -> bool:
        """Check if post passes all filters."""
        # Check minimum upvotes
        min_upvotes = filters.get('min_upvotes', 0)
        if post.score < min_upvotes:
            logger.debug(f"Post {post.id} below min upvotes: {post.score} < {min_upvotes}")
            return False

        # Check maximum upvotes
        max_upvotes = filters.get('max_upvotes')
        if max_upvotes is not None and post.score > max_upvotes:
            logger.debug(f"Post {post.id} above max upvotes: {post.score} > {max_upvotes}")
            return False

        # Check if question mark required
        if filters.get('require_question_mark', False):
            if '?' not in post.title and '?' not in getattr(post, 'selftext', ''):
                logger.debug(f"Post {post.id} missing question mark")
                return False

        # Check avoid keywords
        avoid_keywords = filters.get('avoid_keywords', [])
        if avoid_keywords:
            title = post.title.lower()
            selftext = getattr(post, 'selftext', '').lower()
            text = f"{title} {selftext}"

            if any(keyword.lower() in text for keyword in avoid_keywords):
                logger.debug(f"Post {post.id} contains avoid keyword")
                return False

        return True

    def _passes_anti_detection(self, post) -> bool:
        """Check anti-detection rules."""
        anti_detection = self.config.get('reply', {}).get('anti_detection', {})

        # Check if post has too many replies already
        max_replies = anti_detection.get('skip_if_too_many_replies', 10)
        if post.num_comments > max_replies:
            logger.debug(f"Post {post.id} has too many comments: {post.num_comments}")
            return False

        # Check post age
        import time
        max_age_hours = anti_detection.get('skip_if_post_age_hours', 24)
        age_hours = (time.time() - post.created_utc) / 3600

        if age_hours > max_age_hours:
            logger.debug(f"Post {post.id} too old: {age_hours:.1f} hours")
            return False

        return True

    def notify_new_post(self, post, subreddit_name: str):
        """
        Called when a new post is detected in a monitored subreddit.
        Used for notification purposes.
        """
        logger.info(f"New post in r/{subreddit_name}: {post.title}")
        # This will be handled by the notification system
        return {
            'type': 'new_post',
            'subreddit': subreddit_name,
            'title': post.title,
            'url': f"https://reddit.com{post.permalink}",
            'author': str(post.author) if post.author else "[deleted]"
        }
