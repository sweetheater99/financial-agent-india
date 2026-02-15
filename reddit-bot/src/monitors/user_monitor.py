"""Monitor specific users for new posts and comments."""

from typing import List, Dict
from loguru import logger


class UserMonitor:
    """Monitors specific Reddit users for activity."""

    def __init__(self, reddit_client, database, config: Dict):
        self.reddit_client = reddit_client
        self.database = database
        self.config = config

    def check_users(self, user_configs: List[Dict]) -> List[Dict]:
        """
        Check configured users for new activity.

        Args:
            user_configs: List of user tracking configurations

        Returns:
            List of new activities detected
        """
        new_activities = []

        for user_config in user_configs:
            if not user_config.get('enabled', False):
                continue

            username = user_config['username']
            logger.info(f"Checking user u/{username}")

            try:
                activities = self._check_user_activity(username, user_config)
                new_activities.extend(activities)

            except Exception as e:
                logger.error(f"Error checking user u/{username}: {e}")

        if new_activities:
            logger.info(f"Found {len(new_activities)} new user activities")

        return new_activities

    def _check_user_activity(self, username: str, user_config: Dict) -> List[Dict]:
        """Check activity for a specific user."""
        activities = []
        track_config = user_config.get('track', {})

        # Check posts
        if track_config.get('posts', True):
            post_activities = self._check_user_posts(username, user_config)
            activities.extend(post_activities)

        # Check comments
        if track_config.get('comments', True):
            comment_activities = self._check_user_comments(username, user_config)
            activities.extend(comment_activities)

        return activities

    def _check_user_posts(self, username: str, user_config: Dict) -> List[Dict]:
        """Check for new posts from a user."""
        activities = []

        try:
            # Get recent posts
            posts = self.reddit_client.get_user_posts(username, limit=25)

            for post in posts:
                # Check if we've already tracked this post
                if self.database.has_tracked_user_activity(post.id):
                    continue

                # Record the activity
                self.database.record_user_activity(
                    username=username,
                    activity_type='post',
                    item_id=post.id,
                    subreddit=post.subreddit.display_name,
                    title=post.title,
                    url=f"https://reddit.com{post.permalink}",
                    created_utc=int(post.created_utc)
                )

                # Check if we should notify about this
                if self._should_notify(post, user_config, 'post'):
                    activities.append({
                        'type': 'user_post',
                        'username': username,
                        'post': post,
                        'subreddit': post.subreddit.display_name,
                        'title': post.title,
                        'url': f"https://reddit.com{post.permalink}",
                        'is_hidden': post.hidden if hasattr(post, 'hidden') else False
                    })
                    logger.info(f"New post from u/{username} in r/{post.subreddit.display_name}")

        except Exception as e:
            logger.error(f"Error checking posts for u/{username}: {e}")

        return activities

    def _check_user_comments(self, username: str, user_config: Dict) -> List[Dict]:
        """Check for new comments from a user."""
        activities = []

        try:
            # Get recent comments
            comments = self.reddit_client.get_user_comments(username, limit=25)

            for comment in comments:
                # Check if we've already tracked this comment
                if self.database.has_tracked_user_activity(comment.id):
                    continue

                # Record the activity
                try:
                    # Get the post this comment is on
                    post_title = comment.submission.title if hasattr(comment, 'submission') else 'Unknown'
                    subreddit = comment.subreddit.display_name

                    self.database.record_user_activity(
                        username=username,
                        activity_type='comment',
                        item_id=comment.id,
                        subreddit=subreddit,
                        title=post_title,
                        url=f"https://reddit.com{comment.permalink}",
                        created_utc=int(comment.created_utc)
                    )

                    # Check if we should notify
                    if self._should_notify(comment, user_config, 'comment'):
                        activities.append({
                            'type': 'user_comment',
                            'username': username,
                            'comment': comment,
                            'subreddit': subreddit,
                            'post_title': post_title,
                            'comment_body': comment.body[:200],  # First 200 chars
                            'url': f"https://reddit.com{comment.permalink}"
                        })
                        logger.info(f"New comment from u/{username} in r/{subreddit}")

                except Exception as e:
                    logger.warning(f"Error processing comment {comment.id}: {e}")

        except Exception as e:
            logger.error(f"Error checking comments for u/{username}: {e}")

        return activities

    def _should_notify(self, item, user_config: Dict, item_type: str) -> bool:
        """Determine if we should notify about this activity."""
        notify_config = user_config.get('notify', {})

        # Check if notifications are enabled for this type
        if item_type == 'post' and not notify_config.get('new_post', True):
            return False

        if item_type == 'comment' and not notify_config.get('new_comment', True):
            return False

        # Check subreddit filters
        specific_subreddit = None
        if item_type == 'post':
            specific_subreddit = notify_config.get('post_in_specific_subreddit')
        else:
            specific_subreddit = notify_config.get('comment_in_specific_subreddit')

        if specific_subreddit:
            item_subreddit = item.subreddit.display_name
            if item_subreddit.lower() != specific_subreddit.lower():
                return False

        return True

    def check_hidden_posts(self, username: str) -> List[Dict]:
        """
        Check for posts even if they're hidden.
        This works by checking the user's profile directly.

        Args:
            username: Reddit username to check

        Returns:
            List of activities found
        """
        activities = []

        try:
            # Get posts from user profile
            user = self.reddit_client.get_user(username)
            posts = list(user.submissions.new(limit=10))

            for post in posts:
                if not self.database.has_tracked_user_activity(post.id):
                    activities.append({
                        'type': 'hidden_post',
                        'username': username,
                        'post': post,
                        'subreddit': post.subreddit.display_name,
                        'title': post.title,
                        'url': f"https://reddit.com{post.permalink}",
                        'is_hidden': True
                    })

                    # Record it
                    self.database.record_user_activity(
                        username=username,
                        activity_type='post',
                        item_id=post.id,
                        subreddit=post.subreddit.display_name,
                        title=post.title,
                        url=f"https://reddit.com{post.permalink}",
                        created_utc=int(post.created_utc)
                    )

            if activities:
                logger.info(f"Found {len(activities)} hidden posts from u/{username}")

        except Exception as e:
            logger.error(f"Error checking hidden posts for u/{username}: {e}")

        return activities
