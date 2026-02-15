"""Smart AI-powered reply system with approval queue."""

import time
import random
from typing import Dict, Optional
from loguru import logger


class SmartReply:
    """Handles AI-powered replies with approval and rate limiting."""

    def __init__(self, reddit_client, database, response_generator, config: Dict):
        self.reddit_client = reddit_client
        self.database = database
        self.response_generator = response_generator
        self.config = config
        self.pending_replies = []  # Queue for manual approval

    def process_post(self, post, subreddit_config: Dict) -> Optional[Dict]:
        """
        Process a post and potentially reply to it.

        Args:
            post: PRAW post object
            subreddit_config: Subreddit configuration

        Returns:
            Dict with result info or None
        """
        try:
            # Check rate limits
            if not self._check_rate_limits():
                logger.warning("Rate limit reached, skipping reply")
                return None

            # Generate response
            guidelines = subreddit_config.get('auto_reply', {}).get('guidelines', '')

            logger.info(f"Generating reply for post: {post.title[:50]}...")

            result = self.response_generator.generate_reply(
                post_or_comment=post,
                context_type='post',
                guidelines=guidelines,
                subreddit_config=subreddit_config
            )

            if not result:
                logger.warning("Failed to generate response")
                return None

            response_text = result['response']
            confidence = result['confidence']

            # Validate response
            if not self.response_generator.validate_response(response_text):
                logger.warning("Response failed validation")
                return None

            # Check approval mode
            approval_mode = self.config.get('ai', {}).get('approval_mode', 'manual')

            if approval_mode == 'manual':
                # Add to pending queue for manual approval
                self._add_to_approval_queue(post, response_text, confidence, 'post')
                logger.info("Reply added to approval queue")
                return {'status': 'pending_approval', 'confidence': confidence}

            elif approval_mode == 'threshold':
                # Auto-post if confidence is high enough
                if self.response_generator.should_reply_based_on_confidence(confidence):
                    return self._post_reply(post, response_text, confidence, 'post')
                else:
                    logger.info(f"Confidence too low ({confidence:.2f}), adding to queue")
                    self._add_to_approval_queue(post, response_text, confidence, 'post')
                    return {'status': 'pending_approval', 'confidence': confidence}

            elif approval_mode == 'auto':
                # Auto-post everything
                return self._post_reply(post, response_text, confidence, 'post')

        except Exception as e:
            logger.error(f"Error processing post for reply: {e}")
            return None

    def _post_reply(self, target, response_text: str, confidence: float, target_type: str) -> Dict:
        """Post a reply after delay."""
        try:
            # Apply natural delay
            delay = self._calculate_reply_delay()
            logger.info(f"Waiting {delay} seconds before posting reply...")
            time.sleep(delay)

            # Post the reply
            if target_type == 'post':
                reply_id = self.reddit_client.reply_to_post(target, response_text)
            else:
                reply_id = self.reddit_client.reply_to_comment(target, response_text)

            if reply_id:
                # Record the reply
                self.database.record_reply(
                    target_type=target_type,
                    target_id=target.id,
                    reply_id=reply_id,
                    subreddit=target.subreddit.display_name,
                    reply_text=response_text,
                    confidence_score=confidence
                )

                logger.info(f"Successfully posted reply with confidence {confidence:.2f}")

                return {
                    'status': 'posted',
                    'reply_id': reply_id,
                    'confidence': confidence,
                    'url': f"https://reddit.com{target.permalink}"
                }
            else:
                return {'status': 'failed', 'error': 'Failed to post reply'}

        except Exception as e:
            logger.error(f"Error posting reply: {e}")
            return {'status': 'error', 'error': str(e)}

    def _add_to_approval_queue(self, target, response_text: str, confidence: float, target_type: str):
        """Add reply to approval queue."""
        self.pending_replies.append({
            'target': target,
            'target_type': target_type,
            'response': response_text,
            'confidence': confidence,
            'subreddit': target.subreddit.display_name,
            'title': target.title if target_type == 'post' else 'Comment',
            'url': f"https://reddit.com{target.permalink}",
            'added_at': time.time()
        })

    def get_pending_replies(self) -> list:
        """Get list of replies pending approval."""
        return self.pending_replies.copy()

    def approve_reply(self, index: int) -> Dict:
        """
        Approve and post a pending reply.

        Args:
            index: Index in pending_replies list

        Returns:
            Dict with result
        """
        if index >= len(self.pending_replies):
            return {'status': 'error', 'error': 'Invalid index'}

        pending = self.pending_replies[index]

        result = self._post_reply(
            target=pending['target'],
            response_text=pending['response'],
            confidence=pending['confidence'],
            target_type=pending['target_type']
        )

        # Remove from queue
        self.pending_replies.pop(index)

        return result

    def reject_reply(self, index: int):
        """Reject and remove a pending reply."""
        if index < len(self.pending_replies):
            self.pending_replies.pop(index)
            logger.info(f"Rejected pending reply at index {index}")

    def edit_and_approve_reply(self, index: int, edited_text: str) -> Dict:
        """
        Edit a pending reply and post it.

        Args:
            index: Index in pending_replies list
            edited_text: Edited reply text

        Returns:
            Dict with result
        """
        if index >= len(self.pending_replies):
            return {'status': 'error', 'error': 'Invalid index'}

        pending = self.pending_replies[index]
        pending['response'] = edited_text

        result = self._post_reply(
            target=pending['target'],
            response_text=edited_text,
            confidence=pending['confidence'],
            target_type=pending['target_type']
        )

        # Remove from queue
        self.pending_replies.pop(index)

        return result

    def _check_rate_limits(self) -> bool:
        """Check if we've exceeded rate limits."""
        rate_limit_config = self.config.get('reply', {}).get('rate_limit', {})

        # Check hourly limit
        max_per_hour = rate_limit_config.get('max_replies_per_hour', 5)
        replies_last_hour = self.database.get_reply_count_last_hour()

        if replies_last_hour >= max_per_hour:
            logger.warning(f"Hourly rate limit reached: {replies_last_hour}/{max_per_hour}")
            return False

        # Check daily limit
        max_per_day = rate_limit_config.get('max_replies_per_day', 30)
        replies_today = self.database.get_reply_count_today()

        if replies_today >= max_per_day:
            logger.warning(f"Daily rate limit reached: {replies_today}/{max_per_day}")
            return False

        # Check cooldown
        cooldown = rate_limit_config.get('cooldown_between_replies', 300)
        # This would need database support for last reply time
        # For now, we'll skip this check

        return True

    def _calculate_reply_delay(self) -> int:
        """Calculate a natural-looking delay before posting."""
        timing_config = self.config.get('reply', {}).get('timing', {})

        min_delay = timing_config.get('min_delay_seconds', 120)
        max_delay = timing_config.get('max_delay_seconds', 1800)
        randomize = timing_config.get('randomize', True)

        if randomize:
            # Random delay within range
            return random.randint(min_delay, max_delay)
        else:
            # Use minimum delay
            return min_delay
