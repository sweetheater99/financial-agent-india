"""Main notification dispatcher."""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
from loguru import logger
import requests


class Notifier:
    """Handles notifications through multiple channels."""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('notifications', {}).get('enabled', True)
        self.channels = config.get('notifications', {}).get('channels', {})
        self.notify_on = config.get('notifications', {}).get('notify_on', {})

    def notify(self, event_type: str, data: Dict):
        """
        Send notification for an event.

        Args:
            event_type: Type of event (tracked_user_activity, new_subreddit_posts, etc.)
            data: Event data
        """
        if not self.enabled:
            return

        # Check if we should notify for this event type
        if not self.notify_on.get(event_type, True):
            return

        message = self._format_message(event_type, data)

        # Send through enabled channels
        if self.channels.get('console', True):
            self._notify_console(message)

        if self.channels.get('email', False):
            self._notify_email(message, data)

        if self.channels.get('telegram', False):
            self._notify_telegram(message)

        if self.channels.get('discord', False):
            self._notify_discord(message)

    def _format_message(self, event_type: str, data: Dict) -> str:
        """Format notification message based on event type."""
        if event_type == 'tracked_user_activity':
            return self._format_user_activity(data)

        elif event_type == 'new_subreddit_posts':
            return self._format_new_post(data)

        elif event_type == 'auto_reply_posted':
            return self._format_reply_posted(data)

        elif event_type == 'errors':
            return f"⚠️ Error: {data.get('error', 'Unknown error')}"

        else:
            return f"Event: {event_type}\n{data}"

    def _format_user_activity(self, data: Dict) -> str:
        """Format user activity notification."""
        activity_type = data.get('type', 'activity')
        username = data.get('username', 'unknown')

        if activity_type == 'user_post':
            return (
                f"🔔 New Post from u/{username}\n"
                f"Subreddit: r/{data.get('subreddit')}\n"
                f"Title: {data.get('title')}\n"
                f"URL: {data.get('url')}\n"
                f"Hidden: {data.get('is_hidden', False)}"
            )

        elif activity_type == 'user_comment':
            return (
                f"💬 New Comment from u/{username}\n"
                f"Subreddit: r/{data.get('subreddit')}\n"
                f"Post: {data.get('post_title', 'N/A')}\n"
                f"Comment: {data.get('comment_body', '')[:100]}...\n"
                f"URL: {data.get('url')}"
            )

        return f"User activity: {data}"

    def _format_new_post(self, data: Dict) -> str:
        """Format new subreddit post notification."""
        return (
            f"📝 New Post in r/{data.get('subreddit')}\n"
            f"Title: {data.get('title')}\n"
            f"Author: u/{data.get('author')}\n"
            f"URL: {data.get('url')}"
        )

    def _format_reply_posted(self, data: Dict) -> str:
        """Format reply posted notification."""
        return (
            f"✅ Reply Posted\n"
            f"Subreddit: r/{data.get('subreddit', 'unknown')}\n"
            f"Confidence: {data.get('confidence', 0):.2f}\n"
            f"URL: {data.get('url', 'N/A')}"
        )

    def _notify_console(self, message: str):
        """Print notification to console."""
        logger.info(f"\n{'='*50}\n{message}\n{'='*50}")

    def _notify_email(self, message: str, data: Dict):
        """Send email notification."""
        try:
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_username = os.getenv('SMTP_USERNAME')
            smtp_password = os.getenv('SMTP_PASSWORD')
            notification_email = os.getenv('NOTIFICATION_EMAIL')

            if not all([smtp_username, smtp_password, notification_email]):
                logger.warning("Email credentials not configured")
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_username
            msg['To'] = notification_email
            msg['Subject'] = 'Reddit Bot Notification'

            msg.attach(MIMEText(message, 'plain'))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)

            logger.info("Email notification sent")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def _notify_telegram(self, message: str):
        """Send Telegram notification."""
        try:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')

            if not all([bot_token, chat_id]):
                logger.warning("Telegram credentials not configured")
                return

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }

            response = requests.post(url, data=data, timeout=10)

            if response.status_code == 200:
                logger.info("Telegram notification sent")
            else:
                logger.error(f"Telegram notification failed: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def _notify_discord(self, message: str):
        """Send Discord webhook notification."""
        try:
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

            if not webhook_url:
                logger.warning("Discord webhook URL not configured")
                return

            data = {'content': message}
            response = requests.post(webhook_url, json=data, timeout=10)

            if response.status_code == 204:
                logger.info("Discord notification sent")
            else:
                logger.error(f"Discord notification failed: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    def notify_batch(self, event_type: str, items: List[Dict]):
        """
        Send batch notification for multiple items.

        Args:
            event_type: Type of event
            items: List of items to notify about
        """
        if not items:
            return

        # For now, send individual notifications
        # Could be enhanced to batch multiple items in one message
        for item in items:
            self.notify(event_type, item)
