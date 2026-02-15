"""Database management for tracking posts, comments, and bot actions."""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
from loguru import logger


class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Track seen posts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_posts (
                    post_id TEXT PRIMARY KEY,
                    subreddit TEXT,
                    author TEXT,
                    title TEXT,
                    url TEXT,
                    created_utc INTEGER,
                    seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Track seen comments
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_comments (
                    comment_id TEXT PRIMARY KEY,
                    post_id TEXT,
                    subreddit TEXT,
                    author TEXT,
                    body TEXT,
                    created_utc INTEGER,
                    seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Track our replies
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS my_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_type TEXT,  -- 'post' or 'comment'
                    target_id TEXT,
                    reply_id TEXT,
                    subreddit TEXT,
                    reply_text TEXT,
                    confidence_score REAL,
                    replied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(target_id)
                )
            """)

            # Track user activity
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    activity_type TEXT,  -- 'post' or 'comment'
                    item_id TEXT UNIQUE,
                    subreddit TEXT,
                    title TEXT,
                    url TEXT,
                    created_utc INTEGER,
                    tracked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Track writing style samples (for learning user's style)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS writing_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_id TEXT UNIQUE,
                    subreddit TEXT,
                    text TEXT,
                    length INTEGER,
                    created_utc INTEGER,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_seen_posts_subreddit
                ON seen_posts(subreddit)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_activity_username
                ON user_activity(username)
            """)

            conn.commit()
            logger.info("Database initialized successfully")

    def has_seen_post(self, post_id: str) -> bool:
        """Check if we've seen this post before."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM seen_posts WHERE post_id = ?", (post_id,))
            return cursor.fetchone() is not None

    def mark_post_seen(self, post_id: str, subreddit: str, author: str,
                       title: str, url: str, created_utc: int):
        """Mark a post as seen."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO seen_posts
                (post_id, subreddit, author, title, url, created_utc)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (post_id, subreddit, author, title, url, created_utc))
            conn.commit()

    def has_seen_comment(self, comment_id: str) -> bool:
        """Check if we've seen this comment before."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM seen_comments WHERE comment_id = ?", (comment_id,))
            return cursor.fetchone() is not None

    def mark_comment_seen(self, comment_id: str, post_id: str, subreddit: str,
                          author: str, body: str, created_utc: int):
        """Mark a comment as seen."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO seen_comments
                (comment_id, post_id, subreddit, author, body, created_utc)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (comment_id, post_id, subreddit, author, body, created_utc))
            conn.commit()

    def has_replied_to(self, target_id: str) -> bool:
        """Check if we've already replied to this post/comment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM my_replies WHERE target_id = ?", (target_id,))
            return cursor.fetchone() is not None

    def record_reply(self, target_type: str, target_id: str, reply_id: str,
                     subreddit: str, reply_text: str, confidence_score: float = 0.0):
        """Record that we replied to a post/comment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO my_replies
                (target_type, target_id, reply_id, subreddit, reply_text, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (target_type, target_id, reply_id, subreddit, reply_text, confidence_score))
            conn.commit()

    def get_reply_count_last_hour(self) -> int:
        """Get number of replies in the last hour."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM my_replies
                WHERE replied_at > datetime('now', '-1 hour')
            """)
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_reply_count_today(self) -> int:
        """Get number of replies today."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM my_replies
                WHERE DATE(replied_at) = DATE('now')
            """)
            result = cursor.fetchone()
            return result[0] if result else 0

    def record_user_activity(self, username: str, activity_type: str, item_id: str,
                            subreddit: str, title: str, url: str, created_utc: int):
        """Record tracked user activity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO user_activity
                (username, activity_type, item_id, subreddit, title, url, created_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, activity_type, item_id, subreddit, title, url, created_utc))
            conn.commit()

    def has_tracked_user_activity(self, item_id: str) -> bool:
        """Check if we've already tracked this user activity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM user_activity WHERE item_id = ?", (item_id,))
            return cursor.fetchone() is not None

    def add_writing_sample(self, comment_id: str, subreddit: str,
                          text: str, created_utc: int):
        """Add a writing sample for style learning."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO writing_samples
                (comment_id, subreddit, text, length, created_utc)
                VALUES (?, ?, ?, ?, ?)
            """, (comment_id, subreddit, text, len(text), created_utc))
            conn.commit()

    def get_writing_samples(self, limit: int = 50) -> List[Dict]:
        """Get writing samples for style analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT text, subreddit, length
                FROM writing_samples
                ORDER BY created_utc DESC
                LIMIT ?
            """, (limit,))
            results = cursor.fetchall()
            return [
                {"text": row[0], "subreddit": row[1], "length": row[2]}
                for row in results
            ]

    def get_stats(self) -> Dict:
        """Get bot statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM seen_posts")
            posts_seen = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM seen_comments")
            comments_seen = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM my_replies")
            total_replies = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM my_replies
                WHERE DATE(replied_at) = DATE('now')
            """)
            replies_today = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_activity")
            user_activities = cursor.fetchone()[0]

            return {
                "posts_seen": posts_seen,
                "comments_seen": comments_seen,
                "total_replies": total_replies,
                "replies_today": replies_today,
                "user_activities_tracked": user_activities
            }
