# Reddit Automation Bot 🤖

An intelligent Reddit automation bot that monitors subreddits and users, generates natural AI-powered replies, and creates posts. Designed to appear completely human with sophisticated anti-detection features.

## ✨ Features

### 🎯 Core Capabilities

- **Smart Auto-Replies**: AI-powered natural responses that mimic your writing style
- **Subreddit Monitoring**: Track new posts in specific subreddits with keyword filtering
- **User Tracking**: Monitor specific users for new posts and comments (even hidden ones)
- **Post Creation**: Create text and link posts with context-aware content
- **Multi-Channel Notifications**: Email, Telegram, Discord, and console alerts

### 🧠 AI-Powered Natural Replies

- **Writing Style Learning**: Analyzes your Reddit history to match your tone and style
- **Context-Aware**: Understands post/comment context for relevant replies
- **Confidence Scoring**: Only posts high-quality responses
- **Manual/Auto Approval**: Review replies before posting or trust the AI
- **Anti-Detection**: Natural timing, randomization, and human-like behavior

### 🔒 Safety Features

- **Rate Limiting**: Configurable hourly and daily limits
- **Approval Queue**: Review and edit replies before posting
- **Smart Filtering**: Avoid replying to old posts or over-commented threads
- **Confidence Thresholds**: Only post when AI is confident

## 📋 Prerequisites

1. **Python 3.9+**
2. **Reddit Account** with API credentials
3. **AI API Key** (Claude or OpenAI recommended)

## 🚀 Quick Start

### 1. Create Reddit App

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" at the bottom
3. Select "script"
4. Fill in the details:
   - **name**: Your bot name
   - **redirect uri**: http://localhost:8080
5. Note your **client_id** (under app name) and **client_secret**

### 2. Install Dependencies

```bash
cd reddit-bot
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

**Required Configuration:**
```env
# Reddit API
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password

# AI Provider (choose one)
CLAUDE_API_KEY=your_claude_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Configure Bot Behavior

Edit the configuration files in the `config/` directory:

**`config/config.yaml`** - Main settings (AI provider, approval mode, rate limits)
**`config/subreddits.yaml`** - Subreddits to monitor and auto-reply rules
**`config/users.yaml`** - Users to track

### 5. Learn Your Writing Style (Optional but Recommended)

```bash
python main.py learn-style
```

This analyzes your recent Reddit comments to make replies sound like you.

### 6. Start the Bot

**Test authentication:**
```bash
python main.py test-auth
```

**Run once (for testing):**
```bash
python main.py start --once
```

**Run continuously:**
```bash
python main.py start
```

**Learn style and run:**
```bash
python main.py start --learn-style
```

## 📖 Usage Guide

### CLI Commands

```bash
# Start the bot
python main.py start                    # Run continuously
python main.py start --once             # Run one iteration
python main.py start --learn-style      # Learn style first, then run

# Review pending replies
python main.py review                   # Interactive approval interface

# Create posts
python main.py post "subreddit" "Title" --text "Content"
python main.py post "subreddit" "Title" --url "https://example.com"

# View statistics
python main.py stats

# Learn writing style
python main.py learn-style

# Test authentication
python main.py test-auth
```

### Configuration Guide

#### Main Config (`config/config.yaml`)

```yaml
ai:
  provider: "claude"  # or "openai"
  model: "claude-sonnet-4.5"
  approval_mode: "manual"  # manual, auto, or threshold
  confidence_threshold: 0.8  # for threshold mode

reply:
  timing:
    min_delay_seconds: 120     # Wait at least 2 minutes
    max_delay_seconds: 1800    # Reply within 30 minutes

  rate_limit:
    max_replies_per_hour: 5
    max_replies_per_day: 30

  anti_detection:
    reply_probability: 0.7     # Don't reply to everything
    skip_if_too_many_replies: 10
    skip_if_post_age_hours: 24
```

#### Subreddit Config (`config/subreddits.yaml`)

```yaml
subreddits:
  - name: "learnpython"
    enabled: true

    auto_reply:
      enabled: true
      keywords:
        - "help"
        - "question"
        - "how to"

      guidelines: |
        - Be helpful and encouraging
        - Provide code examples
        - Explain concepts simply

      filters:
        min_upvotes: 0
        avoid_keywords: ["solved", "thanks"]

    reply_probability: 0.6  # Reply to 60% of matches
```

#### User Tracking (`config/users.yaml`)

```yaml
tracked_users:
  - username: "example_user"
    enabled: true

    track:
      posts: true
      comments: true
      hidden_posts: true  # Check even if posts are hidden

    check_interval: 60  # Check every minute

    notify:
      new_post: true
      new_comment: true
```

## 🎨 Approval Modes

### Manual Mode (Recommended)
- Review every reply before posting
- Edit responses if needed
- Maximum control and safety

```bash
python main.py review
```

### Threshold Mode (Balanced)
- Auto-post high-confidence replies
- Review low-confidence ones
- Good balance of automation and control

### Auto Mode (Full Automation)
- Post all generated replies automatically
- Use with caution
- Best after testing with manual mode

## 🔔 Notifications

Configure multiple notification channels in `.env`:

```env
# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=recipient@example.com

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Discord
DISCORD_WEBHOOK_URL=your_webhook_url
```

Enable channels in `config/config.yaml`:

```yaml
notifications:
  enabled: true
  channels:
    console: true
    email: true
    telegram: true
    discord: false
```

## 🛡️ Anti-Detection Best Practices

1. **Use Manual Approval Initially**: Test and tune your settings
2. **Set Realistic Rate Limits**: Don't reply too frequently
3. **Use Reply Probability < 1.0**: Don't reply to every matching post
4. **Learn Your Writing Style**: Makes replies more authentic
5. **Add Natural Delays**: Use random timing (120-1800 seconds)
6. **Avoid Old Posts**: Set `skip_if_post_age_hours`
7. **Skip Popular Posts**: Set `skip_if_too_many_replies`

## 📊 Statistics & Monitoring

View bot statistics:

```bash
python main.py stats
```

Check logs:

```bash
tail -f logs/bot.log
```

Database location: `data/bot.db` (SQLite)

## 🐛 Troubleshooting

### Authentication Failed
- Verify credentials in `.env`
- Ensure Reddit app is created as "script" type
- Check username/password are correct

### No Replies Generated
- Check `config/subreddits.yaml` has enabled subreddits
- Verify keywords match post content
- Check rate limits haven't been reached
- Review logs: `tail -f logs/bot.log`

### AI Provider Error
- Verify API key in `.env`
- Check API quota/billing
- Ensure `praw` and `anthropic`/`openai` packages are installed

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

## 📁 Project Structure

```
reddit-bot/
├── config/              # Configuration files
│   ├── config.yaml      # Main settings
│   ├── subreddits.yaml  # Subreddit monitoring
│   └── users.yaml       # User tracking
├── src/
│   ├── ai/              # AI providers and response generation
│   ├── monitors/        # Subreddit and user monitors
│   ├── actions/         # Reply and post actions
│   ├── notifications/   # Notification system
│   ├── storage/         # Database
│   └── bot.py           # Main orchestrator
├── data/                # Database files
├── logs/                # Log files
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## ⚠️ Important Notes

1. **Reddit Terms of Service**: Ensure your bot usage complies with [Reddit's API Terms](https://www.reddit.com/wiki/api-terms)
2. **Subreddit Rules**: Check each subreddit's rules about bots before auto-replying
3. **Rate Limiting**: Reddit has strict rate limits - use reasonable intervals
4. **Transparency**: Some subreddits require bot identification
5. **Ethical Use**: Use responsibly and don't spam

## 🔐 Security

- Never commit `.env` file
- Keep API keys secure
- Don't share your Reddit password
- Use environment variables for credentials

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

MIT License - Use at your own risk

## 🆘 Support

For issues or questions:
1. Check the logs: `logs/bot.log`
2. Review configuration files
3. Test authentication: `python main.py test-auth`
4. Verify API quotas and credentials

---

**Happy Redditing! 🎉**

*Remember: Use responsibly and respect community guidelines.*
