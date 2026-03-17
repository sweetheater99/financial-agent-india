#!/bin/bash
# Auto-refresh Kite access token. Alert via Telegram on success or failure.
# Crontab: 35 8 * * 1-5
cd ~/financial-agent-india
source venv/bin/activate
set -a; source ~/.config/env/global.env; set +a

# Try auto-login
OUTPUT=$(python3 kite_auth.py 2>&1)
EXIT_CODE=$?

TOKEN=${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}
CHAT=${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}
TOPIC=${TELEGRAM_TOPIC_STOCKS}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Kite token refreshed successfully"
    # Extract username from output
    USER_LINE=$(echo "$OUTPUT" | grep 'Logged in as')
    TMPFILE=$(mktemp)
    echo "<b>Kite token refreshed</b>
$USER_LINE" > "$TMPFILE"
    CURL_ARGS="-d chat_id=$CHAT -d parse_mode=HTML --data-urlencode text@$TMPFILE"
    [ -n "$TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TOPIC"
    curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" $CURL_ARGS > /dev/null 2>&1
    rm -f "$TMPFILE"
else
    echo "Kite token refresh FAILED: $OUTPUT"
    TMPFILE=$(mktemp)
    echo "<b>Kite token refresh FAILED</b>
Run manually: ssh pi && cd financial-agent-india && python3 kite_auth.py" > "$TMPFILE"
    CURL_ARGS="-d chat_id=$CHAT -d parse_mode=HTML --data-urlencode text@$TMPFILE"
    [ -n "$TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TOPIC"
    curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" $CURL_ARGS > /dev/null 2>&1
    rm -f "$TMPFILE"
fi
