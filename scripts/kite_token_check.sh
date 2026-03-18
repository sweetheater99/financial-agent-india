#!/bin/bash
# Auto-refresh Kite access token with retry. Alert via Telegram on success or failure.
# Crontab: 35 8 * * 1-5
cd ~/financial-agent-india
source venv/bin/activate
set -a; source ~/.config/env/global.env; set +a

TOKEN=${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}
CHAT=${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}
TOPIC=${TELEGRAM_TOPIC_STOCKS}

send_tg() {
    local MSG="$1"
    TMPFILE=$(mktemp)
    echo "$MSG" > "$TMPFILE"
    CURL_ARGS="-d chat_id=$CHAT -d parse_mode=HTML --data-urlencode text@$TMPFILE"
    [ -n "$TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TOPIC"
    curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" $CURL_ARGS > /dev/null 2>&1
    rm -f "$TMPFILE"
}

MAX_RETRIES=3
RETRY_DELAY=30
SUCCESS=0

for ATTEMPT in $(seq 1 $MAX_RETRIES); do
    OUTPUT=$(python3 kite_auth.py 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS=1
        USER_LINE=$(echo "$OUTPUT" | grep 'Logged in as')
        if [ $ATTEMPT -gt 1 ]; then
            send_tg "<b>Kite token refreshed</b> (attempt $ATTEMPT/$MAX_RETRIES)
$USER_LINE"
        else
            send_tg "<b>Kite token refreshed</b>
$USER_LINE"
        fi
        echo "Kite token refreshed (attempt $ATTEMPT)"
        break
    fi

    echo "Kite auth attempt $ATTEMPT/$MAX_RETRIES failed: $OUTPUT"
    if [ $ATTEMPT -lt $MAX_RETRIES ]; then
        sleep $RETRY_DELAY
        RETRY_DELAY=$((RETRY_DELAY * 2))
    fi
done

if [ $SUCCESS -eq 0 ]; then
    send_tg "<b>CRITICAL: Kite token refresh FAILED after $MAX_RETRIES attempts</b>
Trading will run in protect-only mode (no new entries).
Run manually: ssh pi and cd financial-agent-india and python3 kite_auth.py"
    echo "Kite token refresh FAILED after $MAX_RETRIES attempts"
    exit 1
fi
