#!/bin/bash
# Fast options monitor — runs every 5 min during market hours
# Only checks option positions, much lighter than full monitor
#
# Crontab: */5 9-15 * * 1-5 /bin/bash ~/financial-agent-india/scripts/options_monitor_cron.sh

export PATH="$HOME/.local/bin:$PATH"

# Source env vars for Telegram notifications
if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

cd ~/financial-agent-india
source venv/bin/activate

LOG=data/paper_trades/options_monitor.log
mkdir -p data/paper_trades

HOUR=$(TZ=Asia/Kolkata date +%H)
MIN=$(TZ=Asia/Kolkata date +%M)

# Only run during market hours (9:15 AM to 3:30 PM IST)
TIME_MINS=$((HOUR * 60 + MIN))
if [ "$TIME_MINS" -lt 555 ] || [ "$TIME_MINS" -gt 930 ]; then
    exit 0
fi

echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"
python options_monitor.py >> "$LOG" 2>&1
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    LAST_LINES=$(tail -5 "$LOG" | head -c 500)
    TMPFILE=$(mktemp)
    echo "[ALERT] options_monitor FAILED (exit $EXIT_CODE)
$LAST_LINES" > "$TMPFILE"
    curl -s -X POST "https://api.telegram.org/bot${DEAL_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_FORUM_CHAT_ID}" \
        -d parse_mode="HTML" \
        --data-urlencode "text@$TMPFILE" > /dev/null 2>&1
    rm -f "$TMPFILE"
fi

echo "" >> "$LOG"
