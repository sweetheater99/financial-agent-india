#!/bin/bash
# watchdog.sh — alert if paper_trade hasn't run in 2+ hours
# Add to cron: 0 */2 * * 1-5 ~/financial-agent-india/scripts/watchdog.sh

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

PORTFOLIO="$HOME/financial-agent-india/data/paper_trades/portfolio.json"

if [ ! -f "$PORTFOLIO" ]; then
    exit 0  # no portfolio yet
fi

# Check last modification time (GNU stat on Linux)
LAST_MOD=$(stat -c %Y "$PORTFOLIO" 2>/dev/null || stat -f %m "$PORTFOLIO" 2>/dev/null)
NOW=$(date +%s)
DIFF=$(( (NOW - LAST_MOD) / 3600 ))

if [ "$DIFF" -ge 2 ]; then
    TMPFILE=$(mktemp)
    echo "[WATCHDOG] paper_trade hasn't run in ${DIFF}h. Check Pi cron." > "$TMPFILE"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d parse_mode="HTML" \
        --data-urlencode "text@$TMPFILE" > /dev/null 2>&1
    rm -f "$TMPFILE"
fi
