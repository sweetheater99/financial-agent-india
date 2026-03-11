#!/bin/bash
# V7 Executor Tick — every 3 min during market hours
# Crontab: */3 9-15 * * 1-5 ~/financial-agent-india/scripts/v7_executor.sh

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

BOT_DIR="$HOME/financial-agent-india"
cd "$BOT_DIR"
source venv/bin/activate

LOG="data/v7/cron.log"
mkdir -p data/v7

# File lock: prevent overlapping ticks
LOCKFILE="/tmp/v7_executor.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "[$(TZ=Asia/Kolkata date)] [SKIP] Previous tick running" >> "$LOG"; exit 0; }

# Log rotation
if [ -f "$LOG" ]; then
    LOG_SIZE=$(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG" 2>/dev/null || echo 0)
    if [ "$LOG_SIZE" -gt 5242880 ]; then
        mv "$LOG" "${LOG}.old"
        echo "--- Log rotated $(TZ=Asia/Kolkata date) ---" > "$LOG"
    fi
fi

HOUR=$(TZ=Asia/Kolkata date +%H)
MIN=$(TZ=Asia/Kolkata date +%M)
TIME_MINS=$((HOUR * 60 + MIN))
TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

# Pre-market: skip
if [ "$TIME_MINS" -lt 555 ]; then
    exit 0
fi

# Post-market: skip
if [ "$TIME_MINS" -gt 930 ]; then
    exit 0
fi

# Holiday check
IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    exit 0
fi

# Kite token check
KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [TICK] Kite expired — protect-only mode" >> "$LOG"
    # TODO: Run protect-only tick (check SL orders, no new trades)
    exit 0
fi

TICK_START=$(date +%s)
echo "[$(TZ=Asia/Kolkata date)] [TICK] Starting" >> "$LOG"

python -m v7.main --paper tick >> "$LOG" 2>&1
EXIT_CODE=$?

TICK_END=$(date +%s)
TICK_DURATION=$((TICK_END - TICK_START))
echo "[TIMING] Tick: ${TICK_DURATION}s (exit $EXIT_CODE)" >> "$LOG"

# Alert if tick took too long (> 120s)
if [ "$TICK_DURATION" -gt 120 ]; then
    TG_TOKEN="${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}"
    TG_CHAT="${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}"
    TG_TOPIC="${TELEGRAM_TOPIC_STOCKS}"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
        TMPFILE=$(mktemp)
        echo "V7: Slow tick ${TICK_DURATION}s (limit 120s)" > "$TMPFILE"
        CURL_ARGS="-d chat_id=${TG_CHAT} -d parse_mode=HTML --data-urlencode text@$TMPFILE"
        [ -n "$TG_TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TG_TOPIC"
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            $CURL_ARGS > /dev/null 2>&1
        rm -f "$TMPFILE"
    fi
fi

echo "" >> "$LOG"
