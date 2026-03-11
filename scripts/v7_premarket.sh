#!/bin/bash
# V7 Pre-Market Playbook — 8:43 AM IST Mon-Fri
# Crontab: 43 8 * * 1-5 ~/financial-agent-india/scripts/v7_premarket.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# Source env vars
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

# Log rotation: cap at 5MB
if [ -f "$LOG" ]; then
    LOG_SIZE=$(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG" 2>/dev/null || echo 0)
    if [ "$LOG_SIZE" -gt 5242880 ]; then
        mv "$LOG" "${LOG}.old"
        echo "--- Log rotated $(TZ=Asia/Kolkata date) ---" > "$LOG"
    fi
fi

TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

# Holiday check
IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [SKIP] Not a trading day ($TODAY_DATE)" >> "$LOG"
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
    echo "[$(TZ=Asia/Kolkata date)] [WARN] Kite token expired — premarket will use fallback" >> "$LOG"
    # Alert but continue — premarket can still generate playbook from global data
    TG_TOKEN="${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}"
    TG_CHAT="${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}"
    TG_TOPIC="${TELEGRAM_TOPIC_STOCKS}"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
        TMPFILE=$(mktemp)
        echo "V7: Kite token expired at premarket. Refresh token." > "$TMPFILE"
        CURL_ARGS="-d chat_id=${TG_CHAT} -d parse_mode=HTML --data-urlencode text@$TMPFILE"
        [ -n "$TG_TOPIC" ] && CURL_ARGS="$CURL_ARGS -d message_thread_id=$TG_TOPIC"
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            $CURL_ARGS > /dev/null 2>&1
        rm -f "$TMPFILE"
    fi
fi

echo "[$(TZ=Asia/Kolkata date)] [PREMARKET] Starting" >> "$LOG"
python -m v7.main premarket >> "$LOG" 2>&1
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[$(TZ=Asia/Kolkata date)] [PREMARKET] FAILED (exit $EXIT_CODE)" >> "$LOG"
    # Error alert already sent by main.py
fi

echo "" >> "$LOG"
