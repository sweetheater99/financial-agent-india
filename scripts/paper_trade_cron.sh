#!/bin/bash
# Paper trade cron wrapper — V6 Claude-First Architecture
#
# Runs every 5 min during market hours. Decides what to do each tick:
# - 9:10 AM: Morning briefing
# - 9:15-9:45 AM: Open mode (screener + new positions)
# - 12:40-12:50 PM: Afternoon re-entry pass
# - Every 5 min: Smart monitor (checks only positions that are due)
# - 3:35 PM: EOD wrap + Obsidian journal
# - Hourly (10-14): Digest
#
# Crontab: */5 9-15 * * 1-5

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# Source env vars
if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

cd ~/financial-agent-india
source venv/bin/activate

LOG=data/paper_trades/cron.log
mkdir -p data/paper_trades

# --- File lock: prevent overlapping ticks ---
LOCKFILE="/tmp/paper_trade.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "[SKIP] Previous tick still running at $(TZ=Asia/Kolkata date)" >> "$LOG"; exit 0; }

# --- Log rotation: cap at 5MB ---
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

# Override mode from argument (e.g., "briefing", "weekly-report")
if [ -n "$1" ]; then
    echo "--- $(TZ=Asia/Kolkata date) [$1] ---" >> "$LOG"
    python paper_trade.py "$1" >> "$LOG" 2>&1
    echo "" >> "$LOG"
    exit $?
fi

# --- Holiday check: skip NSE holidays ---
IS_HOLIDAY=$(python3 -c "
from config import NSE_HOLIDAYS
year = int('${TODAY_DATE}'[:4])
holidays = [str(d) for d in NSE_HOLIDAYS.get(year, [])]
print('yes' if '${TODAY_DATE}' in holidays else 'no')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"
    echo "[SKIP] NSE holiday ($TODAY_DATE)" >> "$LOG"
    echo "" >> "$LOG"
    exit 0
fi

echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"

# Pre-market hours: skip
if [ "$TIME_MINS" -lt 550 ]; then
    echo "[SKIP] Pre-market ($HOUR:$MIN)" >> "$LOG"
    echo "" >> "$LOG"
    exit 0
fi

# Post-market hours: skip
if [ "$TIME_MINS" -gt 940 ]; then
    echo "[SKIP] Post-market ($HOUR:$MIN)" >> "$LOG"
    echo "" >> "$LOG"
    exit 0
fi

MODE=""
EXIT_CODE=0
TICK_START=$(date +%s)

# --- 9:10 AM: Morning briefing ---
if [ "$HOUR" -eq 9 ] && [ "$MIN" -ge 8 ] && [ "$MIN" -le 12 ]; then
    echo "[BRIEFING] Pre-market morning briefing" >> "$LOG"
    python paper_trade.py briefing >> "$LOG" 2>&1
fi

# --- 9:15-9:45 AM: Open mode ---
if [ "$HOUR" -eq 9 ] && [ "$MIN" -ge 15 ] && [ "$MIN" -le 45 ]; then
    MODE="open"
    echo "[OPEN] Running screener + opening positions" >> "$LOG"
    python paper_trade.py open >> "$LOG" 2>&1
    EXIT_CODE=$?
fi

# --- 12:40-12:50 PM: Afternoon re-entry pass ---
if [ "$HOUR" -eq 12 ] && [ "$MIN" -ge 40 ] && [ "$MIN" -le 50 ]; then
    echo "[AFTERNOON] Re-entry screening pass" >> "$LOG"
    python paper_trade.py open >> "$LOG" 2>&1
fi

# --- Every run: Smart monitor ---
echo "[MONITOR] Smart tiered check" >> "$LOG"
python paper_trade.py monitor >> "$LOG" 2>&1
MON_EXIT=$?

# --- 3:15-3:25 PM: Overnight hedge scan ---
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 13 ] && [ "$MIN" -le 27 ]; then
    echo "[HEDGE] Overnight hedge scan" >> "$LOG"
    python overnight_hedge.py >> "$LOG" 2>&1
    HEDGE_EXIT=$?
    if [ "$HEDGE_EXIT" -ne 0 ]; then
        echo "[HEDGE] ERROR: overnight hedge scan failed (exit $HEDGE_EXIT)" >> "$LOG"
    fi
fi

# --- 3:35 PM: EOD wrap + Obsidian journal ---
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 33 ] && [ "$MIN" -le 40 ]; then
    echo "[EOD] End-of-day wrap" >> "$LOG"
    python -c "
from paper_trade import load_portfolio, _telegram_send
from smart_monitor import generate_eod_wrap, write_daily_journal
portfolio = load_portfolio()
msg = generate_eod_wrap(portfolio)
_telegram_send(msg)
print('EOD wrap sent')
# Write trading journal to Obsidian
try:
    path = write_daily_journal(portfolio, {})
    if path:
        print(f'Journal written: {path}')
except Exception as e:
    print(f'Journal write failed: {e}')
" >> "$LOG" 2>&1

    # Clean up old decision replay files (>30 days)
    find data/paper_trades/claude_decisions/ -name "*.json" -mtime +30 -delete 2>/dev/null
fi

# --- Sunday 10 AM: Weekly report ---
DOW=$(TZ=Asia/Kolkata date +%u)  # 7=Sunday
if [ "$DOW" -eq 7 ] && [ "$HOUR" -eq 10 ] && [ "$MIN" -le 5 ]; then
    echo "[WEEKLY] Sending weekly performance report" >> "$LOG"
    python paper_trade.py weekly-report >> "$LOG" 2>&1
fi

# --- Tick timing ---
TICK_END=$(date +%s)
TICK_DURATION=$((TICK_END - TICK_START))
echo "[TIMING] Tick completed in ${TICK_DURATION}s" >> "$LOG"

# --- Heartbeat (only every 30 min to avoid Telegram spam) ---
if [ "$((MIN % 30))" -lt 5 ]; then
    FINAL_EXIT=${EXIT_CODE:-$MON_EXIT}
    OPEN_COUNT=$(python -c "
import json; p=json.load(open('data/paper_trades/portfolio.json'))
print(sum(1 for pos in p.get('positions',[]) if pos.get('status')=='open'))
" 2>/dev/null || echo "?")
    CAPITAL=$(python -c "
import json; p=json.load(open('data/paper_trades/portfolio.json'))
print(f\"₹{p.get('available_capital',0):,.0f}\")
" 2>/dev/null || echo "?")

    if [ "${FINAL_EXIT:-0}" -eq 0 ]; then
        MSG="[hb] ${MODE:-monitor} OK | Open: $OPEN_COUNT | Capital: $CAPITAL | ${TICK_DURATION}s"
    else
        LAST_LINES=$(tail -5 "$LOG" | head -c 500)
        MSG="[ALERT] paper_trade FAILED (exit $FINAL_EXIT)
$LAST_LINES"
    fi

    # Alert if tick took too long
    if [ "$TICK_DURATION" -gt 240 ]; then
        MSG="⚠️ Slow tick: ${TICK_DURATION}s (limit 300s)
$MSG"
    fi

    TG_TOKEN="${DEAL_BOT_TOKEN:-$TELEGRAM_BOT_TOKEN}"
    TG_CHAT="${TELEGRAM_FORUM_CHAT_ID:-$DEAL_BOT_CHAT_ID}"
    TG_TOPIC="${TELEGRAM_TOPIC_STOCKS}"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
        TMPFILE=$(mktemp)
        echo "$MSG" > "$TMPFILE"
        CURL_DATA="-d chat_id=${TG_CHAT} -d parse_mode=HTML --data-urlencode text@$TMPFILE"
        [ -n "$TG_TOPIC" ] && CURL_DATA="$CURL_DATA -d message_thread_id=$TG_TOPIC"
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            $CURL_DATA > /dev/null 2>&1
        rm -f "$TMPFILE"
    fi
fi

echo "" >> "$LOG"
