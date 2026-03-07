#!/bin/bash
# Paper trade cron wrapper — called every 30 minutes during market hours.
#
# V3: Added heartbeat + failure alerts via Telegram.

export PATH="$HOME/.local/bin:$PATH"

# Source env vars for Telegram notifications
if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

cd ~/financial-agent-india
source venv/bin/activate

LOG=data/paper_trades/cron.log
mkdir -p data/paper_trades

echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"

HOUR=$(TZ=Asia/Kolkata date +%H)
MIN=$(TZ=Asia/Kolkata date +%M)

# Override mode from argument (e.g., cron calls with "briefing")
if [ -n "$1" ]; then
    echo "[$1] Running with explicit mode" >> "$LOG"
    python paper_trade.py "$1" >> "$LOG" 2>&1
    EXIT_CODE=$?
    # Skip normal flow for explicit mode
    echo "" >> "$LOG"
    exit $EXIT_CODE
fi

# Determine mode
MODE=""
OPEN_EXIT=0
if [ "$HOUR" -eq 9 ] && [ "$MIN" -le 45 ]; then
    MODE="open"
    echo "[OPEN] Running screener + opening positions" >> "$LOG"
    python paper_trade.py open >> "$LOG" 2>&1
    OPEN_EXIT=$?
fi

echo "[MONITOR] Checking exit conditions" >> "$LOG"
python paper_trade.py monitor >> "$LOG" 2>&1
MON_EXIT=$?

# 3:15-3:30 PM IST: Final monitor pass
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 15 ]; then
    echo "[FINAL] End-of-day monitor pass" >> "$LOG"
    python paper_trade.py monitor >> "$LOG" 2>&1
fi

# V4: Weekly report (Sunday 10 AM IST)
DOW=$(TZ=Asia/Kolkata date +%u)  # 7=Sunday
if [ "$DOW" -eq 7 ] && [ "$HOUR" -eq 10 ] && [ "$MIN" -le 30 ]; then
    echo "[WEEKLY] Sending weekly performance report" >> "$LOG"
    python paper_trade.py weekly-report >> "$LOG" 2>&1
fi

# --- V3 Heartbeat / Failure Alert ---
EXIT_CODE=${OPEN_EXIT:-$MON_EXIT}
OPEN_COUNT=$(python -c "
import json; p=json.load(open('data/paper_trades/portfolio.json'))
print(sum(1 for pos in p.get('positions',[]) if pos.get('status')=='open'))
" 2>/dev/null || echo "?")
CAPITAL=$(python -c "
import json; p=json.load(open('data/paper_trades/portfolio.json'))
print(f\"₹{p.get('available_capital',0):,.0f}\")
" 2>/dev/null || echo "?")

if [ "${EXIT_CODE:-0}" -eq 0 ]; then
    MSG="[hb] ${MODE:-monitor} OK | Open: $OPEN_COUNT | Capital: $CAPITAL"
else
    LAST_LINES=$(tail -5 "$LOG" | head -c 500)
    MSG="[ALERT] paper_trade FAILED (exit $EXIT_CODE)
$LAST_LINES"
fi

# Send via Telegram (use temp file to avoid escaping issues)
TMPFILE=$(mktemp)
echo "$MSG" > "$TMPFILE"
curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" \
    -d parse_mode="HTML" \
    --data-urlencode "text@$TMPFILE" > /dev/null 2>&1
rm -f "$TMPFILE"

echo "" >> "$LOG"
