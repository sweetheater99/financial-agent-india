#!/bin/bash
# Paper trade cron wrapper — called by launchd every 30 minutes during market hours.
#
# Logic:
#   9:00-9:45 AM IST  → Run screener + open new positions
#   3:15+ PM IST      → Final monitor pass before close
#   Every invocation   → Monitor exits on open positions

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

# 9:00-9:45 AM IST: Run screener and open new positions
if [ "$HOUR" -eq 9 ] && [ "$MIN" -le 45 ]; then
    echo "[OPEN] Running screener + opening positions" >> "$LOG"
    python paper_trade.py open >> "$LOG" 2>&1
fi

# 3:15-3:30 PM IST: Final monitor pass before market close
if [ "$HOUR" -eq 15 ] && [ "$MIN" -ge 15 ]; then
    echo "[FINAL] End-of-day monitor pass" >> "$LOG"
    python paper_trade.py monitor >> "$LOG" 2>&1
fi

# Every run: monitor exits
echo "[MONITOR] Checking exit conditions" >> "$LOG"
python paper_trade.py monitor >> "$LOG" 2>&1

echo "" >> "$LOG"
