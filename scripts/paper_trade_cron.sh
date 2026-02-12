#!/bin/bash
# Paper trade cron wrapper — called by launchd every 30 minutes during market hours.
#
# Logic:
#   9:20-9:30 AM IST  → Run screener + open new positions
#   3:15+ PM IST      → Final monitor pass before close
#   Every invocation   → Monitor exits on open positions

cd ~/financial-agent-india
source venv/bin/activate

LOG=data/paper_trades/cron.log
mkdir -p data/paper_trades

echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"

HOUR=$(TZ=Asia/Kolkata date +%H)
MIN=$(TZ=Asia/Kolkata date +%M)

# 9:20-9:30 AM IST: Run screener and open new positions
if [ "$HOUR" -eq 9 ] && [ "$MIN" -ge 20 ] && [ "$MIN" -le 30 ]; then
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
