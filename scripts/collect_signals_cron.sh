#!/bin/bash
# Signal collector cron — runs at 9:45 AM IST daily to capture screener signals.

export PATH="$HOME/.local/bin:$PATH"
source ~/scripts/notify.sh

if [ -f "$HOME/.config/env/global.env" ]; then
    set -a
    source "$HOME/.config/env/global.env"
    set +a
fi

cd ~/financial-agent-india
source venv/bin/activate

LOG=data/signal_collections/collect.log
mkdir -p data/signal_collections

echo "--- $(TZ=Asia/Kolkata date) ---" >> "$LOG"

if python collect_signals.py >> "$LOG" 2>&1; then
    SIGNALS=$(tail -5 "$LOG" | grep -oP \(\d+) signals' 2>/dev/null | head -1 || echo "done")
    notify "Signal Collector" "$SIGNALS"
else
    notify_err "Signal Collector" "Collection failed — check collect.log"
fi

echo "" >> "$LOG"
