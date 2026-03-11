#!/bin/bash
# V7 Monthly Report — 1st of month 10:07 AM IST
# Crontab: 7 10 1 * * ~/financial-agent-india/scripts/v7_monthly_report.sh

set -euo pipefail
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

echo "[$(TZ=Asia/Kolkata date)] [MONTHLY] Starting monthly report" >> "$LOG"
python -m v7.main monthly >> "$LOG" 2>&1
echo "" >> "$LOG"
