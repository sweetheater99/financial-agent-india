#!/bin/bash
# V7 Weekly Review — Sunday 10:03 AM IST
# Crontab: 3 10 * * 0 ~/financial-agent-india/scripts/v7_weekly_review.sh

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

echo "[$(TZ=Asia/Kolkata date)] [WEEKLY] Starting weekly review" >> "$LOG"
python -m v7.main weekly >> "$LOG" 2>&1
echo "" >> "$LOG"
