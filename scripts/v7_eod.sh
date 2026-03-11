#!/bin/bash
# V7 EOD Review + Journal — 3:33 PM IST Mon-Fri
# Crontab: 33 15 * * 1-5 ~/financial-agent-india/scripts/v7_eod.sh

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

TODAY_DATE=$(TZ=Asia/Kolkata date +%Y-%m-%d)

IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    exit 0
fi

echo "[$(TZ=Asia/Kolkata date)] [EOD] Starting EOD review + journal" >> "$LOG"
python -m v7.main --paper eod >> "$LOG" 2>&1
echo "" >> "$LOG"
