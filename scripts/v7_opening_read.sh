#!/bin/bash
# V7 Opening Read — 9:13 AM IST Mon-Fri
# Crontab: 13 9 * * 1-5 ~/financial-agent-india/scripts/v7_opening_read.sh

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
    echo "[$(TZ=Asia/Kolkata date)] [SKIP] Not a trading day" >> "$LOG"
    exit 0
fi

KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [OPENING-READ] Kite expired — skipping" >> "$LOG"
    exit 1
fi

echo "[$(TZ=Asia/Kolkata date)] [OPENING-READ] Starting" >> "$LOG"
python -m v7.main opening-read >> "$LOG" 2>&1
echo "" >> "$LOG"
