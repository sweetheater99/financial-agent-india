#!/bin/bash
# V7 Strategist Check-in — 10:28 AM and 12:58 PM IST Mon-Fri
# Crontab:
#   28 10 * * 1-5 ~/financial-agent-india/scripts/v7_checkin.sh
#   58 12 * * 1-5 ~/financial-agent-india/scripts/v7_checkin.sh

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
HOUR=$(TZ=Asia/Kolkata date +%H)

IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    exit 0
fi

# Determine check-in number from time
if [ "$HOUR" -le 11 ]; then
    CHECKIN_NUM=1
else
    CHECKIN_NUM=2
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
    echo "[$(TZ=Asia/Kolkata date)] [CHECKIN] Kite expired — skipping check-in $CHECKIN_NUM" >> "$LOG"
    exit 0
fi

echo "[$(TZ=Asia/Kolkata date)] [CHECKIN] Check-in #$CHECKIN_NUM starting" >> "$LOG"
python -m v7.main checkin --num "$CHECKIN_NUM" >> "$LOG" 2>&1
echo "" >> "$LOG"
