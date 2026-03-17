#!/bin/bash
# V7 Pre-Open Session — 9:10 AM IST Mon-Fri
# Crontab: 10 9 * * 1-5 ~/financial-agent-india/scripts/v7_preopen.sh

set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

# Source env vars
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

# Holiday check
IS_HOLIDAY=$(python3 -c "
from config import is_trading_day
from datetime import date
print('no' if is_trading_day(date.fromisoformat('${TODAY_DATE}')) else 'yes')
" 2>/dev/null || echo "no")

if [ "$IS_HOLIDAY" = "yes" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [SKIP] Not a trading day ($TODAY_DATE)" >> "$LOG"
    exit 0
fi

echo "[$(TZ=Asia/Kolkata date)] [PREOPEN] Starting" >> "$LOG"
python3 -m v7.main preopen >> "$LOG" 2>&1
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[$(TZ=Asia/Kolkata date)] [PREOPEN] FAILED (exit $EXIT_CODE)" >> "$LOG"
fi

echo "" >> "$LOG"
