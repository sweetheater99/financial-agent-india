#!/bin/bash
# V7 WebSocket Monitor — starts at 9:14 AM, runs until 3:30 PM
# Crontab: 14 9 * * 1-5 ~/financial-agent-india/scripts/v7_ws_monitor.sh

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

LOG="data/v7/ws_monitor.log"
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

# Kill any existing ws_monitor process
pkill -f "v7.ws_monitor" 2>/dev/null || true

# Kite token check
KITE_OK=$(python3 -c "
from kite_data import get_kite
try:
    get_kite()
    print('yes')
except:
    print('no')
" 2>/dev/null || echo "no")

if [ "$KITE_OK" = "no" ]; then
    echo "[$(TZ=Asia/Kolkata date)] [WS] Kite token expired — cannot start WS monitor" >> "$LOG"
    exit 1
fi

echo "[$(TZ=Asia/Kolkata date)] [WS] Starting WebSocket monitor" >> "$LOG"
python3 -m v7.ws_monitor >> "$LOG" 2>&1 &
WS_PID=$!
echo "[$(TZ=Asia/Kolkata date)] [WS] Started (PID $WS_PID)" >> "$LOG"
