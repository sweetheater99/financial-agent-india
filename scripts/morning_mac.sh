#!/bin/bash
# V7 Morning Sequence for Mac
# Run daily at 8:40 AM IST on weekdays
# Cron: 40 8 * * 1-5 ~/financial-agent-india/scripts/morning_mac.sh >> ~/financial-agent-india/logs/morning.log 2>&1

set -e
cd ~/financial-agent-india
source venv/bin/activate

echo "=== $(date '+%Y-%m-%d %H:%M:%S') V7 Morning Sequence ==="

# 1. Kite auth (daily OAuth)
echo "[1/3] Kite auth..."
python kite_auth.py
echo "  Kite auth done."

# 2. Premarket playbook (Claude generates trading plan)
echo "[2/3] Premarket playbook..."
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT python -m v7.main premarket
echo "  Premarket done."

# 3. Wait for 9:45 AM then run opening-read
echo "[3/3] Waiting for 9:45 AM to run opening-read..."
while [ $(date '+%H%M') -lt 0945 ]; do
    sleep 60
done
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT python -m v7.main opening-read
echo "  Opening read done."

echo "=== Morning sequence complete. Start tick loop manually or via cron. ==="
