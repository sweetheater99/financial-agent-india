#!/bin/bash
# Deploy V7 to Pi — run from Mac
# Usage: scripts/v7_deploy_pi.sh

set -euo pipefail

PI_HOST="pi@homepi.local"
REMOTE_DIR="~/financial-agent-india"

echo "=== V7 Deployment to Pi ==="

# 1. Sync code
echo "[1/5] Syncing code to Pi..."
rsync -avz --exclude='venv/' --exclude='data/' --exclude='__pycache__/' \
    --exclude='.git/' --exclude='*.pyc' \
    ~/financial-agent-india/ "$PI_HOST:$REMOTE_DIR/"

# 2. Create data directories
echo "[2/5] Creating data directories..."
ssh "$PI_HOST" "mkdir -p $REMOTE_DIR/data/v7"

# 3. Install any new dependencies
echo "[3/5] Checking dependencies..."
ssh "$PI_HOST" "cd $REMOTE_DIR && source venv/bin/activate && pip install -q anthropic 2>/dev/null || true"

# 4. Make scripts executable
echo "[4/5] Setting permissions..."
ssh "$PI_HOST" "chmod +x $REMOTE_DIR/scripts/v7_*.sh"

# 5. Show crontab instructions
echo "[5/5] Crontab setup:"
echo ""
echo "SSH into Pi and run: crontab -e"
echo "Add the following lines:"
echo ""
cat << 'CRON'
# -- V7 Professional Trader Bot --
43 8  * * 1-5  ~/financial-agent-india/scripts/v7_premarket.sh
13 9  * * 1-5  ~/financial-agent-india/scripts/v7_opening_read.sh
*/3 9-15 * * 1-5  ~/financial-agent-india/scripts/v7_executor.sh
28 10 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
58 12 * * 1-5  ~/financial-agent-india/scripts/v7_checkin.sh
33 15 * * 1-5  ~/financial-agent-india/scripts/v7_eod.sh
50 8  * * *    ~/financial-agent-india/scripts/kite_token_check.sh
3  10 * * 0    ~/financial-agent-india/scripts/v7_weekly_review.sh
7  10 1 * *    ~/financial-agent-india/scripts/v7_monthly_report.sh
CRON

echo ""
echo "=== Deployment complete ==="
echo ""
echo "TRANSITION PLAN:"
echo "1. Keep V6 cron running alongside V7 for 2-4 weeks"
echo "2. V7 runs in --paper mode by default (no real orders)"
echo "3. Monitor data/v7/cron.log for errors"
echo "4. Compare V7 paper P&L vs V6 paper P&L after 50+ trades"
echo "5. When V7 shows positive expectancy: disable V6 cron, switch V7 to live"
echo "6. To go live: remove --paper flag from v7/main.py calls in cron scripts"
