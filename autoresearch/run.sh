#!/bin/bash
# Overnight autoresearch loop orchestrator
# Cron entries (run on Pi):
# 45 15 * * 1-5 ~/financial-agent-india/autoresearch/run.sh
# 0 6 * * 2-6 pkill -f "autoresearch/run.sh"; pkill -f "claude.*program.md"
# 30 10 * * 0 cd ~/financial-agent-india && source venv/bin/activate && python3 autoresearch/cache_data.py

set -euo pipefail

# ── Variables ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/autoresearch/logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d).log"
LOCK_FILE="/tmp/financial_agent_autoresearch.lock"

mkdir -p "$LOG_DIR"

exec >> "$LOG_FILE" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] autoresearch/run.sh started"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
# shellcheck source=/dev/null
source venv/bin/activate
# shellcheck source=/dev/null
source ~/.config/env/global.env

# ── File lock ─────────────────────────────────────────────────────────────────
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another instance is running. Exiting."
    exit 1
fi

# ── Holiday check ─────────────────────────────────────────────────────────────
if python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from datetime import date
from config import NSE_HOLIDAYS_2026
today = date.today()
if today in NSE_HOLIDAYS_2026:
    print(f"[INFO] Today ({today}) is an NSE holiday. Skipping.")
    sys.exit(0)
elif today.weekday() >= 5:
    print(f"[INFO] Today ({today}) is a weekend. Skipping.")
    sys.exit(0)
else:
    sys.exit(1)
PYEOF
then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Holiday/weekend detected — exiting normally."
    exit 0
fi

# ── Disk space check ──────────────────────────────────────────────────────────
AVAIL_KB=$(df -k "$PROJECT_DIR" | awk 'NR==2 {print $4}')
AVAIL_MB=$(( AVAIL_KB / 1024 ))
if (( AVAIL_MB < 500 )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Only ${AVAIL_MB}MB free (need 500MB). Aborting."
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Disk OK: ${AVAIL_MB}MB free"

# ── Baseline validation ────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running baseline dry-run..."
if ! python3 autoresearch/evaluate.py --dry-run; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Baseline validation failed. Aborting."
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Baseline validation passed."

# ── Log rotation ──────────────────────────────────────────────────────────────
for f in "$LOG_DIR"/run_*.log; do
    [[ -f "$f" ]] || continue
    SIZE=$(wc -c < "$f")
    if (( SIZE > 5 * 1024 * 1024 )); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rotating large log: $f (${SIZE} bytes)"
        truncate -s 0 "$f"
    fi
done

# ── Working tree check ────────────────────────────────────────────────────────
if ! git diff --quiet HEAD; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Working tree is dirty. Commit or clean changes before running. Aborting."
    git diff --stat HEAD
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Working tree is clean."

# ── Git branch ────────────────────────────────────────────────────────────────
git checkout -B autoresearch/overnight
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checked out branch: autoresearch/overnight"

# ── Claude Code invocation with retry ─────────────────────────────────────────
MAX_RETRIES=3
ATTEMPT=0
EXIT_CODE=0

while (( ATTEMPT < MAX_RETRIES )); do
    ATTEMPT=$(( ATTEMPT + 1 ))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude Code attempt ${ATTEMPT}/${MAX_RETRIES}..."

    set +e
    timeout 50400 claude -p "Read autoresearch/program.md and begin experimenting. \
This is a fresh overnight session. Check autoresearch/results.tsv for prior experiments. \
Run experiments in a loop until you receive SIGTERM. Do not stop or ask questions." \
        --model claude-sonnet-4-20250514 \
        --allowedTools "Read,Write,Edit,Bash,Glob,Grep"
    EXIT_CODE=$?
    set -e

    if (( EXIT_CODE == 124 )); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude Code timed out (normal). Exiting retry loop."
        break
    elif (( EXIT_CODE == 0 || EXIT_CODE == 143 )); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude Code exited normally (exit ${EXIT_CODE})."
        break
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude Code crashed (exit ${EXIT_CODE}). Attempt ${ATTEMPT}/${MAX_RETRIES}."
        if (( ATTEMPT < MAX_RETRIES )); then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retrying in 30s..."
            sleep 30
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max retries reached. Proceeding to summary."
        fi
    fi
done

# ── Return to main ────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Returning to main branch..."
git checkout main || git checkout master
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Back on main branch."

# ── Morning summary ───────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sending morning summary..."
python3 autoresearch/send_summary.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] autoresearch/run.sh completed."
