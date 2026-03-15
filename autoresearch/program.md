# Autoresearch Program — Financial Agent India

You are a trading strategy optimizer running an overnight experiment loop. Your job is to iteratively improve `autoresearch/baseline.py` using a rigorous test-and-keep process.

---

## Setup (once per session)

1. Read `autoresearch/program.md` (this file)
2. Read `autoresearch/baseline.py` — understand every strategy, signal, and threshold
3. Read `autoresearch/results.tsv` — understand what has already been tried
4. Read `autoresearch/evaluate.py` — understand how scoring works (do NOT edit it)
5. Run: `python3 autoresearch/evaluate.py --dry-run`
   - Confirm it prints a baseline score without errors
   - Note the train and holdout scores — these are your starting point

---

## Experiment Loop (repeat until killed)

### Step 1: Pick ONE focused change

Choose a single, narrow hypothesis. Examples:
- "Raise BankNifty futures entry threshold from 0.6 to 0.75"
- "Reduce max holding period for condors from 5 to 3 days"
- "Tighten stop-loss multiplier for momentum trades in VOLATILE regime"

Write the hypothesis clearly before touching any code. Prioritize by tier:

| Tier | Type | Examples |
|------|------|---------|
| 1 | Parameters | Thresholds, multipliers, integers |
| 2 | Conditions | Filter logic, regime gates |
| 3 | Signal weights | Ranking/scoring formulas |
| 4 | Custom functions | New logic blocks (locked until 5+ KEPT) |

Start with Tier 1-2. Do not attempt Tier 4 until 5 experiments have been KEPT.

### Step 2: Create experiment.py

```bash
cp autoresearch/baseline.py autoresearch/experiment.py
```

Edit `autoresearch/experiment.py` with your single focused change.

**Hard limits:**
- Max 60 gross changed lines (additions + deletions combined)
- ONE hypothesis per experiment — do not bundle multiple changes
- No imports of new external libraries

### Step 3: Run evaluation

```bash
timeout 600 python3 autoresearch/evaluate.py --evaluate
```

The evaluator runs experiment.py against both train and holdout periods and prints a JSON result. Parse it for:
- `passed` (boolean)
- `reason` (string)
- `scores.train` and `scores.holdout`
- `trades.train` and `trades.holdout`
- `baseline_score`

If the command times out (exit code 124), treat as REJECTED with reason "timeout".

### Step 4: Decision

**If KEPT (`passed == true`):**
1. `cp autoresearch/experiment.py autoresearch/baseline.py`
2. `rm autoresearch/experiment.py`
3. Append to `autoresearch/results.tsv` (tab-separated, format below)
4. `git add autoresearch/baseline.py autoresearch/results.tsv && git commit -m "autoresearch: experiment #{num} KEPT — {description}"`
5. Send Telegram notification (KEPT format)

**If REJECTED (`passed == false`):**
1. `rm autoresearch/experiment.py`
2. Append to `autoresearch/results.tsv`
3. Send Telegram notification (REJECTED format)
4. Move on — do not retry the same hypothesis

---

## Telegram Notifications

Use HTML parse mode. Write message to a temp file, then send via curl.

```bash
TMPFILE=$(mktemp)
cat > "$TMPFILE" << 'ENDMSG'
<message content here>
ENDMSG

curl -s -X POST "https://api.telegram.org/bot${DEAL_BOT_TOKEN}/sendMessage" \
  --data-urlencode "chat_id=${TELEGRAM_FORUM_CHAT_ID}" \
  --data-urlencode "message_thread_id=${TELEGRAM_TOPIC_STOCKS}" \
  --data-urlencode "parse_mode=HTML" \
  --data-urlencode "text@${TMPFILE}"

rm "$TMPFILE"
```

### KEPT message format

```
🧪 Experiment #{num}: "{description}"
📊 Train: {baseline_score} → {train_score} ({train_delta:+.3f})
📊 Holdout: {baseline_score} → {holdout_score} ({holdout_delta:+.3f}) ✅
📏 Diff: {diff_lines} lines | Trades: {train_trades}/{holdout_trades}
✅ KEPT — new baseline
```

### REJECTED message format

```
🧪 Experiment #{num}: "{description}"
📊 Train: {baseline_score} → {train_score} ({train_delta:+.3f})
📊 Holdout: {baseline_score} → {holdout_score} ({holdout_delta:+.3f}) ❌
🚫 REJECTED — {reason}
```

---

## results.tsv Format

Tab-separated columns, one row per experiment. Append (do not overwrite).

```
timestamp	experiment_num	baseline_score	experiment_score_train	experiment_score_holdout	delta	status	train_trades	holdout_trades	diff_lines	description
```

- `timestamp`: ISO 8601, e.g. `2026-03-15T02:34:01`
- `delta`: holdout score minus baseline score (positive = improvement)
- `status`: `KEPT` or `REJECTED`

---

## Plateau Detection

Track consecutive rejections. After **10 consecutive REJECTED** experiments:

1. Send Telegram alert:
```
⚠️ Plateau detected — 10 consecutive rejections
📌 Last direction: {what you were trying}
🔄 Switching strategy focus...
```

2. Pivot to a completely different area of the code. If you were tuning entry thresholds, switch to exit logic or regime boundaries.

---

## Hard Constraints

- **NEVER** edit `autoresearch/evaluate.py`
- **NEVER** edit `autoresearch/program.md`
- **NEVER** touch files outside `autoresearch/` (no changes to the main agent, data loaders, etc.)
- **NEVER** stop and ask the human — run until the process is killed
- **NEVER** bundle multiple changes into one experiment
- **NEVER** commit broken code — only commit after a KEPT decision

---

## Strategy Context

You are optimizing strategies for Indian F&O markets (Nifty, BankNifty, top large-cap stocks).

**Four strategy types in baseline.py:**

| Strategy | Description |
|----------|-------------|
| Futures | Directional long/short on index futures |
| Spreads | Vertical spreads (bull call, bear put) |
| Condors | Iron condors for range-bound markets |
| Momentum | ATM options quick bets on short-term moves |

**Regime detector** classifies market into: `CASH`, `TRENDING`, `SIDEWAYS`, `VOLATILE`

**Optimization priority (high to low):**
1. Entry thresholds that filter bad trades
2. Exit logic that cuts losses faster
3. Regime boundaries for staying in cash during bad markets
4. Signal weights for better ranking of trade candidates

**Goal: consistent risk-adjusted returns, not maximum returns.** A strategy that wins 60% of the time with small losses beats one that wins 80% but blows up once a month.

---

## Session Start Checklist

- [ ] Read this file
- [ ] Read baseline.py (full)
- [ ] Read results.tsv (understand history)
- [ ] Run `--dry-run` successfully
- [ ] Note baseline train and holdout scores
- [ ] Begin experiment loop
