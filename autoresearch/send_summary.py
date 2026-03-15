#!/usr/bin/env python3
"""Morning Telegram digest for overnight autoresearch experiments."""

import csv
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timedelta


def parse_results(tsv_path: str, since: datetime) -> list[dict]:
    """Parse results.tsv and return experiments run after `since`."""
    experiments = []
    if not os.path.exists(tsv_path):
        return experiments

    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                # Skip header
                continue
            if not row or not row[0].strip():
                continue
            try:
                ts = datetime.fromisoformat(row[0].strip())
            except ValueError:
                continue
            if ts < since:
                continue
            experiments.append({
                "timestamp": ts,
                "experiment_num": row[1].strip() if len(row) > 1 else "",
                "baseline_score": float(row[2]) if len(row) > 2 and row[2].strip() else None,
                "experiment_score_train": float(row[3]) if len(row) > 3 and row[3].strip() else None,
                "experiment_score_holdout": float(row[4]) if len(row) > 4 and row[4].strip() else None,
                "delta": float(row[5]) if len(row) > 5 and row[5].strip() else None,
                "status": row[6].strip() if len(row) > 6 else "",
                "train_trades": row[7].strip() if len(row) > 7 else "",
                "holdout_trades": row[8].strip() if len(row) > 8 else "",
                "diff_lines": row[9].strip() if len(row) > 9 else "",
                "description": row[10].strip() if len(row) > 10 else "",
            })

    return experiments


def format_summary(experiments: list[dict], date_str: str) -> str:
    """Format HTML summary message."""
    total = len(experiments)
    kept = sum(1 for e in experiments if e["status"].upper() == "KEPT")
    rejected = sum(1 for e in experiments if e["status"].upper() == "REJECTED")
    timeout = sum(1 for e in experiments if e["status"].upper() == "TIMEOUT")

    # Baseline: first experiment's baseline_score; final score: last KEPT holdout score
    baseline_score = None
    final_score = None
    cumulative_delta = 0.0

    for e in experiments:
        if baseline_score is None and e["baseline_score"] is not None:
            baseline_score = e["baseline_score"]
        if e["status"].upper() == "KEPT" and e["experiment_score_holdout"] is not None:
            final_score = e["experiment_score_holdout"]

    for e in experiments:
        if e["delta"] is not None and e["status"].upper() == "KEPT":
            cumulative_delta += e["delta"]

    # Best experiment by delta
    kept_exps = [e for e in experiments if e["status"].upper() == "KEPT" and e["delta"] is not None]
    best = max(kept_exps, key=lambda e: e["delta"]) if kept_exps else None

    lines = [f'&#x1F4CB; <b>Overnight Summary ({date_str})</b>', ""]

    if total == 0:
        lines.append("No experiments ran tonight.")
    else:
        lines.append(f"&#x1F9EA; Experiments: {total} total")
        lines.append(
            f"&#x2705; Kept: {kept} | &#x1F6AB; Rejected: {rejected} | &#x23F1; Timeout: {timeout}"
        )

        if baseline_score is not None and final_score is not None:
            delta_sign = "+" if cumulative_delta >= 0 else ""
            lines.append(
                f"&#x1F4C8; Baseline: {baseline_score:.4f} &#x2192; {final_score:.4f} "
                f"({delta_sign}{cumulative_delta:.4f})"
            )
        elif baseline_score is not None:
            lines.append(f"&#x1F4C8; Baseline: {baseline_score:.4f} (no improvements kept)")

        if best:
            delta_sign = "+" if best["delta"] >= 0 else ""
            desc = best["description"] or f"Experiment {best['experiment_num']}"
            lines.append(
                f"&#x1F3C6; Best: \"{desc}\" ({delta_sign}{best['delta']:.4f})"
            )

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Send message via Telegram using urllib. Returns True on success."""
    token = os.environ.get("DEAL_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_FORUM_CHAT_ID")
    topic_id = os.environ.get("TELEGRAM_TOPIC_STOCKS")

    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }
    if topic_id:
        payload["message_thread_id"] = topic_id

    data = urllib.parse.urlencode(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Telegram send failed: {e}", file=sys.stderr)
        return False


def main():
    # Parse experiments run after 15:45 today
    today = datetime.now().replace(hour=15, minute=45, second=0, microsecond=0)
    date_str = today.strftime("%Y-%m-%d")

    tsv_path = os.path.join(os.path.dirname(__file__), "results.tsv")
    experiments = parse_results(tsv_path, since=today)

    if not experiments:
        print("No experiments ran tonight.")

    message = format_summary(experiments, date_str)
    print(message)
    print()

    sent = send_telegram(message)
    if sent:
        print("Sent to Telegram.")
    else:
        print("Telegram env vars missing or send failed — printed to stdout only.")


if __name__ == "__main__":
    main()
