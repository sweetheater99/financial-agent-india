"""F&O Calibration — compare real Kite option chains against synthetic pricing.

Validates that skew_factor and bid-ask spread model produce realistic premiums.

Usage:
    python fo_calibrate.py                     # NIFTY calibration (requires Kite login)
    python fo_calibrate.py --symbol BANKNIFTY  # BANKNIFTY calibration
"""

import argparse
import logging
import sys
from datetime import date, datetime

import pandas as pd

from fo_data import generate_synthetic_chain, get_nearest_expiry
from fo_strategies import bid_ask_spread_pct

logger = logging.getLogger("fo_calibrate")


def compare_chains(
    real_chain: pd.DataFrame,
    synthetic_chain: pd.DataFrame,
    spot: float,
    symbol: str = "NIFTY",
) -> dict:
    """Compare real vs synthetic chains at key strikes."""
    interval = 50 if symbol == "NIFTY" else 100
    atm = round(spot / interval) * interval

    offsets = [0, 250, 500, 750, -250, -500, -750]
    comparisons = []

    for offset in offsets:
        strike = atm + offset
        for opt_type in ["CE", "PE"]:
            real_row = real_chain[
                (real_chain["strike"] == strike) & (real_chain["option_type"] == opt_type)
            ]
            synth_row = synthetic_chain[
                (synthetic_chain["strike"] == strike) & (synthetic_chain["option_type"] == opt_type)
            ]
            if real_row.empty or synth_row.empty:
                continue

            real_prem = float(real_row.iloc[0]["premium"])
            synth_prem = float(synth_row.iloc[0]["premium"])
            real_iv = float(real_row.iloc[0]["iv"])
            synth_iv = float(synth_row.iloc[0]["iv"])

            diff_pct = (synth_prem - real_prem) / real_prem if real_prem > 0 else 0

            comparisons.append({
                "strike": strike,
                "option_type": opt_type,
                "real_premium": round(real_prem, 1),
                "synth_premium": round(synth_prem, 1),
                "diff_pct": round(diff_pct, 4),
                "real_iv": round(real_iv * 100, 1),
                "synth_iv": round(synth_iv * 100, 1),
            })

    # Condor credit-to-width
    wing_width = 300
    otm_pts = 500

    def _condor_credit(chain):
        def _get(s, t):
            row = chain[(chain["strike"] == s) & (chain["option_type"] == t)]
            return float(row.iloc[0]["premium"]) if not row.empty else 0

        cs = _get(atm + otm_pts, "CE")
        cl = _get(atm + otm_pts + wing_width, "CE")
        ps = _get(atm - otm_pts, "PE")
        pl = _get(atm - otm_pts - wing_width, "PE")
        nc = (cs + ps) - (cl + pl)
        return nc / wing_width if wing_width > 0 else 0

    real_credit_pct = _condor_credit(real_chain)
    synth_credit_pct = _condor_credit(synthetic_chain)

    return {
        "comparisons": comparisons,
        "condor_real_credit_pct": round(real_credit_pct * 100, 1),
        "condor_synth_credit_pct": round(synth_credit_pct * 100, 1),
        "condor_gap_pct": round(abs(real_credit_pct - synth_credit_pct) * 100, 1),
    }


def print_calibration_report(report: dict, spot: float, vix: float, dte: int, symbol: str):
    """Print formatted calibration report."""
    print(f"\nF&O CALIBRATION REPORT — {symbol}")
    print(f"Spot: {spot:.0f}  VIX: {vix:.1f}  DTE: {dte}")
    print("=" * 70)
    print(f"{'Strike':<8} {'Type':<5} {'Real':>8} {'Synth':>8} {'Diff%':>7} {'Real_IV':>8} {'Synth_IV':>9}")
    print("-" * 70)

    for c in sorted(report["comparisons"], key=lambda x: (x["strike"], x["option_type"])):
        print(
            f"{c['strike']:<8} {c['option_type']:<5} "
            f"{c['real_premium']:>8.1f} {c['synth_premium']:>8.1f} "
            f"{c['diff_pct']:>+6.1%} "
            f"{c['real_iv']:>7.1f}% {c['synth_iv']:>8.1f}%"
        )

    print("-" * 70)
    print(f"\nCondor (500pt OTM, 300pt wings):")
    print(f"  Real credit-to-width:      {report['condor_real_credit_pct']:.1f}%")
    print(f"  Synthetic credit-to-width: {report['condor_synth_credit_pct']:.1f}%")
    print(f"  Gap: {report['condor_gap_pct']:.1f}%", end="")
    if report["condor_gap_pct"] < 10:
        print(" — ACCEPTABLE (< 10%)")
    else:
        print(" — NEEDS TUNING (>= 10%)")

    from fo_data import generate_synthetic_chain as _gsc
    import inspect
    sig = inspect.signature(_gsc)
    skew_default = sig.parameters["skew_factor"].default
    print(f"\nCurrent: skew_factor={skew_default}, spread_coeff=3.0")
    if report["condor_gap_pct"] < 10:
        print("Recommendation: No change needed")
    else:
        # Suggest directional adjustment based on whether synthetic overprices or underprices
        if report["condor_synth_credit_pct"] > report["condor_real_credit_pct"]:
            print("Recommendation: Increase spread_coeff (synthetic credit too high vs real)")
        else:
            print("Recommendation: Decrease skew_factor (synthetic premiums too low vs real)")


def main():
    parser = argparse.ArgumentParser(description="F&O Calibration")
    parser.add_argument("--symbol", default="NIFTY")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from kite_data import fetch_option_chain_kite, get_ltp_kite, get_vix_kite

    spot = get_ltp_kite(args.symbol)
    vix = get_vix_kite()

    if spot is None or vix is None:
        print(f"ERROR: Could not fetch spot ({spot}) or VIX ({vix}). Is Kite logged in?")
        sys.exit(1)

    kite_chain = fetch_option_chain_kite(args.symbol)
    if kite_chain is None:
        print(f"ERROR: Could not fetch option chain for {args.symbol}")
        sys.exit(1)

    from fo_chain_collector import convert_kite_chain_to_df
    from kite_data import _find_nearest_expiry

    expiry_str = _find_nearest_expiry(args.symbol)
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    dte = max(1, (expiry_date - date.today()).days)

    real_df = convert_kite_chain_to_df(kite_chain, spot, vix, expiry_str, args.symbol)
    synth_df = generate_synthetic_chain(spot, vix, dte, symbol=args.symbol)

    report = compare_chains(real_df, synth_df, spot, args.symbol)
    print_calibration_report(report, spot, vix, dte, args.symbol)


if __name__ == "__main__":
    main()
