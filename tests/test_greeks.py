"""Tests for greeks.py — Black-Scholes Greeks computation."""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greeks import (
    black_scholes_greeks,
    implied_volatility,
    compute_position_greeks,
    portfolio_greeks_summary,
    generate_greeks_report,
    _norm_cdf,
)


# ---------------------------------------------------------------------------
# Test parameters (Nifty-like values)
# ---------------------------------------------------------------------------
SPOT = 22000.0
STRIKE_ATM = 22000.0
STRIKE_ITM_CALL = 21500.0  # deep ITM call
STRIKE_OTM_CALL = 23000.0  # OTM call
DTE = 30  # 30 days to expiry
RF = 0.065
IV = 0.15  # 15% vol


class TestBlackScholesCallPrice:
    """1. Black-Scholes call price matches known value (ATM Nifty)."""

    def test_atm_call_price_positive(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        # ATM call with 30 DTE, 15% vol on Nifty ~22000 should be ~300-500
        assert g["theoretical_price"] > 100
        assert g["theoretical_price"] < 1000

    def test_atm_call_price_ballpark(self):
        # BS call price for ATM: approx spot * vol * sqrt(T/365) * 0.4
        # ~22000 * 0.15 * sqrt(30/365) * 0.4 ~ 378
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        assert 200 < g["theoretical_price"] < 600


class TestPutCallParity:
    """2. Black-Scholes put price matches via put-call parity.

    C - P = S - K * exp(-rT)
    """

    def test_put_call_parity(self):
        call = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        put = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "PE")
        t = DTE / 365.0
        parity_rhs = SPOT - STRIKE_ATM * math.exp(-RF * t)
        parity_lhs = call["theoretical_price"] - put["theoretical_price"]
        assert abs(parity_lhs - parity_rhs) < 0.5  # within 0.50 rupees


class TestDeltaBounds:
    """3. Delta: call delta between 0 and 1, put delta between -1 and 0."""

    def test_call_delta_bounds(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        assert 0 < g["delta"] < 1

    def test_put_delta_bounds(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "PE")
        assert -1 < g["delta"] < 0

    def test_otm_call_delta_small(self):
        g = black_scholes_greeks(SPOT, STRIKE_OTM_CALL, DTE, RF, IV, "CE")
        assert 0 < g["delta"] < 0.3

    def test_otm_put_delta_small(self):
        g = black_scholes_greeks(SPOT, STRIKE_ITM_CALL, DTE, RF, IV, "PE")
        # Strike 21500 is OTM for put
        assert -0.3 < g["delta"] < 0


class TestATMDelta:
    """4. ATM delta ~0.5."""

    def test_atm_call_delta_near_half(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        assert 0.45 < g["delta"] < 0.60  # slightly above 0.5 due to drift

    def test_atm_put_delta_near_minus_half(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "PE")
        assert -0.55 < g["delta"] < -0.40


class TestDeepITMDelta:
    """5. Deep ITM delta ~1.0."""

    def test_deep_itm_call_delta(self):
        # Strike 21500 with spot 22000 — moderately ITM (500pts on 22000 = ~2.3%)
        g = black_scholes_greeks(SPOT, STRIKE_ITM_CALL, DTE, RF, IV, "CE")
        assert g["delta"] > 0.75

    def test_very_deep_itm_call_delta(self):
        # Strike 20000 — very deep ITM
        g = black_scholes_greeks(SPOT, 20000.0, DTE, RF, IV, "CE")
        assert g["delta"] > 0.95

    def test_deep_itm_put_delta(self):
        # Put with strike 23000, spot 22000 — deep ITM put (~4.5% ITM)
        g = black_scholes_greeks(SPOT, 23000.0, DTE, RF, IV, "PE")
        assert g["delta"] < -0.80


class TestThetaNegative:
    """6. Theta is negative (time decay)."""

    def test_call_theta_negative(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        assert g["theta"] < 0

    def test_put_theta_negative(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "PE")
        # Deep ITM puts can have positive theta due to interest rate,
        # but ATM puts should have negative theta
        assert g["theta"] < 0

    def test_theta_increases_near_expiry(self):
        # Theta decay accelerates near expiry
        g_30 = black_scholes_greeks(SPOT, STRIKE_ATM, 30, RF, IV, "CE")
        g_5 = black_scholes_greeks(SPOT, STRIKE_ATM, 5, RF, IV, "CE")
        assert abs(g_5["theta"]) > abs(g_30["theta"])


class TestVegaPositive:
    """7. Vega is positive."""

    def test_call_vega_positive(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        assert g["vega"] > 0

    def test_put_vega_positive(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "PE")
        assert g["vega"] > 0

    def test_atm_vega_highest(self):
        # ATM has highest vega
        g_atm = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        g_otm = black_scholes_greeks(SPOT, STRIKE_OTM_CALL, DTE, RF, IV, "CE")
        assert g_atm["vega"] > g_otm["vega"]


class TestIVConvergence:
    """8. IV computation converges for known price."""

    def test_iv_roundtrip_call(self):
        # Compute price from known IV, then recover IV
        known_iv = 0.20
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, known_iv, "CE")
        recovered_iv = implied_volatility(
            g["theoretical_price"], SPOT, STRIKE_ATM, DTE, RF, "CE"
        )
        assert abs(recovered_iv - known_iv) < 0.001

    def test_iv_roundtrip_put(self):
        known_iv = 0.25
        g = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, known_iv, "PE")
        recovered_iv = implied_volatility(
            g["theoretical_price"], SPOT, STRIKE_ATM, DTE, RF, "PE"
        )
        assert abs(recovered_iv - known_iv) < 0.001

    def test_iv_otm(self):
        known_iv = 0.18
        g = black_scholes_greeks(SPOT, STRIKE_OTM_CALL, DTE, RF, known_iv, "CE")
        recovered_iv = implied_volatility(
            g["theoretical_price"], SPOT, STRIKE_OTM_CALL, DTE, RF, "CE"
        )
        assert abs(recovered_iv - known_iv) < 0.001


class TestIVImpossiblePrices:
    """9. IV returns NaN for impossible prices."""

    def test_zero_price(self):
        result = implied_volatility(0, SPOT, STRIKE_ATM, DTE, RF, "CE")
        assert math.isnan(result)

    def test_negative_price(self):
        result = implied_volatility(-10, SPOT, STRIKE_ATM, DTE, RF, "CE")
        assert math.isnan(result)

    def test_expired(self):
        result = implied_volatility(100, SPOT, STRIKE_ATM, 0, RF, "CE")
        assert math.isnan(result)

    def test_price_above_spot_for_call(self):
        # Call price can never exceed spot
        result = implied_volatility(SPOT + 100, SPOT, STRIKE_ATM, DTE, RF, "CE")
        assert math.isnan(result)


class TestPortfolioSummary:
    """10. Portfolio summary aggregates correctly."""

    def _make_positions(self):
        return [
            {
                "symbol": "RELIANCE",
                "instrument": "OPT",
                "option_type": "PE",
                "strike": 2400.0,
                "entry_price": 50.0,
                "quantity": 250,
                "expiry": "27MAR2026",
                "greeks_at_entry": {"iv": 22.0},
            },
            {
                "symbol": "TCS",
                "instrument": "OPT",
                "option_type": "PE",
                "strike": 3800.0,
                "entry_price": 80.0,
                "quantity": 150,
                "expiry": "27MAR2026",
                "greeks_at_entry": {"iv": 18.0},
            },
        ]

    def test_summary_has_all_keys(self):
        positions = self._make_positions()
        spots = {"RELIANCE": 2500.0, "TCS": 3900.0}
        pg = compute_position_greeks(positions, spots, RF)
        summary = portfolio_greeks_summary(pg)

        assert "net_delta" in summary
        assert "net_gamma" in summary
        assert "net_theta_daily" in summary
        assert "net_vega" in summary
        assert "total_premium_at_risk" in summary
        assert "theta_decay_daily_rupees" in summary
        assert "positions_by_delta_exposure" in summary

    def test_summary_aggregation(self):
        positions = self._make_positions()
        spots = {"RELIANCE": 2500.0, "TCS": 3900.0}
        pg = compute_position_greeks(positions, spots, RF)
        summary = portfolio_greeks_summary(pg)

        # Net delta should be sum of individual position deltas
        manual_delta = sum(p["position_delta"] for p in pg)
        assert abs(summary["net_delta"] - round(manual_delta, 4)) < 0.01

        # Total premium at risk
        expected_premium = 50.0 * 250 + 80.0 * 150
        assert abs(summary["total_premium_at_risk"] - expected_premium) < 1.0

    def test_empty_portfolio(self):
        summary = portfolio_greeks_summary([])
        assert summary["net_delta"] == 0.0
        assert summary["total_premium_at_risk"] == 0.0


class TestThetaDailyRupees:
    """11. Theta daily rupees calculation correct."""

    def test_theta_rupees_is_negative(self):
        positions = [
            {
                "symbol": "RELIANCE",
                "instrument": "OPT",
                "option_type": "PE",
                "strike": 2400.0,
                "entry_price": 50.0,
                "quantity": 250,
                "expiry": "27MAR2026",
                "greeks_at_entry": {"iv": 22.0},
            },
        ]
        spots = {"RELIANCE": 2500.0}
        pg = compute_position_greeks(positions, spots, RF)
        summary = portfolio_greeks_summary(pg)
        # Theta should be negative (losing money per day for long options)
        assert summary["theta_decay_daily_rupees"] < 0

    def test_theta_rupees_proportional_to_quantity(self):
        pos1 = [{
            "symbol": "RELIANCE", "instrument": "OPT", "option_type": "PE",
            "strike": 2400.0, "entry_price": 50.0, "quantity": 250,
            "expiry": "27MAR2026", "greeks_at_entry": {"iv": 22.0},
        }]
        pos2 = [{
            "symbol": "RELIANCE", "instrument": "OPT", "option_type": "PE",
            "strike": 2400.0, "entry_price": 50.0, "quantity": 500,
            "expiry": "27MAR2026", "greeks_at_entry": {"iv": 22.0},
        }]
        spots = {"RELIANCE": 2500.0}
        pg1 = compute_position_greeks(pos1, spots, RF)
        pg2 = compute_position_greeks(pos2, spots, RF)
        s1 = portfolio_greeks_summary(pg1)
        s2 = portfolio_greeks_summary(pg2)
        # Double quantity should roughly double theta decay
        ratio = s2["theta_decay_daily_rupees"] / s1["theta_decay_daily_rupees"]
        assert 1.9 < ratio < 2.1


class TestGreeksReport:
    """12. Greeks report contains expected sections."""

    def test_report_has_sections(self):
        positions = [{
            "symbol": "RELIANCE", "instrument": "OPT", "option_type": "PE",
            "strike": 2400.0, "entry_price": 50.0, "quantity": 250,
            "expiry": "27MAR2026", "greeks_at_entry": {"iv": 22.0},
        }]
        spots = {"RELIANCE": 2500.0}
        pg = compute_position_greeks(positions, spots, RF)
        summary = portfolio_greeks_summary(pg)
        report = generate_greeks_report(pg, summary)

        assert "Greeks Dashboard" in report
        assert "Portfolio Greeks" in report
        assert "Per-Position Greeks" in report
        assert "RELIANCE" in report
        assert "Net Delta" in report
        assert "Net Theta" in report

    def test_report_empty_portfolio(self):
        summary = portfolio_greeks_summary([])
        report = generate_greeks_report([], summary)
        assert "No option positions" in report

    def test_report_theta_warning(self):
        # Fabricate a summary with high theta relative to premium
        pg = [{
            "symbol": "TEST", "instrument": "OPT", "option_type": "PE",
            "strike": 100, "quantity": 100, "dte": 2,
            "computed_greeks": {"delta": -0.5, "gamma": 0.01, "theta": -5.0, "vega": 0.1, "iv": 0.3, "theoretical_price": 10},
            "position_delta": -50, "position_gamma": 1.0,
            "position_theta": -500, "position_vega": 10,
            "premium_at_risk": 1000, "theoretical_value": 1000,
        }]
        summary = {
            "net_delta": -50, "net_gamma": 1.0,
            "net_theta_daily": -500, "net_vega": 10,
            "total_premium_at_risk": 1000,
            "theta_decay_daily_rupees": -500,
            "positions_by_delta_exposure": pg,
        }
        report = generate_greeks_report(pg, summary)
        assert "WARN" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_dte_call_itm(self):
        g = black_scholes_greeks(22000, 21500, 0, RF, IV, "CE")
        assert g["theoretical_price"] == 500.0
        assert g["delta"] == 1.0

    def test_zero_dte_put_otm(self):
        g = black_scholes_greeks(22000, 21500, 0, RF, IV, "PE")
        assert g["theoretical_price"] == 0.0
        assert g["delta"] == 0.0

    def test_very_high_dte(self):
        g = black_scholes_greeks(SPOT, STRIKE_ATM, 365, RF, IV, "CE")
        assert g["theoretical_price"] > 0
        assert 0 < g["delta"] < 1

    def test_gamma_symmetric(self):
        # Gamma is the same for calls and puts at same strike
        gc = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "CE")
        gp = black_scholes_greeks(SPOT, STRIKE_ATM, DTE, RF, IV, "PE")
        assert abs(gc["gamma"] - gp["gamma"]) < 1e-8

    def test_skips_equity_positions(self):
        positions = [
            {"symbol": "RELIANCE", "instrument": "EQ", "entry_price": 2500, "quantity": 10},
        ]
        result = compute_position_greeks(positions, {"RELIANCE": 2500}, RF)
        assert len(result) == 0

    def test_momentum_positions_included(self):
        positions = [{
            "symbol": "NIFTY", "instrument": "MOMENTUM", "option_type": "CE",
            "strike": 22000.0, "entry_price": 200.0, "quantity": 75,
            "expiry": "27MAR2026", "greeks_at_entry": {"iv": 15.0},
        }]
        result = compute_position_greeks(positions, {"NIFTY": 22100.0}, RF)
        assert len(result) == 1
