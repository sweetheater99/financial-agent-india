import datetime
import pytest
import pandas as pd


class TestLotSizes:
    def test_nifty_lot_pre_dec_2024(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2024, 6, 15)) == 50

    def test_nifty_lot_dec_2024_to_nov_2025(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2025, 3, 15)) == 75

    def test_nifty_lot_dec_2025_onwards(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2026, 1, 15)) == 65

    def test_banknifty_lot_pre_dec_2024(self):
        from fo_data import get_lot_size
        assert get_lot_size("BANKNIFTY", datetime.date(2024, 6, 15)) == 25

    def test_banknifty_lot_dec_2024_to_nov_2025(self):
        from fo_data import get_lot_size
        assert get_lot_size("BANKNIFTY", datetime.date(2025, 6, 15)) == 30

    def test_banknifty_lot_dec_2025_onwards(self):
        from fo_data import get_lot_size
        assert get_lot_size("BANKNIFTY", datetime.date(2026, 2, 15)) == 28

    def test_nifty_lot_boundary_nov_30_2024(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2024, 11, 30)) == 50

    def test_nifty_lot_boundary_dec_1_2024(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2024, 12, 1)) == 75

    def test_nifty_lot_boundary_nov_30_2025(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2025, 11, 30)) == 75

    def test_nifty_lot_boundary_dec_1_2025(self):
        from fo_data import get_lot_size
        assert get_lot_size("NIFTY", datetime.date(2025, 12, 1)) == 65


class TestExpiryCalendar:
    def test_monthly_expiry_pre_sep_2025_is_thursday(self):
        from fo_data import get_monthly_expiry
        exp = get_monthly_expiry(2025, 3)
        assert exp == datetime.date(2025, 3, 27)
        assert exp.weekday() == 3

    def test_monthly_expiry_post_sep_2025_is_tuesday(self):
        from fo_data import get_monthly_expiry
        exp = get_monthly_expiry(2025, 10)
        assert exp == datetime.date(2025, 10, 28)
        assert exp.weekday() == 1

    def test_weekly_expiry_pre_sep_2025_is_thursday(self):
        from fo_data import get_weekly_expiries
        expiries = get_weekly_expiries(2025, 3)
        for exp in expiries:
            assert exp.weekday() == 3

    def test_weekly_expiry_post_sep_2025_is_tuesday(self):
        from fo_data import get_weekly_expiries
        expiries = get_weekly_expiries(2025, 10)
        for exp in expiries:
            assert exp.weekday() == 1

    def test_monthly_expiry_sep_2025_is_tuesday(self):
        from fo_data import get_monthly_expiry
        exp = get_monthly_expiry(2025, 9)
        assert exp == datetime.date(2025, 9, 30)
        assert exp.weekday() == 1

    def test_nearest_expiry_respects_min_dte(self):
        from fo_data import get_nearest_expiry
        exp = get_nearest_expiry(datetime.date(2025, 10, 1), min_dte=20)
        assert exp.month >= 10
        assert (exp - datetime.date(2025, 10, 1)).days >= 20

    def test_nearest_expiry_weekly(self):
        from fo_data import get_nearest_expiry
        exp = get_nearest_expiry(datetime.date(2025, 10, 1), weekly=True)
        # Should return nearest weekly expiry (Tuesday) on or after Oct 1
        assert exp >= datetime.date(2025, 10, 1)
        assert exp.weekday() == 1  # Tuesday
        # First Tuesday in Oct 2025 is Oct 7
        assert exp == datetime.date(2025, 10, 7)

    def test_nearest_expiry_min_dte_zero(self):
        from fo_data import get_nearest_expiry
        # Oct 28 2025 is the last Tuesday (monthly expiry day itself)
        exp = get_nearest_expiry(datetime.date(2025, 10, 28), min_dte=0)
        # With min_dte=0, should return that same day's expiry
        assert exp == datetime.date(2025, 10, 28)


class TestSyntheticChain:
    def test_chain_has_ce_and_pe(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        assert isinstance(chain, pd.DataFrame)
        assert "CE" in chain["option_type"].values
        assert "PE" in chain["option_type"].values

    def test_chain_strike_interval_nifty(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        ce_strikes = sorted(chain[chain["option_type"] == "CE"]["strike"].values)
        diffs = [ce_strikes[i+1] - ce_strikes[i] for i in range(len(ce_strikes)-1)]
        assert all(d == 50 for d in diffs)

    def test_chain_strike_interval_banknifty(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=50000.0, vix=16.0, dte=30, symbol="BANKNIFTY")
        ce_strikes = sorted(chain[chain["option_type"] == "CE"]["strike"].values)
        diffs = [ce_strikes[i+1] - ce_strikes[i] for i in range(len(ce_strikes)-1)]
        assert all(d == 100 for d in diffs)

    def test_chain_atm_delta_approximately_0_5(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        atm_ce = chain[(chain["option_type"] == "CE") & (chain["strike"] == 23000.0)]
        assert len(atm_ce) == 1
        assert 0.4 < atm_ce.iloc[0]["delta"] < 0.65

    def test_chain_premium_positive(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        assert (chain["premium"] > 0).all()

    def test_chain_filters_low_premium(self):
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=14.0, dte=30, symbol="NIFTY")
        assert (chain["premium"] >= 5.0).all()

    def test_skew_otm_put_iv_higher_than_atm(self):
        """OTM puts should have higher IV than ATM due to put skew."""
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY")
        atm_strike = 23000.0
        otm_put_strike = 22500.0  # 500pt OTM

        atm_row = chain[(chain["strike"] == atm_strike) & (chain["option_type"] == "PE")]
        otm_row = chain[(chain["strike"] == otm_put_strike) & (chain["option_type"] == "PE")]

        assert not atm_row.empty and not otm_row.empty
        assert otm_row.iloc[0]["iv"] > atm_row.iloc[0]["iv"], \
            f"OTM put IV {otm_row.iloc[0]['iv']:.4f} should exceed ATM IV {atm_row.iloc[0]['iv']:.4f}"

    def test_skew_otm_call_iv_mildly_higher(self):
        """OTM calls should have mildly higher IV (smile), less than put skew."""
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY")
        atm_strike = 23000.0
        otm_call_strike = 23500.0
        otm_put_strike = 22500.0

        atm_pe = chain[(chain["strike"] == atm_strike) & (chain["option_type"] == "PE")]
        otm_ce = chain[(chain["strike"] == otm_call_strike) & (chain["option_type"] == "CE")]
        otm_pe = chain[(chain["strike"] == otm_put_strike) & (chain["option_type"] == "PE")]

        assert not otm_ce.empty
        call_iv_bump = otm_ce.iloc[0]["iv"] - atm_pe.iloc[0]["iv"]
        put_iv_bump = otm_pe.iloc[0]["iv"] - atm_pe.iloc[0]["iv"]

        assert call_iv_bump > 0, "OTM call IV should be slightly above ATM"
        assert call_iv_bump < put_iv_bump, \
            f"Call IV bump ({call_iv_bump:.4f}) should be less than put IV bump ({put_iv_bump:.4f})"

    def test_skew_factor_zero_gives_flat_iv(self):
        """With skew_factor=0, all strikes should have same IV (backward compat)."""
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY", skew_factor=0.0)
        ivs = chain["iv"].unique()
        assert len(ivs) == 1, f"Expected flat IV with skew_factor=0, got {len(ivs)} unique IVs"
        assert abs(ivs[0] - 0.15) < 0.001

    def test_skew_increases_with_distance(self):
        """Further OTM puts should have progressively higher IV."""
        from fo_data import generate_synthetic_chain
        chain = generate_synthetic_chain(spot=23000.0, vix=15.0, dte=30, symbol="NIFTY")
        pe_chain = chain[chain["option_type"] == "PE"].sort_values("strike", ascending=False)

        # Get IVs for strikes below ATM (OTM puts)
        otm_puts = pe_chain[pe_chain["strike"] < 23000.0].head(5)
        if len(otm_puts) >= 2:
            ivs = otm_puts["iv"].tolist()
            # IVs should be increasing as strike decreases (further OTM = higher IV)
            for i in range(len(ivs) - 1):
                assert ivs[i] <= ivs[i + 1], \
                    f"IV should increase for further OTM puts: {ivs[i]:.4f} vs {ivs[i+1]:.4f}"


class TestFuturesPrice:
    def test_futures_above_spot(self):
        from fo_data import get_futures_price
        fut = get_futures_price(spot=23000.0, dte=30)
        assert fut > 23000.0  # cost of carry makes futures > spot

    def test_futures_zero_dte_equals_spot(self):
        from fo_data import get_futures_price
        fut = get_futures_price(spot=23000.0, dte=0)
        assert abs(fut - 23000.0) < 0.01


class TestTransactionCosts:
    def test_options_cost_pre_apr_2025(self):
        from fo_data import calc_options_costs
        cost = calc_options_costs(premium=100.0, quantity=75, side="sell", date=datetime.date(2025, 3, 15))
        assert cost > 0
        assert cost < 50

    def test_options_cost_post_apr_2026(self):
        from fo_data import calc_options_costs
        cost_new = calc_options_costs(premium=100.0, quantity=75, side="sell", date=datetime.date(2026, 5, 15))
        cost_old = calc_options_costs(premium=100.0, quantity=75, side="sell", date=datetime.date(2025, 3, 15))
        assert cost_new > cost_old

    def test_futures_cost_round_trip(self):
        from fo_data import calc_futures_round_trip
        cost = calc_futures_round_trip(entry_price=23000.0, exit_price=23200.0, quantity=75, date=datetime.date(2026, 1, 15))
        assert cost > 0
        assert cost < 600

    def test_options_round_trip(self):
        from fo_data import calc_options_round_trip
        cost = calc_options_round_trip(entry_premium=100.0, exit_premium=150.0, quantity=75, date=datetime.date(2026, 1, 15))
        assert cost > 0

    def test_exercise_stt_much_higher(self):
        from fo_data import calc_exercise_stt
        stt = calc_exercise_stt(spot=23000.0, quantity=75)
        assert stt > 2000
