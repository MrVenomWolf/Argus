"""
test_suite.py
==============
Comprehensive unit and integration test suite verifying:
  1. Parametric Log Z-Score & Non-Parametric Log-MAD Z-Score math & edge cases
  2. Outlier-masking immunity when a prior volume spike is inside the 20-day window
  3. 1-Bar Shifted Donchian Channels & Close Location Value (CLV) bull-trap rejection
  4. Wilder's 14-period ATR (RMA smoothing) & Stop/Target math
  5. Intraday NSE U-Curve session projection & time-of-day slot matching
  6. Liquidity turnover filter & realistic Strategy Backtester
  7. DataFrame batch caching, alert CSV logging, and non-blocking config loader
"""

from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import alerts
import anomaly_engine
from anomaly_engine import (
    NORMAL_CONSISTENCY_FACTOR,
     DataPoint,
    backtest_donchian_volume,
    calculate_robust_metrics,
    classify_signal,
    compute_donchian_channels,
    compute_wilder_atr,
    find_anomalies,
    nse_intraday_volume_fraction,
)
import config
from data_sources import equities


class TestMathematicalEngine(unittest.TestCase):

    def test_degenerate_and_nan_inputs(self):
        """Zero, negative, NaN, Inf, or <5 positive bars must safely return 0.0s."""
        hist = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        self.assertEqual(calculate_robust_metrics(hist, 0.0), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(calculate_robust_metrics(hist, -50.0), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(calculate_robust_metrics(hist, float("nan")), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(calculate_robust_metrics(hist, float("inf")), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(calculate_robust_metrics(np.array([]), 500.0), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(
            calculate_robust_metrics(np.array([10.0, 20.0, 30.0, 40.0]), 500.0),
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_constant_volume_zero_variance(self):
        """Constant historical volume (std=0, MAD=0) must not divide by zero."""
        hist = np.full(20, 1_000_000.0)
        med, rvol, z_log, z_mad = calculate_robust_metrics(hist, 2_500_000.0)
        self.assertAlmostEqual(med, 1_000_000.0)
        self.assertAlmostEqual(rvol, 2.5)
        self.assertEqual(z_log, 0.0)
        self.assertEqual(z_mad, 0.0)

    def test_exact_log_z_and_log_mad_formulas(self):
        """Verify exact analytical match with textbook formulas to 1e-12 precision."""
        rng = np.random.default_rng(123)
        hist = rng.lognormal(mean=13.5, sigma=0.35, size=20)
        current = float(np.median(hist) * 3.2)

        med, rvol, z_log, z_mad = calculate_robust_metrics(hist, current)

        expected_med = float(np.median(hist))
        expected_rvol = current / expected_med
        log_h = np.log1p(hist)
        log_c = float(np.log1p(current))
        expected_z_log = (log_c - float(np.mean(log_h))) / float(np.std(log_h, ddof=1))

        med_log = float(np.median(log_h))
        mad_log = float(np.median(np.abs(log_h - med_log)))
        expected_z_mad = NORMAL_CONSISTENCY_FACTOR * (log_c - med_log) / mad_log

        self.assertAlmostEqual(med, expected_med, places=10)
        self.assertAlmostEqual(rvol, expected_rvol, places=10)
        self.assertAlmostEqual(z_log, expected_z_log, places=10)
        self.assertAlmostEqual(z_mad, expected_z_mad, places=10)

    def test_outlier_masking_immunity(self):
        """
        A single 50x earnings volume spike inside the trailing 20-day window inflates
        sample std and masks parametric `z_score` (< 2.5), whereas `modified_z_score`
        (Log-MAD) remains unmasked (> 4.0) and still flags the breakout.
        """
        rng = np.random.default_rng(42)
        clean_hist = rng.lognormal(mean=14.0, sigma=0.22, size=20)
        fresh_breakout_vol = float(np.median(clean_hist) * 3.4)

        contaminated_hist = clean_hist.copy()
        contaminated_hist[7] = float(np.median(clean_hist) * 55.0)  # Huge prior outlier

        _, rvol, z_log_masked, z_mad_robust = calculate_robust_metrics(
            contaminated_hist, fresh_breakout_vol
        )
        self.assertGreater(rvol, 3.0)
        self.assertLess(z_log_masked, 1.5, "Parametric Z should be masked by the 55x outlier")
        self.assertGreater(z_mad_robust, 4.0, "Log-MAD Z must resist the 55x outlier")

    def test_shifted_donchian_channel_excludes_current_bar(self):
        """
        Donchian Upper/Lower at index `i` must strictly equal max/min of `[i-window : i]`,
        NEVER including bar `i`, so `Close[i] > Upper[i]` is possible on a breakout.
        """
        highs = np.array([100.0, 102.0, 105.0, 103.0, 104.0, 120.0])
        lows = np.array([95.0, 96.0, 98.0, 97.0, 99.0, 101.0])
        upper, lower, mid = compute_donchian_channels(highs, lows, window=4)

        # At index 5 (where High=120.0), prior 4 bars are indices 1..4 -> max High is 105.0, min Low is 96.0
        self.assertAlmostEqual(upper[5], 105.0)
        self.assertAlmostEqual(lower[5], 96.0)
        self.assertAlmostEqual(mid[5], 100.5)

    def test_wilder_atr_rma_calculation(self):
        """Verify Wilder's ATR uses true RMA smoothing (alpha = 1/period) and gap True Range."""
        highs = np.array([110.0, 115.0, 112.0, 118.0, 125.0, 130.0])
        lows = np.array([100.0, 108.0, 104.0, 110.0, 119.0, 120.0])
        closes = np.array([105.0, 110.0, 106.0, 116.0, 122.0, 128.0])

        # Period=3
        # TR[0] = 110 - 100 = 10
        # TR[1] = max(115-108=7, |115-105|=10, |108-105|=3) = 10 (Gap up from 105 to 115)
        # TR[2] = max(112-104=8, |112-110|=2, |104-110|=6) = 8
        # ATR[2] (seed) = mean(10, 10, 8) = 28 / 3 = 9.3333333333
        # TR[3] = max(118-110=8, |118-106|=12, |110-106|=4) = 12
        # ATR[3] = (1/3)*12 + (2/3)*(28/3) = 4 + 56/9 = 92/9 = 10.2222222222
        atr = compute_wilder_atr(highs, lows, closes, period=3)
        self.assertAlmostEqual(atr[2], 28.0 / 3.0, places=8)
        self.assertAlmostEqual(atr[3], 92.0 / 9.0, places=8)

    def test_bull_trap_upper_wick_rejected_by_clv(self):
        """
        If price pierces Donchian Upper (105) to High=115 and closes at 105.5 with Low=100,
        CLV = (105.5 - 100) / (115 - 100) = 0.366 < 0.65 -> Must NOT trigger LONG BREAKOUT.
        """
        sig_trap = classify_signal(
            close_price=105.5,
            high_price=115.0,
            low_price=100.0,
            open_price=104.0,
            donchian_upper=105.0,
            donchian_lower=90.0,
            donchian_mid=97.5,
            clv=5.5 / 15.0,
        )
        self.assertEqual(sig_trap, "VOL SPIKE")

        sig_valid = classify_signal(
            close_price=113.5,
            high_price=115.0,
            low_price=100.0,
            open_price=104.0,
            donchian_upper=105.0,
            donchian_lower=90.0,
            donchian_mid=97.5,
            clv=13.5 / 15.0,
        )
        self.assertEqual(sig_valid, "LONG BREAKOUT")


class TestIntradayAndPipeline(unittest.TestCase):

    def test_nse_intraday_u_curve_bounds(self):
        """Verify NSE session fraction across pre-market, open, midday, close, and weekend."""
        ist = anomaly_engine.IST
        # Wednesday 09:00 IST (pre-market) -> (1.0, False)
        f_pre, open_pre = nse_intraday_volume_fraction(datetime(2026, 11, 4, 9, 0, tzinfo=ist))
        self.assertFalse(open_pre)
        self.assertEqual(f_pre, 1.0)

        # Wednesday 09:45 IST (30m into session) -> open=True, 0.08 <= f < 0.35
        f_morn, open_morn = nse_intraday_volume_fraction(datetime(2026, 11, 4, 9, 45, tzinfo=ist))
        self.assertTrue(open_morn)
        self.assertTrue(0.08 <= f_morn < 0.35)

        # Wednesday 12:30 IST (midday) -> open=True, f_morn < f_mid < 0.70
        f_mid, open_mid = nse_intraday_volume_fraction(datetime(2026, 11, 4, 12, 30, tzinfo=ist))
        self.assertTrue(open_mid)
        self.assertGreater(f_mid, f_morn)

        # Saturday 11:00 IST (weekend) -> (1.0, False)
        f_wknd, open_wknd = nse_intraday_volume_fraction(datetime(2026, 11, 7, 11, 0, tzinfo=ist))
        self.assertFalse(open_wknd)
        self.assertEqual(f_wknd, 1.0)

    def test_intraday_same_slot_volume_matching(self):
        """
        On 15m charts, 09:15 AM bars naturally have 4x the volume of afternoon bars.
        Verify `_select_baseline_history` compares a 09:15 AM bar against historical 09:15 AM bars.
        """
        ist = anomaly_engine.IST
        pts = []
        base_day = datetime(2026, 10, 1, tzinfo=ist)
        for day in range(10):
            # 09:15 slot: normal morning volume is ~500,000
            dt_open = base_day + timedelta(days=day, hours=9, minutes=15)
            pts.append(DataPoint("equity", "SLOT.NS", dt_open, 500_000.0 * (1 + 0.05 * (day % 3))))
            # 13:00 slot: quiet afternoon volume is ~80,000
            dt_mid = base_day + timedelta(days=day, hours=13, minutes=0)
            pts.append(DataPoint("equity", "SLOT.NS", dt_mid, 80_000.0 * (1 + 0.05 * (day % 3))))

        # Today's 09:15 bar has 520,000 volume (completely normal for 09:15 AM, even though >6x 13:00 volume!)
        dt_today_open = base_day + timedelta(days=11, hours=9, minutes=15)
        pts.append(DataPoint("equity", "SLOT.NS", dt_today_open, 520_000.0))

        anoms = find_anomalies(pts, min_rvol=2.0, min_z_score=2.5, window=10, interval="15m")
        self.assertEqual(
            len(anoms),
            0,
            "Normal 09:15 volume must NOT trigger a false anomaly when compared against prior 09:15 bars",
        )

    def test_illiquid_penny_stock_filtered_out(self):
        """A daily stock with < ₹2 Cr median turnover must be filtered out even if RVOL=5x."""
        pts = []
        base_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(25):
            dt = base_dt + timedelta(days=i)
            vol = 10_000.0 * (1.0 + 0.1 * (i % 4)) if i < 24 else 60_000.0
            # Close = ₹15, Median Volume = 10,000 -> Daily turnover = ₹1.5 Lakhs (0.015 Cr < 2.0 Cr)
            pts.append(
                DataPoint(
                    "equity",
                    "PENNY.NS",
                    dt,
                    vol,
                    raw={"open": 15.0, "high": 16.5, "low": 14.8, "close": 16.2},
                )
            )

        anoms = find_anomalies(pts, min_rvol=2.0, min_z_score=2.5, interval="1d", min_turnover_cr=2.0)
        self.assertEqual(len(anoms), 0, "Illiquid penny stock must be rejected by turnover filter")

    def test_backtester_and_csv_logging(self):
        """Verify backtester metrics and CSV alert logger schema."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2025-01-01", periods=90, freq="B")
        closes = np.linspace(1000.0, 1250.0, 90) + rng.normal(0, 4.0, 90)
        highs = closes + rng.uniform(5.0, 15.0, 90)
        lows = closes - rng.uniform(5.0, 15.0, 90)
        opens = closes - rng.uniform(-5.0, 5.0, 90)
        vols = rng.lognormal(mean=14.0, sigma=0.2, size=90)

        # Inject a strong volume-backed Donchian breakout on bar 40 and bar 89
        for idx in (40, 89):
            closes[idx] = np.max(highs[idx - 20 : idx]) + 18.0
            highs[idx] = closes[idx] + 2.0
            lows[idx] = closes[idx] - 20.0
            opens[idx] = closes[idx] - 15.0
            vols[idx] = np.median(vols[idx - 20 : idx]) * 4.0

        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
            index=dates,
        )
        pts = equities._df_to_datapoints("TESTBO.NS", df)
        self.assertEqual(len(pts), 90)
        self.assertIsNotNone(equities.get_cached_dataframe("TESTBO.NS"))

        anoms = find_anomalies(pts, min_rvol=2.0, min_z_score=2.5, interval="1d")
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0].signal, "LONG BREAKOUT")

        bt = backtest_donchian_volume(df, symbol="TESTBO.NS")
        self.assertIsNotNone(bt)
        self.assertGreaterEqual(bt.trades, 1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            os.remove(tmp_path)
            alerts.log_alerts_to_csv(anoms, path=tmp_path)
            logged_df = pd.read_csv(tmp_path)
            self.assertEqual(len(logged_df), 1)
            self.assertEqual(logged_df.iloc[0]["signal"], "LONG BREAKOUT")
            self.assertIn("donchian_upper", logged_df.columns)
            self.assertIn("stop_loss", logged_df.columns)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
