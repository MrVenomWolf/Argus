"""
test_suite.py
==============
Comprehensive unit and integration test suite verifying:
  1. Parametric Log Z-Score & Non-Parametric Log-MAD Z-Score math & edge cases
  2. Outlier-masking immunity & anti-masking promotion gate
  3. Exact `scipy.stats.t.sf` Student-t p-values & conservative tie-grouped BH-FDR
  4. 1-Bar Shifted Donchian Channels & Close Location Value (CLV) bull-trap rejection
  5. Wilder's 14-period ATR (RMA smoothing) & Stop/Target math
  6. Intraday NSE U-Curve session projection (with 10:00 AM IST guard)
  7. Multi-timezone (`Asia/Kolkata`, `America/New_York`, `UTC`) intraday same-slot matching
  8. Shared signal & ₹1 Cr turnover filter across live scanner and long-only backtester
  9. Strict OHLCV bar validation (rejecting malformed bars without silent clamping)
"""

from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t as scipy_student_t

import alerts
import anomaly_engine
from anomaly_engine import (
    NORMAL_CONSISTENCY_FACTOR,
    DataPoint,
    apply_benjamini_hochberg,
    backtest_donchian_volume,
    calculate_robust_metrics,
    classify_signal,
    compute_donchian_channels,
    compute_wilder_atr,
    find_anomalies,
    nse_intraday_volume_fraction,
    student_t_one_sided_pvalue,
)
import config
from data_sources import equities


class TestMathematicalEngine(unittest.TestCase):

    def test_degenerate_and_nan_inputs(self):
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
        hist = np.full(20, 1_000_000.0)
        med, rvol, z_log, z_mad = calculate_robust_metrics(hist, 2_500_000.0)
        self.assertAlmostEqual(med, 1_000_000.0)
        self.assertAlmostEqual(rvol, 2.5)
        self.assertEqual(z_log, 0.0)
        self.assertEqual(z_mad, 0.0)

    def test_exact_log_z_and_log_mad_formulas(self):
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
        rng = np.random.default_rng(42)
        clean_hist = rng.lognormal(mean=14.0, sigma=0.22, size=20)
        fresh_breakout_vol = float(np.median(clean_hist) * 3.4)

        contaminated_hist = clean_hist.copy()
        contaminated_hist[7] = float(np.median(clean_hist) * 55.0)

        _, rvol, z_log_masked, z_mad_robust = calculate_robust_metrics(
            contaminated_hist, fresh_breakout_vol
        )
        self.assertGreater(rvol, 3.0)
        self.assertLess(z_log_masked, 1.5)
        self.assertGreater(z_mad_robust, 4.0)
        self.assertGreater(z_mad_robust, z_log_masked * 1.40)

    def test_exact_scipy_student_t_and_tied_benjamini_hochberg_fdr(self):
        # Verify exact match with scipy.stats.t.sf across degrees of freedom and t-values
        for df in (5, 10, 19, 60):
            for t_val in (0.5, 1.5, 2.5, 3.5, 5.0):
                p = student_t_one_sided_pvalue(t_val, df=df)
                expected = float(scipy_student_t.sf(t_val, df=df))
                self.assertAlmostEqual(p, expected, places=15)

        # Verify conservative tie-handling in Benjamini-Hochberg
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t1 = anomaly_engine.Anomaly(DataPoint("eq", "TIE1.NS", dt, 100), 30, 3.0, 3.5, 3.5, p_value=0.0010)
        t2 = anomaly_engine.Anomaly(DataPoint("eq", "TIE2.NS", dt, 100), 30, 3.0, 3.5, 3.5, p_value=0.0010)
        t3 = anomaly_engine.Anomaly(DataPoint("eq", "OTHER.NS", dt, 100), 30, 2.5, 2.6, 2.6, p_value=0.0090)

        apply_benjamini_hochberg(
            [t3, t1, t2],
            total_tested=100,
            all_p_values=[0.0010, 0.0010, 0.0090] + [0.5] * 97,
        )
        # Tied p=0.0010 at ranks 1 and 2 both get conservative rank-1 q = 0.0010 * 100 / 1 = 0.10
        self.assertAlmostEqual(t1.fdr_q, 0.10, places=6)
        self.assertAlmostEqual(t2.fdr_q, 0.10, places=6)
        self.assertEqual(t1.fdr_q, t2.fdr_q)
        # Rank 3 (p=0.0090) gets 0.0090 * 100 / 3 = 0.30
        self.assertAlmostEqual(t3.fdr_q, 0.30, places=6)

    def test_shifted_donchian_channel_excludes_current_bar(self):
        highs = np.array([100.0, 102.0, 105.0, 103.0, 104.0, 120.0])
        lows = np.array([95.0, 96.0, 98.0, 97.0, 99.0, 101.0])
        upper, lower, mid = compute_donchian_channels(highs, lows, window=4)
        self.assertAlmostEqual(upper[0], 100.0)
        self.assertAlmostEqual(lower[0], 95.0)
        self.assertAlmostEqual(upper[5], 105.0)
        self.assertAlmostEqual(lower[5], 96.0)
        self.assertAlmostEqual(mid[5], 100.5)

    def test_wilder_atr_rma_calculation(self):
        highs = np.array([110.0, 115.0, 112.0, 118.0, 125.0, 130.0])
        lows = np.array([100.0, 108.0, 104.0, 110.0, 119.0, 120.0])
        closes = np.array([105.0, 110.0, 106.0, 116.0, 122.0, 128.0])
        atr = compute_wilder_atr(highs, lows, closes, period=3)
        self.assertAlmostEqual(atr[2], 28.0 / 3.0, places=8)
        self.assertAlmostEqual(atr[3], 92.0 / 9.0, places=8)

    def test_bull_trap_upper_wick_rejected_by_clv(self):
        sig_trap = classify_signal(105.5, 115.0, 100.0, 104.0, 105.0, 90.0, 97.5, 5.5 / 15.0)
        self.assertEqual(sig_trap, "VOL SPIKE")

        sig_valid = classify_signal(113.5, 115.0, 100.0, 104.0, 105.0, 90.0, 97.5, 13.5 / 15.0)
        self.assertEqual(sig_valid, "LONG BREAKOUT")


class TestIntradayAndPipeline(unittest.TestCase):

    def test_nse_intraday_u_curve_bounds_and_early_guard(self):
        ist = anomaly_engine.IST
        self.assertEqual(nse_intraday_volume_fraction(datetime(2026, 11, 4, 9, 0, tzinfo=ist)), (1.0, False))
        self.assertEqual(nse_intraday_volume_fraction(datetime(2026, 11, 4, 9, 30, tzinfo=ist)), (1.0, False))

        f_morn, open_morn = nse_intraday_volume_fraction(datetime(2026, 11, 4, 10, 15, tzinfo=ist))
        self.assertTrue(open_morn)
        self.assertTrue(0.20 <= f_morn < 0.40)

        f_mid, open_mid = nse_intraday_volume_fraction(datetime(2026, 11, 4, 12, 30, tzinfo=ist))
        self.assertTrue(open_mid)
        self.assertGreater(f_mid, f_morn)

        self.assertEqual(nse_intraday_volume_fraction(datetime(2026, 11, 7, 11, 0, tzinfo=ist)), (1.0, False))

    def test_intraday_same_slot_volume_matching_across_timezones(self):
        for tz_obj, sym in (
            (ZoneInfo("Asia/Kolkata"), "SLOT.NS"),
            (ZoneInfo("America/New_York"), "SPY"),
            (timezone.utc, "AAPL"),
        ):
            pts = []
            base_open = datetime(2026, 10, 1, 9, 30, tzinfo=tz_obj)
            for day in range(10):
                pts.append(DataPoint("equity", sym, base_open + timedelta(days=day), 500_000.0 * (1 + 0.05 * (day % 3))))
                pts.append(DataPoint("equity", sym, base_open + timedelta(days=day, hours=3), 80_000.0 * (1 + 0.05 * (day % 3))))

            pts.append(DataPoint("equity", sym, base_open + timedelta(days=11), 520_000.0))
            anoms = find_anomalies(pts, min_rvol=2.0, min_z_score=2.5, window=10, interval="15m")
            self.assertEqual(len(anoms), 0, f"False morning open alarm triggered for {sym} in {tz_obj}")

    def test_illiquid_penny_stock_filtered_in_scanner_and_backtester(self):
        """Both `find_anomalies` and `backtest_donchian_volume` reject < ₹1 Cr daily stocks."""
        dates = pd.date_range("2026-01-01", periods=35, freq="B")
        opens = np.full(35, 15.0)
        highs = np.full(35, 16.0)
        lows = np.full(35, 14.8)
        closes = np.full(35, 15.5)
        vols = np.array([10_000.0 * (1.0 + 0.1 * (i % 4)) for i in range(35)])
        # Inject a 6x volume breakout on bar 30, but daily turnover is only ₹1.5 Lakhs (< ₹1 Cr)
        closes[30] = 18.0
        highs[30] = 18.2
        vols[30] = 60_000.0

        df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols}, index=dates)
        pts, rej = equities._df_to_datapoints("PENNY.NS", df)
        self.assertEqual(rej, 0)
        self.assertEqual(len(find_anomalies(pts[:31], interval="1d")), 0)

        bt = backtest_donchian_volume(df, symbol="PENNY.NS")
        self.assertIsNotNone(bt)
        self.assertEqual(bt.trades, 0, "Backtester must also reject < ₹1 Cr illiquid NSE stocks")

    def test_malformed_ohlc_bars_rejected_not_clamped(self):
        """Bars with High < Low, High < Close, or negative volume must be rejected and counted."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0, -5.0, 100.0],
                "High": [110.0, 95.0, 105.0, 110.0, 110.0],   # Row 1 has High(95) < Low(98); Row 2 has High(105) < Close(108)
                "Low": [95.0, 98.0, 95.0, 95.0, 95.0],
                "Close": [105.0, 96.0, 108.0, 105.0, 105.0],
                "Volume": [1000.0, 1000.0, 1000.0, 1000.0, -50.0],  # Row 3 has Open=-5; Row 4 has Volume=-50
            },
            index=dates,
        )
        pts, rejected = equities._df_to_datapoints("BAD.NS", df)
        self.assertEqual(len(pts), 1)
        self.assertEqual(rejected, 4)

    def test_backtester_both_exit_modes_and_csv_logging(self):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2025-01-01", periods=90, freq="B")
        closes = np.linspace(1000.0, 1250.0, 90) + rng.normal(0, 4.0, 90)
        highs = closes + rng.uniform(5.0, 15.0, 90)
        lows = closes - rng.uniform(5.0, 15.0, 90)
        opens = closes - rng.uniform(-5.0, 5.0, 90)
        vols = rng.lognormal(mean=14.0, sigma=0.2, size=90)

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
        pts, rej = equities._df_to_datapoints("TESTBO.NS", df)
        self.assertEqual(rej, 0)
        anoms = find_anomalies(pts, min_rvol=2.0, min_z_score=2.5, interval="1d")
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0].signal, "LONG BREAKOUT")

        bt_trail = backtest_donchian_volume(df, symbol="TESTBO.NS", use_target=False)
        bt_target = backtest_donchian_volume(df, symbol="TESTBO.NS", use_target=True)
        self.assertIsNotNone(bt_trail)
        self.assertIsNotNone(bt_target)
        self.assertGreaterEqual(bt_trail.trades, 1)
        self.assertGreaterEqual(bt_target.trades, 1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            os.remove(tmp_path)
            alerts.log_alerts_to_csv(anoms, path=tmp_path)
            logged_df = pd.read_csv(tmp_path)
            self.assertEqual(len(logged_df), 1)
            self.assertEqual(logged_df.iloc[0]["signal"], "LONG BREAKOUT")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
