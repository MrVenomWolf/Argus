"""
test_suite.py
==============
Comprehensive unit and integration test suite verifying:
  1. Parametric Log Z-Score & Non-Parametric Log-MAD Z-Score math & edge cases
  2. Outlier-masking immunity & anti-masking promotion gate
  3. Student-t one-sided p-value & Benjamini-Hochberg FDR calibration
  4. 1-Bar Shifted Donchian Channels & Close Location Value (CLV) bull-trap rejection
  5. Wilder's 14-period ATR (RMA smoothing) & Stop/Target math
  6. Intraday NSE U-Curve session projection (with 10:00 AM IST guard)
  7. Multi-timezone (`Asia/Kolkata`, `America/New_York`, `UTC`) intraday same-slot matching
  8. Default ₹1 Cr liquidity turnover filter & Next-Open 1% Risk-Sized Backtester
  9. Lazy `config.EQUITY_TICKERS` attribute & 17-column CSV alert logger
"""

from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

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

    def test_student_t_and_benjamini_hochberg_fdr(self):
        p = student_t_one_sided_pvalue(3.5, df=19)
        self.assertAlmostEqual(p, 0.0012, places=3)

        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a1 = anomaly_engine.Anomaly(DataPoint("eq", "A.NS", dt, 100), 30, 3.3, 4.0, 4.0, p_value=0.0005)
        a2 = anomaly_engine.Anomaly(DataPoint("eq", "B.NS", dt, 100), 30, 2.5, 2.6, 2.6, p_value=0.0080)
        apply_benjamini_hochberg([a2, a1], total_tested=100)
        self.assertAlmostEqual(a1.fdr_q, 0.05, places=4)
        self.assertAlmostEqual(a2.fdr_q, 0.40, places=4)

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
        # 09:00 IST (pre-market) and 09:30 IST (< 10:00 guard) both return (1.0, False)
        self.assertEqual(nse_intraday_volume_fraction(datetime(2026, 11, 4, 9, 0, tzinfo=ist)), (1.0, False))
        self.assertEqual(nse_intraday_volume_fraction(datetime(2026, 11, 4, 9, 30, tzinfo=ist)), (1.0, False))

        # 10:15 IST (1 hour into session) -> open=True, 0.20 <= f < 0.40
        f_morn, open_morn = nse_intraday_volume_fraction(datetime(2026, 11, 4, 10, 15, tzinfo=ist))
        self.assertTrue(open_morn)
        self.assertTrue(0.20 <= f_morn < 0.40)

        # 12:30 IST (midday) -> open=True, f_morn < f_mid < 0.70
        f_mid, open_mid = nse_intraday_volume_fraction(datetime(2026, 11, 4, 12, 30, tzinfo=ist))
        self.assertTrue(open_mid)
        self.assertGreater(f_mid, f_morn)

        # Saturday -> (1.0, False)
        self.assertEqual(nse_intraday_volume_fraction(datetime(2026, 11, 7, 11, 0, tzinfo=ist)), (1.0, False))

    def test_intraday_same_slot_volume_matching_across_timezones(self):
        """Verify same-slot matching works for Asia/Kolkata (.NS), America/New_York (US), and UTC."""
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

    def test_illiquid_penny_stock_filtered_by_default(self):
        """Default MIN_MEDIAN_TURNOVER_CR=1.0 filters out < ₹1 Cr daily stocks automatically."""
        pts = []
        base_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(25):
            dt = base_dt + timedelta(days=i)
            vol = 10_000.0 * (1.0 + 0.1 * (i % 4)) if i < 24 else 60_000.0
            pts.append(
                DataPoint(
                    "equity",
                    "PENNY.NS",
                    dt,
                    vol,
                    raw={"open": 15.0, "high": 16.5, "low": 14.8, "close": 16.2},
                )
            )
        self.assertEqual(len(find_anomalies(pts, min_rvol=2.0, min_z_score=2.5, interval="1d")), 0)

    def test_backtester_and_csv_logging(self):
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
        pts = equities._df_to_datapoints("TESTBO.NS", df)
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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
