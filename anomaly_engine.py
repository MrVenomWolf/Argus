"""
anomaly_engine.py
==================
Quantitative Statistical Anomaly & Donchian Breakout Engine using vectorized
NumPy and Pandas primitives.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import config

IST = timezone(timedelta(hours=5, minutes=30))
NORMAL_CONSISTENCY_FACTOR = 0.6744897501960817  # Phi^-1(0.75)


@dataclass
class DataPoint:
    source: str
    symbol: str
    timestamp: datetime
    value: float
    value_label: str = "Volume"
    raw: dict = field(default_factory=dict)


@dataclass
class Anomaly:
    point: DataPoint
    baseline_median: float
    rvol: float
    z_score: float
    modified_z_score: float
    close_price: float = 0.0
    donchian_upper: float = 0.0
    donchian_lower: float = 0.0
    donchian_mid: float = 0.0
    atr: float = 0.0
    close_location_value: float = 0.5
    channel_width_pct: float = 0.0
    signal: str = "VOL SPIKE"
    stop_loss: float = 0.0
    target_price: float = 0.0
    turnover_cr: float = 0.0
    is_live_projected: bool = False

    @property
    def currency(self) -> str:
        return "\u20b9" if self.point.symbol.endswith(".NS") else "$"

    @property
    def effective_z(self) -> float:
        return max(self.z_score, self.modified_z_score)

    @property
    def delta_pct(self) -> str:
        return f"{(self.rvol - 1.0) * 100:+.1f}%"

    @property
    def is_actionable_breakout(self) -> bool:
        return self.signal in ("LONG BREAKOUT", "SHORT BREAKDOWN")

    def __str__(self) -> str:
        cur = self.currency
        price_str = f"{cur}{self.close_price:,.2f}" if self.close_price > 0 else "N/A"
        proj_tag = " [LIVE-PROJ]" if self.is_live_projected else ""
        return (
            f"[{self.signal}] {self.point.symbol:<13} @ {price_str:<10} | "
            f"RVOL {self.rvol:5.2f}x | Log-Z {self.z_score:+5.2f}\u03c3 | "
            f"MAD-Z {self.modified_z_score:+5.2f}\u03c3 | CLV {self.close_location_value*100:4.0f}% | "
            f"Donchian [{self.donchian_lower:,.1f} - {self.donchian_upper:,.1f}] | "
            f"Stop {cur}{self.stop_loss:,.2f} / Target {cur}{self.target_price:,.2f}{proj_tag}"
        )


@dataclass
class BacktestResult:
    symbol: str
    total_bars: int
    trades: int
    win_rate_pct: float
    profit_factor: float
    avg_r_multiple: float
    net_return_pct: float
    max_drawdown_pct: float
    unfiltered_trades: int
    unfiltered_win_rate_pct: float
    unfiltered_net_return_pct: float


def nse_intraday_volume_fraction(now_ist: Optional[datetime] = None) -> Tuple[float, bool]:
    """Expected cumulative fraction of daily NSE volume completed at `now_ist` (09:15-15:30 IST)."""
    now_ist = (now_ist or datetime.now(IST)).astimezone(IST)
    mins = now_ist.hour * 60 + now_ist.minute
    if now_ist.weekday() >= 5 or not (555 <= mins < 930):
        return 1.0, False

    t = max(5.0, float(mins - 555)) / 375.0
    frac = max(0.08, min(1.0, t + 0.14 * math.sin(2.0 * math.pi * t)))
    return frac, True


def calculate_robust_metrics(history: np.ndarray, current: float) -> Tuple[float, float, float, float]:
    """Computes Median Baseline, RVOL, Parametric Log-Z, and Non-Parametric Log-MAD Z-score."""
    if not np.isfinite(current) or current <= 0 or history.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    pos = history[np.isfinite(history) & (history > 0)]
    if pos.size < 5:
        return 0.0, 0.0, 0.0, 0.0

    baseline_med = float(np.median(pos))
    rvol = current / baseline_med if baseline_med > 0 else 1.0

    log_cur = float(np.log1p(current))
    log_hist = np.log1p(pos)

    log_std = float(np.std(log_hist, ddof=1))
    z_score = (log_cur - float(np.mean(log_hist))) / log_std if log_std > 1e-9 else 0.0

    log_med = float(np.median(log_hist))
    log_mad = float(np.median(np.abs(log_hist - log_med)))
    modified_z = (NORMAL_CONSISTENCY_FACTOR * (log_cur - log_med)) / log_mad if log_mad > 1e-9 else z_score

    return baseline_med, rvol, z_score, modified_z


def compute_wilder_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """Wilder's ATR using True Range and RMA smoothing (alpha = 1/period)."""
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float)

    prev_c = np.roll(closes, 1)
    prev_c[0] = closes[0]
    tr = np.maximum.reduce([highs - lows, np.abs(highs - prev_c), np.abs(lows - prev_c)])

    atr = pd.Series(tr).expanding().mean().to_numpy(dtype=float, copy=True)
    if n > period:
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def compute_donchian_channels(
    highs: np.ndarray, lows: np.ndarray, window: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 1-bar shifted Donchian Channels [t-window ... t-1] via Pandas rolling."""
    if len(highs) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    upper = pd.Series(highs, dtype=float).shift(1).rolling(window, min_periods=1).max().bfill().to_numpy()
    lower = pd.Series(lows, dtype=float).shift(1).rolling(window, min_periods=1).min().bfill().to_numpy()
    return upper, lower, 0.5 * (upper + lower)


def _select_baseline_history(
    points: List[DataPoint], window: int, interval: str = "1d"
) -> Tuple[np.ndarray, float, bool]:
    latest = points[-1]
    raw_vol = float(latest.value)
    default_hist = np.array([p.value for p in points[-(window + 1) : -1]], dtype=float)

    if interval.endswith(("m", "h")):
        t_ist = latest.timestamp.astimezone(IST) if latest.timestamp.tzinfo else latest.timestamp
        slot = (t_ist.hour, t_ist.minute)
        same_slot = [
            float(p.value)
            for p in points[:-1]
            if ((p.timestamp.astimezone(IST) if p.timestamp.tzinfo else p.timestamp).hour, p.timestamp.minute) == slot
            and p.value > 0
        ]
        if len(same_slot) >= 5:
            return np.array(same_slot[-window:], dtype=float), raw_vol, False
        return default_hist, raw_vol, False

    if latest.symbol.endswith(".NS"):
        now_ist = datetime.now(IST)
        latest_date = (latest.timestamp.astimezone(IST) if latest.timestamp.tzinfo else latest.timestamp).date()
        if latest_date == now_ist.date():
            frac, is_open = nse_intraday_volume_fraction(now_ist)
            if is_open and 0.08 <= frac < 0.98:
                return default_hist, raw_vol / frac, True

    return default_hist, raw_vol, False


def classify_signal(
    close_price: float,
    high_price: float,
    low_price: float,
    open_price: float,
    donchian_upper: float,
    donchian_lower: float,
    donchian_mid: float,
    clv: float,
    min_clv: float = None,
) -> str:
    min_clv = min_clv if min_clv is not None else config.MIN_CLOSE_STRENGTH
    if donchian_upper <= 0 or close_price <= 0:
        return "VOL SPIKE"
    if close_price > donchian_upper and clv >= min_clv and close_price >= open_price:
        return "LONG BREAKOUT"
    if close_price < donchian_lower and clv <= (1.0 - min_clv) and close_price <= open_price:
        return "SHORT BREAKDOWN"
    if close_price >= donchian_mid and clv >= min_clv and close_price > open_price:
        return "BULLISH ACCUM"
    if close_price < donchian_mid and clv <= (1.0 - min_clv) and close_price < open_price:
        return "BEARISH DISTRIB"
    return "VOL SPIKE"


def find_anomalies(
    points: List[DataPoint],
    min_rvol: float = None,
    min_z_score: float = None,
    window: int = None,
    donchian_window: int = None,
    interval: str = "1d",
    only_breakouts: bool = False,
    min_turnover_cr: float = None,
) -> List[Anomaly]:
    window = window or config.BASELINE_WINDOW
    donchian_window = donchian_window or config.DONCHIAN_WINDOW
    min_rvol = min_rvol if min_rvol is not None else config.MIN_RVOL
    min_z_score = min_z_score if min_z_score is not None else config.MIN_Z_SCORE
    min_turnover_cr = min_turnover_cr if min_turnover_cr is not None else config.MIN_MEDIAN_TURNOVER_CR

    if not points or len(points) < window + 1:
        return []

    points = sorted(points, key=lambda p: p.timestamp)
    hist_vols, eff_vol, is_proj = _select_baseline_history(points, window=window, interval=interval)
    baseline_med, rvol, z_score, modified_z = calculate_robust_metrics(hist_vols, eff_vol)

    if max(z_score, modified_z) < min_z_score or rvol < min_rvol:
        return []

    ohlc = np.array(
        [
            (
                float(c := p.raw.get("close", 0.0)),
                float(p.raw.get("high", c)),
                float(p.raw.get("low", c)),
                float(p.raw.get("open", c)),
            )
            for p in points
        ],
        dtype=float,
    )
    closes, highs, lows, opens = ohlc.T
    close_t, high_t, low_t, open_t = closes[-1], highs[-1], lows[-1], opens[-1]

    turnover_cr = (baseline_med * close_t) / 1e7 if close_t > 0 else 0.0
    if interval == "1d" and min_turnover_cr > 0 and close_t > 0 and turnover_cr < min_turnover_cr:
        return []

    d_upper = float(np.max(highs[-(donchian_window + 1) : -1]))
    d_lower = float(np.min(lows[-(donchian_window + 1) : -1]))
    d_mid = 0.5 * (d_upper + d_lower)
    width_pct = ((d_upper - d_lower) / d_mid) * 100.0 if d_mid > 0 else 0.0

    rng = high_t - low_t
    clv = float(np.clip((close_t - low_t) / rng, 0.0, 1.0)) if rng > 1e-9 else 0.5
    atr_val = float(compute_wilder_atr(highs, lows, closes, period=config.ATR_PERIOD)[-1])

    signal = classify_signal(close_t, high_t, low_t, open_t, d_upper, d_lower, d_mid, clv)
    if only_breakouts and signal not in ("LONG BREAKOUT", "SHORT BREAKDOWN"):
        return []

    direction = -1.0 if signal == "SHORT BREAKDOWN" else 1.0
    stop_loss = max(0.0, close_t - direction * config.ATR_STOP_MULT * atr_val)
    target_price = max(0.0, close_t + direction * config.ATR_TARGET_MULT * atr_val)

    return [
        Anomaly(
            point=points[-1],
            baseline_median=baseline_med,
            rvol=rvol,
            z_score=z_score,
            modified_z_score=modified_z,
            close_price=close_t,
            donchian_upper=d_upper,
            donchian_lower=d_lower,
            donchian_mid=d_mid,
            atr=atr_val,
            close_location_value=clv,
            channel_width_pct=width_pct,
            signal=signal,
            stop_loss=stop_loss,
            target_price=target_price,
            turnover_cr=turnover_cr,
            is_live_projected=is_proj,
        )
    ]


def backtest_donchian_volume(
    df: pd.DataFrame,
    symbol: str = "",
    window: int = None,
    donchian_window: int = None,
    min_rvol: float = None,
    min_z_score: float = None,
    cost_pct: float = None,
) -> Optional[BacktestResult]:
    if df is None or df.empty:
        return None

    window = window or config.BASELINE_WINDOW
    donchian_window = donchian_window or config.DONCHIAN_WINDOW
    min_rvol = min_rvol if min_rvol is not None else config.MIN_RVOL
    min_z_score = min_z_score if min_z_score is not None else config.MIN_Z_SCORE
    cost_pct = cost_pct if cost_pct is not None else config.NSE_ROUND_TRIP_COST_PCT

    start_idx = max(window, donchian_window, config.ATR_PERIOD) + 1
    if len(df) < start_idx + 4:
        return None

    closes, highs, lows, opens, vols = (
        pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        for col in ("Close", "High", "Low", "Open", "Volume")
    )
    atr = compute_wilder_atr(highs, lows, closes, period=config.ATR_PERIOD)
    d_upper, _, d_mid = compute_donchian_channels(highs, lows, window=donchian_window)

    def _simulate(require_vol: bool):
        in_pos, entry, risk, stop, hi_close = False, 0.0, 0.0, 0.0, 0.0
        r_mults, rets = [], []
        eq, peak, max_dd = 1.0, 1.0, 0.0

        for i in range(start_idx, len(closes)):
            c, h, l, o, v = closes[i], highs[i], lows[i], opens[i], vols[i]
            if not np.isfinite(c) or c <= 0:
                continue

            if in_pos:
                hit_stop = l <= stop
                if hit_stop or c < d_mid[i] or i == len(closes) - 1:
                    exit_px = (min(o, stop) if o < stop else stop) if hit_stop else c
                    net_ret = ((exit_px - entry) / entry) - (cost_pct / 100.0)
                    rets.append(net_ret * 100.0)
                    r_mults.append((net_ret * entry) / risk if risk > 1e-9 else 0.0)
                    eq *= 1.0 + net_ret
                    peak = max(peak, eq)
                    max_dd = max(max_dd, (peak - eq) / peak)
                    in_pos = False
                else:
                    hi_close = max(hi_close, c)
                    stop = max(stop, d_mid[i], hi_close - 2.5 * atr[i])
                continue

            if c <= d_upper[i] or c < o:
                continue
            if require_vol:
                clv = (c - l) / (h - l) if (h - l) > 1e-9 else 0.5
                if clv < config.MIN_CLOSE_STRENGTH:
                    continue
                _, rvol_i, z_i, mod_z_i = calculate_robust_metrics(vols[i - window : i], v)
                if rvol_i < min_rvol or max(z_i, mod_z_i) < min_z_score:
                    continue

            in_pos, entry, hi_close = True, c, c
            risk = max(c * 0.01, config.ATR_STOP_MULT * atr[i])
            stop = entry - risk

        if not rets:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0
        wins = [r for r in rets if r > 0]
        loss_sum = sum(-r for r in rets if r <= 0)
        pf = (sum(wins) / loss_sum) if loss_sum > 1e-9 else (9.99 if wins else 0.0)
        return len(rets), len(wins) / len(rets) * 100.0, pf, float(np.mean(r_mults)), (eq - 1.0) * 100.0, max_dd * 100.0

    t, wr, pf, avg_r, net, dd = _simulate(True)
    ut, uwr, _, _, unet, _ = _simulate(False)
    return BacktestResult(symbol, len(closes), t, wr, pf, avg_r, net, dd, ut, uwr, unet)
