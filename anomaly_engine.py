"""
anomaly_engine.py
==================
Quantitative Statistical Volume-Breakout & Strategy Engine using vectorized
NumPy, Pandas, and SciPy primitives.
"""

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t as scipy_student_t

import config

IST = ZoneInfo("Asia/Kolkata")
NY_TZ = ZoneInfo("America/New_York")
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
    z_score: float                      # Conventional parametric log-volume Z-score
    modified_z_score: float             # Non-parametric log-MAD heuristic anti-masking score
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
    p_value: float = 1.0                # Parametric predictive Student-t p-value (from z_score)
    fdr_q: Optional[float] = None       # Benjamini-Hochberg adjusted q-value from parametric p_value

    @property
    def currency(self) -> str:
        return "\u20b9" if self.point.symbol.endswith(".NS") else "$"

    @property
    def effective_z(self) -> float:
        """
        Uses Parametric Log-Z by default, promoting the heuristic Log-MAD score
        only when a prior outlier in the window inflated sample std (modified_z > 1.40 * z_score).
        Note: `p_value` and `fdr_q` always report the parametric Student-t significance of `z_score`.
        """
        if self.modified_z_score > self.z_score * 1.40 and self.z_score > 0.5:
            return self.modified_z_score
        return self.z_score

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
    """Long-only historical backtest metrics comparing volume-confirmed vs raw Donchian breakouts."""
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
    """
    Parametric U-curve approximation of cumulative daily NSE volume (09:15-15:30 IST).
    Requires at least 45 minutes elapsed (>= 10:00 IST, mins >= 600) before projecting.
    """
    now_ist = (now_ist or datetime.now(IST)).astimezone(IST)
    mins = now_ist.hour * 60 + now_ist.minute
    if now_ist.weekday() >= 5 or not (600 <= mins < 930):
        return 1.0, False

    t = float(mins - 555) / 375.0
    frac = max(0.20, min(1.0, t + 0.14 * math.sin(2.0 * math.pi * t)))
    return frac, True


def student_t_one_sided_pvalue(t_stat: float, df: int = 19) -> float:
    """Exact one-sided Student-t survival probability P(T >= t_stat) via `scipy.stats.t.sf`."""
    if not np.isfinite(t_stat) or df < 1:
        return 1.0
    return float(scipy_student_t.sf(t_stat, df=df))


def apply_benjamini_hochberg(
    anomalies: List[Anomaly],
    total_tested: int,
    all_p_values: Optional[List[float]] = None,
) -> List[Anomaly]:
    """
    Computes Benjamini-Hochberg (1995) False Discovery Rate q-values across `total_tested` symbols.
    Tied p-values are handled conservatively as a group using `side="left"` ranks so all tied
    observations receive the same conservative adjusted q-value.
    """
    if not anomalies:
        return anomalies

    if all_p_values and len(all_p_values) >= len(anomalies):
        p_arr = np.sort(np.asarray(all_p_values, dtype=float))
    else:
        p_arr = np.sort(np.asarray([a.p_value for a in anomalies], dtype=float))

    m = max(total_tested, len(p_arr))
    # Conservative tie-handling: assign the first (smallest) 1-based rank to all members of a tie
    ranks = np.searchsorted(p_arr, p_arr, side="left").astype(float) + 1.0
    raw_q = (p_arr * m) / ranks
    bh_q = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0.0, 1.0)

    for a in anomalies:
        idx = int(np.searchsorted(p_arr, a.p_value, side="left"))
        idx = max(0, min(len(p_arr) - 1, idx))
        a.fdr_q = float(bh_q[idx])
    return anomalies


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

    upper = pd.Series(highs, dtype=float).shift(1).rolling(window, min_periods=1).max().fillna(highs[0]).to_numpy()
    lower = pd.Series(lows, dtype=float).shift(1).rolling(window, min_periods=1).min().fillna(lows[0]).to_numpy()
    return upper, lower, 0.5 * (upper + lower)


def _to_exchange_hm(ts: datetime, tz: ZoneInfo) -> Tuple[int, int]:
    t = ts.astimezone(tz) if ts.tzinfo else ts
    return t.hour, t.minute


def _select_baseline_history(
    points: List[DataPoint], window: int, interval: str = "1d"
) -> Tuple[np.ndarray, float, bool]:
    latest = points[-1]
    raw_vol = float(latest.value)
    default_hist = np.array([p.value for p in points[-(window + 1) : -1]], dtype=float)
    tz = IST if latest.symbol.endswith(".NS") else NY_TZ

    if interval.endswith(("m", "h")):
        slot = _to_exchange_hm(latest.timestamp, tz)
        same_slot = [
            float(p.value)
            for p in points[:-1]
            if _to_exchange_hm(p.timestamp, tz) == slot and p.value > 0
        ]
        if len(same_slot) >= 5:
            return np.array(same_slot[-window:], dtype=float), raw_vol, False
        return default_hist, raw_vol, False

    if latest.symbol.endswith(".NS"):
        now_ist = datetime.now(IST)
        latest_date = (latest.timestamp.astimezone(IST) if latest.timestamp.tzinfo else latest.timestamp).date()
        if latest_date == now_ist.date():
            frac, is_open = nse_intraday_volume_fraction(now_ist)
            if is_open and 0.20 <= frac < 0.98:
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


def _evaluate_bar_volume_and_turnover(
    hist_vols: np.ndarray,
    prior_closes: np.ndarray,
    eff_vol: float,
    close_t: float,
    min_rvol: float,
    min_z_score: float,
    min_turnover_cr: float,
    is_daily: bool,
) -> Tuple[bool, float, float, float, float, float]:
    """
    Shared volume & turnover evaluation gate called by BOTH `find_anomalies` (live scanner)
    and `backtest_donchian_volume` (historical backtester) to guarantee signal parity.
    Returns (passed_gate, baseline_med, rvol, z_score, modified_z, turnover_cr).
    """
    baseline_med, rvol, z_score, modified_z = calculate_robust_metrics(hist_vols, eff_vol)
    eff_z = modified_z if (modified_z > z_score * 1.40 and z_score > 0.5) else z_score

    if len(prior_closes) == len(hist_vols) and np.all(prior_closes > 0):
        turnover_cr = float(np.median(hist_vols * prior_closes)) / 1e7
    else:
        turnover_cr = (baseline_med * close_t) / 1e7 if close_t > 0 else 0.0

    if eff_z < min_z_score or rvol < min_rvol:
        return False, baseline_med, rvol, z_score, modified_z, turnover_cr
    if is_daily and min_turnover_cr > 0 and close_t > 0 and turnover_cr < min_turnover_cr:
        return False, baseline_med, rvol, z_score, modified_z, turnover_cr
    return True, baseline_med, rvol, z_score, modified_z, turnover_cr


def compute_symbol_pvalue(
    points: List[DataPoint],
    window: int = None,
    interval: str = "1d",
) -> Optional[float]:
    """Computes the parametric predictive Student-t p-value for a symbol."""
    window = window if (window is not None and window > 0) else config.BASELINE_WINDOW
    if not points or len(points) < window + 1:
        return None
    points = sorted(points, key=lambda p: p.timestamp)
    hist_vols, eff_vol, _ = _select_baseline_history(points, window=window, interval=interval)
    n_valid = int(np.sum(np.isfinite(hist_vols) & (hist_vols > 0)))
    if n_valid < 5:
        return None
    _, _, z_score, _ = calculate_robust_metrics(hist_vols, eff_vol)
    t_stat = z_score / math.sqrt(1.0 + 1.0 / n_valid)
    return student_t_one_sided_pvalue(t_stat, df=max(4, n_valid - 1))


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
    window = window if (window is not None and window > 0) else config.BASELINE_WINDOW
    donchian_window = (
        donchian_window if (donchian_window is not None and donchian_window > 0) else config.DONCHIAN_WINDOW
    )
    min_rvol = min_rvol if min_rvol is not None else config.MIN_RVOL
    min_z_score = min_z_score if min_z_score is not None else config.MIN_Z_SCORE
    min_turnover_cr = min_turnover_cr if min_turnover_cr is not None else config.MIN_MEDIAN_TURNOVER_CR

    if not points or len(points) < max(window, donchian_window) + 1:
        return []

    points = sorted(points, key=lambda p: p.timestamp)
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

    hist_vols, eff_vol, is_proj = _select_baseline_history(points, window=window, interval=interval)
    prior_closes = closes[-(window + 1) : -1]

    passed, baseline_med, rvol, z_score, modified_z, turnover_cr = _evaluate_bar_volume_and_turnover(
        hist_vols=hist_vols,
        prior_closes=prior_closes,
        eff_vol=eff_vol,
        close_t=close_t,
        min_rvol=min_rvol,
        min_z_score=min_z_score,
        min_turnover_cr=min_turnover_cr,
        is_daily=(interval == "1d"),
    )
    if not passed:
        return []

    n_valid = int(np.sum(np.isfinite(hist_vols) & (hist_vols > 0)))
    t_stat = z_score / math.sqrt(1.0 + 1.0 / n_valid) if n_valid > 0 else 0.0
    p_val = student_t_one_sided_pvalue(t_stat, df=max(4, n_valid - 1))

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

    direction = -1.0 if signal in ("SHORT BREAKDOWN", "BEARISH DISTRIB") else 1.0
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
            p_value=p_val,
            fdr_q=None,
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
    use_target: bool = False,
    min_turnover_cr: float = None,
) -> Optional[BacktestResult]:
    """
    Long-only historical backtester evaluating `LONG BREAKOUT` signals using the
    shared `_evaluate_bar_volume_and_turnover` and `classify_signal` rules.
      - Default exit (`use_target=False`): 2.5*ATR Chandelier trail + next-open DonchianMid exit.
      - Target exit (`use_target=True`)  : Also exits at 4.0*ATR target (with stop-first precedence).
    """
    if df is None or df.empty:
        return None

    window = window or config.BASELINE_WINDOW
    donchian_window = donchian_window or config.DONCHIAN_WINDOW
    min_rvol = min_rvol if min_rvol is not None else config.MIN_RVOL
    min_z_score = min_z_score if min_z_score is not None else config.MIN_Z_SCORE
    cost_pct = cost_pct if cost_pct is not None else config.NSE_ROUND_TRIP_COST_PCT
    min_turnover_cr = (
        min_turnover_cr
        if min_turnover_cr is not None
        else (config.MIN_MEDIAN_TURNOVER_CR if symbol.endswith(".NS") else 0.0)
    )

    start_idx = max(window, donchian_window, config.ATR_PERIOD) + 1
    if len(df) < start_idx + 4:
        return None

    closes, highs, lows, opens, vols = (
        pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        for col in ("Close", "High", "Low", "Open", "Volume")
    )
    atr = compute_wilder_atr(highs, lows, closes, period=config.ATR_PERIOD)
    d_upper, d_lower, d_mid = compute_donchian_channels(highs, lows, window=donchian_window)

    tz = IST if symbol.endswith(".NS") else NY_TZ
    slots: Optional[List[Tuple[int, int]]] = None
    if hasattr(df.index, "tz") or isinstance(df.index, pd.DatetimeIndex):
        sample_slots = [
            _to_exchange_hm(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts, tz)
            for ts in df.index[: min(len(df), 30)]
        ]
        if len(set(sample_slots)) > 1:
            slots = [
                _to_exchange_hm(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts, tz)
                for ts in df.index
            ]
    is_daily = slots is None

    def _simulate(require_vol: bool):
        in_pos, pending_entry, pending_exit = False, False, False
        entry, risk, stop, target, hi_close, pos_weight = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        r_mults, rets = [], []
        eq, peak, max_dd = 1.0, 1.0, 0.0
        slip_frac = getattr(config, "STOP_SLIPPAGE_ATR_FRAC", 0.10)
        risk_pct = getattr(config, "RISK_PER_TRADE_PCT", 1.0) / 100.0

        for i in range(start_idx, len(closes)):
            c, h, l, o, v = closes[i], highs[i], lows[i], opens[i], vols[i]
            if not np.isfinite(c) or c <= 0:
                continue

            if pending_entry and not in_pos:
                pending_entry, pending_exit = False, False
                entry = o if (np.isfinite(o) and o > 0) else c
                risk = max(entry * 0.01, config.ATR_STOP_MULT * atr[i - 1])
                stop = entry - risk
                target = entry + config.ATR_TARGET_MULT * atr[i - 1]
                hi_close = entry
                stop_dist_frac = risk / entry
                pos_weight = min(1.0, risk_pct / stop_dist_frac) if stop_dist_frac > 1e-9 else 1.0
                in_pos = True

            if in_pos:
                prev_atr = atr[i - 1]
                hit_stop = l <= stop
                hit_target = use_target and (not hit_stop) and (h >= target)
                is_last = i == len(closes) - 1
                if pending_exit or hit_stop or hit_target or is_last:
                    if pending_exit:
                        exit_px = o if (np.isfinite(o) and o > 0) else c
                    elif hit_stop:
                        raw_exit = min(o, stop) if o < stop else stop
                        exit_px = max(0.01, raw_exit - slip_frac * prev_atr)
                    elif hit_target:
                        exit_px = max(o, target) if o > target else target
                    else:
                        exit_px = c
                    net_ret = ((exit_px - entry) / entry) - (cost_pct / 100.0)
                    r_mult = (net_ret * entry) / risk if risk > 1e-9 else 0.0
                    sized_ret = pos_weight * net_ret
                    rets.append(sized_ret * 100.0)
                    r_mults.append(r_mult)
                    eq *= 1.0 + sized_ret
                    peak = max(peak, eq)
                    max_dd = max(max_dd, (peak - eq) / peak)
                    in_pos, pending_exit = False, False
                else:
                    # Track marked-to-market unrealized peak and drawdown while position is open
                    mtm_close = eq * (1.0 + pos_weight * ((c - entry) / entry))
                    mtm_low = eq * (1.0 + pos_weight * ((l - entry) / entry))
                    peak = max(peak, mtm_close)
                    max_dd = max(max_dd, (peak - mtm_low) / peak, (peak - mtm_close) / peak)

                    hi_close = max(hi_close, c)
                    stop = max(stop, hi_close - 2.5 * atr[i])
                    if c < d_mid[i]:
                        pending_exit = True
                continue

            if i >= len(closes) - 1:
                continue

            if not require_vol:
                if c > d_upper[i] and c >= o:
                    pending_entry = True
                continue

            if slots is not None:
                slot_vols = [vols[j] for j in range(i) if slots[j] == slots[i] and vols[j] > 0]
                hist_v = np.array(slot_vols[-window:], dtype=float) if len(slot_vols) >= 5 else vols[i - window : i]
            else:
                hist_v = vols[i - window : i]
            prior_c = closes[i - window : i]

            passed, _, _, _, _, _ = _evaluate_bar_volume_and_turnover(
                hist_vols=hist_v,
                prior_closes=prior_c,
                eff_vol=v,
                close_t=c,
                min_rvol=min_rvol,
                min_z_score=min_z_score,
                min_turnover_cr=min_turnover_cr,
                is_daily=is_daily,
            )
            if not passed:
                continue

            clv = float(np.clip((c - l) / (h - l), 0.0, 1.0)) if (h - l) > 1e-9 else 0.5
            sig = classify_signal(c, h, l, o, d_upper[i], d_lower[i], d_mid[i], clv)
            if sig == "LONG BREAKOUT":
                pending_entry = True

        if not rets:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0
        wins = [r for r in rets if r > 0]
        loss_sum = sum(-r for r in rets if r <= 0)
        pf = (sum(wins) / loss_sum) if loss_sum > 1e-9 else (float("inf") if wins else 0.0)
        return len(rets), len(wins) / len(rets) * 100.0, pf, float(np.mean(r_mults)), (eq - 1.0) * 100.0, max_dd * 100.0

    t, wr, pf, avg_r, net, dd = _simulate(True)
    ut, uwr, _, _, unet, _ = _simulate(False)
    return BacktestResult(symbol, len(closes), t, wr, pf, avg_r, net, dd, ut, uwr, unet)
