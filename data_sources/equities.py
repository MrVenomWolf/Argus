"""
data_sources/equities.py
=========================
Batched, rate-limit-resistant NSE & US OHLCV fetcher with strict OHLCV
validation and in-memory DataFrame caching.
"""

from datetime import timezone
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
import warnings

import pandas as pd

from anomaly_engine import DataPoint

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

_CACHE_LOCK = threading.Lock()
DATAFRAME_CACHE: Dict[str, pd.DataFrame] = {}
LAST_SCAN_STATS: Dict[str, int] = {
    "requested": 0,
    "fetched": 0,
    "skipped": 0,
    "failed": 0,
    "rejected_bars": 0,
}

DEFAULT_PERIOD_BY_INTERVAL = {
    "5m": "1mo",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "1d": "1y",
}


def get_cached_dataframe(symbol: str) -> Optional[pd.DataFrame]:
    with _CACHE_LOCK:
        df = DATAFRAME_CACHE.get(symbol)
        return df.copy() if df is not None else None


def clear_dataframe_cache() -> None:
    with _CACHE_LOCK:
        DATAFRAME_CACHE.clear()


def _df_to_datapoints(ticker: str, df: pd.DataFrame) -> Tuple[List[DataPoint], int]:
    """
    Validates OHLCV rows, rejecting any bar with missing/non-positive prices,
    negative volume, or inverted OHLC relationships (High < Low, High < max(O,C),
    or Low > min(O,C)) rather than silently repairing malformed data.
    Returns (valid_points, rejected_bar_count).
    """
    if df is None or df.empty:
        return [], 0
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not set(cols).issubset(df.columns):
        return [], 0

    sub = df[cols].dropna(subset=["Close", "Volume"]).copy()
    raw_len = len(sub)
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()

    valid_rows = []
    points: List[DataPoint] = []
    rejected = raw_len - len(sub)

    for row in sub.itertuples():
        o, h, l, c, v = row.Open, row.High, row.Low, row.Close, row.Volume
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c) or pd.isna(v):
            rejected += 1
            continue
        o, h, l, c, v = float(o), float(h), float(l), float(c), float(v)
        if min(o, h, l, c) <= 0 or v < 0 or h < max(o, c, l) or l > min(o, c, h):
            rejected += 1
            continue

        valid_rows.append(row.Index)
        ts = row.Index.to_pydatetime() if hasattr(row.Index, "to_pydatetime") else row.Index
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        points.append(
            DataPoint(
                source="equity",
                symbol=ticker,
                timestamp=ts,
                value=v,
                value_label="Volume (shares)",
                raw={"open": o, "high": h, "low": l, "close": c},
            )
        )

    if points:
        clean_df = sub.loc[valid_rows].copy()
        with _CACHE_LOCK:
            DATAFRAME_CACHE[ticker] = clean_df

    return points, rejected


def fetch_equity_volume(
    ticker: str, period: Optional[str] = None, interval: str = "1d"
) -> List[DataPoint]:
    import yfinance as yf

    df = yf.Ticker(ticker).history(
        period=period or DEFAULT_PERIOD_BY_INTERVAL.get(interval, "1y"),
        interval=interval,
    )
    pts, _ = _df_to_datapoints(ticker, df)
    return pts


def fetch_equities_batch(
    tickers: List[str],
    period: Optional[str] = None,
    interval: str = "1d",
    chunk_size: int = 80,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, List[DataPoint]]:
    import yfinance as yf

    resolved_period = period or DEFAULT_PERIOD_BY_INTERVAL.get(interval, "1y")
    results: Dict[str, List[DataPoint]] = {}
    total = len(tickers)
    failed_count = 0
    rejected_bars_total = 0

    for start in range(0, total, chunk_size):
        if cancel_event and cancel_event.is_set():
            break

        chunk = tickers[start : start + chunk_size]
        raw_df = None
        for attempt in range(3):
            try:
                raw_df = yf.download(
                    tickers=chunk,
                    period=resolved_period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.8 * (2 ** attempt))

        if raw_df is None or raw_df.empty:
            failed_count += len(chunk)
        elif isinstance(raw_df.columns, pd.MultiIndex):
            available = set(raw_df.columns.get_level_values(0))
            for t in chunk:
                if t in available:
                    pts, rej = _df_to_datapoints(t, raw_df[t])
                    rejected_bars_total += rej
                    if pts:
                        results[t] = pts
        elif len(chunk) == 1:
            pts, rej = _df_to_datapoints(chunk[0], raw_df)
            rejected_bars_total += rej
            if pts:
                results[chunk[0]] = pts

        if progress_callback:
            progress_callback(min(total, start + len(chunk)), total)

    fetched_count = len(results)
    LAST_SCAN_STATS.update({
        "requested": total,
        "fetched": fetched_count,
        "skipped": max(0, total - fetched_count - failed_count),
        "failed": failed_count,
        "rejected_bars": rejected_bars_total,
    })
    return results
