"""
data_sources/equities.py
=========================
Batched, rate-limit-resistant NSE OHLCV fetcher with in-memory DataFrame cache.
"""

from datetime import timezone
import threading
import time
from typing import Callable, Dict, List, Optional
import warnings

import pandas as pd

from anomaly_engine import DataPoint

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

_CACHE_LOCK = threading.Lock()
DATAFRAME_CACHE: Dict[str, pd.DataFrame] = {}

DEFAULT_PERIOD_BY_INTERVAL = {
    "5m": "5d",
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


def _df_to_datapoints(ticker: str, df: pd.DataFrame) -> List[DataPoint]:
    if df is None or df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not set(cols).issubset(df.columns):
        return []

    clean_df = df[cols].dropna(subset=["Close", "Volume"]).copy()
    if clean_df.empty:
        return []

    with _CACHE_LOCK:
        DATAFRAME_CACHE[ticker] = clean_df

    points: List[DataPoint] = []
    for row in clean_df.itertuples():
        c, v = float(row.Close), float(row.Volume)
        if pd.isna(c) or pd.isna(v) or c <= 0:
            continue
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
                raw={
                    "open": float(row.Open) if not pd.isna(row.Open) else c,
                    "high": float(row.High) if not pd.isna(row.High) else c,
                    "low": float(row.Low) if not pd.isna(row.Low) else c,
                    "close": c,
                },
            )
        )
    return points


def fetch_equity_volume(
    ticker: str, period: Optional[str] = None, interval: str = "1d"
) -> List[DataPoint]:
    import yfinance as yf

    df = yf.Ticker(ticker).history(
        period=period or DEFAULT_PERIOD_BY_INTERVAL.get(interval, "1y"),
        interval=interval,
    )
    return _df_to_datapoints(ticker, df)


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

    for start in range(0, total, chunk_size):
        if cancel_event and cancel_event.is_set():
            break

        chunk = tickers[start : start + chunk_size]
        raw_df = None
        for attempt in range(2):
            try:
                raw_df = yf.download(
                    tickers=chunk,
                    period=resolved_period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                break
            except Exception:
                if attempt == 0:
                    time.sleep(1.0)

        if raw_df is not None and not raw_df.empty:
            if isinstance(raw_df.columns, pd.MultiIndex):
                available = set(raw_df.columns.get_level_values(0))
                for t in chunk:
                    if t in available and (pts := _df_to_datapoints(t, raw_df[t])):
                        results[t] = pts
            elif len(chunk) == 1:
                if pts := _df_to_datapoints(chunk[0], raw_df):
                    results[chunk[0]] = pts

        if progress_callback:
            progress_callback(min(total, start + len(chunk)), total)

    return results
