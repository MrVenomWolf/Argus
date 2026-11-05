"""
main.py
========
Run this to scan NSE equities in batches and print/log statistically
anomalous volume activity + Donchian Channel breakouts.

Examples:
    python main.py                                    -> Nifty 500, 1d interval
    python main.py --universe NIFTY_200 --interval 15m -> Intraday 15m live scan
    python main.py --breakouts-only --rvol 2.5        -> Only confirmed Donchian breakouts
"""

import argparse
import threading
from typing import Callable, List, Optional

from alerts import log_alerts_to_csv, print_alerts
from anomaly_engine import Anomaly, apply_benjamini_hochberg, compute_symbol_pvalue, find_anomalies
import config
from data_sources import equities

SIGNAL_PRIORITY = {
    "LONG BREAKOUT": 0,
    "SHORT BREAKDOWN": 1,
    "BULLISH ACCUM": 2,
    "BEARISH DISTRIB": 3,
    "VOL SPIKE": 4,
}


def fetch_single_equity(ticker: str):
    """Backward-compatible single-ticker fetcher."""
    try:
        return equities.fetch_equity_volume(ticker)
    except Exception:
        return []


def scan_equities(universe: str = "ALL_NSE", interval: str = "1d"):
    """Fetches NSE equity DataPoints in rate-limit-safe batches."""
    tickers = config.get_nse_stocks(universe)
    by_symbol = equities.fetch_equities_batch(tickers=tickers, interval=interval, chunk_size=100)
    all_points = []
    for pts in by_symbol.values():
        all_points.extend(pts)
    return all_points


def run(
    min_rvol: Optional[float] = None,
    min_z_score: Optional[float] = None,
    universe: str = "ALL_NSE",
    interval: str = "1d",
    only_breakouts: bool = False,
    donchian_window: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[Anomaly]:
    tickers = config.get_nse_stocks(universe)
    if not tickers:
        print("No tickers resolved.")
        return []

    print(
        f"Scanning {len(tickers)} NSE equities [{universe} | interval={interval}] "
        f"in rate-limit-safe batches..."
    )
    by_symbol = equities.fetch_equities_batch(
        tickers=tickers,
        interval=interval,
        chunk_size=100,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    if not by_symbol:
        print("No data fetched — check network connection.")
        return []

    all_anomalies: List[Anomaly] = []
    all_p_values: List[float] = []
    for symbol, points in by_symbol.items():
        if cancel_event is not None and cancel_event.is_set():
            break
        if (p_val := compute_symbol_pvalue(points, interval=interval)) is not None:
            all_p_values.append(p_val)
        all_anomalies.extend(
            find_anomalies(
                points,
                min_rvol=min_rvol,
                min_z_score=min_z_score,
                donchian_window=donchian_window,
                interval=interval,
                only_breakouts=only_breakouts,
            )
        )

    # Apply Benjamini-Hochberg FDR calibration over the complete family of tested p-values
    apply_benjamini_hochberg(all_anomalies, total_tested=len(all_p_values), all_p_values=all_p_values)

    # Sort: Actionable Donchian Breakouts first, then highest statistical Z-score
    all_anomalies.sort(
        key=lambda a: (SIGNAL_PRIORITY.get(a.signal, 9), -a.effective_z, -a.rvol)
    )

    print_alerts(all_anomalies, min_rvol=min_rvol, min_z_score=min_z_score)
    if all_anomalies:
        log_alerts_to_csv(all_anomalies)

    return all_anomalies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE Statistical Volume Anomaly & Donchian Breakout Scanner"
    )
    parser.add_argument("--rvol", type=float, default=None, help="Minimum relative volume multiplier (default: 2.0)")
    parser.add_argument("--z-score", type=float, default=None, help="Minimum log Z-score (default: 2.5)")
    parser.add_argument(
        "--universe",
        type=str,
        default="ALL_NSE",
        choices=["ALL_NSE", "NIFTY_500", "NIFTY_200", "SPX_500"],
        help="Equity universe to scan: ALL_NSE, NIFTY_500, NIFTY_200, or SPX_500 (default: ALL_NSE)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["5m", "15m", "30m", "1h", "1d"],
        help="Candle timeframe interval (default: 1d)",
    )
    parser.add_argument(
        "--breakouts-only",
        action="store_true",
        help="Only flag stocks breaking 20-period Donchian High/Low with strong CLV",
    )
    args = parser.parse_args()

    run(
        min_rvol=args.rvol,
        min_z_score=args.z_score,
        universe=args.universe,
        interval=args.interval,
        only_breakouts=args.breakouts_only,
    )


if __name__ == "__main__":
    main()
