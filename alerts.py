"""
alerts.py
==========
Console output formatting and CSV logging for detected volume anomalies and
Donchian breakouts.
"""

import csv
import os
from datetime import datetime, timezone
from typing import List

from anomaly_engine import Anomaly
import config


def print_alerts(anomalies: List[Anomaly], min_rvol: float = None, min_z_score: float = None) -> None:
    if not anomalies:
        print("No trades crossed the threshold this run. Nothing to flag.")
        return

    active_rvol = min_rvol if min_rvol is not None else config.MIN_RVOL
    active_z = min_z_score if min_z_score is not None else config.MIN_Z_SCORE

    breakout_count = sum(1 for a in anomalies if a.is_actionable_breakout)
    print(f"\n{'=' * 95}")
    print(
        f"  {len(anomalies)} ANOMALOUS VOLUME EVENT(S) FOUND "
        f"({breakout_count} DONCHIAN BREAKOUTS | RVOL >= {active_rvol:.1f}x, Z >= {active_z}\u03c3)"
    )
    print(f"{'=' * 95}")
    for a in anomalies:
        print(f"  {a}")


def log_alerts_to_csv(anomalies: List[Anomaly], path: str = None) -> None:
    path = path or config.ALERT_LOG_FILE
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "detected_at",
                "source",
                "symbol",
                "signal",
                "trade_timestamp",
                "close_price",
                "value",
                "baseline_median",
                "rvol",
                "z_score",
                "modified_z_score",
                "clv_pct",
                "donchian_upper",
                "donchian_lower",
                "atr_14",
                "stop_loss",
                "target_2r",
                "p_value",
                "fdr_q",
            ])
        detected_at = datetime.now(timezone.utc).isoformat()
        for a in anomalies:
            q_str = f"{a.fdr_q:.6f}" if a.fdr_q is not None else ""
            writer.writerow([
                detected_at,
                a.point.source,
                a.point.symbol,
                a.signal,
                a.point.timestamp.isoformat(),
                f"{a.close_price:.2f}",
                f"{a.point.value:.0f}",
                f"{a.baseline_median:.0f}",
                f"{a.rvol:.2f}",
                f"{a.z_score:.2f}",
                f"{a.modified_z_score:.2f}",
                f"{a.close_location_value * 100:.1f}",
                f"{a.donchian_upper:.2f}",
                f"{a.donchian_lower:.2f}",
                f"{a.atr:.2f}",
                f"{a.stop_loss:.2f}",
                f"{a.target_price:.2f}",
                f"{a.p_value:.6f}",
                q_str,
            ])
    print(f"\nLogged {len(anomalies)} alert(s) to {path}")
