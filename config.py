"""
config.py
=========
Central configuration and dynamic universe loader for ALL publicly listed NSE
equities (`EQUITY_L.csv`), Nifty indices, and the S&P 500 (`SPX_500`), with
local disk caching for instant startup.
"""

from io import StringIO
import json
from pathlib import Path
import time
from typing import List

import pandas as pd
import requests

# Core Statistical & Strategy Thresholds
MIN_Z_SCORE = 2.5
MIN_RVOL = 2.0
BASELINE_WINDOW = 20

DONCHIAN_WINDOW = 20
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0
ATR_TARGET_MULT = 4.0
MIN_CLOSE_STRENGTH = 0.65
MIN_MEDIAN_TURNOVER_CR = 1.0
NSE_ROUND_TRIP_COST_PCT = 0.25   # 0.25% round-trip NSE delivery (0.1% STT/side + exchange/GST/stamp)
STOP_SLIPPAGE_ATR_FRAC = 0.10    # 0.10 * ATR adverse slippage on stop-loss fills
RISK_PER_TRADE_PCT = 1.0         # 1.0% fixed-fractional equity risk per trade sized by 2xATR stop

ALERT_LOG_FILE = "alerts_log.csv"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_TTL_SECONDS = 7 * 86400

UNIVERSE_URLS = {
    "ALL_NSE": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "NIFTY_500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NIFTY_200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "SPX_500": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
}


def get_nse_stocks(universe: str = "ALL_NSE", timeout: int = 10) -> List[str]:
    """
    Fetch every ticker for `universe` ("ALL_NSE", "NIFTY_500", "NIFTY_200", "SPX_500")
    directly from official exchange/index sources with local JSON caching.
    """
    key = universe.upper() if universe.upper() in UNIVERSE_URLS else "ALL_NSE"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"tickers_{key.lower()}.json"

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime < CACHE_TTL_SECONDS):
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    resp = requests.get(
        UNIVERSE_URLS[key],
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        timeout=timeout,
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = df.columns.str.strip()
    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    symbols = df[col].dropna().astype(str).str.strip()

    if key == "SPX_500":
        # Yahoo Finance uses '-' instead of '.' for US share classes (e.g., BRK-B, BF-B)
        us_tickers = [s.replace(".", "-") for s in symbols if s]
        tickers = ["^GSPC", "SPY"] + us_tickers
    else:
        tickers = [f"{s}.NS" for s in symbols if s]

    cache_file.write_text(json.dumps(tickers), encoding="utf-8")
    return tickers


def get_all_nse_stocks() -> List[str]:
    """Fetch every publicly listed stock on NSE (~2,600+ stocks)."""
    return get_nse_stocks("ALL_NSE")


def __getattr__(name: str):
    """PEP 562 lazy module attribute so `import config` triggers zero network I/O."""
    if name == "EQUITY_TICKERS":
        tickers = get_all_nse_stocks()
        globals()["EQUITY_TICKERS"] = tickers
        return tickers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
