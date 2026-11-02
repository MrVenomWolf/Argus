# NSE & S&P 500 Trade Anomaly + Donchian Breakout Detector

A high-throughput, quantitative volume-anomaly and price-structure scanner for the **Indian Stock Market (NSE — 2,635+ listed equities)** and the **US S&P 500 (`SPX_500`)**. It scans entire market universes in rate-limit-safe batches, combining **outlier-resistant log-volume statistics (Parametric Log-Z + Non-Parametric Log-MAD Z)** with **1-bar shifted Donchian Channels, Close Location Value (CLV), Wilder's ATR(14) risk levels, and an embedded Strategy Backtester**.

Includes both a **dark-themed Python Quantitative Desktop Terminal (`gui.py`)** for market-wide screening and **TradingView Pine Script v5 Strategies (`pinescript/`)** for real-time tick-by-tick chart execution on NSE and SPX/US equities.

![NSE Trade Anomaly Detector dashboard](docs/dashboard.png)

---

## Key Features

* **Multi-Market Universe Coverage (`ALL_NSE`, `NIFTY_500`, `NIFTY_200`, `SPX_500`)**: Dynamically fetches all **2,635+ active NSE equities** directly from `EQUITY_L.csv` as well as **`^GSPC`, `SPY`, and all 503 S&P 500 US constituents** (`SPX_500`), automatically formatting prices and turnover in **`₹` (`₹ Cr`)** or **`$` (`$M`)**.
* **Dual-Z Statistical Volume Engine**: Combines **Parametric Log-Volume Z-Score** (`ddof=1`) with **Non-Parametric Log-MAD Modified Z-Score** (Iglewicz & Hoaglin, 1993 in log-space) so a single prior earnings spike inside the 20-bar window cannot mask a fresh institutional breakout (*variance-masking immunity*).
* **Directional Price Structure (Shifted Donchian + CLV)**: Evaluates volume anomalies against a **1-bar shifted 20-period Donchian Channel** (`[t-20 ... t-1]`) and **Close Location Value (`CLV >= 0.65`)** to separate genuine institutional breakouts (`LONG BREAKOUT` / `SHORT BREAKDOWN`) from upper-wick bull traps.
* **Intraday & Live Session Volume Normalization**:
  * **Live Daily Projection (`1d`)**: During live NSE market hours (`09:15–15:30 IST`), partial daily volume is normalized against an empirical NSE intraday U-curve participation profile so breakouts are caught **live** rather than 1 day late.
  * **Intraday Slot Matching (`15m`, `1h`)**: Matches intraday bars against historical volume from the **exact same time-of-day slot** to eliminate morning open U-curve false alarms.
* **Rate-Limit-Safe Batch Pipeline**: Uses chunked multi-ticker batching (`yf.download`) and an in-memory OHLCV DataFrame cache so scanning thousands of tickers avoids Yahoo Finance HTTP 429 bans and clicking any stock in the GUI renders its chart in `<5ms`.
* **Built-In Historical Strategy Backtester**: Simulates taking every Volume-Confirmed Donchian Breakout with a `2× Wilder ATR(14)` initial stop, a Chandelier / Donchian Mid trailing stop, and **0.15% round-trip transaction costs**, directly comparing net performance against raw unfiltered Donchian breakouts.
* **TradingView Pine Script v5 Strategies**: Ready-to-paste scripts for **NSE (`nse_anomaly_donchian.pine`)** and **SPX / US Equities (`spx_anomaly_donchian.pine` with automatic `AMEX:SPY` volume proxy)**.

---

## Quantitative Methodology

![Anomaly detection pipeline](docs/methodology_diagram.svg)

Trading volume is **strictly positive and right-skewed (approximately log-normal)** — a naive percentage-change or simple moving average (SMA) baseline is easily distorted by prior spikes. This engine pairs robust log-space statistics with price-structure confirmation:

| Metric | Formula | Quantitative Purpose |
|---|---|---|
| **Volume Baseline** | `Med_V = median(V_{t-20} ... V_{t-1})` | Outlier-resistant baseline (50% breakdown point), strictly excluding bar `t` |
| **Relative Volume (RVOL)** | `RVOL_t = V_t / Med_V` | Enforces **economic magnitude** vs. typical trading activity |
| **Parametric Log Z-Score** | `Z_log = (ln(1 + V_t) - μ_log) / σ_log` | Corrects for volume's log-normal skewness using Bessel-corrected sample std (`ddof=1`) |
| **Robust Log-MAD Z-Score** | `Z_mad = 0.6745 × (ln(1 + V_t) - Med_log) / MAD_log` | Prevents **variance masking** when a prior earnings spike sits inside the 20-bar window |
| **Shifted Donchian Channel** | `Upper_t = max(H_{t-20...t-1})`, `Lower_t = min(L_{t-20...t-1})` | Strictly shifted by 1 bar so `Close_t > Upper_t` confirms absorption of overhead supply |
| **Close Location Value (CLV)** | `CLV_t = (Close_t - Low_t) / (High_t - Low_t)` | Rejects upper-wick "shooting star" bull traps (requires `CLV_t >= 0.65` for longs) |
| **Wilder's ATR(14)** | `ATR_t = (1/14) × TR_t + (13/14) × ATR_{t-1}` | Sizes volatility-adjusted initial Stop-Loss (`Close - 2×ATR`) and 2R Target (`Close + 4×ATR`) |

**Actionable Breakout Rules:**
* **Volume Anomaly Gate:** `RVOL_t >= 2.0x` **AND** `max(Z_log, Z_mad) >= 2.5σ`
* **Long Breakout (`LONG BREAKOUT`):** `Close_t > Upper_t` **AND** `CLV_t >= 0.65` **AND** `Close_t >= Open_t`
* **Short Breakdown (`SHORT BREAKDOWN`):** `Close_t < Lower_t` **AND** `CLV_t <= 0.35` **AND** `Close_t <= Open_t`
* **Pre-Breakout Accumulation (`BULLISH ACCUM`):** Volume Gate passed **AND** `Close_t >= DonchianMid_t` **AND** `CLV_t >= 0.65`

---

## Live Market Execution Workflow

![Live Execution Workflow](docs/execution_workflow.svg)

Because `DonchianUpper_t = max(High_{t-20 ... t-1})` only depends on the **prior 20 completed bars**, the exact breakout price ceiling is known *before* the current bar finishes:
1. **Spot Pre-Breakout Accumulation (`BULLISH ACCUM`)**: The scanner flags stocks where institutional volume (`RVOL >= 2.0x`, `Z >= 2.5σ`) is already absorbing supply just underneath the 20-period Donchian High.
2. **Park a Resting Buy Stop-Limit (`SL` Buy) Order**: Place a trigger order at `DonchianUpper + 1 tick` directly on the exchange server. If price rejects off resistance, the order never triggers (`0` loss). If an institution sweeps the book in 1 second, the exchange fills your order at the exact breakout level in microseconds.
3. **Manage Risk with Wilder's ATR(14)**: Place the initial hard stop at `Entry - 2.0 × ATR(14)` and trail winners with `max(DonchianMid, HighestClose - 2.5 × ATR(14))`.

---

## Setup & Installation

```bash
git clone https://github.com/MrVenomWolf/NSE-Trade-Anomaly-Detector.git
cd NSE-Trade-Anomaly-Detector
pip install -r requirements.txt
```

---

## Running the Application

### 1. Quantitative Desktop Terminal (Recommended)
```bash
python gui.py
```
* Select your **Universe** (`ALL_NSE` for all 2,635+ listed NSE stocks, `NIFTY_500`, `NIFTY_200`, or `SPX_500` for `^GSPC` + `SPY` + 503 S&P 500 US equities) and **Timeframe** (`15m`, `1h`, or `1d`).
* Click **RUN LIVE SCAN** (and toggle **Donchian Breakouts Only** to filter directly to confirmed breakouts).
* Click any flagged stock in the table to inspect its **Donchian Channel overlay**, **2xATR Stop / 4xATR Target lines**, **20-bar Median Volume Baseline**, and **Historical Strategy Backtest**.

### 2. Command Line Scanner (CLI)
```bash
# Scan all 2,635+ NSE stocks on daily candles
python main.py

# Scan S&P 500 (^GSPC, SPY + 503 US equities) on daily or intraday candles
python main.py --universe SPX_500 --interval 1d

# Scan Nifty 200 on live 15-minute intraday candles for confirmed Donchian breakouts only
python main.py --universe NIFTY_200 --interval 15m --breakouts-only

# Custom thresholds
python main.py --rvol 2.5 --z-score 3.0
```

### 3. TradingView Pine Script v5 Strategies (NSE & SPX / US Markets)
* **For NSE Stocks:** Copy `pinescript/nse_anomaly_donchian.pine`
* **For S&P 500 (`SPX` / `SPY` / `ES1!`) & US Stocks:** Copy `pinescript/spx_anomaly_donchian.pine` *(includes automatic `AMEX:SPY` ETF volume proxy fallback when loaded on the cash `SPX` index, which lacks native tick volume on TradingView, plus bidirectional Long/Short execution)*.
1. In TradingView, open the **Pine Editor**, clear any existing template (`Ctrl+A` then `Backspace`), paste the script, and click **Add to chart**.
2. Open TradingView's **Strategy Tester** tab to inspect live backtest metrics across `5m`, `15m`, `1H`, or `1D` timeframes.

### 4. Automated Test Suite
```bash
python test_suite.py
```

---

## Project Architecture

```text
├── config.py                        <- Thresholds & cached dynamic universe loader (ALL_NSE / Nifty 500 / Nifty 200 / SPX_500)
├── anomaly_engine.py                <- Vectorized math engine (Log-Z, Log-MAD Z, Donchian, CLV, Wilder ATR, Backtester)
├── alerts.py                        <- Console reporting & 17-column CSV alert logger
├── main.py                          <- CLI entry point with batched scanning & signal prioritization
├── gui.py                           <- Non-blocking dark Tkinter + Matplotlib quantitative terminal
├── test_suite.py                    <- 11 unit & integration tests verifying math, edge cases, and pipeline
├── MATH_AND_CHANGES_EXPLAINED.txt   <- Deep-dive mathematical documentation & technical interview Q&A guide
├── docs/
│   ├── dashboard.png                <- v2 Quantitative Desktop Terminal preview
│   ├── methodology_diagram.svg      <- 5-stage statistical & price-structure pipeline diagram
│   └── execution_workflow.svg       <- 3-stage live market SL order execution diagram
├── data_sources/
│   └── equities.py                  <- Rate-limit-resistant batched OHLCV fetcher & in-memory DataFrame cache
└── pinescript/
    ├── nse_anomaly_donchian.pine    <- TradingView Pine Script v5 Strategy for NSE equities
    └── spx_anomaly_donchian.pine    <- TradingView Pine Script v5 Strategy for SPX / SPY / US markets
```

---

## Disclaimer

Built for **educational and quantitative research purposes only**. Statistical volume anomalies and Donchian breakouts represent quantitative filtering rules and historical backtest simulations — they do not constitute financial advice. Always account for live market liquidity, circuit limits, and execution slippage.
