# NSE Trade Anomaly & Donchian Breakout Detector

A high-throughput, quantitative volume-anomaly and price-structure scanner for the **Indian Stock Market (NSE)**. It dynamically scans **2,600+ publicly listed NSE equities** (`EQUITY_L.csv`) in rate-limit-safe batches, combining **outlier-resistant log-volume statistics (Parametric Log-Z + Non-Parametric Log-MAD Z)** with **1-bar shifted Donchian Channels, Close Location Value (CLV), Wilder's ATR(14) risk levels, and an embedded Strategy Backtester**.

Includes both a **dark-themed Python Desktop Terminal (`gui.py`)** for full-market screening and a **TradingView Pine Script v5 Strategy (`pinescript/nse_anomaly_donchian.pine`)** for real-time tick-by-tick chart execution.

![NSE Trade Anomaly Detector dashboard](docs/dashboard.png)

---

## Key Features

* **Dual-Z Statistical Volume Engine**: Combines **Parametric Log-Volume Z-Score** (`ddof=1`) with **Non-Parametric Log-MAD Modified Z-Score** (Iglewicz & Hoaglin, 1993 in log-space) so a single prior earnings spike inside the 20-bar window cannot mask a fresh institutional breakout (*variance-masking immunity*).
* **Directional Price Structure (Shifted Donchian + CLV)**: Evaluates volume anomalies against a **1-bar shifted 20-period Donchian Channel** (`[t-20 ... t-1]`) and **Close Location Value (`CLV >= 0.65`)** to separate genuine institutional breakouts (`LONG BREAKOUT` / `SHORT BREAKDOWN`) from upper-wick bull traps.
* **Intraday & Live Session Volume Normalization**:
  * **Live Daily Projection (`1d`)**: During live NSE market hours (`09:15–15:30 IST`), partial daily volume is normalized against an empirical NSE intraday U-curve participation profile so breakouts are caught **live** rather than 1 day late.
  * **Intraday Slot Matching (`15m`, `1h`)**: Matches intraday bars against historical volume from the **exact same IST time-of-day slot** to eliminate morning open U-curve false alarms.
* **Rate-Limit-Safe Batch Pipeline**: Uses chunked multi-ticker batching (`yf.download`) and an in-memory OHLCV DataFrame cache so scanning thousands of tickers avoids Yahoo Finance HTTP 429 bans and clicking any stock in the GUI renders its chart in `<5ms`.
* **Built-In Historical Strategy Backtester**: Simulates taking every Volume-Confirmed Donchian Breakout with a `2x Wilder ATR(14)` initial stop, a Chandelier / Donchian Mid trailing stop, and **0.15% round-trip NSE transaction costs**, directly comparing net performance against raw unfiltered Donchian breakouts.
* **TradingView Pine Script v5 Strategy**: Ready-to-paste `pinescript/nse_anomaly_donchian.pine` script with live HUD table, dynamic trailing stops, and real-time alert hooks.

---

## Quantitative Methodology

![Anomaly detection pipeline](docs/methodology_diagram.svg)

Trading volume is **strictly positive and right-skewed (approximately log-normal)** — a naive percentage-change or simple moving average (SMA) baseline is easily distorted by prior spikes. This engine pairs robust log-space statistics with price-structure confirmation:

| Metric | Formula | Quantitative Purpose |
|---|---|---|
| **Volume Baseline** | $\text{Med}_V = \text{median}(V_{t-20 \dots t-1})$ | Outlier-resistant baseline (50% breakdown point), strictly excluding bar $t$ |
| **Relative Volume (RVOL)** | $\text{RVOL}_t = \frac{V_t}{\text{Med}_V}$ | Enforces **economic magnitude** vs. typical trading activity |
| **Parametric Log Z-Score** | $Z_{\ln} = \frac{\ln(1+V_t) - \mu_{\ln}}{\sigma_{\ln (\text{ddof}=1)}}$ | Corrects for volume's log-normal skewness using Bessel-corrected sample std |
| **Robust Log-MAD Z-Score** | $Z_{\text{MAD}} = \frac{0.6745 \cdot (\ln(1+V_t) - \text{Med}_{\ln})}{\text{MAD}_{\ln}}$ | Prevents **variance masking** when a prior earnings spike sits inside the 20-bar window |
| **Shifted Donchian Channel** | $U_t = \max(H_{t-20 \dots t-1}), \; L_t = \min(L_{t-20 \dots t-1})$ | Strictly shifted by 1 bar so $\text{Close}_t > U_t$ confirms absorption of overhead supply |
| **Close Location Value (CLV)** | $\text{CLV}_t = \frac{\text{Close}_t - \text{Low}_t}{\text{High}_t - \text{Low}_t} \in [0, 1]$ | Rejects upper-wick "shooting star" bull traps (requires $\text{CLV}_t \ge 0.65$ for longs) |
| **Wilder's ATR(14)** | $\text{ATR}_t = \frac{1}{14}\text{TR}_t + \frac{13}{14}\text{ATR}_{t-1}$ | Sizes volatility-adjusted initial Stop-Loss ($\text{Close} - 2\times\text{ATR}$) and $2\text{R}$ Target ($\text{Close} + 4\times\text{ATR}$) |

**Actionable Breakout Rule:**
* **Volume Anomaly Gate:** $\text{RVOL}_t \ge 2.0\text{x}$ **AND** $\max(Z_{\ln}, Z_{\text{MAD}}) \ge 2.5\sigma$
* **Long Breakout Confirmation:** $\text{Close}_t > U_t$ **AND** $\text{CLV}_t \ge 0.65$ **AND** $\text{Close}_t \ge \text{Open}_t$
* **Short Breakdown Confirmation:** $\text{Close}_t < L_t$ **AND** $\text{CLV}_t \le 0.35$ **AND** $\text{Close}_t \le \text{Open}_t$

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
* Select your **Universe** (`ALL_NSE` for all 2,600+ listed stocks, `NIFTY_500`, or `NIFTY_200`) and **Timeframe** (`15m`, `1h`, or `1d`).
* Click **RUN LIVE SCAN** (and toggle **Donchian Breakouts Only** to filter directly to confirmed breakouts).
* Click any flagged stock in the table to inspect its **Donchian Channel overlay**, **2xATR Stop / 4xATR Target lines**, **20-bar Median Volume Baseline**, and **Historical Strategy Backtest**.

### 2. Command Line Scanner (CLI)
```bash
# Scan all 2,600+ NSE stocks on daily candles
python main.py

# Scan Nifty 200 on live 15-minute intraday candles for confirmed Donchian breakouts only
python main.py --universe NIFTY_200 --interval 15m --breakouts-only

# Custom thresholds
python main.py --rvol 2.5 --z-score 3.0
```

### 3. TradingView Pine Script v5 Strategy (Live Charts)
1. Open `pinescript/nse_anomaly_donchian.pine` and copy the entire script.
2. In TradingView, open the **Pine Editor**, clear any existing template (`Ctrl+A` then `Backspace`), paste the script, and click **Add to chart**.
3. Open TradingView's **Strategy Tester** tab to inspect live backtest metrics across `5m`, `15m`, `1H`, or `1D` timeframes.

### 4. Automated Test Suite
```bash
python test_suite.py
```

---

## Project Architecture

```text
├── config.py                        <- Thresholds & cached dynamic NSE universe loader (ALL_NSE / Nifty 500 / Nifty 200)
├── anomaly_engine.py                <- Vectorized math engine (Log-Z, Log-MAD Z, Donchian, CLV, Wilder ATR, Backtester)
├── alerts.py                        <- Console reporting & 17-column CSV alert logger
├── main.py                          <- CLI entry point with batched scanning & signal prioritization
├── gui.py                           <- Non-blocking dark Tkinter + Matplotlib quantitative terminal
├── test_suite.py                    <- 11 unit & integration tests verifying math, edge cases, and pipeline
├── MATH_AND_CHANGES_EXPLAINED.txt   <- Deep-dive mathematical documentation & technical interview Q&A guide
├── data_sources/
│   └── equities.py                  <- Rate-limit-resistant batched OHLCV fetcher & in-memory DataFrame cache
└── pinescript/
    └── nse_anomaly_donchian.pine    <- TradingView Pine Script v5 Strategy & live HUD indicator
```

---

## Disclaimer

Built for **educational and quantitative research purposes only**. Statistical volume anomalies and Donchian breakouts represent quantitative filtering rules and historical backtest simulations — they do not constitute financial advice. Always account for live market liquidity, circuit limits, and execution slippage.
