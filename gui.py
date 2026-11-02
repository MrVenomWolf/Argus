"""
gui.py
=======
Non-Blocking Quantitative Dark Terminal for NSE Volume Anomaly & Donchian
Breakout Detection.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anomaly_engine
from anomaly_engine import Anomaly
import config
from data_sources import equities
import main as engine_main


class DashboardApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NSE Quantitative Volume Anomaly & Donchian Breakout Terminal")
        self.geometry("1360x800")
        self.minsize(1100, 660)
        self.configure(bg="#121212")

        self.anomalies_map: Dict[str, Anomaly] = {}
        self.all_last_anomalies: List[Anomaly] = []
        self._scan_cancel_event: Optional[threading.Event] = None
        self._is_scanning = False
        self._chart_request_id = 0

        self._apply_dark_theme()
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_dark_theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        BG, PANEL, FG, BLUE, GREEN = "#121212", "#1b1b1f", "#e4e4e8", "#007acc", "#00e676"

        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 9))
        s.configure("TFrame", background=BG)
        s.configure("TLabelframe", background=PANEL, foreground=GREEN, relief="solid")
        s.configure("TLabelframe.Label", background=PANEL, foreground=GREEN, font=("Segoe UI", 9, "bold"))
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("TEntry", fieldbackground="#26262b", foreground="#ffffff", insertcolor="#ffffff")
        s.configure("TCombobox", fieldbackground="#26262b", background="#26262b", foreground="#ffffff", arrowcolor="#ffffff")
        s.map("TCombobox", fieldbackground=[("readonly", "#26262b")], selectbackground=[("readonly", BLUE)])
        s.configure("TButton", background=BLUE, foreground="#ffffff", borderwidth=0, padding=(12, 5), font=("Segoe UI", 9, "bold"))
        s.map("TButton", background=[("active", "#0098ff"), ("disabled", "#333338")])
        s.configure("Horizontal.TProgressbar", troughcolor="#1e1e24", background=GREEN)
        s.configure("Treeview", background="#18181c", foreground=FG, fieldbackground="#18181c", rowheight=27, font=("Consolas", 9))
        s.configure("Treeview.Heading", background="#26262c", foreground=GREEN, font=("Segoe UI", 8, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", "#ffffff")])

    def _setup_ui(self):
        ctrl = ttk.Frame(self, padding=(12, 10, 12, 6))
        ctrl.pack(fill=tk.X)

        self.universe_var = tk.StringVar(value="ALL_NSE")
        self.interval_var = tk.StringVar(value="1d")
        self.rvol_var = tk.StringVar(value=str(config.MIN_RVOL))
        self.z_var = tk.StringVar(value=str(config.MIN_Z_SCORE))
        self.donchian_var = tk.StringVar(value=str(config.DONCHIAN_WINDOW))

        for label, var, values, width in (
            ("Universe:", self.universe_var, ["ALL_NSE", "NIFTY_500", "NIFTY_200"], 11),
            ("Timeframe:", self.interval_var, ["15m", "1h", "1d"], 6),
            ("Min RVOL (x):", self.rvol_var, None, 5),
            ("Min Z (\u03c3):", self.z_var, None, 5),
            ("Donchian (N):", self.donchian_var, None, 4),
        ):
            ttk.Label(ctrl, text=label, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            widget = (
                ttk.Combobox(ctrl, textvariable=var, values=values, state="readonly", width=width)
                if values
                else ttk.Entry(ctrl, textvariable=var, width=width)
            )
            widget.pack(side=tk.LEFT, padx=(5, 14))

        self.breakouts_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl, text="Donchian Breakouts Only", variable=self.breakouts_only_var, command=self._refresh_table_filter
        ).pack(side=tk.LEFT, padx=(4, 14))

        self.btn_run = ttk.Button(ctrl, text="RUN LIVE SCAN", command=self.toggle_scan)
        self.btn_run.pack(side=tk.RIGHT, ipadx=8)

        status_bar = ttk.Frame(self, padding=(12, 0, 12, 8))
        status_bar.pack(fill=tk.X)
        self.lbl_status = ttk.Label(
            status_bar, text="Ready. Defaulting to ALL_NSE (2,200+ listed stocks). Click RUN LIVE SCAN.", foreground="#9cdcfe", font=("Consolas", 9)
        )
        self.lbl_status.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(status_bar, orient=tk.HORIZONTAL, length=260, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left_frame = ttk.Frame(paned, padding=4)
        paned.add(left_frame, weight=5)

        headers = {
            "symbol": ("SYMBOL", 105, "w"),
            "signal": ("SIGNAL", 130, "w"),
            "price": ("PRICE", 82, "e"),
            "rvol": ("RVOL", 65, "e"),
            "z_score": ("LOG-Z", 68, "e"),
            "mad_z": ("MAD-Z", 68, "e"),
            "clv": ("CLV", 55, "e"),
            "stop": ("STOP (2ATR)", 88, "e"),
            "target": ("TARGET (2R)", 88, "e"),
        }
        self.tree = ttk.Treeview(left_frame, columns=tuple(headers), show="headings", selectmode="browse")
        for col, (text, width, anchor) in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)

        for tag, color in (
            ("LONG BREAKOUT", "#00e676"),
            ("SHORT BREAKDOWN", "#ff5252"),
            ("BULLISH ACCUM", "#40c4ff"),
            ("BEARISH DISTRIB", "#ffab40"),
            ("VOL SPIKE", "#d4d4d8"),
        ):
            self.tree.tag_configure(tag, foreground=color)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_stock_select)

        right_frame = ttk.Frame(paned, padding=4)
        paned.add(right_frame, weight=6)

        self.info_box = ttk.LabelFrame(
            right_frame, text=" QUANTITATIVE SIGNAL, DONCHIAN STRUCTURE & HISTORICAL BACKTEST ", padding=10
        )
        self.info_box.pack(fill=tk.X, pady=(0, 8))

        self.lbl_price_info = tk.Label(
            self.info_box,
            text="Select any flagged stock on the left to inspect Donchian channels, Wilder ATR risk levels, and Strategy Backtest.",
            font=("Consolas", 9),
            bg="#1b1b1f",
            fg="#b0b0b8",
            anchor="w",
            justify="left",
        )
        self.lbl_price_info.pack(fill=tk.X)

        self.fig, (self.ax_price, self.ax_vol) = plt.subplots(
            2, 1, figsize=(6.8, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.7, 1.0]}, facecolor="#1b1b1f"
        )
        self._style_dark_axes()
        self.fig.tight_layout(pad=1.8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_dark_axes(self):
        for ax in (self.ax_price, self.ax_vol):
            ax.set_facecolor("#121212")
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2e2e36")

    def toggle_scan(self):
        if self._is_scanning:
            if self._scan_cancel_event:
                self._scan_cancel_event.set()
            self.lbl_status.config(text="Stopping scan after current batch...", foreground="#ffab40")
            return
        self.start_scan()

    def start_scan(self):
        try:
            min_rvol, min_z, donchian_n = float(self.rvol_var.get()), float(self.z_var.get()), int(self.donchian_var.get())
            if min_rvol <= 0 or min_z <= 0 or donchian_n < 5:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Enter positive RVOL/Z-Score and Donchian N >= 5.")
            return

        universe, interval = self.universe_var.get(), self.interval_var.get()
        self._is_scanning = True
        self._scan_cancel_event = threading.Event()
        self.btn_run.config(text="STOP SCAN")
        self.progress["value"] = 0
        self.lbl_status.config(text=f"Scanning {universe} ({interval})...", foreground="#00e676")

        self.tree.delete(*self.tree.get_children())
        self.anomalies_map.clear()
        self.all_last_anomalies.clear()
        equities.clear_dataframe_cache()

        def worker():
            try:
                anomalies = engine_main.run(
                    min_rvol=min_rvol,
                    min_z_score=min_z,
                    universe=universe,
                    interval=interval,
                    only_breakouts=False,
                    donchian_window=donchian_n,
                    progress_callback=lambda d, t: self.after(0, self._update_progress_ui, d, t),
                    cancel_event=self._scan_cancel_event,
                )
                self.after(0, self._on_scan_complete, anomalies, None)
            except Exception as exc:
                self.after(0, self._on_scan_complete, [], str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _update_progress_ui(self, done: int, total: int):
        if self.winfo_exists() and total > 0:
            pct = (done / total) * 100.0
            self.progress["value"] = pct
            self.lbl_status.config(text=f"Downloaded & analyzed {done} / {total} NSE tickers ({pct:.0f}%)...", foreground="#9cdcfe")

    def _on_scan_complete(self, anomalies: List[Anomaly], error_msg: Optional[str]):
        if not self.winfo_exists():
            return
        self._is_scanning = False
        self.btn_run.config(text="RUN LIVE SCAN")
        self.progress["value"] = 0 if error_msg else 100
        if error_msg:
            self.lbl_status.config(text=f"Scan failed: {error_msg}", foreground="#ff5252")
            messagebox.showerror("Scan Error", f"Failed: {error_msg}")
            return
        self.all_last_anomalies = anomalies
        self._refresh_table_filter()

    def _refresh_table_filter(self):
        self.tree.delete(*self.tree.get_children())
        self.anomalies_map.clear()
        only_bo = self.breakouts_only_var.get()
        displayed = [a for a in self.all_last_anomalies if not only_bo or a.is_actionable_breakout]

        for a in displayed:
            iid = self.tree.insert(
                "",
                tk.END,
                values=(
                    a.point.symbol.replace(".NS", ""),
                    a.signal,
                    f"\u20b9{a.close_price:,.2f}",
                    f"{a.rvol:.2f}x",
                    f"{a.z_score:+.2f}\u03c3",
                    f"{a.modified_z_score:+.2f}\u03c3",
                    f"{a.close_location_value * 100:.0f}%",
                    f"\u20b9{a.stop_loss:,.2f}",
                    f"\u20b9{a.target_price:,.2f}",
                ),
                tags=(a.signal,),
            )
            self.anomalies_map[iid] = a

        bo_total = sum(1 for a in self.all_last_anomalies if a.is_actionable_breakout)
        self.lbl_status.config(
            text=f"Scan complete: {len(displayed)} shown ({len(self.all_last_anomalies)} anomalies, {bo_total} Donchian Breakouts).",
            foreground="#00e676" if displayed else "#ffab40",
        )
        if children := self.tree.get_children():
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])

    def on_stock_select(self, _event=None):
        if not (sel := self.tree.selection()) or not (anomaly := self.anomalies_map.get(sel[0])):
            return
        self._chart_request_id += 1
        req_id, symbol = self._chart_request_id, anomaly.point.symbol

        if (cached_df := equities.get_cached_dataframe(symbol)) is not None and not cached_df.empty:
            self._prepare_and_render_symbol(req_id, symbol, cached_df, anomaly)
            return

        self.lbl_price_info.config(text=f"Loading {symbol}...", fg="#9cdcfe")
        interval = self.interval_var.get()

        def fetch_worker():
            try:
                equities.fetch_equity_volume(symbol, interval=interval)
                if (df := equities.get_cached_dataframe(symbol)) is not None and not df.empty:
                    self.after(0, self._prepare_and_render_symbol, req_id, symbol, df, anomaly)
                else:
                    self.after(0, lambda: self.lbl_price_info.config(text=f"No data for {symbol}", fg="#ff5252"))
            except Exception as e:
                self.after(0, lambda: self.lbl_price_info.config(text=f"Error: {e}", fg="#ff5252"))

        threading.Thread(target=fetch_worker, daemon=True).start()

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        if df is None or df.empty:
            return 0.0
        h, l, c = (pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float) for k in ("High", "Low", "Close"))
        atr_arr = anomaly_engine.compute_wilder_atr(h, l, c, period=period)
        return float(atr_arr[-1]) if atr_arr.size > 0 else 0.0

    def _safe_float(self, val_str: str, fallback: float) -> float:
        try:
            return float(val_str)
        except ValueError:
            return fallback

    def _prepare_and_render_symbol(self, req_id: int, symbol: str, df: pd.DataFrame, anomaly: Anomaly):
        if req_id != self._chart_request_id or not self.winfo_exists():
            return

        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        donchian_n = max(5, int(self._safe_float(self.donchian_var.get(), config.DONCHIAN_WINDOW)))
        highs, lows, vols = (
            pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in ("High", "Low", "Volume")
        )
        d_upper, d_lower, d_mid = anomaly_engine.compute_donchian_channels(highs, lows, window=donchian_n)
        med_vol = pd.Series(vols).shift(1).rolling(config.BASELINE_WINDOW, min_periods=5).median().to_numpy()

        bt = anomaly_engine.backtest_donchian_volume(
            df,
            symbol=symbol,
            window=config.BASELINE_WINDOW,
            donchian_window=donchian_n,
            min_rvol=self._safe_float(self.rvol_var.get(), config.MIN_RVOL),
            min_z_score=self._safe_float(self.z_var.get(), config.MIN_Z_SCORE),
        )

        bt_line = (
            f"STRATEGY BACKTEST ({bt.total_bars} bars, 0.15% NSE cost) -> "
            f"Vol+Donchian: {bt.trades} trades | WinRate: {bt.win_rate_pct:.1f}% | "
            f"PF: {bt.profit_factor:.2f} | Avg R: {bt.avg_r_multiple:+.2f}R | "
            f"Net: {bt.net_return_pct:+.1f}% (vs Raw Donchian: {bt.unfiltered_net_return_pct:+.1f}%)"
            if bt
            else "Backtest: Insufficient historical bars"
        )
        proj_note = " (Intraday Normalized)" if anomaly.is_live_projected else ""
        info_text = (
            f"SYMBOL: {symbol:<13} | SIGNAL: {anomaly.signal:<16} | CLOSE: \u20b9{anomaly.close_price:,.2f} "
            f"| MEDIAN TURNOVER: \u20b9{anomaly.turnover_cr:,.1f} Cr\n"
            f"VOLUME STATS -> RVOL: {anomaly.rvol:.2f}x{proj_note} | Parametric Log-Z: {anomaly.z_score:+.2f}\u03c3 "
            f"| Robust Log-MAD Z: {anomaly.modified_z_score:+.2f}\u03c3 | CLV: {anomaly.close_location_value*100:.0f}%\n"
            f"DONCHIAN ({donchian_n}) -> Upper: \u20b9{anomaly.donchian_upper:,.2f} | Mid: \u20b9{anomaly.donchian_mid:,.2f} "
            f"| Lower: \u20b9{anomaly.donchian_lower:,.2f} | Width: {anomaly.channel_width_pct:.1f}%\n"
            f"WILDER ATR(14): \u20b9{anomaly.atr:,.2f} | INITIAL STOP (2xATR): \u20b9{anomaly.stop_loss:,.2f} "
            f"| 2R TARGET (4xATR): \u20b9{anomaly.target_price:,.2f}\n{bt_line}"
        )
        sig_color = {
            "LONG BREAKOUT": "#00e676",
            "SHORT BREAKDOWN": "#ff5252",
            "BULLISH ACCUM": "#40c4ff",
            "BEARISH DISTRIB": "#ffab40",
        }.get(anomaly.signal, "#e4e4e8")
        self.lbl_price_info.config(text=info_text, fg=sig_color)

        n_plot = min(len(df), 80)
        sub_df = df.iloc[-n_plot:]
        x = np.arange(n_plot)

        self.ax_price.clear()
        self.ax_vol.clear()
        self._style_dark_axes()

        self.ax_price.plot(x, sub_df["Close"], label="Close", color="#00b0ff", lw=1.9)
        self.ax_price.plot(x, d_upper[-n_plot:], label=f"Donchian High ({donchian_n})", color="#00e676", lw=1.2, ls="--")
        self.ax_price.plot(x, d_mid[-n_plot:], label="Donchian Mid (Trail)", color="#ffab40", lw=1.0, ls=":")
        self.ax_price.plot(x, d_lower[-n_plot:], label=f"Donchian Low ({donchian_n})", color="#ff5252", lw=1.2, ls="--")
        self.ax_price.fill_between(x, d_lower[-n_plot:], d_upper[-n_plot:], color="#007acc", alpha=0.08)

        if anomaly.stop_loss > 0:
            self.ax_price.axhline(anomaly.stop_loss, color="#ff5252", lw=0.9, ls="-.", alpha=0.7, label="2xATR Stop")
        if anomaly.target_price > 0:
            self.ax_price.axhline(anomaly.target_price, color="#00e676", lw=0.9, ls="-.", alpha=0.7, label="4xATR Target (2R)")

        self.ax_price.set_title(
            f"{symbol} ({self.interval_var.get()}) \u2014 {anomaly.signal} [RVOL {anomaly.rvol:.2f}x | Z {anomaly.effective_z:+.2f}\u03c3]",
            color="#ffffff",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_price.grid(True, linestyle=":", alpha=0.25, color="#666666")
        self.ax_price.legend(loc="upper left", fontsize=7.5, facecolor="#1b1b1f", edgecolor="#333338", labelcolor="#e4e4e8", ncol=3)

        colors = ["#00e676" if c >= o else "#ff5252" for c, o in zip(sub_df["Close"], sub_df["Open"])]
        self.ax_vol.bar(x, sub_df["Volume"], color=colors, alpha=0.82, width=0.7)
        self.ax_vol.plot(x, med_vol[-n_plot:], color="#ffab40", lw=1.3, label=f"Median Vol ({config.BASELINE_WINDOW})")
        self.ax_vol.grid(True, linestyle=":", alpha=0.25, color="#666666")
        self.ax_vol.legend(loc="upper left", fontsize=7.5, facecolor="#1b1b1f", edgecolor="#333338", labelcolor="#e4e4e8")

        fmt = "%d-%b %H:%M" if self.interval_var.get() != "1d" else "%d-%b"
        ticks = x[:: max(1, len(x) // 8)]
        self.ax_vol.set_xticks(ticks)
        self.ax_vol.set_xticklabels([sub_df.index[i].strftime(fmt) for i in ticks], rotation=20, ha="right")

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    def _on_close(self):
        if self._scan_cancel_event:
            self._scan_cancel_event.set()
        plt.close("all")
        self.destroy()


if __name__ == "__main__":
    DashboardApp().mainloop()
