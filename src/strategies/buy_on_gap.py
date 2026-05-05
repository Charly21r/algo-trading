import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter


# ─────────────────────────────────────────────────────────────────────────────
#  BUY ON GAP MODEL (BOG)
#
#  Original rules (from the screenshot):
#    1. Select stocks near market open whose return from prev day's low to
#       today's open is below -1 std (90-day close-to-close vol). = "gapped down"
#    2. Filter: open price must be above the 20-day MA of closing prices.
#    3. Buy the 10 stocks with the lowest gap return. Buy all if < 10.
#    4. Liquidate all positions at the close.
#
#  Extended parameters allow customisation of all thresholds.
# ─────────────────────────────────────────────────────────────────────────────


class BuyOnGapStrategy:
    """
    Intraday gap-down mean reversion strategy.

    Stocks that gap down sharply but remain above their 20-day MA are
    bet on to recover intraday. Positions are opened at the open and
    closed at the close — no overnight exposure.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        {ticker: OHLCV DataFrame with DatetimeIndex}
        Required columns: open, high, low, close, volume
    vol_window : int
        Lookback for historical volatility (default 90 days)
    ma_window : int
        MA window for trend filter (default 20 days)
    gap_threshold : float
        Gap must be below gap_threshold * hist_vol (default -1.0)
    n_stocks : int
        Max stocks to buy per day (default 10)
    capital : float
        Starting capital, equally split across selected stocks each day
    position_sizing : str
        'equal'   — equal dollar allocation per stock
        'inverse' — allocate more capital to the biggest gaps
                    (larger gap → larger position, mean-reversion bet)
    """

    def __init__(
        self,
        data,
        vol_window       = 90,
        ma_window        = 20,
        gap_threshold    = -1.0,
        n_stocks         = 10,
        capital          = 100_000.0,
        position_sizing  = "equal",
    ):
        self.data             = data
        self.vol_window       = vol_window
        self.ma_window        = ma_window
        self.gap_threshold    = gap_threshold
        self.n_stocks         = n_stocks
        self.capital          = capital
        self.position_sizing  = position_sizing

        self.tickers          = list(data.keys())
        self.results          = None
        self._prepared        = None

    # ──────────────────────────────────────────────────────────────────────
    #  PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────

    def _prepare(self):
        """
        Pre-compute for each ticker:
          - 90-day rolling std of close-to-close returns  (hist_vol)
          - 20-day rolling MA of close                    (ma_20)
          - gap return: (today open - yesterday low) / yesterday low
          - intraday return: (close - open) / open
        """

        prepared = {}

        for ticker, df in self.data.items():
            d = df.copy().sort_index()
            d.columns = [c.lower() for c in d.columns]

            d["close_ret"]    = d["close"].pct_change()
            d["hist_vol"]     = d["close_ret"].rolling(self.vol_window).std()
            d["ma"]           = d["close"].rolling(self.ma_window).mean()
            d["prev_low"]     = d["low"].shift(1)
            d["gap_return"]   = (d["open"] - d["prev_low"]) / d["prev_low"]
            d["intraday_ret"] = (d["close"] - d["open"]) / d["open"]

            prepared[ticker] = d

        self._prepared = prepared
        return prepared

    # ──────────────────────────────────────────────────────────────────────
    #  DAILY SELECTION
    # ──────────────────────────────────────────────────────────────────────

    def _select(self, date):
        """
        Apply the three selection rules for a given date.

        Returns
        -------
        pd.DataFrame  columns: ticker, gap_return, open_price, intraday_ret
            Sorted by gap_return ascending (most oversold first),
            truncated to n_stocks.
        """

        candidates = []

        for ticker, df in self._prepared.items():
            if date not in df.index:
                continue

            row = df.loc[date]

            if pd.isna(row["hist_vol"]) or pd.isna(row["ma"]) or pd.isna(row["gap_return"]):
                continue

            # Rule 1 — gap below threshold * hist_vol
            if row["gap_return"] >= self.gap_threshold * row["hist_vol"]:
                continue

            # Rule 2 — open above 20-day MA
            if row["open"] <= row["ma"]:
                continue

            candidates.append({
                "ticker"      : ticker,
                "gap_return"  : row["gap_return"],
                "open_price"  : row["open"],
                "intraday_ret": row["intraday_ret"],
                "hist_vol"    : row["hist_vol"],
            })

        if not candidates:
            return pd.DataFrame()

        df_c = pd.DataFrame(candidates).sort_values("gap_return")
        return df_c.head(self.n_stocks)

    # ──────────────────────────────────────────────────────────────────────
    #  POSITION SIZING
    # ──────────────────────────────────────────────────────────────────────

    def _allocate(self, selected):
        """
        Compute dollar allocation per stock.

        equal    — capital / n  for each stock
        inverse  — weight proportional to abs(gap_return);
                   bigger gap → bigger bet (more oversold → more reversion)
        """

        n = len(selected)
        if n == 0:
            return selected

        if self.position_sizing == "equal":
            selected = selected.copy()
            selected["allocation"] = self.capital / n

        elif self.position_sizing == "inverse":
            # Weight by normalised abs gap magnitude
            gaps    = selected["gap_return"].abs()
            weights = gaps / gaps.sum()
            selected = selected.copy()
            selected["allocation"] = weights * self.capital

        return selected

    # ──────────────────────────────────────────────────────────────────────
    #  BACKTEST
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        """
        Run the full backtest across all available dates.

        Each day:
          - Select qualifying stocks at open
          - Allocate capital
          - Record intraday P&L (open → close)

        Returns
        -------
        pd.DataFrame — daily metrics indexed by date
        """

        if self._prepared is None:
            self._prepare()

        all_dates = sorted(
            set.intersection(*[set(df.index) for df in self._prepared.values()])
        )

        records = []

        for date in all_dates:

            selected = self._select(date)

            if selected.empty:
                records.append({
                    "date"           : date,
                    "n_stocks"       : 0,
                    "tickers"        : [],
                    "gap_returns"    : [],
                    "intraday_rets"  : [],
                    "daily_pnl"      : 0.0,
                    "daily_return"   : 0.0,
                })
                continue

            selected = self._allocate(selected)

            day_pnl  = 0.0
            day_rets = []

            for _, row in selected.iterrows():
                shares   = row["allocation"] / row["open_price"]
                pnl      = shares * (row["open_price"] * (1 + row["intraday_ret"])
                                     - row["open_price"])
                day_pnl += pnl
                day_rets.append(row["intraday_ret"])

            records.append({
                "date"          : date,
                "n_stocks"      : len(selected),
                "tickers"       : selected["ticker"].tolist(),
                "gap_returns"   : selected["gap_return"].tolist(),
                "intraday_rets" : day_rets,
                "daily_pnl"     : day_pnl,
                "daily_return"  : np.mean(day_rets),
            })

        df = pd.DataFrame(records).set_index("date")
        df["cumulative_pnl"]    = df["daily_pnl"].cumsum()
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

        self.results = df
        return df

    # ──────────────────────────────────────────────────────────────────────
    #  PERFORMANCE SUMMARY
    # ──────────────────────────────────────────────────────────────────────

    def summary(self):
        """
        Compute and print key performance metrics.

        Returns
        -------
        dict
        """

        if self.results is None:
            self.run()

        rets   = self.results["daily_return"]
        active = rets[self.results["n_stocks"] > 0]

        mean_r  = active.mean()
        std_r   = active.std()
        sharpe  = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else np.nan

        n_years   = len(rets) / 252
        total_ret = (1 + rets).prod() - 1
        cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan

        cum      = (1 + rets).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd   = drawdown.min()

        win_rate   = (active > 0).mean()
        avg_stocks = self.results.loc[self.results["n_stocks"] > 0, "n_stocks"].mean()
        n_active   = (self.results["n_stocks"] > 0).sum()
        n_total    = len(self.results)

        # Average gap on active days
        avg_gap = np.mean([
            np.mean(g) for g in self.results.loc[self.results["n_stocks"] > 0, "gap_returns"]
            if g
        ])

        print(f"\n{'═' * 58}")
        print(f"  Buy on Gap Strategy — Performance Summary")
        print(f"  Sizing: {self.position_sizing.upper()}")
        print(f"{'═' * 58}")
        print(f"  Gap threshold   : {self.gap_threshold} σ  ({self.vol_window}-day vol)")
        print(f"  MA filter       : {self.ma_window}-day MA")
        print(f"  Max stocks/day  : {self.n_stocks}")
        print(f"{'─' * 58}")
        print(f"  CAGR            : {cagr * 100:>10.2f}%")
        print(f"  Sharpe ratio    : {sharpe:>10.4f}")
        print(f"  Max drawdown    : {max_dd * 100:>10.2f}%")
        print(f"  Win rate        : {win_rate * 100:>10.1f}%")
        print(f"  Total return    : {total_ret * 100:>10.2f}%")
        print(f"  Avg gap (entry) : {avg_gap * 100:>10.2f}%")
        print(f"  Avg stocks/day  : {avg_stocks:>10.1f}")
        print(f"  Active days     : {n_active:>10} / {n_total}")

        return {
            "sharpe"      : sharpe,
            "cagr"        : cagr,
            "max_drawdown": max_dd,
            "win_rate"    : win_rate,
            "total_return": total_ret,
            "avg_gap"     : avg_gap,
            "avg_stocks"  : avg_stocks,
            "n_active"    : n_active,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  SENSITIVITY ANALYSIS
    # ──────────────────────────────────────────────────────────────────────

    def sensitivity(self, thresholds=(-0.5, -1.0, -1.5, -2.0, -2.5),
                    n_stocks_list=(5, 10, 20)):
        """
        Grid search over gap_threshold and n_stocks.
        Prints a table sorted by Sharpe.

        Returns
        -------
        pd.DataFrame
        """

        rows = []

        for thresh in thresholds:
            for n in n_stocks_list:
                s = BuyOnGapStrategy(
                    self.data,
                    vol_window      = self.vol_window,
                    ma_window       = self.ma_window,
                    gap_threshold   = thresh,
                    n_stocks        = n,
                    capital         = self.capital,
                    position_sizing = self.position_sizing,
                )
                s.run()
                m = s.summary()
                rows.append({
                    "gap_thresh" : thresh,
                    "n_stocks"   : n,
                    "Sharpe"     : round(m["sharpe"],        4),
                    "CAGR (%)"   : round(m["cagr"] * 100,   2),
                    "MaxDD (%)"  : round(m["max_drawdown"] * 100, 2),
                    "WinRate (%)": round(m["win_rate"] * 100, 1),
                    "ActiveDays" : m["n_active"],
                })

        df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

        print(f"\n{'═' * 70}")
        print(f"  Sensitivity Analysis — Buy on Gap")
        print(f"{'═' * 70}")
        print(df.to_string(index=False))

        return df


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_bog(strategy, pair_name="Universe", save_path=None):
    """
    3-panel chart for the Buy on Gap strategy:
      Panel 1 — Cumulative return with active-day markers
      Panel 2 — Daily return distribution (histogram)
      Panel 3 — Rolling 63-day Sharpe ratio
    """

    if strategy.results is None:
        strategy.run()

    df      = strategy.results
    rets    = df["daily_return"]
    cum     = df["cumulative_return"]
    active  = df["n_stocks"] > 0

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("#0F172A")
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.2, 1.2], hspace=0.10)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax in axes:
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(axis="y", color="#1E293B", linewidth=0.7)
        ax.grid(axis="x", color="#1E293B", linewidth=0.4, linestyle=":")

    BLUE   = "#2563EB"
    GREEN  = "#22C55E"
    RED    = "#DC2626"
    PURPLE = "#A78BFA"

    # ── Panel 1: Cumulative return ───────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(cum.index, cum.values, color=BLUE, linewidth=1.8, zorder=3)
    ax0.fill_between(cum.index, 0, cum.values,
                     where=cum.values >= 0, alpha=0.15, color=BLUE)
    ax0.fill_between(cum.index, 0, cum.values,
                     where=cum.values < 0,  alpha=0.20, color=RED)
    ax0.axhline(0, color="#475569", linewidth=0.8, linestyle="--")

    # Mark active trading days as small dots on the x-axis
    active_dates = df.index[active]
    ax0.scatter(active_dates,
                [cum.min() * 1.05] * len(active_dates),
                marker="|", s=12, color=GREEN, alpha=0.4, zorder=2)

    # Drawdown shading
    running_max = cum.cummax()
    ax0.fill_between(cum.index, cum.values, running_max,
                     where=(cum.values < running_max),
                     alpha=0.12, color=RED, label="Drawdown")

    sharpe = (rets[active].mean() / rets[active].std()) * np.sqrt(252)
    total  = cum.iloc[-1]
    ax0.set_title(
        f"{pair_name}  |  Buy on Gap  |  "
        f"Sharpe {sharpe:.3f}  |  "
        f"Total Return {total:.3f}  |  "
        f"Win Rate {(rets[active] > 0).mean() * 100:.1f}%",
        color="#F1F5F9", fontsize=12, fontweight="bold", pad=10
    )
    ax0.set_ylabel("Cumulative Return", color="#94A3B8", fontsize=9)
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    active_patch = mpatches.Patch(color=GREEN, alpha=0.6, label="Active trading day")
    dd_patch     = mpatches.Patch(color=RED,   alpha=0.4, label="Drawdown period")
    ax0.legend(handles=[active_patch, dd_patch],
               loc="upper left", framealpha=0.15, fontsize=8,
               labelcolor="#E2E8F0", facecolor="#1E293B", edgecolor="#334155")
    ax0.tick_params(labelbottom=False)

    # ── Panel 2: Return distribution ────────────────────────────────────────
    ax1         = axes[1]
    active_rets = rets[active].dropna() * 100   # in percent

    bins = np.linspace(active_rets.min(), active_rets.max(), 50)
    ax1.hist(active_rets[active_rets >= 0], bins=bins, color=GREEN,
             alpha=0.7, label="Positive days", edgecolor="none")
    ax1.hist(active_rets[active_rets < 0],  bins=bins, color=RED,
             alpha=0.7, label="Negative days", edgecolor="none")
    ax1.axvline(0,                   color="#CBD5E1", linewidth=0.9, linestyle="--")
    ax1.axvline(active_rets.mean(),  color=BLUE,     linewidth=1.2,
                linestyle="--", label=f"Mean {active_rets.mean():.2f}%")
    ax1.set_ylabel("Frequency", color="#94A3B8", fontsize=9)
    ax1.set_xlabel("Daily Return (%)", color="#94A3B8", fontsize=9)
    ax1.legend(loc="upper left", framealpha=0.15, fontsize=8,
               labelcolor="#E2E8F0", facecolor="#1E293B", edgecolor="#334155",
               ncol=3)
    ax1.tick_params(labelbottom=True)

    # ── Panel 3: Rolling 63-day Sharpe ──────────────────────────────────────
    ax2 = axes[2]

    roll_sharpe = (
        rets.rolling(63).mean() / rets.rolling(63).std()
    ) * np.sqrt(252)

    ax2.plot(roll_sharpe.index, roll_sharpe.values,
             color=PURPLE, linewidth=1.2, label="63-day rolling Sharpe")
    ax2.axhline(0,   color="#475569", linewidth=0.8, linestyle="--")
    ax2.axhline(1.0, color=GREEN,     linewidth=0.7, linestyle=":",
                alpha=0.7, label="Sharpe = 1.0")
    ax2.axhline(-1.0, color=RED,      linewidth=0.7, linestyle=":", alpha=0.7)
    ax2.fill_between(roll_sharpe.index, 0, roll_sharpe.values,
                     where=(roll_sharpe.values >= 0), alpha=0.10, color=GREEN)
    ax2.fill_between(roll_sharpe.index, 0, roll_sharpe.values,
                     where=(roll_sharpe.values < 0),  alpha=0.10, color=RED)
    ax2.set_ylabel("Rolling Sharpe", color="#94A3B8", fontsize=9)
    ax2.set_xlabel("Date",           color="#94A3B8", fontsize=9)
    ax2.legend(loc="upper left", framealpha=0.15, fontsize=8,
               labelcolor="#E2E8F0", facecolor="#1E293B", edgecolor="#334155",
               ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_universe(tickers, start, end):
    """Download OHLCV data for a list of tickers via yfinance."""

    import yfinance as yf

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"  Warning: no data for {ticker}, skipping.")
                continue
            df.columns = [
                c[0].lower() if isinstance(c, tuple) else c.lower()
                for c in df.columns
            ]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            data[ticker] = df
        except Exception as e:
            print(f"  Warning: failed to download {ticker}: {e}")

    print(f"  Loaded {len(data)} / {len(tickers)} tickers.")
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    US_LARGE_CAPS = [
        "TSLA","BRK.B","LLY","AVGO","COST","ABBV","PEP","MRK","TMO","ACN",
        "DHR","MCD","LIN","NEE","TXN","LOW","UNP","PM","UPS","RTX",
        "HON","IBM","CAT","GS","SPGI","BLK","AMGN","INTU","PLD","DE"
    ]
    TECH_GROWTH = [
        "SNOW","CRWD","ZS","NET","DDOG","MDB","OKTA","SHOP","SQ","ROKU",
        "U","AI","PLTR","COIN","RBLX","DOCU","TWLO","TEAM","WDAY","PANW"
    ]
    SEMIS = [
        "TSM","ASML","MU","AMAT","LRCX","KLAC","MRVL","ON","NXPI","ADI",
        "MCHP","MPWR"
    ]
    FINANCIALS = [
        "C","WFC","MS","SCHW","AXP","BK","USB","PNC","TFC","COF",
        "ICE","CME","SPGI","MCO"
    ]
    HEALTHCARE = [
        "ABT","BMY","GILD","ISRG","VRTX","REGN","ZTS","HCA","CI","ELV",
        "MRNA","BIIB","ILMN"
    ]

    CONSUMER = [
        "NKE","SBUX","TGT","COST","DG","DLTR","EBAY","ETSY","BBY","ORLY",
        "AZO","YUM","CMG"
    ]
    ENERGY = [
        "CVX","SLB","COP","EOG","PSX","MPC","OXY","KMI","HAL"
    ]

    INDUSTRIALS = [
        "BA","GE","MMM","LMT","NOC","GD","FDX","CSX","NSC","WM","RSG"
    ]

    INTERNATIONAL = [
        "BABA","JD","PDD","TCEHY","NIO","XPEV","LI",
        "SAP","ASML","UL","NSRGY","TM","SONY","SHEL","HSBC"
    ]

    ETFS = [
        "SPY","QQQ","IWM","DIA","VTI",
        "ARKK","XLF","XLK","XLE","XLV","XLY","XLP","XLI","XLU",
        "SMH","SOXX","TAN","ICLN",
        "EEM","FXI","EWJ","VGK"
    ]

    BASE_UNIVERSE = [ "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "PFE", "CSCO", "ADBE", "CRM", "NFLX", "INTC", "AMD", "PYPL", "QCOM", "T", "VZ", "WMT", "KO", "MCD", ]
    
    INDEXES = [
        "^GSPC",   # S&P 500
        "^IXIC",   # Nasdaq
        "^DJI",    # Dow Jones
        "^RUT",    # Russell 2000
        "^VIX",    # Volatility Index
        "^FTSE",   # FTSE 100
        "^N225",   # Nikkei
        "^HSI"     # Hang Seng
    ]

    UNIVERSE = (
    BASE_UNIVERSE +
    US_LARGE_CAPS +
    SEMIS +
    FINANCIALS
    #ENERGY +
    #INDUSTRIALS +
    #INTERNATIONAL +
    #ETFS +
    #INDEXES
)

    print("Downloading universe...")
    data = load_universe(UNIVERSE, start="2020-01-01", end="2025-01-01")

    # ── Run with default parameters (equal sizing) ───────────────────────────
    strat = BuyOnGapStrategy(
        data,
        vol_window      = 90,
        ma_window       = 20,
        gap_threshold   = -1.0,
        n_stocks        = 10,
        capital         = 100_000.0,
        position_sizing = "equal",
    )
    strat.run()
    strat.summary()

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_bog(strat, pair_name="S&P 500 Sample", save_path="bog_equal.png")

    # ── Inverse sizing (bigger gap = bigger position) ────────────────────────
    strat_inv = BuyOnGapStrategy(
        data,
        vol_window      = 20,
        ma_window       = 20,
        gap_threshold   = -1.0,
        n_stocks        = 10,
        capital         = 100_000.0,
        position_sizing = "inverse",
    )
    strat_inv.run()
    strat_inv.summary()

    plot_bog(strat_inv, pair_name="S&P 500 Sample (Inverse Sizing)",
             save_path="bog_inverse.png")

    # ── Sensitivity: vary gap threshold and n_stocks ─────────────────────────
    strat.sensitivity(
        thresholds    = (-0.5, -1.0, -1.5, -2.0, -2.5),
        n_stocks_list = (5, 10, 20),
    )