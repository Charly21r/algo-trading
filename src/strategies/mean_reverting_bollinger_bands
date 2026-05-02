import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ─────────────────────────────────────────────────────────────────────────────
#  HEDGE RATIO METHODS
#  Four ways to compute the hedge ratio beta between two series Y and X.
#  The spread is then defined as: spread = Y - beta * X
# ─────────────────────────────────────────────────────────────────────────────

def hedge_ratio_prices(series_y, series_x):
    """
    OLS regression on raw price levels.

        Y = beta * X + epsilon

    Simple and fast. Assumes a linear relationship in price space.
    Sensitive to non-stationarity and scale differences between the two series.

    Returns
    -------
    beta : float
        Hedge ratio
    spread : pd.Series
        Residual spread: Y - beta * X
    """

    df = pd.concat([series_y, series_x], axis=1).dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]

    X = add_constant(x)
    beta = OLS(y, X).fit().params.iloc[1]
    spread = y - beta * x

    return beta, spread


def hedge_ratio_log(series_y, series_x):
    """
    OLS regression on log-price levels.

        log(Y) = beta * log(X) + epsilon

    More appropriate when prices are multiplicative (which they usually are).
    The hedge ratio beta here represents an elasticity: a 1% move in X
    corresponds to a beta% move in Y.

    Returns
    -------
    beta : float
        Log-price hedge ratio
    spread : pd.Series
        Log spread: log(Y) - beta * log(X)
    """

    df = pd.concat([series_y, series_x], axis=1).dropna()
    log_y = np.log(df.iloc[:, 0])
    log_x = np.log(df.iloc[:, 1])

    X = add_constant(log_x)
    beta = OLS(log_y, X).fit().params.iloc[1]
    spread = log_y - beta * log_x

    return beta, spread


def hedge_ratio_ratio(series_y, series_x):
    """
    Price ratio as the spread.

        spread = Y / X

    No OLS involved: the hedge ratio is implicitly 1 unit of X per 1 unit of Y.
    Simple and interpretable. Works well when both series are in the same units
    and trade at similar price levels (e.g. two ETFs tracking related indices).

    Returns
    -------
    beta : float
        Always 1.0 (ratio hedge)
    spread : pd.Series
        Y / X
    """

    df = pd.concat([series_y, series_x], axis=1).dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]

    spread = y / x

    return 1.0, spread


def hedge_ratio_kalman(series_y, series_x, delta=1e-4, vt=1.0):
    """
    Kalman Filter to estimate a time-varying hedge ratio.

    Models the system as:
        Y(t) = beta(t) * X(t) + alpha(t) + noise      (observation)
        [alpha, beta](t) = [alpha, beta](t-1) + w(t)  (state transition)

    The state [alpha, beta] evolves as a random walk. The Kalman filter
    recursively updates the hedge ratio as new data arrives, making it
    adaptive to structural breaks and regime changes.

    Parameters
    ----------
    delta : float
        Process noise scaling. Higher = faster adaptation to new regimes,
        lower = smoother, slower-adapting hedge ratio.
    vt : float
        Observation noise variance.

    Returns
    -------
    beta : pd.Series
        Time-varying hedge ratio at each timestep
    spread : pd.Series
        Y - alpha(t) - beta(t) * X  (Kalman residual)
    alpha : pd.Series
        Time-varying intercept
    """

    df = pd.concat([series_y, series_x], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    n = len(y)

    # State: [alpha, beta], dim=2
    # Observation matrix F_t = [1, x_t]
    # Transition matrix = identity (random walk)

    # Process noise covariance
    Wt = delta / (1 - delta) * np.eye(2)

    # Initial state and covariance
    theta = np.zeros(2)       # [alpha_0, beta_0]
    P     = np.zeros((2, 2))  # state covariance

    alphas = np.zeros(n)
    betas  = np.zeros(n)
    spreads = np.zeros(n)

    for t in range(n):
        F = np.array([[1.0, x[t]]])   # observation matrix (1x2)

        # Prediction step
        P_pred = P + Wt

        # Innovation (prediction error)
        y_hat  = float(F @ theta)
        innov  = y[t] - y_hat

        # Innovation covariance
        S = float(F @ P_pred @ F.T) + vt

        # Kalman gain
        K = (P_pred @ F.T) / S        # (2x1)

        # Update state
        theta = theta + K.flatten() * innov
        P     = P_pred - K @ F @ P_pred

        alphas[t]  = theta[0]
        betas[t]   = theta[1]
        spreads[t] = innov   # innovation = Y - alpha - beta*X

    index = df.index
    return (
        pd.Series(betas,   index=index, name="kalman_beta"),
        pd.Series(spreads, index=index, name="spread"),
        pd.Series(alphas,  index=index, name="kalman_alpha"),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BOLLINGER BANDS MEAN REVERSION STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

class BollingerBandsMeanReversion:
    """
    Pairs trading strategy using Bollinger Bands on a cointegrated spread.

    Entry logic:
        - Go LONG  the spread when z-score <= -entry_z  (spread too low)
        - Go SHORT the spread when z-score >=  entry_z  (spread too high)

    Exit logic:
        - Close LONG  when z-score >= -exit_z  (spread reverts toward mean)
        - Close SHORT when z-score <=  exit_z  (spread reverts toward mean)

    Stop-loss logic (optional):
        - Close any position when |z-score| >= stop_z

    For each hedge method the spread is defined differently:
        prices  ->  spread = Y - beta * X
        log     ->  spread = log(Y) - beta * log(X)
        ratio   ->  spread = Y / X
        kalman  ->  spread = Kalman filter innovation (time-varying beta)

    Parameters
    ----------
    series_y : pd.Series
        First leg of the pair (the series being traded)
    series_x : pd.Series
        Second leg of the pair (the hedge)
    hedge_method : str
        One of: 'prices', 'log', 'ratio', 'kalman'
    window : int
        Rolling window for z-score calculation (mean and std of spread)
    entry_z : float
        Z-score threshold to open a position (default 2.0)
    exit_z : float
        Z-score threshold to close a position (default 0.5)
    stop_z : float or None
        Z-score threshold for stop-loss (default None = no stop)
    kalman_delta : float
        Process noise for Kalman filter (only used when hedge_method='kalman')
    """

    HEDGE_METHODS = ("prices", "log", "ratio", "kalman")

    def __init__(
        self,
        series_y,
        series_x,
        hedge_method  = "log",
        window        = 20,
        entry_z       = 2.0,
        exit_z        = 0.5,
        stop_z        = None,
        kalman_delta  = 1e-4,
    ):
        if hedge_method not in self.HEDGE_METHODS:
            raise ValueError(f"hedge_method must be one of {self.HEDGE_METHODS}")

        self.series_y     = series_y.dropna()
        self.series_x     = series_x.dropna()
        self.hedge_method = hedge_method
        self.window       = window
        self.entry_z      = entry_z
        self.exit_z       = exit_z
        self.stop_z       = stop_z
        self.kalman_delta = kalman_delta

        self.spread       = None
        self.z_score      = None
        self.signals      = None
        self.beta         = None   # scalar or Series depending on method

    # ──────────────────────────────────────────
    #  STEP 1: compute spread
    # ──────────────────────────────────────────

    def compute_spread(self):
        """Compute the spread using the selected hedge ratio method."""

        if self.hedge_method == "prices":
            self.beta, self.spread = hedge_ratio_prices(self.series_y, self.series_x)

        elif self.hedge_method == "log":
            self.beta, self.spread = hedge_ratio_log(self.series_y, self.series_x)

        elif self.hedge_method == "ratio":
            self.beta, self.spread = hedge_ratio_ratio(self.series_y, self.series_x)

        elif self.hedge_method == "kalman":
            self.beta, self.spread, self.alpha_kalman = hedge_ratio_kalman(
                self.series_y, self.series_x, delta=self.kalman_delta
            )

        return self.spread

    # ──────────────────────────────────────────
    #  STEP 2: compute rolling z-score
    # ──────────────────────────────────────────

    def compute_zscore(self):
        """
        Compute rolling z-score of the spread:
            z(t) = (spread(t) - mean(spread, window)) / std(spread, window)

        For Kalman, the spread is already zero-mean (innovations), so we
        normalise by rolling std only.
        """

        if self.spread is None:
            self.compute_spread()

        roll_mean = self.spread.rolling(self.window).mean()
        roll_std  = self.spread.rolling(self.window).std()

        if self.hedge_method == "kalman":
            # Kalman innovations are approximately zero-mean; normalise by std only
            self.z_score = self.spread / roll_std
        else:
            self.z_score = (self.spread - roll_mean) / roll_std

        return self.z_score

    # ──────────────────────────────────────────
    #  STEP 3: generate signals
    # ──────────────────────────────────────────

    def generate_signals(self):
        """
        Generate entry/exit signals from the z-score.

        Signal convention:
             1 = long  the spread (long Y, short X)
            -1 = short the spread (short Y, long X)
             0 = flat

        Returns
        -------
        pd.DataFrame with columns:
            spread, z_score, signal, position
        """

        if self.z_score is None:
            self.compute_zscore()

        z = self.z_score.copy()

        signal   = pd.Series(0, index=z.index)
        position = pd.Series(0, index=z.index)

        current_pos = 0

        for t in range(len(z)):
            zt = z.iloc[t]

            if np.isnan(zt):
                position.iloc[t] = 0
                continue

            # Stop-loss: close any open position
            if self.stop_z is not None and current_pos != 0:
                if abs(zt) >= self.stop_z:
                    current_pos = 0

            # Entry signals (only when flat)
            if current_pos == 0:
                if zt <= -self.entry_z:
                    current_pos =  1   # long spread
                    signal.iloc[t] =  1
                elif zt >= self.entry_z:
                    current_pos = -1   # short spread
                    signal.iloc[t] = -1

            # Exit signals (when in a position)
            elif current_pos == 1:
                if zt >= -self.exit_z:
                    current_pos = 0
                    signal.iloc[t] = 0

            elif current_pos == -1:
                if zt <= self.exit_z:
                    current_pos = 0
                    signal.iloc[t] = 0

            position.iloc[t] = current_pos

        self.signals = pd.DataFrame({
            "spread"  : self.spread,
            "z_score" : z,
            "signal"  : signal,
            "position": position,
        })

        return self.signals

    # ──────────────────────────────────────────
    #  STEP 4: compute returns
    # ──────────────────────────────────────────

    def compute_returns(self):
        """
        Compute strategy returns.

        Position at t is determined at close of t, so returns are realised
        from t to t+1. The position is held in the spread, so:

            spread_return(t) = spread(t) - spread(t-1)   [for price/ratio spreads]
            spread_return(t) = spread(t)                  [for Kalman innovations,
                                                           which are already differences]

        Returns
        -------
        pd.DataFrame with additional columns:
            spread_return, strategy_return, cumulative_return
        """

        if self.signals is None:
            self.generate_signals()

        df = self.signals.copy()

        if self.hedge_method == "kalman":
            # Kalman spread is already an innovation (difference-like)
            df["spread_return"] = df["spread"]
        else:
            df["spread_return"] = df["spread"].diff()

        # Strategy return = position(t-1) * spread_return(t)
        df["strategy_return"]  = df["position"].shift(1) * df["spread_return"]
        df["cumulative_return"] = df["strategy_return"].cumsum()

        return df

    # ──────────────────────────────────────────
    #  STEP 5: performance summary
    # ──────────────────────────────────────────

    def summary(self):
        """
        Print and return key performance metrics.

        Returns
        -------
        dict with Sharpe ratio, total return, max drawdown, win rate,
        number of trades, and average holding period.
        """

        df = self.compute_returns()
        rets = df["strategy_return"].dropna()

        total_return  = df["cumulative_return"].iloc[-1]
        sharpe        = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else np.nan
        cum           = df["cumulative_return"]
        drawdown      = cum - cum.cummax()
        max_drawdown  = drawdown.min()

        # Trades: count transitions into a non-zero position
        pos       = df["position"]
        entries   = ((pos != 0) & (pos.shift(1) == 0)).sum()
        exits     = ((pos == 0) & (pos.shift(1) != 0)).sum()
        n_trades  = min(entries, exits)

        # Win rate
        trade_rets = []
        in_trade   = False
        entry_cum  = 0.0
        for t in range(len(df)):
            p = df["position"].iloc[t]
            c = df["cumulative_return"].iloc[t]
            if not in_trade and p != 0:
                in_trade   = True
                entry_cum  = c
            elif in_trade and p == 0:
                in_trade = False
                trade_rets.append(c - entry_cum)
        win_rate = np.mean([r > 0 for r in trade_rets]) if trade_rets else np.nan

        # Average holding period
        holding = (pos != 0).sum() / max(n_trades, 1)

        print(f"\n{'─' * 55}")
        print(f"  Bollinger Bands Strategy — {self.hedge_method.upper()} hedge")
        print(f"  entry_z={self.entry_z}  exit_z={self.exit_z}  window={self.window}")
        print(f"{'─' * 55}")
        print(f"  Total return      : {total_return:>10.4f}")
        print(f"  Sharpe ratio      : {sharpe:>10.4f}")
        print(f"  Max drawdown      : {max_drawdown:>10.4f}")
        print(f"  Number of trades  : {n_trades:>10}")
        print(f"  Win rate          : {win_rate * 100:>9.1f}%")
        print(f"  Avg holding (days): {holding:>10.1f}")

        return {
            "hedge_method"  : self.hedge_method,
            "total_return"  : total_return,
            "sharpe"        : sharpe,
            "max_drawdown"  : max_drawdown,
            "n_trades"      : n_trades,
            "win_rate"      : win_rate,
            "avg_holding"   : holding,
            "returns_df"    : df,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARE ALL FOUR HEDGE METHODS
# ─────────────────────────────────────────────────────────────────────────────

def compare_hedge_methods(
    series_y,
    series_x,
    window   = 20,
    entry_z  = 2.0,
    exit_z   = 0.5,
    stop_z   = None,
):
    """
    Run the Bollinger Bands strategy with all four hedge methods and
    print a comparison table.

    Parameters
    ----------
    series_y, series_x : pd.Series
        The two legs of the pair
    window, entry_z, exit_z, stop_z : strategy parameters

    Returns
    -------
    pd.DataFrame
        Summary metrics for all four methods, sorted by Sharpe ratio
    """

    results = []

    for method in BollingerBandsMeanReversion.HEDGE_METHODS:
        strat = BollingerBandsMeanReversion(
            series_y,
            series_x,
            hedge_method = method,
            window       = window,
            entry_z      = entry_z,
            exit_z       = exit_z,
            stop_z       = stop_z,
        )
        res = strat.summary()
        results.append({
            "Method"          : method,
            "Total Return"    : round(res["total_return"],  4),
            "Sharpe"          : round(res["sharpe"],        4),
            "Max Drawdown"    : round(res["max_drawdown"],  4),
            "Trades"          : res["n_trades"],
            "Win Rate (%)"    : round(res["win_rate"] * 100, 1),
            "Avg Hold (days)" : round(res["avg_holding"],   1),
        })

    df_results = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

    print(f"\n{'═' * 75}")
    print(f"  COMPARISON — All Hedge Methods")
    print(f"  entry_z={entry_z}  exit_z={exit_z}  window={window}")
    print(f"{'═' * 75}")
    print(df_results.to_string(index=False))

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
#  USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import yfinance as yf

    print("Downloading data...")
    spy = yf.download("SPY", start="2020-01-01", end="2025-01-01")["Close"].squeeze()
    qqq = yf.download("QQQ", start="2020-01-01", end="2025-01-01")["Close"].squeeze()
    spy.name, qqq.name = "SPY", "QQQ"

    # ── Single strategy with log hedge ──────────────────────────────────────
    strat = BollingerBandsMeanReversion(
        spy, qqq,
        hedge_method = "log",
        window       = 20,
        entry_z      = 2.0,
        exit_z       = 0.5,
        stop_z       = 3.5,
    )
    result = strat.summary()

    # ── Single strategy with Kalman hedge ───────────────────────────────────
    strat_k = BollingerBandsMeanReversion(
        spy, qqq,
        hedge_method  = "kalman",
        window        = 20,
        entry_z       = 2.0,
        exit_z        = 0.5,
        kalman_delta  = 1e-4,
    )
    result_k = strat_k.summary()

    # ── Compare all four methods ─────────────────────────────────────────────
    comparison = compare_hedge_methods(
        spy, qqq,
        window  = 20,
        entry_z = 2.0,
        exit_z  = 0.5,
        stop_z  = 3.5,
    )
