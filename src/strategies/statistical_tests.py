import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ─────────────────────────────────────────────
#  1. AUGMENTED DICKEY-FULLER TEST
# ─────────────────────────────────────────────

def adf_test(series, name="Series", significance=0.05):
    """
    Augmented Dickey-Fuller test for stationarity.

    H0: The series has a unit root (non-stationary)
    H1: The series is stationary

    Parameters
    ----------
    series : pd.Series
        Time series to test
    name : str
        Label for display purposes
    significance : float
        Significance level (default 0.05)

    Returns
    -------
    dict with keys: statistic, p_value, is_stationary, critical_values
    """

    series = series.dropna()

    result = adfuller(series, autolag="AIC")

    adf_stat   = result[0]
    p_value    = result[1]
    crit_vals  = result[4]
    is_stationary = p_value < significance

    print(f"\n{'─' * 50}")
    print(f"  Augmented Dickey-Fuller Test: {name}")
    print(f"{'─' * 50}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Critical values:")
    for key, val in crit_vals.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result        : {'STATIONARY ✓' if is_stationary else 'NON-STATIONARY ✗'}")

    return {
        "statistic"      : adf_stat,
        "p_value"        : p_value,
        "is_stationary"  : is_stationary,
        "critical_values": crit_vals,
    }


# ─────────────────────────────────────────────
#  2. HURST EXPONENT
# ─────────────────────────────────────────────

def hurst_exponent(series, min_lag=2, max_lag=100):
    """
    Estimate the Hurst exponent using the rescaled range (R/S) method.

    Interpretation:
        H < 0.5  ->  Mean-reverting (anti-persistent)
        H = 0.5  ->  Random walk (no memory)
        H > 0.5  ->  Trending (persistent)

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Price or return series
    min_lag : int
        Minimum lag for R/S calculation
    max_lag : int
        Maximum lag for R/S calculation

    Returns
    -------
    dict with keys: hurst, interpretation
    """

    series = np.array(series.dropna()) if isinstance(series, pd.Series) else np.array(series)

    lags   = range(min_lag, min(max_lag, len(series) // 2))
    rs_vals = []

    for lag in lags:
        # Split into non-overlapping subseries of length lag
        sub_series = [series[i:i + lag] for i in range(0, len(series) - lag, lag)]
        rs_sub = []

        for sub in sub_series:
            if len(sub) < 2:
                continue
            mean   = np.mean(sub)
            devs   = np.cumsum(sub - mean)
            R      = devs.max() - devs.min()
            S      = np.std(sub, ddof=1)
            if S > 0:
                rs_sub.append(R / S)

        if rs_sub:
            rs_vals.append(np.mean(rs_sub))

    lags_arr = np.array(list(lags)[:len(rs_vals)])
    rs_arr   = np.array(rs_vals)

    # Log-log regression to estimate H
    log_lags = np.log(lags_arr)
    log_rs   = np.log(rs_arr)
    H = np.polyfit(log_lags, log_rs, 1)[0]

    if H < 0.45:
        interpretation = "Mean-reverting (anti-persistent)"
    elif H > 0.55:
        interpretation = "Trending (persistent)"
    else:
        interpretation = "Random walk"

    print(f"\n{'─' * 50}")
    print(f"  Hurst Exponent")
    print(f"{'─' * 50}")
    print(f"  H             : {H:.4f}")
    print(f"  Interpretation: {interpretation}")

    return {"hurst": H, "interpretation": interpretation}


# ─────────────────────────────────────────────
#  3. VARIANCE RATIO TEST (Lo-MacKinlay)
# ─────────────────────────────────────────────

def variance_ratio_test(series, lags=(2, 4, 8, 16)):
    """
    Lo-MacKinlay Variance Ratio test.

    Under a random walk, Var(k-period returns) = k * Var(1-period returns),
    so VR(k) = 1.

    VR > 1  ->  Positive autocorrelation (momentum / trending)
    VR < 1  ->  Negative autocorrelation (mean-reverting)

    Parameters
    ----------
    series : pd.Series
        Price series (NOT returns)
    lags : tuple of int
        Holding periods to test

    Returns
    -------
    pd.DataFrame with VR statistics and z-scores per lag
    """

    series  = series.dropna()
    log_ret = np.log(series).diff().dropna()
    n       = len(log_ret)
    mu      = log_ret.mean()

    sigma_1 = np.sum((log_ret - mu) ** 2) / (n - 1)

    print(f"\n{'─' * 50}")
    print(f"  Variance Ratio Test (Lo-MacKinlay)")
    print(f"{'─' * 50}")
    print(f"  {'Lag (k)':>8}  {'VR(k)':>8}  {'Z-score':>10}  {'Signal':>20}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 10}  {'─' * 20}")

    results = []

    for k in lags:
        # k-period returns
        k_ret   = log_ret.rolling(k).sum().dropna()
        nk      = len(k_ret)
        sigma_k = np.sum((k_ret - k * mu) ** 2) / (nk * (k - 1))

        vr = sigma_k / sigma_1

        # Asymptotic variance (homoskedastic)
        delta = 2 * (2 * k - 1) * (k - 1) / (3 * k * n)
        z     = (vr - 1) / np.sqrt(delta)

        if abs(z) > 1.96:
            signal = "Trending" if vr > 1 else "Mean-reverting"
        else:
            signal = "Random walk"

        print(f"  {k:>8}  {vr:>8.4f}  {z:>10.4f}  {signal:>20}")

        results.append({"lag": k, "VR": vr, "z_score": z, "signal": signal})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  4. HALF-LIFE OF MEAN REVERSION
# ─────────────────────────────────────────────

def half_life_mean_reversion(series, name="Series"):
    """
    Estimate the half-life of mean reversion using an OLS regression
    on the Ornstein-Uhlenbeck process:

        delta_y(t) = lambda * y(t-1) + epsilon

    Half-life = -log(2) / lambda

    A shorter half-life means faster mean reversion.

    Parameters
    ----------
    series : pd.Series
        Price or spread series
    name : str
        Label for display

    Returns
    -------
    dict with keys: half_life, lambda_, is_mean_reverting
    """

    series = series.dropna()

    y      = series.values
    y_lag  = y[:-1]
    delta_y = np.diff(y)

    # OLS: delta_y = lambda * y_lag + epsilon
    X      = add_constant(y_lag)
    model  = OLS(delta_y, X).fit()
    lambda_ = model.params[1]

    if lambda_ >= 0:
        half_life = np.nan
        is_mean_reverting = False
    else:
        half_life = -np.log(2) / lambda_
        is_mean_reverting = True

    print(f"\n{'─' * 50}")
    print(f"  Half-Life of Mean Reversion: {name}")
    print(f"{'─' * 50}")
    print(f"  Lambda (speed): {lambda_:.6f}")
    if is_mean_reverting:
        print(f"  Half-life     : {half_life:.2f} periods")
        print(f"  Result        : Mean-reverting ✓")
    else:
        print(f"  Half-life     : N/A (lambda >= 0, not mean-reverting)")
        print(f"  Result        : NOT mean-reverting ✗")

    return {
        "half_life"        : half_life,
        "lambda_"          : lambda_,
        "is_mean_reverting": is_mean_reverting,
    }


# ─────────────────────────────────────────────
#  5. COINTEGRATED AUGMENTED DICKEY-FULLER (CADF)
# ─────────────────────────────────────────────

def cadf_test(series_y, series_x, name_y="Y", name_x="X", significance=0.05):
    """
    Cointegrated Augmented Dickey-Fuller (CADF) test.

    Tests whether two non-stationary series are cointegrated:
    1. Run OLS regression: Y = beta * X + epsilon
    2. Run ADF on the residuals (spread)
    3. If residuals are stationary -> series are cointegrated

    Parameters
    ----------
    series_y : pd.Series
        Dependent price series
    series_x : pd.Series
        Independent price series
    name_y : str
        Label for Y
    name_x : str
        Label for X
    significance : float
        Significance level

    Returns
    -------
    dict with keys: hedge_ratio, adf_stat, p_value, is_cointegrated, spread
    """

    # Align series
    df = pd.concat([series_y, series_x], axis=1).dropna()
    df.columns = [name_y, name_x]

    # Step 1: OLS regression to find hedge ratio
    X = add_constant(df[name_x])
    model = OLS(df[name_y], X).fit()
    hedge_ratio = model.params[name_x]
    intercept   = model.params["const"]

    # Step 2: Compute spread (residuals)
    spread = df[name_y] - hedge_ratio * df[name_x] - intercept

    # Step 3: ADF on spread
    result = adfuller(spread, autolag="AIC")
    adf_stat = result[0]
    p_value  = result[1]
    crit_vals = result[4]
    is_cointegrated = p_value < significance

    print(f"\n{'─' * 50}")
    print(f"  CADF Test: {name_y} ~ {name_x}")
    print(f"{'─' * 50}")
    print(f"  Hedge ratio   : {hedge_ratio:.4f}")
    print(f"  Intercept     : {intercept:.4f}")
    print(f"  ADF on spread : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print(f"  Critical values:")
    for key, val in crit_vals.items():
        print(f"    {key}: {val:.4f}")
    print(f"  Result        : {'COINTEGRATED ✓' if is_cointegrated else 'NOT COINTEGRATED ✗'}")

    return {
        "hedge_ratio"    : hedge_ratio,
        "intercept"      : intercept,
        "adf_stat"       : adf_stat,
        "p_value"        : p_value,
        "is_cointegrated": is_cointegrated,
        "spread"         : spread,
    }



# ─────────────────────────────────────────────
#  6. JOHANSEN COINTEGRATION TEST
# ─────────────────────────────────────────────

def johansen_test(series_list, names=None, det_order=0, k_ar_diff=1, significance=0.05):
    """
    Johansen cointegration test for a system of N time series.

    Unlike CADF (which tests pairs), Johansen tests an entire group at once
    and determines how many cointegrating relationships exist (the rank r).

    Two test statistics are reported:
      - Trace statistic    : tests H0: rank <= r  vs  H1: rank > r
      - Max-eigenvalue     : tests H0: rank = r   vs  H1: rank = r+1

    Parameters
    ----------
    series_list : list of pd.Series
        List of price series to test (all must be I(1))
    names : list of str, optional
        Labels for each series
    det_order : int
        Deterministic term: -1 = none, 0 = constant, 1 = constant + trend
    k_ar_diff : int
        Number of lagged differences in the VECM (default 1)
    significance : float
        Significance level: 0.10, 0.05, or 0.01

    Returns
    -------
    dict with keys:
        n_cointegrating_vectors : int, number of significant cointegrating relationships
        hedge_ratios            : np.ndarray, eigenvectors (columns = cointegrating vectors)
        trace_stats             : list of trace statistics
        max_eigen_stats         : list of max-eigenvalue statistics
        spreads                 : pd.DataFrame of cointegrating spreads
    """

    sig_map = {0.10: 0, 0.05: 1, 0.01: 2}
    if significance not in sig_map:
        raise ValueError("significance must be 0.10, 0.05, or 0.01")
    sig_idx = sig_map[significance]

    n = len(series_list)
    if names is None:
        names = [f"S{i+1}" for i in range(n)]

    df = pd.concat(series_list, axis=1).dropna()
    df.columns = names

    result = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stats     = result.lr1
    trace_crits     = result.cvt
    max_eigen_stats = result.lr2
    max_eigen_crits = result.cvm
    eigenvectors    = result.evec

    # Determine rank: sequential testing stops at first non-rejection
    n_cointegrating = 0
    for i in range(n):
        if trace_stats[i] > trace_crits[i, sig_idx]:
            n_cointegrating += 1
        else:
            break

    print(f"\n{chr(9472) * 60}")
    print(f"  Johansen Cointegration Test")
    print(f"  Series : {', '.join(names)}")
    print(f"  Significance: {int(significance * 100)}%")
    print(f"{chr(9472) * 60}")
    print(f"\n  Trace Statistic")
    print(f"  {'H0: rank':>12}  {'Statistic':>12}  {'Crit. Value':>12}  {'Reject H0':>10}")
    print(f"  {chr(9472) * 12}  {chr(9472) * 12}  {chr(9472) * 12}  {chr(9472) * 10}")
    for i in range(n):
        reject = "Yes ✓" if trace_stats[i] > trace_crits[i, sig_idx] else "No  ✗"
        print(f"  {'<= ' + str(i):>12}  {trace_stats[i]:>12.4f}  {trace_crits[i, sig_idx]:>12.4f}  {reject:>10}")

    print(f"\n  Max-Eigenvalue Statistic")
    print(f"  {'H0: rank':>12}  {'Statistic':>12}  {'Crit. Value':>12}  {'Reject H0':>10}")
    print(f"  {chr(9472) * 12}  {chr(9472) * 12}  {chr(9472) * 12}  {chr(9472) * 10}")
    for i in range(n):
        reject = "Yes ✓" if max_eigen_stats[i] > max_eigen_crits[i, sig_idx] else "No  ✗"
        print(f"  {'= ' + str(i):>12}  {max_eigen_stats[i]:>12.4f}  {max_eigen_crits[i, sig_idx]:>12.4f}  {reject:>10}")

    print(f"\n  Cointegrating rank : {n_cointegrating}")
    if n_cointegrating == 0:
        print(f"  Result             : No cointegration found ✗")
    elif n_cointegrating == 1:
        print(f"  Result             : 1 cointegrating relationship found ✓")
        hedge = eigenvectors[:, 0]
        print(f"  Cointegrating vector (hedge ratios):")
        for name, h in zip(names, hedge):
            print(f"    {name}: {h:.6f}")
    else:
        print(f"  Result             : {n_cointegrating} cointegrating relationships found ✓")

    spreads = pd.DataFrame(index=df.index)
    for i in range(n_cointegrating):
        vec = eigenvectors[:, i]
        spreads[f"spread_{i+1}"] = df.values @ vec

    return {
        "n_cointegrating_vectors": n_cointegrating,
        "hedge_ratios"           : eigenvectors[:, :n_cointegrating],
        "eigenvectors_all"       : eigenvectors,
        "trace_stats"            : trace_stats,
        "max_eigen_stats"        : max_eigen_stats,
        "spreads"                : spreads,
    }

# ─────────────────────────────────────────────
#  USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("Downloading data...")
    ewa = yf.download("EWA", start="2015-01-01", end="2020-01-01")["Close"].squeeze()
    ewc = yf.download("EWC", start="2015-01-01", end="2020-01-01")["Close"].squeeze()
    gld = yf.download("GLD", start="2015-01-01", end="2020-01-01")["Close"].squeeze()
    ewa.name, ewc.name, gld.name = "EWA", "EWC", "GLD"

    # 1. Stationarity — prices should be I(1), returns should be I(0)
    adf_test(ewa, name="EWA Price")
    adf_test(ewa.pct_change().dropna(), name="EWA Returns")

    # 2. Hurst on EWA — expect H > 0.5 (prices trend), H < 0.5 on spread
    hurst_exponent(ewa)

    # 3. Variance ratio on EWA
    variance_ratio_test(ewa, lags=(2, 4, 8, 16, 32))

    # 4. CADF: EWA vs EWC — the pair you're actually trading
    cadf_result = cadf_test(ewa, ewc, name_y="EWA", name_x="EWC")

    # 5. Half-life on the SPREAD, not on returns — this is what matters for the strategy
    half_life_mean_reversion(cadf_result["spread"], name="EWA/EWC spread")

    # 6. Hurst on the spread — expect H < 0.5 if genuinely mean-reverting
    hurst_exponent(cadf_result["spread"])

    # 7. Johansen on EWA + EWC
    johansen_test([ewa, ewc], names=["EWA", "EWC"])

    # 8. Also test EWA, EWC, GLD as a trio
    johansen_test([ewa, ewc, gld], names=["EWA", "EWC", "GLD"])