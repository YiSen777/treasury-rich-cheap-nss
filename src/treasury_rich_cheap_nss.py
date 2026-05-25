"""
Treasury rich-cheap strategy using Nelson-Siegel-Svensson curve fitting.

This script downloads U.S. Treasury yield data from FRED, calibrates a
Nelson-Siegel-Svensson fair-value curve, identifies rich and cheap maturities
from residual z-scores, and backtests a duration-adjusted relative-value
strategy against simple Treasury benchmarks.
"""

# The warning filter keeps optimization and plotting output readable during an
# end-to-end research run.
import warnings
warnings.filterwarnings("ignore")

# These libraries provide numerical arrays, tabular data handling, visualization,
# date ranges, and FRED data access.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as web

# These scientific-computing utilities support NSS calibration and PCA-based
# yield-curve factor analysis.
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Configuration
# ============================================================

# This configuration block defines the historical sample, trading thresholds,
# maturity universe, and risk-scaling assumptions used by the strategy.
START_DATE = dt.datetime(2010, 1, 1)
END_DATE = dt.datetime(2025, 1, 1)

ZSCORE_WINDOW = 252          # 1-year rolling window
ENTRY_Z = 1.5                # signal threshold
TRANSACTION_COST_BPS = 1.0   # one-way cost in bps
GROSS_LEVERAGE = 1.0         # total absolute weights sum to 1
TOP_N = 2                    # long top N cheap, short top N rich

USE_MATURITIES = [
    "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"
]

MATURITY_YEARS = {
    "3M": 0.25,
    "6M": 0.50,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# Rough modified-duration proxies.
# For a real implementation, replace this with actual bond or futures DV01.
DURATION_PROXY = {
    "3M": 0.25,
    "6M": 0.50,
    "1Y": 0.95,
    "2Y": 1.8,
    "3Y": 2.7,
    "5Y": 4.5,
    "7Y": 6.2,
    "10Y": 8.0,
    "20Y": 15.0,
    "30Y": 20.0,
}


# ============================================================
# 2. Data Fetching from FRED
# ============================================================

# This function downloads Treasury yield curve data from FRED and standardizes
# the columns into readable maturity labels.
def fetch_fred_yields(start_date, end_date):
    fred_tickers = {
        "3M": "DGS3MO",
        "6M": "DGS6MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "3Y": "DGS3",
        "5Y": "DGS5",
        "7Y": "DGS7",
        "10Y": "DGS10",
        "20Y": "DGS20",
        "30Y": "DGS30",
    }

    tickers = [fred_tickers[m] for m in USE_MATURITIES]

    data = web.DataReader(tickers, "fred", start_date, end_date)
    data.columns = USE_MATURITIES

    # Keep business-day observations where all maturities exist.
    data = data.ffill().dropna()

    return data


# ============================================================
# 3. Nelson-Siegel-Svensson Model
# ============================================================

# This function evaluates the Nelson-Siegel-Svensson yield curve for one or more
# maturities.
def nss_yield(maturity, beta0, beta1, beta2, tau1, beta3, tau2):
    """
    Nelson-Siegel-Svensson yield curve.

    y(m) = beta0
         + beta1 * A(m, tau1)
         + beta2 * [A(m, tau1) - exp(-m/tau1)]
         + beta3 * [A(m, tau2) - exp(-m/tau2)]

    where A(m,tau) = (1 - exp(-m/tau)) / (m/tau)
    """
    m = np.asarray(maturity, dtype=float)

    tau1 = max(tau1, 1e-6)
    tau2 = max(tau2, 1e-6)

    x1 = m / tau1
    x2 = m / tau2

    A1 = (1 - np.exp(-x1)) / x1
    A2 = (1 - np.exp(-x2)) / x2

    term1 = beta0
    term2 = beta1 * A1
    term3 = beta2 * (A1 - np.exp(-x1))
    term4 = beta3 * (A2 - np.exp(-x2))

    return term1 + term2 + term3 + term4


# This objective function measures the squared fitting error between NSS model
# yields and observed market yields.
def nss_sse(params, maturities, market_yields):
    fitted = nss_yield(maturities, *params)
    residuals = fitted - market_yields
    return np.sum(residuals ** 2)


# This function calibrates NSS parameters for one trading day using bounded
# numerical optimization.
def calibrate_nss_single_day(maturities, market_yields, initial_guess):
    """
    Calibrate NSS parameters for one date.
    market_yields should be in decimal form, e.g. 0.045.
    """

    bounds = [
        (0.0001, 0.20),    # beta0
        (-0.30, 0.30),     # beta1
        (-0.30, 0.30),     # beta2
        (0.05, 30.0),      # tau1
        (-0.30, 0.30),     # beta3
        (0.05, 30.0),      # tau2
    ]

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    initial_guess = np.clip(np.asarray(initial_guess), lower, upper)

    result = minimize(
        nss_sse,
        initial_guess,
        args=(maturities, market_yields),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        return result.x, True, result.fun

    return np.full(6, np.nan), False, np.nan


# This function calibrates the NSS model across the full historical yield
# dataset and returns parameters, fitted curves, and residuals.
def calibrate_nss_history(yields_percent):
    """
    Fit NSS curve every day.

    Returns:
        params_df
        fitted_df
        residual_df
    """

    yields_decimal = yields_percent / 100.0
    maturities = np.array([MATURITY_YEARS[m] for m in USE_MATURITIES], dtype=float)

    params_history = []
    fitted_history = []
    residual_history = []

    first_curve = yields_decimal.iloc[0].values

    initial_guess = np.array([
        first_curve[-1],               # beta0, long-end level
        first_curve[0] - first_curve[-1],  # beta1, short-long spread
        0.0,                           # beta2
        1.5,                           # tau1
        0.0,                           # beta3
        5.0,                           # tau2
    ])

    last_good_params = initial_guess.copy()

    print("Calibrating NSS curve day by day...")

    for date, row in yields_decimal.iterrows():
        market_yields = row.values.astype(float)

        params, success, sse = calibrate_nss_single_day(
            maturities=maturities,
            market_yields=market_yields,
            initial_guess=last_good_params,
        )

        if success:
            last_good_params = params
            fitted = nss_yield(maturities, *params)
            residual = market_yields - fitted
        else:
            fitted = np.full(len(USE_MATURITIES), np.nan)
            residual = np.full(len(USE_MATURITIES), np.nan)

        params_history.append(params)
        fitted_history.append(fitted)
        residual_history.append(residual)

    params_df = pd.DataFrame(
        params_history,
        index=yields_percent.index,
        columns=["beta0", "beta1", "beta2", "tau1", "beta3", "tau2"],
    )

    fitted_df = pd.DataFrame(
        fitted_history,
        index=yields_percent.index,
        columns=USE_MATURITIES,
    )

    residual_df = pd.DataFrame(
        residual_history,
        index=yields_percent.index,
        columns=USE_MATURITIES,
    )

    return params_df, fitted_df, residual_df

# ============================================================
# 3.5 KMO Test Before PCA
# ============================================================

# This function calculates the Kaiser-Meyer-Olkin statistic to check whether the
# yield-change data are suitable for PCA.
def calculate_kmo(data: pd.DataFrame):
    """
    Calculate Kaiser-Meyer-Olkin (KMO) measure for sampling adequacy.

    KMO is used to evaluate whether variables share enough common variance
    to justify PCA or factor analysis.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of variables.
        In this project, use daily Treasury yield changes.

    Returns
    -------
    overall_kmo : float
        Overall KMO statistic.

    kmo_per_variable : pd.Series
        KMO statistic for each maturity.
    """

    data = data.dropna()

    # Correlation matrix
    corr = data.corr().values

    # Use pseudo-inverse for numerical stability
    inv_corr = np.linalg.pinv(corr)

    # Partial correlation matrix
    partial_corr = np.zeros_like(corr)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i == j:
                partial_corr[i, j] = 0.0
            else:
                partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(
                    inv_corr[i, i] * inv_corr[j, j]
                )

    # Squared correlations and partial correlations
    corr_squared = corr ** 2
    partial_corr_squared = partial_corr ** 2

    # Remove diagonal terms
    np.fill_diagonal(corr_squared, 0.0)
    np.fill_diagonal(partial_corr_squared, 0.0)

    # Overall KMO
    numerator = np.sum(corr_squared)
    denominator = numerator + np.sum(partial_corr_squared)

    overall_kmo = numerator / denominator

    # KMO per variable
    kmo_per_variable_values = np.sum(corr_squared, axis=0) / (
        np.sum(corr_squared, axis=0) + np.sum(partial_corr_squared, axis=0)
    )

    kmo_per_variable = pd.Series(
        kmo_per_variable_values,
        index=data.columns,
        name="KMO"
    )

    return overall_kmo, kmo_per_variable


# This helper converts the numeric KMO statistic into an easy-to-read quality
# label.
def interpret_kmo(kmo_value: float) -> str:
    """
    Interpret KMO statistic.
    """

    if kmo_value >= 0.90:
        return "excellent"
    elif kmo_value >= 0.80:
        return "great"
    elif kmo_value >= 0.70:
        return "acceptable"
    elif kmo_value >= 0.60:
        return "mediocre"
    elif kmo_value >= 0.50:
        return "poor"
    else:
        return "not suitable"


# This function runs the KMO test on daily Treasury yield changes and prints a
# compact diagnostic report.
def run_kmo_test(yields_percent: pd.DataFrame):
    """
    Run KMO test on Treasury yield changes.

    We use yield changes rather than yield levels because PCA on yield curve
    risk is usually applied to changes in yields.
    """

    yield_changes = yields_percent.diff().dropna()

    overall_kmo, kmo_by_maturity = calculate_kmo(yield_changes)

    print("\nKMO Test Results:")
    print(f"Overall KMO: {overall_kmo:.4f}")
    print(f"Interpretation: {interpret_kmo(overall_kmo)}")

    print("\nKMO by Maturity:")
    print(kmo_by_maturity.round(4))

    return overall_kmo, kmo_by_maturity

# ============================================================
# 4. PCA Analysis
# ============================================================

# This function extracts the first three principal components of yield changes,
# commonly interpreted as level, slope, and curvature.
def run_pca_analysis(yields_percent):
    """
    PCA on daily yield changes.

    Usually:
        PC1 ≈ level
        PC2 ≈ slope
        PC3 ≈ curvature
    """

    yield_changes = yields_percent.diff().dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(yield_changes)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=USE_MATURITIES,
        columns=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
    )

    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
    )

    pc_scores = pd.DataFrame(
        pcs,
        index=yield_changes.index,
        columns=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
    )

    return loadings, explained, pc_scores


# This plotting function visualizes PCA loadings across the maturity curve.
def plot_pca_loadings(loadings, explained):
    plt.figure(figsize=(10, 6))

    for col in loadings.columns:
        plt.plot(loadings.index, loadings[col], marker="o", label=col)

    plt.title("PCA Loadings of Treasury Yield Changes")
    plt.xlabel("Maturity")
    plt.ylabel("Loading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nPCA Explained Variance:")
    print(explained)

# ============================================================
# 5. Residual Z-Score and Rich-Cheap Signals
# ============================================================

# This function converts NSS residuals into rolling z-scores for rich-cheap
# signal generation.
def compute_residual_zscores(residual_df, window=252):
    """
    rolling z-score of NSS residuals.

    residual = market yield - fitted yield.

    positive z-score:
        market yield unusually above fair yield
        bond price unusually cheap

    negative z-score:
        market yield unusually below fair yield
        bond price unusually rich
    """

    rolling_mean = residual_df.rolling(window, min_periods=window // 2).mean()
    rolling_std = residual_df.rolling(window, min_periods=window // 2).std()

    zscores = (residual_df - rolling_mean) / rolling_std
    zscores = zscores.replace([np.inf, -np.inf], np.nan)

    return zscores

# This function builds daily long-cheap and short-rich positions using
# inverse-duration scaling.
def build_rich_cheap_positions(zscores, top_n=2, entry_z=1.0, gross_leverage=1.0):
    """
    Build daily long-cheap / short-rich portfolio.

    Long:
        maturities with highest positive residual z-score.

    Short:
        maturities with lowest negative residual z-score.

    We use inverse-duration scaling so that long and short legs
    have more balanced rate-risk exposure.
    """

    positions = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)

    durations = pd.Series(DURATION_PROXY)

    for date, row in zscores.iterrows():
        row = row.dropna()

        if row.empty:
            continue

        cheap = row[row > entry_z].sort_values(ascending=False).head(top_n)
        rich = row[row < -entry_z].sort_values(ascending=True).head(top_n)

        daily_pos = pd.Series(0.0, index=zscores.columns)

        if len(cheap) > 0:
            cheap_weights = 1.0 / durations.loc[cheap.index]
            cheap_weights = cheap_weights / cheap_weights.abs().sum()
            daily_pos.loc[cheap.index] = cheap_weights

        if len(rich) > 0:
            rich_weights = 1.0 / durations.loc[rich.index]
            rich_weights = rich_weights / rich_weights.abs().sum()
            daily_pos.loc[rich.index] = -rich_weights

        # If both long and short legs exist, scale each side to 50%.
        # If only one side exists, use gross leverage on that side.
        long_gross = daily_pos[daily_pos > 0].sum()
        short_gross = daily_pos[daily_pos < 0].abs().sum()

        final_pos = pd.Series(0.0, index=zscores.columns)

        if long_gross > 0 and short_gross > 0:
            final_pos[daily_pos > 0] = daily_pos[daily_pos > 0] / long_gross * (gross_leverage / 2)
            final_pos[daily_pos < 0] = daily_pos[daily_pos < 0] / short_gross * (gross_leverage / 2)
        elif long_gross > 0:
            final_pos[daily_pos > 0] = daily_pos[daily_pos > 0] / long_gross * gross_leverage
        elif short_gross > 0:
            final_pos[daily_pos < 0] = daily_pos[daily_pos < 0] / short_gross * gross_leverage

        positions.loc[date] = final_pos

    return positions


# ============================================================
# 6. Duration-Based Return Approximation
# ============================================================

# This function approximates Treasury bucket returns from yield changes and
# modified-duration proxies.
def compute_bond_proxy_returns(yields_percent):
    """
    Approximate Treasury bucket returns using duration:

        bond return ≈ -modified_duration × yield_change

    yields are in percent, so divide by 100 before differencing.
    """

    yields_decimal = yields_percent / 100.0
    yield_changes = yields_decimal.diff()

    durations = pd.Series(DURATION_PROXY)

    bond_returns = -yield_changes.mul(durations, axis=1)

    return bond_returns


# This function backtests the strategy with lagged positions and turnover-based
# transaction costs.
def backtest_strategy(yields_percent, positions, transaction_cost_bps=1.0):
    """
    Strategy return:

        return_t = sum(position_{t-1} * bond_return_t) - transaction_cost_t

    Lagged positions avoid look-ahead bias.
    """

    bond_returns = compute_bond_proxy_returns(yields_percent)

    lagged_positions = positions.shift(1).fillna(0.0)

    gross_returns = (lagged_positions * bond_returns).sum(axis=1)

    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)

    transaction_cost = turnover * (transaction_cost_bps / 10000.0)

    net_returns = gross_returns - transaction_cost

    results = pd.DataFrame(index=yields_percent.index)
    results["Gross Return"] = gross_returns
    results["Turnover"] = turnover
    results["Transaction Cost"] = transaction_cost
    results["Net Return"] = net_returns
    results["Equity Curve"] = (1 + net_returns.fillna(0)).cumprod()

    return results, bond_returns


# ============================================================
# 7. Benchmark Construction
# ============================================================

# This function builds simple passive Treasury benchmarks for performance
# comparison.
def build_benchmarks(yields_percent):
    """
    Build simple duration-proxy benchmarks:
        1. Long 10Y buy-and-hold
        2. Long 20Y buy-and-hold
        3. Equal-weight duration buckets
    """

    bond_returns = compute_bond_proxy_returns(yields_percent)

    benchmarks = pd.DataFrame(index=yields_percent.index)

    benchmarks["Long 10Y"] = bond_returns["10Y"]
    benchmarks["Long 20Y"] = bond_returns["20Y"]
    benchmarks["Equal Weight Curve"] = bond_returns.mean(axis=1)

    benchmark_equity = (1 + benchmarks.fillna(0)).cumprod()

    return benchmarks, benchmark_equity


# ============================================================
# 8. Performance Metrics
# ============================================================

# This function calculates standard investment performance metrics including
# CAGR, volatility, Sharpe ratio, drawdown, and hit rate.
def performance_metrics(returns, name="Strategy"):
    returns = returns.dropna()

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252

    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = annual_return / downside_vol if downside_vol != 0 else np.nan

    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    hit_rate = (returns > 0).mean()

    return pd.Series({
        "Name": name,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate,
        "Observations": len(returns),
    })


# This function prints and returns a combined strategy-versus-benchmark
# performance table.
def print_performance_table(results, benchmarks):
    rows = []

    rows.append(performance_metrics(results["Net Return"], "Rich-Cheap Strategy"))

    for col in benchmarks.columns:
        rows.append(performance_metrics(benchmarks[col], col))

    summary = pd.DataFrame(rows)

    print("\nPerformance Summary:")
    print(summary.to_string(index=False))

    return summary


# ============================================================
# 9. Plots
# ============================================================

# This plotting function compares the latest observed yield curve with the NSS
# fitted curve.
def plot_yield_curve_snapshot(yields_percent, fitted_df):
    date = yields_percent.index[-1]

    x = np.array([MATURITY_YEARS[m] for m in USE_MATURITIES])

    plt.figure(figsize=(10, 6))
    plt.scatter(x, yields_percent.loc[date, USE_MATURITIES] / 100, label="Market Yield")
    plt.plot(x, fitted_df.loc[date, USE_MATURITIES], marker="o", label="NSS Fitted Yield")

    plt.title(f"NSS Yield Curve Fit on {date.date()}")
    plt.xlabel("Maturity, years")
    plt.ylabel("Yield, decimal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# This plotting function shows residual z-scores through time for key Treasury
# maturities.
def plot_residual_zscores(zscores):
    plt.figure(figsize=(12, 6))

    for col in ["2Y", "5Y", "10Y", "20Y", "30Y"]:
        if col in zscores.columns:
            plt.plot(zscores.index, zscores[col], label=col)

    plt.axhline(ENTRY_Z, color="black", linestyle="--", linewidth=1)
    plt.axhline(-ENTRY_Z, color="black", linestyle="--", linewidth=1)

    plt.title("NSS Residual Z-Scores")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# This plotting function compares the strategy equity curve with benchmark
# equity curves.
def plot_equity_curves(results, benchmark_equity):
    plt.figure(figsize=(12, 6))

    plt.plot(results.index, results["Equity Curve"], label="Rich-Cheap Strategy", linewidth=2)

    for col in benchmark_equity.columns:
        plt.plot(benchmark_equity.index, benchmark_equity[col], label=col, alpha=0.7)

    plt.title("Equity Curve: Rich-Cheap Strategy vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# This plotting function visualizes the strategy drawdown profile.
def plot_drawdown(results):
    equity = results["Equity Curve"]
    drawdown = equity / equity.cummax() - 1

    plt.figure(figsize=(12, 5))
    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4)

    plt.title("Rich-Cheap Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# This plotting function creates a heatmap of daily position weights by
# maturity.
def plot_positions(positions):
    plt.figure(figsize=(12, 5))

    plt.imshow(
        positions.T.values,
        aspect="auto",
        interpolation="nearest",
        extent=[0, len(positions), 0, len(positions.columns)]
    )

    plt.yticks(
        ticks=np.arange(len(positions.columns)) + 0.5,
        labels=positions.columns
    )

    step = max(1, len(positions.index) // 8)
    x_ticks = np.arange(0, len(positions.index), step)
    x_labels = [positions.index[i].strftime("%Y-%m-%d") for i in x_ticks]

    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.colorbar(label="Position Weight")
    plt.title("Position Heatmap: Long Cheap / Short Rich")
    plt.xlabel("Date")
    plt.ylabel("Maturity")
    plt.tight_layout()
    plt.show()


# This plotting function displays daily portfolio turnover used for cost
# estimation.
def plot_turnover(results):
    plt.figure(figsize=(12, 4))
    plt.plot(results.index, results["Turnover"])

    plt.title("Daily Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. Main Pipeline
# ============================================================

# This main function runs the complete research pipeline from data download to
# backtest, diagnostics, plots, and reusable output objects.
def main():
    print("Fetching FRED Treasury yield data...")
    yields = fetch_fred_yields(START_DATE, END_DATE)

    print("\nYield data sample:")
    print(yields.head())

    # KMO test before PCA

    print("\nRunning KMO test...")

    overall_kmo, kmo_by_maturity = run_kmo_test(yields)

    # PCA analysis
    print("\nRunning PCA analysis...")
    loadings, explained, pc_scores = run_pca_analysis(yields)
    plot_pca_loadings(loadings, explained)

    # NSS calibration
    params_df, fitted_df, residual_df = calibrate_nss_history(yields)

    print("\nNSS parameter sample:")
    print(params_df.tail())

    # Residual z-scores
    zscores = compute_residual_zscores(residual_df, window=ZSCORE_WINDOW)

    # Build rich-cheap positions
    positions = build_rich_cheap_positions(
        zscores=zscores,
        top_n=TOP_N,
        entry_z=ENTRY_Z,
        gross_leverage=GROSS_LEVERAGE,
    )

    # Backtest
    results, bond_returns = backtest_strategy(
        yields_percent=yields,
        positions=positions,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    # Benchmarks
    benchmarks, benchmark_equity = build_benchmarks(yields)

    # Performance table
    summary = print_performance_table(results, benchmarks)

    # Plots
    plot_yield_curve_snapshot(yields, fitted_df)
    plot_residual_zscores(zscores)
    plot_equity_curves(results, benchmark_equity)
    plot_drawdown(results)
    plot_positions(positions)
    plot_turnover(results)

    # Save outputs
    output = {
        "yields": yields,
        "params": params_df,
        "fitted": fitted_df,
        "residuals": residual_df,
        "zscores": zscores,
        "positions": positions,
        "results": results,
        "benchmarks": benchmarks,
        "summary": summary,
        "overall_kmo": overall_kmo,
        "kmo_by_maturity": kmo_by_maturity,
        "pca_loadings": loadings,
        "pca_explained": explained,
        "pc_scores": pc_scores,
    }

    return output


# This guard lets the file run as a script while keeping the functions importable
# for notebooks or future tests.
if __name__ == "__main__":
    output = main()
