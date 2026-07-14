import pandas as pd
import numpy as np
import regression
from backtest import Backtester, plot_results
from datasets import Dataset
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---- Configuration ----
HOLDING_PERIOD_DAYS = 5

# ---- 1. Load data ----
data_filename = 'massive_final_all.parquet'
dataset = Dataset(filename=data_filename)

# ---- 2. Run regression to get factor exposures and optimize portfolio weights ----
factor_engine = regression.FactorEngine(dataset, regression.factor_list)
'''
# Setup regression with 5-day holding period
regression_options = {
    "regularization": "none",
    "alpha": 0.1,
    "cv_folds": 5,
    "tune_alpha": False,
    "holding_period_days": HOLDING_PERIOD_DAYS,
}
reg_model = regression.RegressionModel(factor_engine.dataset, factor_engine, options=regression_options)

# Setup risk model and optimizer
risk_model = regression.RiskModel(factor_engine.dataset)
optimizer_options = {
    "max_weight": 0.08,
    "risk_aversion": 0.5,
    "allow_short": False,
    "holding_period_days": HOLDING_PERIOD_DAYS,
    "rebalance_interval_days": HOLDING_PERIOD_DAYS,
}
optimizer = regression.PortfolioOptimizer(factor_engine.dataset, reg_model, risk_model, options=optimizer_options)

optimizer.optimize('test')
target_weights = optimizer.weights
'''

weights_path = os.path.join(script_dir, '..', 'data', f'optimized_weights.parquet')


if weights_path is not None:
    print(f"Loading optimized weights from {weights_path}")
    target_weights = pd.read_parquet(weights_path)
else:
    raise FileNotFoundError(
        "optimized_weights.parquet not found. Generate weights by running the optimizer "
        "or place the parquet at the repo root or factor-model/results/optimized_weights.parquet"
    )


print(f"\n{'='*70}")
print(f"Optimized target weights for test set ({HOLDING_PERIOD_DAYS}-day holding period):")
print(f"{'='*70}")
print(f"\nWeights shape: {target_weights.shape}")
print(f"\nFirst 5 dates' weights (showing first 5 stocks):")
print(target_weights.iloc[:5, :5])
print(f"\nWeight statistics across all dates:")
print(f"  Mean allocation per stock: {target_weights.mean().mean():.4f}")
print(f"  Min weight: {target_weights.min().min():.4f}")
print(f"  Max weight: {target_weights.max().max():.4f}")
print(f"  Std dev of weights: {target_weights.std().std():.4f}")

# ---- 3. Run the backtest ----
print(f"\n{'='*70}")
print(f"Running backtest on test set ({HOLDING_PERIOD_DAYS}-day holding period)")
print(f"{'='*70}")

engine = Backtester(
    data=dataset.test_data,
    target_weights=target_weights,
    initial_capital=1_000_000,
    commission_per_share=0.005,
    commission_min_per_order=1.0,
    slippage_tiers_bps=((1, 500, 3.0), (501, 1500, 10.0), (1501, 3000, 30.0)),
    short_borrow_rate_annual=0.01,
)
results = engine.run()

# ---- 4. Calculate and report results ----
print(f"\n{'='*70}")
print(f"BACKTEST RESULTS ({HOLDING_PERIOD_DAYS}-day holding period)")
print(f"{'='*70}")

starting_capital = results.config.initial_capital
dollar_equity = results.equity_curve * starting_capital
total_pnl = dollar_equity.iloc[-1] - starting_capital
total_return = total_pnl / starting_capital

# Calculate additional metrics
num_days = len(results.equity_curve)
daily_returns = results.equity_curve.pct_change().dropna()
annual_return = (results.equity_curve.iloc[-1] / results.equity_curve.iloc[0]) ** (252 / num_days) - 1
volatility = daily_returns.std() * np.sqrt(252)
sharpe = annual_return / volatility if volatility > 0 else 0
max_dd = (results.equity_curve / results.equity_curve.cummax()).min() - 1

# ---- Benchmark & Tracking Error ----
benchmark_path = os.path.join(script_dir, '..', 'data', 'benchmark.csv')
benchmark_equity = None
tracking_error = None
benchmark_beta = None
benchmark_alpha_daily = None
benchmark_alpha_annual = None
try:
    # CSV has two header rows (attribute, stock). Try multi-header first.
    bdf = pd.read_csv(benchmark_path, header=[0, 1], index_col=0)
    # try to find a close column
    if ('close',) in bdf.columns:
        # unlikely, but handle
        bclose = bdf[('close',)].astype(float)
    else:
        # pick the first close column available
        close_cols = [c for c in bdf.columns if c[0] == 'close']
        if close_cols:
            bclose = bdf[close_cols[0]].astype(float)
        else:
            # fallback to first numeric column
            bclose = bdf.iloc[:, 0].astype(float)
    # ensure bclose is a Series (pick first column if DataFrame)
    if isinstance(bclose, pd.DataFrame):
        bclose = bclose.iloc[:, 0]
    bclose.index = pd.to_datetime(bclose.index)
    # align to backtest dates
    b_close_aligned = bclose.reindex(results.equity_curve.index).ffill().bfill()
    # normalize to 1 at start to produce equity-like series
    benchmark_equity = b_close_aligned / b_close_aligned.iloc[0]

    # compute tracking error using period returns
    b_returns = bclose.pct_change()
    aligned = pd.concat([daily_returns, b_returns.reindex(daily_returns.index)], axis=1).dropna()
    if not aligned.empty:
        strategy_returns = aligned.iloc[:, 0]
        benchmark_returns = aligned.iloc[:, 1]
        active = strategy_returns - benchmark_returns
        tracking_error = float(active.std() * np.sqrt(engine.cfg.periods_per_year))
        benchmark_var = float(benchmark_returns.var())
        if benchmark_var > 0:
            benchmark_beta = float(strategy_returns.cov(benchmark_returns) / benchmark_var)
            benchmark_alpha_daily = float(strategy_returns.mean() - benchmark_beta * benchmark_returns.mean())
            benchmark_alpha_annual = benchmark_alpha_daily * engine.cfg.periods_per_year
        results.metrics.update({
            'benchmark_beta': benchmark_beta,
            'benchmark_alpha_daily': benchmark_alpha_daily,
            'benchmark_alpha_annual': benchmark_alpha_annual,
        })
        results.metrics['tracking_error'] = tracking_error
except Exception as e:
    print(f"[BENCHMARK] Could not load benchmark for tracking error: {e}")

print(f"\nCapital & Returns:")
print(f"  Starting capital:     ${starting_capital:>15,.0f}")
print(f"  Ending equity:        ${dollar_equity.iloc[-1]:>15,.0f}")
print(f"  Total PnL:            ${total_pnl:>+15,.0f}")
print(f"  Total Return:         {total_return:>15.2%}")

print(f"\nRisk Metrics:")
print(f"  Annualized Return:    {annual_return:>15.2%}")
print(f"  Annualized Volatility:{volatility:>15.2%}")
print(f"  Sharpe Ratio:         {sharpe:>15.2f}")
print(f"  Maximum Drawdown:     {max_dd:>15.2%}")

print(f"\nBacktest Details:")
print(f"  Trading Days:         {num_days:>15,.0f}")
print(f"  Backtest Period:      {results.equity_curve.index[0].date()} to {results.equity_curve.index[-1].date()}")
if tracking_error is not None:
    print(f"  Tracking Error (annualized): {tracking_error:>15.2%}")

# Benchmark comparison metrics
benchmark_total_return = None
benchmark_annual_return = None
active_annual_return = None
information_ratio = None
if benchmark_equity is not None:
    try:
        benchmark_total_return = benchmark_equity.iloc[-1] - 1.0
        # annualize similarly to strategy
        benchmark_annual_return = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) ** (252 / num_days) - 1
        if benchmark_annual_return is not None:
            active_annual_return = annual_return - benchmark_annual_return
        if tracking_error is not None and tracking_error > 0:
            information_ratio = active_annual_return / tracking_error

        results.metrics.update({
            'benchmark_total_return': benchmark_total_return,
            'benchmark_annual_return': benchmark_annual_return,
            'active_annual_return': active_annual_return,
            'information_ratio': information_ratio,
        })
    except Exception as _:
        pass

if benchmark_total_return is not None:
    print(f"\nBenchmark Comparison:")
    print(f"  Benchmark Total Return: {benchmark_total_return:>15.2%}")
    print(f"  Benchmark Annual Return: {benchmark_annual_return:>15.2%}")
    print(f"  Active Annual Return:   {active_annual_return:>15.2%}")
    if tracking_error is not None:
        print(f"  Tracking Error (ann):   {tracking_error:>15.2%}")
    if information_ratio is not None:
        print(f"  Information Ratio:      {information_ratio:>15.3f}")
    if benchmark_beta is not None:
        print(f"  Beta vs Benchmark:      {benchmark_beta:>15.3f}")
    if benchmark_alpha_annual is not None:
        print(f"  Alpha (annualized):     {benchmark_alpha_annual:>15.2%}")

print(f"\n{'='*70}")

# ---- 5. Chart ----
plot_results(results, benchmark_equity=benchmark_equity, save_path=f'./factor-model/results/backtest_equity_{HOLDING_PERIOD_DAYS}d_low_slippage.png')
