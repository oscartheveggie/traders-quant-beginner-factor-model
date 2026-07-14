# -*- coding: utf-8 -*-
"""
Simple integration test for regression -> weights -> backtest flow.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import factors
import regression
from backtest import Backtester

print("="*70)
print("SIMPLE BACKTEST TEST")
print("="*70)

print("\n[OK] Loading example data...")
example_data = pd.read_csv(
    factors.data_filepath,
    header=[0, 1],
    dtype=float,
    index_col=0,
    parse_dates=True,
)
print(f"[OK] Example data shape: {example_data.shape}")
print(f"[OK] Date range: {example_data.index[0]} to {example_data.index[-1]}")
print(f"[OK] Attributes: {list(example_data.columns.get_level_values(0).unique())}")
print(f"[OK] Stocks: {list(example_data.columns.get_level_values(1).unique())}")

factor_names = regression.factor_list

print("\n[RUN] Running regression and building weights...")
try:
    reg_results = regression.multiple_regression(example_data, factor_names)
    betas = regression.get_regression_betas(reg_results)
    factor_values = regression.get_factor_values(example_data, factor_names)

    # Placeholder risk model from regression.py
    covariance = regression.estimate_covariance_matrix_placeholder(example_data)

    target_weights = regression.build_weight_panel(example_data, reg_results, factor_names)

    # Force some NaNs to confirm that NaN means exiting those positions.
    target_weights.iloc[10:20, 0] = np.nan

    print(f"[OK] Betas: {betas.to_dict()}")
    print(f"[OK] Factor panels: {list(factor_values.keys())}")
    print(f"[OK] Placeholder covariance shape: {covariance.shape}")

    print("\n[RUN] Running Backtester with target weights...")
    engine = Backtester(
        data=example_data,
        target_weights=target_weights,
        initial_capital=1_000_000,
        commission_per_share=0.005,
        commission_min_per_order=1.0,
        slippage_tiers_bps=((1, 500, 13.0), (501, 1500, 20.0), (1501, 3000, 50.0)),
        short_borrow_rate_annual=0.01,
    )
    bt_results = engine.run()
    
    # Verify results
    print("\n[OK] Backtest completed successfully!")
    print(f"[RESULT] Total Return: {bt_results.metrics['total_return']*100:.2f}%")
    print(f"[RESULT] Annual Return: {bt_results.metrics['annual_return']*100:.2f}%")
    print(f"[RESULT] Annual Volatility: {bt_results.metrics['annual_volatility']*100:.2f}%")
    print(f"[RESULT] Sharpe Ratio: {bt_results.metrics['sharpe_ratio']:.4f}")
    print(f"[RESULT] Sortino Ratio: {bt_results.metrics['sortino_ratio']:.4f}")
    print(f"[RESULT] UPI: {bt_results.metrics['upi']:.4f}")
    print(f"[RESULT] Max Drawdown: {bt_results.metrics['max_drawdown']*100:.2f}%")
    
    # Verify all required keys
    required_keys = [
        'total_return', 'annual_return', 'annual_volatility', 'sharpe_ratio',
        'sortino_ratio', 'upi', 'max_drawdown', 'avg_turnover'
    ]
    missing_keys = [k for k in required_keys if k not in bt_results.metrics]
    
    if missing_keys:
        print(f"\n[ERROR] Missing keys: {missing_keys}")
        sys.exit(1)
    else:
        print(f"\n[OK] All required metrics present!")

    if bt_results.executed_weights.iloc[11, 0] != 0.0:
        print("\n[ERROR] NaN target weight did not force an exit to zero executed weight.")
        sys.exit(1)
    print("[OK] NaN target weights correctly translate to zero executed weights.")
    
    print("\n" + "="*70)
    print("TEST PASSED!")
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] Test failed with error:")
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
