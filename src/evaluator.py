import ast
import os
import tokenize
import pandas as pd
import statsmodels.api as sm
import factors
import functions as fc
from datasets import Dataset
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
custom_factors_list = []

"""
===== EVALUATOR FUNCTIONS =====
This module defines functions to evaluate factors defined in `factors.py`.

Format:
def evaluation_function(stock_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:

Each `stock_data` is a DataFrame with attribute columns
(e.g., 'open', 'close', 'high', 'low', 'volume') and index as datetime.

Output: A DataFrame with columns for each evaluation metric and rows for each factor

Note: You may want to use training set for evaluation to avoid overfitting
"""
def ic_ir(stock_data: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    """Calculate IC/ICIR for each factor against 1-day future return."""
    return ic_ir_with_holding_period(
        stock_data=stock_data,
        factor_names=factor_names,
        holding_period_days=1,
        rebalance_interval_days=1,
    )


def ic_ir_with_holding_period(
    stock_data: pd.DataFrame,
    factor_names: list[str],
    holding_period_days: int = 1,
    rebalance_interval_days: int = 1,
) -> pd.DataFrame:
    """Calculate IC/ICIR for each factor with configurable holding and rebalance intervals."""

    if holding_period_days <= 0:
        raise ValueError("holding_period_days must be a positive integer")
    if rebalance_interval_days <= 0:
        raise ValueError("rebalance_interval_days must be a positive integer")
    
    cols = pd.MultiIndex.from_product([factor_names, ['ic']], names=['Factor', 'Metric'])
    ics = pd.DataFrame(columns=cols)
    future_return_col = f'future_{holding_period_days}d_return'

    for factor_name in factor_names:
        comparison_df = compare_factor_with_future_return(
            stock_data=stock_data,
            factor_name=factor_name,
            holding_period_days=holding_period_days,
        )

        if rebalance_interval_days > 1:
            comparison_df = comparison_df.iloc[::rebalance_interval_days]

        if comparison_df.empty:
            print(f"No valid data for factor '{factor_name}' to calculate IC.")
            continue
        
        # Calculate cross-sectional IC for each timestamp
        for index in tqdm(comparison_df.index):

            ics_df = comparison_df.loc[index:index, future_return_col].T
            ics_df.columns = [future_return_col]

            factor_df = comparison_df.loc[index:index, factor_name].T
            factor_df.columns = [factor_name]

            ics_df = pd.concat([ics_df, factor_df], axis=1)
            ics_df.columns = [future_return_col, f'{factor_name}']
            
            # Drop NaN values for this timestamp (handle sparse data)
            ics_df = ics_df.dropna()
            
            # Skip if insufficient valid data for correlation
            if len(ics_df) < 2:
                continue
                
            ic_value = ics_df.corr(method='pearson').iloc[0, 1]  # Correlation between future returns and factor values
            ics.loc[index, (factor_name, 'ic')] = ic_value
        
    ics.sort_index(inplace=True)  # Ensure results are sorted by timestamp

    results = ics.mean().unstack(level=0).T.rename(columns={'ic': 'ic'})

    icir_values = ics.mean() / ics.std()  # Mean IC divided by its standard deviation
    
    results['ic_stdev'] = ics.std().unstack(level=0).T
    results['icir'] = icir_values.unstack(level=0).T
    results = results.replace([float('inf'), float('-inf')], pd.NA)
    
    results = results.sort_values('icir', key=abs, ascending=False)

    return results


def ic_ir_across_intervals(
    stock_data: pd.DataFrame,
    factor_names: list[str],
    interval_days: list[int] = [1, 5, 10, 30],
) -> pd.DataFrame:
    """
    Evaluate all factors across multiple interval days.

    For each interval n:
    - holding period return uses n-day future return
    - rebalance uses every n-th row (n-day intervals)
    """

    interval_results = []

    for n_days in interval_days:
        period_result = ic_ir_with_holding_period(
            stock_data=stock_data,
            factor_names=factor_names,
            holding_period_days=n_days,
            rebalance_interval_days=n_days,
        )

        period_result.columns = pd.MultiIndex.from_product(
            [[f'{n_days}d'], period_result.columns],
            names=['interval', 'metric']
        )
        interval_results.append(period_result)

    if not interval_results:
        return pd.DataFrame(index=factor_names)

    summary = pd.concat(interval_results, axis=1)
    summary = summary.reindex(factor_names)
    return summary



"""
===== HELPER FUNCTIONS: Misc =====
"""

def compare_factor_with_future_return(
    stock_data: pd.DataFrame,
    factor_name: str,
    holding_period_days: int = 1,
) -> pd.DataFrame:
    """Zip factor values with n-day future returns for cross-sectional IC testing."""
    factor_func = getattr(factors, factor_name, None)
    if factor_func is None:
        raise ValueError(f"Factor function '{factor_name}' not found in factors.py")
    
    factor_values = factor_func(stock_data)
    print("Factor values calculated for factor:", factor_name)

    close = stock_data[['close']]
    future_returns = close.shift(-holding_period_days).div(close).sub(1)
    print(f"Future {holding_period_days}-day returns calculated.")

    factor_values = fc.rename_attribute(factor_values, factor_name)
    future_returns = fc.rename_attribute(future_returns, f'future_{holding_period_days}d_return')

    # Only drop rows where ALL values are NaN (sparse data tolerance)
    comparison_df = pd.concat([factor_values, future_returns], axis=1).dropna(how='all')

    return comparison_df

def get_functions_from_file(filepath):
    with tokenize.open(filepath) as file:
        tree = ast.parse(file.read(), filename=filepath)
    
    # Extract names of top-level function definitions
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


"""
===== MAIN EXECUTION =====
This block will execute when the script is run directly. You can use it to test the evaluation functions with example data, 
or to run a full evaluation pipeline.
"""

if __name__ == "__main__":

    is_multi_period = input("Run multi-period evaluation? (y/n): ").strip().lower() == 'y'
    
    dataset = Dataset(filename='massive_final_all.parquet')

    if custom_factors_list:
        factor_names = custom_factors_list
    else:
        factor_names = get_functions_from_file(os.path.join(script_dir, "factors.py"))

    if is_multi_period:
        # Multi-period run across interval days (hold n-days, rebalance every n days)
        interval_days = [1, 5, 10, 30]
        interval_results = ic_ir_across_intervals(dataset.train_data, factor_names, interval_days)
        
        # Print separate tables for each interval
        for n_days in interval_days:
            print(f"\n=== IC/ICIR ({n_days}d) ===")
            interval_table = interval_results.xs(f'{n_days}d', level='interval', axis=1)
            print(interval_table)

    else:
        # Default single-period run (1-day hold, daily rebalance)
        ic_ir_results = ic_ir(dataset.train_data, factor_names)
        print("\n=== IC/ICIR (1-day) ===")
        print(ic_ir_results)

    
