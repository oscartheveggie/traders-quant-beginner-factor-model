import numpy as np
import pandas as pd

"""
File: functions.py
Description: This module defines all functions that can be used in factor definitions.

Format for each function:
```python
def function_name(stock_data: pd.DataFrame) -> pd.DataFrame:
    # function expression
    <df name> = rename_attribute(<df name>, <attribute name>/'signal')
    return <df name>
```
(IMPORTANT: Remember to rename the output DataFrame using `rename_attribute` helper
function for readability and debugging purposes)

Each `stock_data` is a DataFrame with attribute columns at the top level (e.g., 'open', 'close',
'high', 'low', 'volume'), stock tickers at the bottom level (e.g., 'AAPL', 'GOOGL'), and index as
datetime.

The output (factor_values) is a DataFrame with the same index as `stock_data` (i.e., datetime)
and columns corresponding to the factor values for each stock.

Functions are split into 3 sections:
1) Derived attribute functions: Converting existing attributes into new ones
   (e.g., daily_return, vwap, pe_ratio, etc.)
2) Time-series functions: Applying time-series operations like rolling, expanding, etc.
   (e.g., ts_mean, ts_std, ts_rank, etc.)
3) Cross-sectional functions: Applying cross-sectional operations across stocks at each time point
   (e.g., rank, zscore, etc.)
4) Miscellaneous factor function
   (e.g., if_else, etc.)
4) Helper functions: Functions that can be used in factor definitions but are not factors themselves
   (e.g., rename_attribute, neutralize_by, etc.)

"""

"""
===== 1) Derived Attribute Functions =====

Format: 
def function_name(stock_data: pd.DataFrame) -> pd.DataFrame:

"""

def daily_return(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily return (= today's close / yesterday's close - 1) for all stocks"""
    _require_multiindex_columns(stock_data, 'daily_return')
    close = stock_data[['close']]
    returns = close.pct_change()
    returns = rename_attribute(returns, 'daily_return')
    return returns

def rsi(stock_data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI) for all stocks"""
    _require_multiindex_columns(stock_data, 'rsi')
    close = stock_data[['close']]
    delta = close.diff()
    gain = delta.where(delta > 0).rolling(window, min_periods=window).mean()
    loss = (-delta.where(delta < 0)).rolling(window, min_periods=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rename_attribute(rsi, 'rsi')
    return rsi

def cap(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate market capitalization (cap = close * shares_outstanding) for all stocks"""
    _require_multiindex_columns(stock_data, 'cap')
    close = rename_attribute(stock_data[['close']], 'signal')
    shares_outstanding = rename_attribute(stock_data[['diluted_shares_outstanding']], 'signal')
    cap = close.mul(shares_outstanding)
    cap = rename_attribute(cap, 'cap')
    return cap

def enterprise_value(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate enterprise value (EV = cap + total_liabilities - cash_and_equivalents) for all stocks"""
    _require_multiindex_columns(stock_data, 'enterprise_value')
    market_cap = rename_attribute(cap(stock_data), 'signal')
    total_liabilities = rename_attribute(stock_data[['total_liabilities']], 'signal')
    cash_and_equivalents = rename_attribute(stock_data[['cash_and_equivalents']], 'signal')
    ev = market_cap.add(total_liabilities).sub(cash_and_equivalents).replace([np.inf, -np.inf], np.nan)
    ev = rename_attribute(ev, 'enterprise_value')
    return ev

def estimated_market_cap(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate estimated market capitalization (cap_latest / latest_close * close) for all stocks"""
    _require_multiindex_columns(stock_data, 'estimated_market_cap')
    # Extract single-level DataFrames for 'cap_latest' and 'close' (select top-level labels)
    # Use label-based selection which works even if MultiIndex level names differ
    cap_latest_df = stock_data['cap_latest'].copy()
    close_df = stock_data['close'].copy()

    # Convert to numeric and handle non-numeric/None by coercing to NaN
    cap_latest_series = pd.to_numeric(cap_latest_df.iloc[-1], errors='coerce')
    latest_close_series = pd.to_numeric(close_df.iloc[-1], errors='coerce')

    # Compute per-stock ratio, guarding divide-by-zero and infinities
    ratio = cap_latest_series.div(latest_close_series).replace([np.inf, -np.inf], np.nan)

    # Apply ratio across each stock's close time-series
    market_cap_df = close_df.mul(ratio, axis=1)

    # Restore MultiIndex columns with attribute name 'estimated_market_cap'
    cols = pd.MultiIndex.from_product([['estimated_market_cap'], market_cap_df.columns], names=['attribute', 'stock'])
    market_cap_df.columns = cols
    return market_cap_df


"""
===== 2) Time-Series Functions =====

Format: 
def ts_<operation>(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:

"""


def ts_rank(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series rank over a rolling window"""
    _require_multiindex_columns(stock_attribute, 'ts_rank')
    ts_rank = stock_attribute.rolling(window).apply(
        lambda x: np.nan if np.isnan(x).any() or len(x) < 2 else (x < x[-1]).sum() / (len(x) - 1), 
        raw=True,
        engine='numba'
    )
    ts_rank = rename_attribute(ts_rank, 'signal')
    return ts_rank

def ts_mean(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series mean over a rolling window"""
    _require_multiindex_columns(stock_attribute, 'ts_mean')
    ts_mean = stock_attribute.rolling(window, min_periods=window).mean()
    ts_mean = rename_attribute(ts_mean, 'signal')
    return ts_mean

def ts_sum(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series sum over a rolling window"""
    _require_multiindex_columns(stock_attribute, 'ts_sum')
    ts_sum = stock_attribute.rolling(window).sum().fillna(0)
    ts_sum = rename_attribute(ts_sum, 'ts_sum()')
    return ts_sum

def ts_std(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series standard deviation over a rolling window"""
    _require_multiindex_columns(stock_attribute, 'ts_std')
    ts_std = stock_attribute.rolling(window, min_periods=window).std()
    ts_std = rename_attribute(ts_std, 'signal')
    return ts_std

def ts_zscore(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series z-score over a rolling window"""
    _require_multiindex_columns(stock_attribute, 'ts_zscore')
    mean = stock_attribute.rolling(window, min_periods=window).mean()
    std = stock_attribute.rolling(window, min_periods=window).std()
    zscore = stock_attribute.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], np.nan)
    zscore = rename_attribute(zscore, 'signal')
    return zscore

def ts_delta(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series delta (current value - value from 'window' periods ago)"""
    _require_multiindex_columns(stock_attribute, 'ts_delta')
    ts_delta = stock_attribute.diff(window)
    ts_delta = rename_attribute(ts_delta, 'signal')
    return ts_delta

def ts_delay(stock_attribute: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calculate time-series delay (value from 'window' periods ago)"""
    _require_multiindex_columns(stock_attribute, 'ts_delay')
    ts_delay = stock_attribute.shift(window)
    ts_delay = rename_attribute(ts_delay, 'signal')
    return ts_delay


"""
===== 3) Cross-Sectional Functions =====

Format:
def function_name(stock_attribute: pd.DataFrame) -> pd.DataFrame:

"""

def rank(stock_attribute: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-sectional rank at each time point"""
    _require_multiindex_columns(stock_attribute, 'rank')
    rank = stock_attribute.rank(axis=1, pct=True)
    rank = rename_attribute(rank, 'signal')
    return rank

def zscore(stock_attribute: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-sectional z-score at each time point"""
    _require_multiindex_columns(stock_attribute, 'zscore')
    mean = stock_attribute.mean(axis=1)
    std = stock_attribute.std(axis=1)
    zscore = stock_attribute.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], np.nan)
    zscore = rename_attribute(zscore, 'signal')
    return zscore


"""
===== 4) Miscellaneous Factor Functions =====
"""


def if_else(condition: pd.DataFrame, true_value: pd.DataFrame, false_value: pd.DataFrame) -> pd.DataFrame:
    """Apply if-else logic to create a new factor based on a condition"""
    _require_multiindex_columns(condition, 'if_else(condition)')
    _require_multiindex_columns(true_value, 'if_else(true_value)')
    _require_multiindex_columns(false_value, 'if_else(false_value)')
    result = true_value.where(condition.astype(bool), false_value)
    result = result.where(condition.notna())
    result = rename_attribute(result, 'signal')
    return result


"""
===== 5) Helper Functions =====
"""

def ones(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame of ones for the given date range and stocks"""
    _require_multiindex_columns(stock_data, 'ones')
    cols = pd.MultiIndex.from_product([['ones()'], stock_data.columns.get_level_values(1).unique()], 
                                      names=['attribute', 'stock'])
    df = pd.DataFrame(1.0, index=stock_data.index, columns=cols)
    df = rename_attribute(df, 'signal')
    return df

def zeros(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame of zeros for the given date range and stocks"""
    _require_multiindex_columns(stock_data, 'zeros')
    cols = pd.MultiIndex.from_product([['zeros()'], stock_data.columns.get_level_values(1).unique()], 
                                      names=['attribute', 'stock'])
    df = pd.DataFrame(0.0, index=stock_data.index, columns=cols)
    df = rename_attribute(df, 'signal')
    return df

def rename_attribute(stock_attribute: pd.DataFrame, new_name: str) -> pd.DataFrame:
    """Helper function to rename the attribute name in the column MultiIndex"""
    _require_multiindex_columns(stock_attribute, 'rename_attribute')
    stock_attribute.columns = pd.MultiIndex.from_tuples(
        [(new_name, col[1]) for col in stock_attribute.columns],
        names=['attribute', 'stock']
    )
    return stock_attribute

def _sic_to_division(code) -> str | None:
    """Map a 4-digit SIC code to its division letter (A–J) per the Standard Industrial Classification."""
    if pd.isna(code):
        return None
    major = int(code) // 100
    if 1 <= major <= 9:   return 'A'
    if 10 <= major <= 14: return 'B'
    if 15 <= major <= 17: return 'C'
    if 20 <= major <= 39: return 'D'
    if 40 <= major <= 49: return 'E'
    if 50 <= major <= 51: return 'F'
    if 52 <= major <= 59: return 'G'
    if 60 <= major <= 67: return 'H'
    if 70 <= major <= 89: return 'I'
    if 91 <= major <= 99: return 'J'
    return None

def neutralize_by(factor_values: pd.DataFrame, option: str, sic_codes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Neutralize factor values by a given option.
    Available options: ['market', 'industry', 'sub_industry']

    'industry'     — demean within SIC division (A-J), grouped per row since codes can change over time.
    'sub_industry' — demean within full 4-digit SIC code, grouped per row.

    `sic_codes` must be provided for 'industry' and 'sub_industry': a DataFrame with the same
    index as `factor_values` and MultiIndex columns ('sic_code', <stock_ticker>), i.e.
    stock_data[['sic_code']].
    """

    _require_multiindex_columns(factor_values, 'neutralize_by')
    match option:
        case 'market':
            factor_values = factor_values.sub(factor_values.mean(axis=1), axis=0)

        case 'industry' | 'sub_industry':
            stocks = factor_values.columns.get_level_values('stock')
            # Align sic codes to the same stock ordering as factor_values
            sic_by_stock = sic_codes.xs('sic_code', axis=1, level=0).reindex(columns=stocks)

            if option == 'industry':
                group_labels = sic_by_stock.apply(lambda col: col.map(_sic_to_division))
            else:
                group_labels = sic_by_stock

            '''
            else:
                # Group by full 4-digit SIC code; normalise to str so int/float variants match
                sic_by_stock[sic_by_stock.notna()] = sic_by_stock[sic_by_stock.notna()].astype(int).astype(str)
                group_labels = sic_by_stock.where(sic_by_stock.isna(), sic_by_stock.astype(str))
            '''

            attribute_label = factor_values.columns.get_level_values(0)[0]

            stocks_only = factor_values.copy()
            stocks_only.columns = stocks
            group_labels = group_labels.copy()
            group_labels.columns = stocks

            stacked_values = stocks_only.stack()
            stacked_groups = group_labels.stack().reindex(stacked_values.index)
            row_keys = stacked_values.index.get_level_values(0)
            group_means = stacked_values.groupby([row_keys, stacked_groups]).transform('mean')

            stacked_values = stacked_values.sub(group_means).where(stacked_groups.notna(), stacked_values)
            factor_values = stacked_values.unstack().reindex(columns=stocks)
            factor_values.columns = pd.MultiIndex.from_product(
                [[attribute_label], factor_values.columns],
                names=['attribute', 'stock']
            )

    return factor_values  # Return adjusted factor values

def winsorize(factor_values: pd.DataFrame, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    """Windsorize factor values to limit the influence of outliers"""
    _require_multiindex_columns(factor_values, 'winsorize')
    # Cross-sectional winsorization by date (row-wise quantiles across stocks).
    lower_bound = factor_values.quantile(lower_quantile, axis=1)
    upper_bound = factor_values.quantile(upper_quantile, axis=1)
    factor_values = factor_values.clip(lower=lower_bound, upper=upper_bound, axis=0)
    return factor_values  # Return windsorized factor values

def decay(factor_values: pd.DataFrame, decay_days: int) -> pd.DataFrame:
    """Apply linear decay to factor values over time"""

    _require_multiindex_columns(factor_values, 'decay')
    decay_weights = np.arange(decay_days, 0, -1)
    decay_weights = decay_weights / decay_weights.sum()  # Normalize weights to sum to 1
    factor_values = factor_values.rolling(window=decay_days).apply(lambda x: np.dot(x, decay_weights), raw=True)
    
    return factor_values  # Return decayed factor values

def ts_ewm_sum(stock_attribute: pd.DataFrame, window: int, half_life: float) -> pd.DataFrame:
    """Exponentially weighted rolling sum (Barra-style time-series weighting).

    weights w_i = 0.5 ** (i / half_life), with i counting backwards from the
    most recent observation. Weights are normalized to sum to 1 across the
    window, so the result is a weighted *average*-style rolling sum suitable
    for momentum-type accumulators.
    """
    _require_multiindex_columns(stock_attribute, 'ts_ewm_sum')
    decay = 0.5 ** (np.arange(window - 1, -1, -1) / float(half_life))
    decay = decay / decay.sum()
    out = stock_attribute.rolling(window, min_periods=window).apply(
        lambda x: float(np.dot(x, decay)), raw=True
    )
    out = rename_attribute(out, 'signal')
    return out


def safe_settings(factor_values: pd.DataFrame, settings_dict: dict, stock_data: pd.DataFrame) -> pd.DataFrame:
    """Apply `settings` while gracefully falling back to market neutralization
    when `sic_code` is not present in `stock_data` (e.g., on the example.csv
    which lacks fundamentals/classification data).
    """
    _require_multiindex_columns(factor_values, 'safe_settings')
    if 'sic_code' in stock_data.columns.get_level_values(0):
        return settings(factor_values, settings_dict, stock_data[['sic_code']])

    local = {**settings_dict}
    if 'neutralize_by' in local and local['neutralize_by'].get('option') in ('industry', 'sub_industry'):
        local['neutralize_by'] = {'option': 'market'}
    return settings(factor_values, local, None)


def settings(factor_values: pd.DataFrame, settings_dict: dict, sic_codes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply various settings to factor values based on provided keyword arguments

    `settings_dict` format:
    {
        <function_name>: {<arg name>: <arg_value>, <arg name>: <arg_value>, ...},
        'neutralize_by': {'option': 'market'},
        'winsorize': {'lower_quantile': 0.01, 'upper_quantile': 0.99},
        'decay': {'decay_days': 5},
    }

    If you do not wish to use certain functions, do not include them in the `settings_dict`.
    e.g., If you only want to neutralize by market, `settings dict = {'neutralize_by': {'option': 'market'}}`

    Pass `sic_codes=stock_data[['sic_code']]` when using 'industry' or 'sub_industry' neutralization.

    Order of preference for setting functions (default values in brackets):
    winsorize (0.01, 0.99) -> neutralize_by (market) -> decay (4) -> normalize
    
    """

    _require_multiindex_columns(factor_values, 'settings')
    # Apply functions in order: winsorize, neutralize_by, decay

    if 'winsorize' in settings_dict:
        kwargs = settings_dict['winsorize']
        factor_values = winsorize(factor_values, kwargs.get('lower_quantile', 0.01), kwargs.get('upper_quantile', 0.99))

    if 'neutralize_by' in settings_dict:
        kwargs = settings_dict['neutralize_by']
        factor_values = neutralize_by(factor_values, kwargs.get('option', 'market'), sic_codes)

    if 'decay' in settings_dict:
        kwargs = settings_dict['decay']
        factor_values = decay(factor_values, kwargs.get('decay_days', 4))
    
    # Normalize the final factor values using cross-sectional z-score
    factor_values = zscore(factor_values)
    
    return factor_values  # Return adjusted factor values
    

def _require_multiindex_columns(df: pd.DataFrame, function_name: str) -> None:
    """Raise a clear error if input columns are not a 2-level MultiIndex."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"{function_name} expects columns as a pandas MultiIndex with levels ['attribute', 'stock']")
    if df.columns.nlevels != 2:
        raise ValueError(f"{function_name} expects a 2-level MultiIndex, got {df.columns.nlevels} levels")

"""
===== TESTING =====
You can run this file directly to test the functions with example data.
"""

if __name__ == "__main__":
    # Example usage with multiple stocks using MultiIndex (stock, attribute) in columns
    dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
    stocks = ['AAPL', 'GOOGL', 'MSFT']
    attributes = ['open', 'close', 'high', 'low', 'volume', 'cap_latest']
    
    # Create MultiIndex columns (stock, attribute) with stock at top level
    columns = pd.MultiIndex.from_product([attributes, stocks], names=['attribute', 'stock'])
    
    # Create data in the correct order
    # New sample data: 5 days, 3 stocks (AAPL, GOOGL, MSFT), attributes (open, close, high, low, volume, cap_latest)
    example_data = [
        [125.5, 127.1, 128.0, 124.8, 1500000,  210e9,  1502.3, 1510.0, 1520.0, 1498.0, 1100000,  980e9,  210.5,  230.0, 231.5, 229.0, 1200000, 180e9],
        [126.0, 125.8, 127.5, 124.5, 1400000,  211e9,  1505.0, 1498.0, 1512.0, 1490.0, 1300000,  985e9,  211.2,  231.0, 230.5, 228.0, 1250000, 181e9],
        [124.0, 124.6, 125.5, 123.2, 1600000,  209e9,  1490.0, 1495.5, 1500.0, 1485.0, 1250000, 970e9,  209.8,  229.0, 229.5, 227.5, 1100000, 179e9],
        [125.8, 126.4, 127.0, 125.0, 1550000,  212e9,  1508.0, 1512.0, 1525.0, 1502.0, 1300000, 990e9,  212.0,  232.0, 233.0, 231.0, 1300000, 182e9],
        [127.2, 128.0, 128.5, 126.5, 1450000,  213e9,  1515.0, 1522.0, 1530.0, 1510.0, 2000000, 1000e9,  213.3,  233.5, 234.0, 232.5, 1350000, 183e9]
    ]

    example_data = pd.DataFrame(example_data, index=dates, columns=columns)
    print(example_data)

    estimated_market_cap = estimated_market_cap(example_data)
    print("\nEstimated Market Cap:\n")
    print(estimated_market_cap)

    """
    daily_returns = daily_return(example_data)
    print(daily_returns)
    print()
    """

    '''
    settings_dict = {
        'winsorize': {'lower_quantile': 0.01, 'upper_quantile': 0.99},
        'neutralize_by': {'option': 'market'},
        'decay': {'decay_days': 4}
    }

    neutralized = settings(example_data[['close']], settings_dict)
    print("\nNeutralized Factor Values (with winsorize and neutralize_by settings):")
    print(neutralized)
    '''

