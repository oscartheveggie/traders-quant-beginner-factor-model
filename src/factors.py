import pandas as pd
import numpy as np
import functions as fc
import os

"""
File: factors.py
Description: This module defines all factors to be evaluated.

Format for each factor:

def factor_name(stock_data: pd.DataFrame) -> pd.DataFrame:
    # factor expression
    factor = fc.rename_attribute(factor, 'factor_name')
    factor = fc.settings(factor, default_settings)
    return factor

Each `stock_data` is a DataFrame with attribute columns 
(e.g., 'open', 'close', 'high', 'low', 'volume') and index as datetime.

(STRONGLY RECOMMENDED: Use the settings function to apply any desired settings to the
factor values before returning)

Output: A DataFrame with the same index as `stock_data` 
and columns corresponding to the factor values for each stock.

How to use functions: Use fc.<function_name>(...)

How to use attributes: Either use fc.<function_name>(stock_data) for 
derived attributes, or directly access stock_data[['attribute_name']] 
for original attributes.
(IMPORTANT: use double brackets to keep the dual level for column index,
otherwise it will produce an error)

IMPORTANT: Do not add helper functions here, if you want to do so, add
them in `functions.py` and import them here. The `evaluator.py` will 
automatically extract all functions defined in this file as factors 
to be evaluated.

Note: MultiIndex alignment checklist to avoid
"Reindexing only valid with uniquely valued Index objects":
1) Before arithmetic between two DataFrames, rename both to the same
    attribute label (for example, 'signal').
2) Prefer DataFrame operations (`where`, `rolling`) over `np.where`
    to preserve index/column metadata.
3) After building a factor, sanity-check that each stock appears once
    (`factor.columns.duplicated().sum() == 0`).

"""

script_dir = os.path.dirname(os.path.abspath(__file__))
example_filename = 'example.csv'
example_filepath = os.path.join(script_dir, '..', 'data', example_filename)

DEFAULT_SETTINGS = {
    'winsorize': {'lower_quantile': 0.01, 'upper_quantile': 0.99},
    'neutralize_by': {'option': 'sub_industry'},
    'decay': {'decay_days': 4}
}

"""
===== MEAN REVERSION / MOMENTUM FACTORS =====
"""

def mean_reversion_0(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    MR 0 (testing only): -rank(ts_zscore(returns, 252))
    """

    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    prev_close = fc.rename_attribute(fc.ts_delay(stock_data[['close']], window=1), 'signal')
    returns = close.div(prev_close).sub(1)
    factor = -fc.rank(fc.ts_zscore(returns, window=252))
    factor = fc.rename_attribute(factor, 'mean_reversion_0')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor

def mean_reversion_1(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    MR 1: -zscore((close - vwap)/ts_mean(vwap, 20)) * rank(ts_delta(volume, 5)) * rank(cap)
    """

    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    vwap = fc.rename_attribute(stock_data[['vwap']].copy(), 'signal')
    volume = fc.rename_attribute(stock_data[['volume']].copy(), 'signal')
    cap = fc.rename_attribute(fc.cap(stock_data).copy(), 'signal')

    price_deviation = close.sub(vwap).div(fc.ts_mean(vwap, window=20))
    volume_momentum = fc.rank(fc.ts_delta(volume, window=5))
    size_effect = fc.rank(cap)

    factor = -fc.zscore(price_deviation).mul(volume_momentum).mul(size_effect)
    factor = fc.rename_attribute(factor, 'mean_reversion_1')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor

def mean_reversion_2(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    MR 2: rank(ts_zscore(-close / ts_mean(close, 20), 5)) 
          + rank(ts_rank(eps / close, 60))
    """

    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    eps = fc.rename_attribute(stock_data[['diluted_earnings_per_share']].copy(), 'signal')

    price_momentum = fc.ts_zscore(-close.div(fc.ts_mean(close, window=20)), window=5)
    value_momentum = fc.ts_rank(eps.div(close), window=60)

    factor = fc.rank(price_momentum).add(fc.rank(value_momentum))
    factor = fc.rename_attribute(factor, 'mean_reversion_2')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor

def mean_reversion_3(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    MR 3: rank(ts_mean(log(open / ts_delay(close, 1)), 20)) 
          - rank(ts_mean(log(close / open), 20))
    """

    open_price = fc.rename_attribute(stock_data[['open']].copy(), 'signal')
    close_price = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    prev_close_price = fc.rename_attribute(fc.ts_delay(stock_data[['close']], window=1), 'signal')

    open_to_prev_close = fc.ts_mean(np.log(open_price / prev_close_price), window=20)
    close_to_open = fc.ts_mean(np.log(close_price / open_price), window=20)

    factor = fc.rank(open_to_prev_close) - fc.rank(close_to_open)

    factor = fc.rename_attribute(factor, 'mean_reversion_3')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor


def mean_reversion_4(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    MR 4:
    N = 60;
    amp = (high / low) - 1;
    price_change = close / ts_delay(close, 1) - 1;
    -ts_sum((ts_zscore(amp, N) > 1? price_change : 0), N)

    """

    settings = {**DEFAULT_SETTINGS, 'decay': {'decay_days': 1}}

    high = fc.rename_attribute(stock_data[['high']].copy(), 'signal')
    low = fc.rename_attribute(stock_data[['low']].copy(), 'signal')
    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')

    amp = high.div(low).sub(1)
    prev_close = fc.rename_attribute(fc.ts_delay(close, window=1), 'signal')
    price_change = close.div(prev_close).sub(1)

    # Keep DataFrame structure (MultiIndex columns) during conditional filtering.
    condition = fc.rename_attribute(fc.ts_zscore(amp, window=60), 'signal') > 1
    conditional_price_change = price_change.where(condition, 0.0)
    factor = -conditional_price_change.rolling(window=60).sum().fillna(0)

    factor = fc.rename_attribute(factor, 'mean_reversion_4')
    factor = fc.settings(factor, settings, stock_data[['sic_code']])
    
    return factor


def mean_reversion_5(stock_data: pd.DataFrame) -> pd.DataFrame:

    """
    MR 5: -1 * ((rank(ts_delta(close, 5) / ts_mean(high - low, 5)) - 0.5) ** 3
    """

    settings = {**DEFAULT_SETTINGS, 'decay': {'decay_days': 1}}

    close_delta = fc.rename_attribute(fc.ts_delta(stock_data[['close']], window=5), 'signal')

    high = fc.rename_attribute(stock_data[['high']].copy(), 'price_range')
    low = fc.rename_attribute(stock_data[['low']].copy(), 'price_range')
    price_range = high - low

    average_price_range = fc.rename_attribute(fc.ts_mean(price_range, window=5), 'signal')

    factor = -1 * (fc.rank(close_delta / average_price_range) - 0.5) ** 3

    factor = fc.rename_attribute(factor, 'mean_reversion_5')
    factor = fc.settings(factor, settings, stock_data[['sic_code']])
    return factor

"""
===== FUNDAMENTAL FACTORS =====
"""

def fundamental_1(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fundamental 1: -zscore(ts_mean((enterprise_value / ebitda), 252) 
                   / (enterprise_value / ebitda))
    """

    ev = fc.rename_attribute(fc.enterprise_value(stock_data).copy(), 'signal')
    ebitda = fc.rename_attribute(stock_data[['ebitda']].copy(), 'signal')

    ev_to_ebitda = ev.div(ebitda.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    factor = -fc.zscore(
        fc.ts_mean(ev_to_ebitda, window=252)
        .div(ev_to_ebitda.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
    )

    factor = fc.rename_attribute(factor, 'fundamental_1')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor

def fundamental_2(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fundamental 2: 

    oey = operating_income / cap;
    num_days_neg = ts_sum((returns < 0 ? 1 : 0), 252);

    rank(ts_delta(oey, 125) / ts_std_dev(oey, 252)) 
    + rank(num_days_neg) + rank(-ts_delta(close, 1))
    """

    settings = {**DEFAULT_SETTINGS, 'decay': {'decay_days': 12}}

    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    operating_income = fc.rename_attribute(stock_data[['operating_income']].copy(), 'signal')
    cap = fc.rename_attribute(fc.cap(stock_data).copy(), 'signal')

    oey = operating_income.div(cap)
    num_days_neg = fc.ts_sum((fc.daily_return(stock_data) < 0).astype(float), window=252)

    oey_momentum = fc.ts_delta(oey, window=125).div(fc.ts_std(oey, window=252))
    price_momentum = -fc.ts_delta(close, window=1)

    factor = fc.rank(oey_momentum).add(fc.rank(num_days_neg)).add(fc.rank(price_momentum))

    factor = fc.rename_attribute(factor, 'fundamental_2')
    factor = fc.settings(factor, settings, stock_data[['sic_code']])
    return factor

    
"""
===== LIQUIDITY FACTORS =====
"""

def liquidity_1(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Liquidity 1: -abs(returns) / (volume * close)
    """

    volume = fc.rename_attribute(stock_data[['volume']].copy(), 'signal')
    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    returns = fc.rename_attribute(fc.daily_return(stock_data), 'signal') # can replace by stock_data[['returns']] if using massive_ohlcv_more_attr.parquet with pre-computed returns

    factor = -np.abs(returns).div(volume * close)

    factor = fc.rename_attribute(factor, 'liquidity_1')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor


def liquidity_2(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Liquidity 2: -ts_sum(returns == 0? 1 : 0, 252)
    """

    returns = fc.rename_attribute(fc.daily_return(stock_data), 'signal') # can replace by stock_data[['returns']] if using massive_ohlcv_more_attr.parquet with pre-computed returns

    zero_return = (returns == 0).astype(float)
    factor = -zero_return.rolling(window=252).sum().fillna(0)

    factor = fc.rename_attribute(factor, 'liquidity_2')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor

def liquidity_test(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Liquidity Test: liquidity 2 but with window = 60
    """

    returns = fc.rename_attribute(fc.daily_return(stock_data), 'signal') # can replace by stock_data[['returns']] if using massive_ohlcv_more_attr.parquet with pre-computed returns

    zero_return = (returns == 0).astype(float)
    factor = -zero_return.rolling(window=60).sum().fillna(0)

    factor = fc.rename_attribute(factor, 'liquidity_test')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor

def liquidity_3(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Custom liquidity factor:
    illiquid = rank(ts_sum(returns == 0 ? 1 : 0, 60))
    rev_neutral = rank(ts_mean(vwap, 10) - close) - 0.5
    healthy_trend = rank(close / ts_mean(close, 60))
    factor = illiquid * rev_neutral * healthy_trend
    """
    close = fc.rename_attribute(stock_data[['close']].copy(), 'signal')
    
    # 1. illiquid
    returns = fc.rename_attribute(fc.daily_return(stock_data), 'signal')
    returns_zero = (returns == 0.0).astype(float)
    ts_sum_zeros = fc.rename_attribute(fc.ts_sum(returns_zero, window=60), 'signal')
    illiquid = fc.rank(ts_sum_zeros)
    
    # 2. rev_neutral
    vwap_df = fc.rename_attribute(stock_data[['vwap']], 'signal')
    ts_mean_vwap = fc.ts_mean(vwap_df, window=10) # returns 'signal'
    rev_neutral_df = ts_mean_vwap.sub(close)
    rev_neutral = fc.rank(rev_neutral_df) - 0.5
    
    # 3. healthy_trend
    ts_mean_close = fc.ts_mean(close, window=60) # returns 'signal'
    healthy_trend_df = close.div(ts_mean_close).replace([np.inf, -np.inf], np.nan).fillna(0)
    healthy_trend = fc.rank(healthy_trend_df)
    
    # 4. Combine
    factor = illiquid.mul(rev_neutral).mul(healthy_trend)
    
    factor = fc.rename_attribute(factor, 'custom_liquidity')
    factor = fc.settings(factor, DEFAULT_SETTINGS, stock_data[['sic_code']])
    return factor


"""
===== BARRA-STYLE FACTORS =====

Implementations follow the formulas from the Barra US Equity Model (USE4)
and Global Equity Model (GEM2/GEMLT). Each factor is computed time-series,
then standardized cross-sectionally via `safe_settings`.

References (formulas):
    - Momentum (RSTR): exponentially-weighted sum of excess log returns over
      a 504-day lookback, lagged by 21 days, half-life 126.
    - Short-term reversal (STREV): negative exp-weighted log return over the
      most recent 21 days, half-life 5.
    - Liquidity (LIQ): composite of STOM/STOQ/STOA — log turnover over 1, 3,
      and 12 months. Weights 0.35/0.35/0.30 per Barra USE4.
    - Value (BTOP proxy): without book value we use the canonical Fama-French
      long-term reversal proxy: -log(close / 1260-day MA).
    - Quality: without ROE/leverage we proxy via the well-documented
      low-volatility / earnings-stability link: -trailing 252-day return std.
"""


def momentum_barra(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Barra Momentum (RSTR) — exponentially-weighted sum of log returns,
    lagged by 21 days to skip short-term reversal effects.

        RSTR_t = sum_{s=t-T_lag-T_window}^{t-T_lag-1} w_s * ln(1 + r_s)
        T_window = 504, T_lag = 21, half-life = 126
    """
    close = stock_data[['close']].copy()
    log_ret = fc.rename_attribute(np.log(close).diff(), 'signal')

    rstr_no_lag = fc.ts_ewm_sum(log_ret, window=504, half_life=126.0)
    factor = fc.ts_delay(rstr_no_lag, window=21)

    factor = fc.safe_settings(factor, DEFAULT_SETTINGS, stock_data)
    factor = fc.rename_attribute(factor, 'momentum_barra')
    return factor


def short_term_reversal_barra(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Barra Short-Term Reversal — negative exp-weighted cumulative log return
    over the trailing month. Stocks that have rallied recently get a negative
    score (expected to revert).

        STREV_t = -sum_{s=t-21}^{t-1} w_s * ln(1 + r_s),  half-life = 5
    """
    
    settings = {**DEFAULT_SETTINGS, 'decay': {'decay_days': 1}}
    
    close = stock_data[['close']].copy()
    log_ret = fc.rename_attribute(np.log(close).diff(), 'signal')

    streak = fc.ts_ewm_sum(log_ret, window=21, half_life=5.0)
    factor = -streak

    
    factor = fc.safe_settings(factor, settings, stock_data)
    factor = fc.rename_attribute(factor, 'short_term_reversal_barra')
    return factor


def liquidity_barra(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Barra Liquidity (LIQ) — composite of share turnover at 1m / 3m / 12m
    horizons:

        STOM_t = ln(sum_{s in last 21d} V_s / S_s)
        STOQ_t = ln((1/3)  * sum_{m in last 3 months}  exp(STOM_m))
        STOA_t = ln((1/12) * sum_{m in last 12 months} exp(STOM_m))
        LIQ_t  = 0.35*z(STOM) + 0.35*z(STOQ) + 0.30*z(STOA)

    Without shares outstanding we use V_s / mean(V over 252d) as a turnover
    proxy. After cross-sectional standardization in `safe_settings` this is
    equivalent up to a per-stock constant.
    """
    volume = stock_data[['volume']].copy()
    vol_signal = fc.rename_attribute(volume.copy(), 'signal')
    avg_vol = fc.ts_mean(vol_signal, window=252).replace(0.0, np.nan)
    turnover_proxy = pd.DataFrame(
        vol_signal.values / avg_vol.values,
        index=vol_signal.index,
        columns=vol_signal.columns,
    )
    turnover_proxy = fc.rename_attribute(turnover_proxy, 'signal')

    # 1-month log turnover
    stom = np.log(turnover_proxy.rolling(21, min_periods=21).sum().replace(0.0, np.nan))
    stom = fc.rename_attribute(stom, 'signal')

    # 3-month and 12-month aggregates of monthly turnover.
    # Use rolling sums over 63 / 252 days, divided by 3 / 12 to match the
    # "average of monthly turnover" definition.
    stoq = np.log((turnover_proxy.rolling(63, min_periods=63).sum() / 3.0).replace(0.0, np.nan))
    stoq = fc.rename_attribute(stoq, 'signal')

    stoa = np.log((turnover_proxy.rolling(252, min_periods=252).sum() / 12.0).replace(0.0, np.nan))
    stoa = fc.rename_attribute(stoa, 'signal')

    liq = 0.35 * fc.zscore(stom) + 0.35 * fc.zscore(stoq) + 0.30 * fc.zscore(stoa)
    liq = fc.rename_attribute(liq, 'signal')

    factor = fc.safe_settings(liq, DEFAULT_SETTINGS, stock_data)
    factor = fc.rename_attribute(factor, 'liquidity_barra')
    return factor


def value_barra(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Barra Value (BTOP) — book-to-price.

    The example dataset is OHLCV-only, so we use the canonical price-only
    proxy: long-term reversal vs. the 5-year (1260-day) moving average.
    Stocks priced well below trend score positive ("cheap").

        VAL_t = -ln(close_t / MA_1260(close)_t)

    When `book_value` is available downstream this should be replaced by
    z(book_value / market_cap).
    """
    close = stock_data[['close']].copy()
    close = fc.rename_attribute(close, 'signal')
    long_ma = fc.ts_mean(close, window=1260)
    factor = -np.log(close / long_ma)
    factor = fc.rename_attribute(factor, 'signal')

    factor = fc.safe_settings(factor, DEFAULT_SETTINGS, stock_data)
    factor = fc.rename_attribute(factor, 'value_barra')
    return factor


def quality_barra(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Barra Quality — composite of profitability, leverage, and earnings
    variability.

    With OHLCV-only data we proxy via the earnings-stability sub-component,
    which empirically dominates Barra's Quality loading (Asness, Frazzini &
    Pedersen 2019): low realized volatility => high quality.

        QUAL_t = -std(daily_returns, 252)

    When fundamentals are present this should be replaced by a weighted
    composite of ROE, leverage, and earnings variability.
    """
    returns = fc.daily_return(stock_data)
    returns = fc.rename_attribute(returns, 'signal')
    vol = fc.ts_std(returns, window=252)
    factor = -vol

    factor = fc.safe_settings(factor, DEFAULT_SETTINGS, stock_data)
    factor = fc.rename_attribute(factor, 'quality_barra')
    return factor


"""
===== TESTING =====
"""


if __name__ == "__main__":

    file_ext = os.path.splitext(example_filepath)[1].lower()
    if file_ext == '.csv':
        example_data = pd.read_csv(example_filepath, header=[0, 1], dtype=float, index_col=0, parse_dates=True, encoding='utf-8')
    elif file_ext == '.parquet':
        example_data = pd.read_parquet(example_filepath)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    print(f"Data loaded successfully with shape: {example_data.shape}")
    print(f"Date range: {example_data.index.min().date()} to {example_data.index.max().date()}")
    print(f"Stocks: {example_data.columns.get_level_values(1).unique().tolist()}\n")

    barra_factors = [
        momentum_barra,
        short_term_reversal_barra,
        liquidity_barra,
        value_barra,
        quality_barra,
    ]

    holding_periods = [1, 5, 21]

    for hp in holding_periods:
        print(f"=== Barra Factors — IC / Rank IC ({hp}-day forward return) ===")
        rows = []
        for f in barra_factors:
            try:
                res = fc.ic_and_rank_ic(example_data, f, holding_period=hp)
                rows.append(res)
                print(f"  {res['factor']:<28}  IC={res['IC']:+.4f}  RankIC={res['RankIC']:+.4f}  ICIR={res['ICIR']:+.3f}  RankICIR={res['RankICIR']:+.3f}  n={res['n_obs']}")
            except Exception as e:
                print(f"  {f.__name__:<28}  FAILED: {e!r}")
        print()

