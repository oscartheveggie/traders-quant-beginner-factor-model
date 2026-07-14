from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
from massive import RESTClient
import os
import asyncio

load_dotenv()
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in the .env file.")

client = RESTClient(api_key=API_KEY)

def get_ticker_mktcap(client: RESTClient, ticker: str) -> float:
    details = client.get_ticker_details(ticker)
    return details.market_cap if details and details.market_cap is not None else 0.0

def get_all_tickers_mktcap(client: RESTClient, tickers: list) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([['cap'], tickers], names=['attribute', 'stock'])
    mktcap = pd.DataFrame(columns=cols)
    for ticker in tqdm(tickers, desc="Fetching market caps"):
        mktcap[('cap', ticker)] = get_ticker_mktcap(client, ticker)
    return mktcap


async def remove_non_stocks(stock_data: pd.DataFrame, client: RESTClient) -> pd.DataFrame:
    """
    Remove non-stock instruments (e.g., ETFs) from the dataframe based on ticker symbols.
    This is a placeholder function and should be implemented based on the specific criteria you want to use for identifying non-stock instruments.
    
    Args:
        stock_data: DataFrame with MultiIndex columns (attribute, stock) and DatetimeIndex
    Returns:
        DataFrame with non-stock instruments removed
    """

    tickers = stock_data.columns.get_level_values('stock').unique()
    filtered_tickers = []

    sem = asyncio.Semaphore(20)

    async def check_ticker(ticker: str):
        async with sem:
            try:
                details = await asyncio.to_thread(client.get_ticker_details, ticker)
                if details is None:
                    return None
                if getattr(details, 'type', None) == 'CS':
                    return ticker
            except Exception:
                return None

    tasks = [asyncio.create_task(check_ticker(t)) for t in tickers]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Checking tickers"):
        res = await coro
        if res:
            filtered_tickers.append(res)

    tickers = filtered_tickers

    stock_data = stock_data.loc[:, stock_data.columns.get_level_values('stock').isin(tickers)]

    return stock_data


def nan_handling(stock_data: pd.DataFrame, nan_threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle NaN values in stock data by identifying and removing stocks with excessive missing data.
    
    Steps:
    1. Calculate NaN ratio for each stock across all dates and attributes
    2. Remove stocks with NaN ratio > threshold (default 50%) - likely delisted or with poor data quality
    3. Return cleaned dataframe with same MultiIndex structure
    """
    
    # Get all stocks (level 1 of MultiIndex columns)
    stocks = stock_data.columns.get_level_values('stock').unique()
    
    stocks_to_keep = []
    stocks_to_remove = []
    
    print(f"\n🔍 NaN Handling Analysis (threshold: {nan_threshold*100}%)")
    print(f"Total stocks found: {len(stocks)}")
    print("-" * 60)
    
    # Analyze each stock
    for stock in tqdm(stocks):

        # Get all data for this stock across all attributes
        stock_data_slice = stock_data.xs(stock, level='stock', axis=1)  
        
        # Calculate NaN ratio
        total_values = stock_data_slice.size
        nan_count = stock_data_slice.isna().sum().sum()
        nan_ratio = nan_count / total_values if total_values > 0 else 0
        
        if nan_ratio > nan_threshold:
            stocks_to_remove.append(stock)
        else:
            stocks_to_keep.append(stock)
    
    print("-" * 60)
    print(f"Removing {len(stocks_to_remove)} stocks, keeping {len(stocks_to_keep)} stocks\n")
    
    # Create cleaned dataframe with only stocks_to_keep
    cleaned_df = stock_data.loc[:, stock_data.columns.get_level_values('stock').isin(stocks_to_keep)]
    
    return cleaned_df


async def top_n_filter(stock_data: pd.DataFrame, top_n: int = 3000, rank_by: str = 'volume') -> pd.DataFrame:
    """
    Filter stocks to keep only top N stocks by market ranking (historical filtering).
    Stocks not in top N are marked with -1 for all attributes.
    
    Data Structure (MultiIndex):
    - Columns (MultiIndex, 2 levels):
      Level 0 (attribute): 'open', 'high', 'low', 'close', 'volume', 'vwap', etc.
      Level 1 (stock): ticker symbols (e.g., 'AAPL', 'GOOGL', 'MSFT')
    - Index: DatetimeIndex with dates
    - Each cell: price/volume data
    
    Strategy:
    1. For each timestamp, rank all stocks by specified metric (default: volume)
    2. Identify stocks in top N at that timestamp
    3. For stocks NOT in top N:
       - Set all attributes (open, high, low, close, volume, vwap) to -1
       - This distinguishes from NaN (missing data) vs out-of-range (not in top N)
    4. Return filtered dataframe with same MultiIndex structure
    
    Args:
        stock_data: DataFrame with MultiIndex columns (attribute, stock) and DatetimeIndex
        top_n: Number of top stocks to keep per timestamp (default: 3000)
        rank_by: Attribute to use for ranking stocks (default: 'volume')
    
    Returns:
        DataFrame with same structure, where out-of-range stocks have -1 values
    """
    
    print(f"\n🏆 Top-N Stock Filtering (N={top_n}, rank_by='{rank_by}')")
    print("-" * 60)
    
    # Get the ranking attribute for all stocks at all times
    if rank_by not in stock_data.columns.get_level_values('attribute'):
        print(f"⚠️  Warning: '{rank_by}' not found in attributes. Using 'volume' instead.")
        rank_by = 'volume'

    # Create a copy of the original data to modify
    filtered_data = stock_data.copy()

    # Get all attributes
    attributes = stock_data.columns.get_level_values('attribute').unique()

    # Rank with NumPy argsort, matching pandas rank(method='min', ascending=False).
    def rank_desc_min(values: pd.Series) -> pd.Series:
        valid = values.dropna()
        if valid.empty:
            return pd.Series(index=values.index, dtype=float)

        arr = valid.to_numpy()
        # Stable descending sort so equal values keep deterministic order.
        order = np.argsort(-arr, kind='mergesort')
        sorted_vals = arr[order]

        ranks_sorted = np.empty(len(sorted_vals), dtype=float)
        current_rank = 1
        ranks_sorted[0] = current_rank
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] != sorted_vals[i - 1]:
                current_rank = i + 1
            ranks_sorted[i] = current_rank

        ranks_by_position = np.empty(len(arr), dtype=float)
        ranks_by_position[order] = ranks_sorted

        ranks = pd.Series(np.nan, index=values.index, dtype=float)
        ranks.loc[valid.index] = ranks_by_position
        return ranks

    # Helper to process a single date (ticker-level)
    async def process_date_no_cik(date_idx: int, rank_data):
        date = rank_data.index[date_idx]
        ranks_at_date = rank_data.iloc[date_idx]
        if ranks_at_date.dropna().empty:
            return None

        # Higher values -> lower rank number (1 is top)
        ranks = rank_desc_min(ranks_at_date)
        return (date, ranks)

    # Helper to process a single date (cik-level)
    async def process_date_with_cik(date_idx: int, rank_data, cik_data):
        date = rank_data.index[date_idx]
        ranks_at_date = rank_data.iloc[date_idx]
        cik_at_date = cik_data.iloc[date_idx]

        valid_mask = ranks_at_date.notna() & cik_at_date.notna()
        valid_ranks = ranks_at_date[valid_mask]
        if valid_ranks.empty:
            return None

        # Sum ranking scores by company and rank cik groups.
        cik_scores = valid_ranks.groupby(cik_at_date[valid_mask]).sum()
        cik_ranks = rank_desc_min(cik_scores)
        
        # Map each stock to its CIK rank
        stock_cik_ranks = cik_at_date.map(cik_ranks)
        return (date, stock_cik_ranks)

    # Process timestamps concurrently using threads for per-date work
    rank_data = stock_data[rank_by]
    stocks_filtered_count = 0
    tasks = []
    sem = asyncio.Semaphore(10)

    async def sem_task(fn, *args):
        async with sem:
            return await fn(*args)

    if 'cik' not in stock_data.columns.get_level_values('attribute'):
        print("⚠️  Warning: 'cik' not found in attributes. Falling back to ticker-level filtering.")
        for date_idx in range(len(rank_data)):
            tasks.append(asyncio.create_task(sem_task(process_date_no_cik, date_idx, rank_data)))
    else:
        cik_data = stock_data['cik']
        for date_idx in range(len(rank_data)):
            tasks.append(asyncio.create_task(sem_task(process_date_with_cik, date_idx, rank_data, cik_data)))

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Filtering by Top-N"):
        res = await coro
        if res is None:
            continue
        date, ranks = res
        for stock in ranks.index:
            if pd.notna(ranks[stock]) and ranks[stock] > top_n:
                stocks_filtered_count += 1
                for attr in attributes:
                    filtered_data.loc[date, (attr, stock)] = -1

    print("-" * 60)
    print(f"✅ Filtering complete: {stocks_filtered_count} stock-date pairs marked as -1")
    print(f"   (representing stocks outside top {top_n})\n")

    return filtered_data


def clean_data(stock_data: pd.DataFrame, options: dict, client: RESTClient=None) -> pd.DataFrame:
    """
    Main function to run the data cleaning steps in sequence based on provided options.

    `options` format:
    {
        <function_name>: {<arg name>: <arg_value>, <arg name>: <arg_value>, ...},
        'remove_non_stocks': {},
        'nan_handling': {'nan_threshold': 0.5},
        'top_n_filter': {'top_n': 3000, 'rank_by': 'volume'},
    }
    If you do not wish to run a particular step, simply omit it from the options dictionary.
    """

    if 'nan_handling' in options:
        print("\nStep 2: Handling NaN values...")
        kwargs = options['nan_handling']
        stock_data = nan_handling(stock_data=stock_data, nan_threshold=kwargs.get('nan_threshold', 0.5))

    if 'remove_non_stocks' in options:
        print("\nStep 1: Removing non-stock instruments...")
        stock_data = asyncio.run(remove_non_stocks(client=client, stock_data=stock_data))

    if 'top_n_filter' in options:
        print("\nStep 3: Filtering top N stocks...")
        kwargs = options['top_n_filter']
        stock_data = asyncio.run(top_n_filter(stock_data=stock_data, top_n=kwargs.get('top_n', 3000), rank_by=kwargs.get('rank_by', 'volume')))

    print("\nData cleaning complete.")
    print(f"Final cleaned data shape: {stock_data.shape}\n")

    return stock_data
    

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "..", "data", "massive_ohlcv_cleaned.parquet")
    

    options = {
        # 'remove_non_stocks': {},  # Uncomment if you have a RESTClient instance and want to remove non-stock instruments
        'top_n_filter': {'top_n': 3000, 'rank_by': 'cap'},
    }

    # Load raw data
    stock_data = pd.read_parquet(data_path)
    print(stock_data.head(), end="\n\n")

    
    # Clean data
    cleaned_data = clean_data(stock_data, options, client=client)
    print(cleaned_data.head(), end="\n\n")
    

    # Save data
    cleaned_data.to_parquet(os.path.join(script_dir, "..", "..", "data", "massive_ohlcv_filtered.parquet"))
    

if __name__ == "__main__":
    main()
