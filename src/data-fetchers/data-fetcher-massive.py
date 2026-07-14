import os
import asyncio
from dotenv import load_dotenv
from massive import RESTClient
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# =====================================
# Global Settings, change here
start_date = '2016-01-01'
end_date = '2025-12-31'

ohlcv_raw_filename = 'massive_ohlcv_raw'
fundamentals_raw_filename = 'massive_fundamentals_raw_annual'
ohlcv_debug_filename = 'massive_ohlcv_debug'
fundamentals_debug_filename = 'massive_fundamentals_debug_annual'
combined_raw_filename = 'massive_combined_raw'
timeframe = 'annual'  # 'annual' or 'quarterly'


output_filetype = 'parquet'  # 'csv' or 'parquet'

debug_mode = False

max_workers = 20  # Number of threads for parallel fetching
checkpoint_every = 25  # Save progress every N successful fetched dates

datatype_to_fetch = ['fundamentals_daily']  # Options: 'ohlcv_daily', 'ohlcv_intraday', 'fundamentals_daily'

ohlcv_attributes = ['open', 'high', 'low', 'close', 'volume', 'vwap']  # OHLCV attributes to fetch and process


fundamental_attributes = ['cash_and_equivalents', 'total_assets', 
                          'total_liabilities', 'total_equity',
                          'dividends', 'net_income', 'ebitda',
                          'operating_income', 'diluted_earnings_per_share',
                          'diluted_shares_outstanding']


# fundamental_attributes = ['diluted_earnings_per_share', 'diluted_shares_outstanding']  # For testing, we will only fetch this attribute to speed up the process


# =====================================

script_dir = os.path.dirname(os.path.abspath(__file__))
if output_filetype == 'csv':
    ohlcv_raw_filepath = os.path.join(script_dir, '..', '..', 'data', f"{ohlcv_raw_filename}.csv")
    fundamentals_raw_filepath = os.path.join(script_dir, '..', '..', 'data', f"{fundamentals_raw_filename}.csv")
    combined_raw_filepath = os.path.join(script_dir, '..', '..', 'data', f"{combined_raw_filename}.csv")
else:
    ohlcv_raw_filepath = os.path.join(script_dir, '..', '..', 'data', f"{ohlcv_raw_filename}.parquet")
    fundamentals_raw_filepath = os.path.join(script_dir, '..', '..', 'data', f"{fundamentals_raw_filename}.parquet")
    combined_raw_filepath = os.path.join(script_dir, '..', '..', 'data', f"{combined_raw_filename}.parquet")

if debug_mode:
    debug_output_filepath = os.path.join(script_dir, '..', '..', 'data', f"{ohlcv_debug_filename}.parquet")
    fundamentals_debug_output_filepath = os.path.join(script_dir, '..', '..', 'data', f"{fundamentals_debug_filename}.parquet")
else:
    debug_output_filepath = None
    fundamentals_debug_output_filepath = None

load_dotenv()
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in the .env file.")

client = RESTClient(api_key=API_KEY)


"""
===== FETCHER FUNCTIONS =====
"""

def get_ohlcv_daily(client: RESTClient, start_date: str, end_date: str, debug_mode=True) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data from Massive API
    
    Args:
        client: RESTClient instance for Massive API
        start_date: Start date as string (format: 'YYYY-MM-DD')
        end_date: End date as string (format: 'YYYY-MM-DD')
        debug_mode: If True, fetch data for the last 10 days ending at end_date; if False, fetch data for all dates in range
    
    Returns:
        DataFrame with MultiIndex columns (attribute, stock) and DatetimeIndex
    """

    
    async def fetch_single_day(date: pd.Timestamp) -> pd.DataFrame | tuple[str, str] | None:
        date_str = date.strftime('%Y-%m-%d')
        try:
            # Massive client is sync; run it in a worker thread under asyncio.
            data = await asyncio.to_thread(client.get_grouped_daily_aggs, date_str, adjusted="true")
            if data:
                result = pd.DataFrame(data)
                return to_multiindex_ohlcv(result, date)
        except Exception as e:
            return (date_str, str(e))
        return None

    async def fetch_all_dates(
        date_range: pd.DatetimeIndex,
        existing_data: pd.DataFrame,
        filepath: str
    ) -> pd.DataFrame:
        semaphore = asyncio.Semaphore(max_workers)

        async def bounded_fetch(date: pd.Timestamp) -> pd.DataFrame | tuple[str, str] | None:
            async with semaphore:
                return await fetch_single_day(date)

        tasks = [asyncio.create_task(bounded_fetch(d)) for d in date_range]
        results_buffer: list[pd.DataFrame] = []
        merged_data = existing_data.copy()
        success_count = 0

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="🚀 Fast Fetching OHLCV"):
            res = await task
            if isinstance(res, tuple):
                print(f"⚠️ Failed for {res[0]}: {res[1]}")
            elif res is not None:
                results_buffer.append(res)
                success_count += 1

                if success_count % checkpoint_every == 0:
                    checkpoint_df = pd.concat(results_buffer, axis=0)
                    merged_data = merge_and_deduplicate(merged_data, checkpoint_df)
                    save_dataframe(merged_data, filepath, output_filetype)
                    print(f"💾 Checkpoint saved after {success_count} fetched dates.")
                    results_buffer.clear()

        if results_buffer:
            checkpoint_df = pd.concat(results_buffer, axis=0)
            merged_data = merge_and_deduplicate(merged_data, checkpoint_df)
            save_dataframe(merged_data, filepath, output_filetype)
            print(f"💾 Final checkpoint saved after {success_count} fetched dates.")

        return merged_data
    

    # ==================================================
    # Main fetching logic

    if debug_mode:
        start_dt = pd.to_datetime(end_date) - pd.Timedelta(days=9)  # Last 10 days including end_date
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    else:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    fetch_start_time = pd.Timestamp.now()

    if debug_mode:
        existing_data = load_existing_dataframe(debug_output_filepath, output_filetype)
    else:
        existing_data = load_existing_dataframe(ohlcv_raw_filepath, output_filetype)
    existing_dates = set(existing_data.index)

    dates_to_fetch = pd.DatetimeIndex([d for d in date_range if d not in existing_dates])
    skipped_count = len(date_range) - len(dates_to_fetch)
    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} dates already present in {os.path.basename(ohlcv_raw_filepath if not debug_mode else debug_output_filepath)}")

    if len(dates_to_fetch) == 0:
        print("✅ All requested dates already exist. Nothing new to fetch.")
        requested_slice = existing_data.reindex(date_range)
        return sort_rows(sort_columns(requested_slice.dropna(how='all')))

    merged_data = asyncio.run(fetch_all_dates(dates_to_fetch, existing_data, ohlcv_raw_filepath if not debug_mode else debug_output_filepath))

    fetch_end_time = pd.Timestamp.now()
    total_fetch_time = fetch_end_time - fetch_start_time

    if debug_mode:
        print(f"Total time elapsed for fetching 10 days: {tqdm.format_interval(total_fetch_time.total_seconds())}")
    else:
        print(f"Total time elapsed for fetching {len(dates_to_fetch)} dates: {tqdm.format_interval(total_fetch_time.total_seconds())}")

    # Final processing of results
    requested_slice = merged_data.reindex(date_range)
    requested_slice = requested_slice.dropna(how='all')

    if not requested_slice.empty:
        requested_slice = sort_columns(requested_slice)
        requested_slice = sort_rows(requested_slice)
        print(f"✅ Success: Available data for {len(requested_slice)} dates")
        return requested_slice
    
    return pd.DataFrame()


def get_fundamentals_daily(client: RESTClient, start_date: str, end_date: str, debug_mode: bool = True) -> pd.DataFrame:
    """
    Fetch fundamentals data from the Massive API and normalize it to a
    MultiIndex dataframe with dates as rows and (attribute, stock) columns.
    
    Args:
        client: RESTClient instance for Massive API
        start_date: Start date as string (format: 'YYYY-MM-DD')
        end_date: End date as string (format: 'YYYY-MM-DD')
    """

    LOOKBACK_DAYS = 400  # Number of past days to fetch for each date to ensure we get the latest available data for each date in the range

    async def fetch_single_day(date: pd.Timestamp) -> pd.DataFrame | tuple[str, str] | None:
        date_str = date.strftime('%Y-%m-%d')
        try:
            # Massive client is sync; run it in a worker thread under asyncio.
            bs_data = await asyncio.to_thread(client.list_financials_balance_sheets, filing_date_gte=date_str, filing_date_lte=date_str, timeframe=timeframe, sort="period_end.asc")
            # bs_data = []
            is_data = await asyncio.to_thread(client.list_financials_income_statements, filing_date_gte=date_str, filing_date_lte=date_str, timeframe=timeframe, sort="period_end.asc")
            cfs_data = await asyncio.to_thread(client.list_financials_cash_flow_statements, filing_date_gte=date_str, filing_date_lte=date_str, timeframe=timeframe, sort="period_end.asc")
            # cfs_data = []

            bs_data = pd.DataFrame(bs_data); is_data = pd.DataFrame(is_data); cfs_data = pd.DataFrame(cfs_data)

            # Combine the data from all three sources
            combined_data = pd.concat([bs_data, is_data, cfs_data], ignore_index=True)
            if combined_data.empty:
                return pd.DataFrame()
            
            available_attributes = set(combined_data.columns)
            combined_data = combined_data[list(available_attributes & set(fundamental_attributes) | {'tickers', 'timeframe'})]
            return to_multiindex_fundamentals(combined_data, date)
            
        except Exception as e:
            raise e
            return (date_str, str(e))
        return None

    async def fetch_all_dates(
        date_range: pd.DatetimeIndex,
        existing_data: pd.DataFrame,
        filepath: str
    ) -> pd.DataFrame:
        semaphore = asyncio.Semaphore(max_workers)

        async def bounded_fetch(date: pd.Timestamp) -> pd.DataFrame | tuple[str, str] | None:
            async with semaphore:
                return await fetch_single_day(date)

        tasks = [asyncio.create_task(bounded_fetch(d)) for d in date_range]
        results_buffer: list[pd.DataFrame] = []
        merged_data = existing_data.copy()
        success_count = 0

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="🚀 Fast Fetching Fundamentals"):
            res = await task
            if isinstance(res, tuple):
                print(f"⚠️ Failed for {res[0]}: {res[1]}")
            elif res is not None:
                results_buffer.append(res)
                success_count += 1

                if success_count % checkpoint_every == 0:
                    checkpoint_df = pd.concat(results_buffer, axis=0)
                    merged_data = merge_and_deduplicate(merged_data, checkpoint_df)
                    save_dataframe(merged_data, filepath, output_filetype)
                    print(f"💾 Checkpoint saved after {success_count} fetched dates.")
                    results_buffer.clear()

        if results_buffer:
            checkpoint_df = pd.concat(results_buffer, axis=0)
            merged_data = merge_and_deduplicate(merged_data, checkpoint_df)
            save_dataframe(merged_data, filepath, output_filetype)
            print(f"💾 Final checkpoint saved after {success_count} fetched dates.")

        return merged_data
    

    # ==================================================
    # Main fetching logic

    if debug_mode:
        start_dt = pd.to_datetime(end_date) - pd.Timedelta(days=14)  # Last 15 days including end_date
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    else:
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    fetch_start_time = pd.Timestamp.now()

    if debug_mode:
        existing_data = load_existing_dataframe(fundamentals_debug_output_filepath, output_filetype)
    else:
        existing_data = load_existing_dataframe(fundamentals_raw_filepath, output_filetype)
    existing_dates = set(existing_data.index)

    dates_to_fetch = pd.DatetimeIndex([d for d in date_range if d not in existing_dates])
    skipped_count = len(date_range) - len(dates_to_fetch)
    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} dates already present in {os.path.basename(fundamentals_raw_filepath if not debug_mode else fundamentals_debug_output_filepath)}")

    if len(dates_to_fetch) == 0:
        print("✅ All requested dates already exist. Nothing new to fetch.")
        requested_slice = existing_data.reindex(date_range)
        return sort_rows(sort_columns(requested_slice.dropna(how='all')))

    merged_data = asyncio.run(fetch_all_dates(dates_to_fetch, existing_data, fundamentals_raw_filepath if not debug_mode else fundamentals_debug_output_filepath))

    fetch_end_time = pd.Timestamp.now()
    total_fetch_time = fetch_end_time - fetch_start_time

    if debug_mode:
        print(f"Total time elapsed for fetching 15 days: {tqdm.format_interval(total_fetch_time.total_seconds())}")
    else:
        print(f"Total time elapsed for fetching {len(dates_to_fetch)} dates: {tqdm.format_interval(total_fetch_time.total_seconds())}")

    # Final processing of results
    requested_slice = merged_data.reindex(date_range)
    requested_slice = requested_slice.dropna(how='all')

    if not requested_slice.empty:
        requested_slice = sort_columns(requested_slice)
        requested_slice = sort_rows(requested_slice)
        print(f"✅ Success: Available data for {len(requested_slice)} dates")
        return requested_slice
    
    return pd.DataFrame()


"""
===== HELPER FUNCTIONS =====
These functions are used to transform and clean the data after fetching from the API.
"""

def to_multiindex_ohlcv(df: pd.DataFrame, current_date: pd.Timestamp):
    """
    Transform a flat dataframe with columns ['ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp', 'transactions', 'otc']
    into a one-row MultiIndex dataframe with:
    - Top level columns: attributes (open, high, low, close, volume, vwap)
    - Bottom level columns: tickers (sorted alphabetically within each attribute)
    - Index: current_date (as datetime)
    """

    # Create a dictionary with MultiIndex columns (attribute, ticker)
    data = {}
    for _, row in df.iterrows():
        ticker = row['ticker']
        for attr in ohlcv_attributes:
            data[(attr, ticker)] = row[attr]
    
    # Create dataframe with single row and MultiIndex columns
    result = pd.DataFrame([data], index=pd.DatetimeIndex([current_date]))
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=['attribute', 'stock'])
    
    return result

def to_multiindex_fundamentals(df: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:

    tickers = [ticker for sublist in df['tickers'] for ticker in sublist]  # Flatten list of lists
    tickers = sorted(set(tickers))  # Get unique tickers and sort alphabetically
    attributes = [col for col in df.columns if col != 'tickers']  # All columns except 'tickers'
    cols = pd.MultiIndex.from_product([attributes, tickers], names=['attribute', 'stock'])

    result = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([current_date]))
    for _, row in df.iterrows():
        if row['timeframe'] != timeframe:
            continue
        for attr in attributes:
            for ticker in row['tickers']:
                result.at[current_date, (attr, ticker)] = row[attr]

    return result


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort columns of a MultiIndex dataframe first by attribute 
    (open, high, low, close, volume, vwap for ohlcv data, alphabetical for fundamentals) 
    and then by stock ticker alphabetically.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DataFrame columns must be a MultiIndex with levels ['attribute', 'stock'].")
    
    attribute_cols = df.columns.get_level_values('attribute').unique()
    if attribute_cols.tolist() == list(ohlcv_attributes):
        # If attributes are the expected OHLCV attributes, sort by the predefined order
        attribute_order = {attr: i for i, attr in enumerate(ohlcv_attributes)}
    else:
        # Otherwise, sort attributes alphabetically
        attribute_order = {attr: i for i, attr in enumerate(sorted(attribute_cols))}
    
    sorted_columns = sorted(df.columns, key=lambda x: (attribute_order.get(x[0], len(attribute_cols)), x[1]))
    return df.reindex(columns=sorted_columns)

def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows of a dataframe by index (date) in ascending order.
    """
    return df.sort_index()


def merge_and_deduplicate(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge fetched data into existing data and keep one row per date (latest wins)."""
    if existing_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing_df, new_df], axis=0)
    merged = merged[~merged.index.duplicated(keep='last')]
    merged = sort_rows(merged)
    merged = sort_columns(merged)
    return merged


def load_existing_dataframe(filepath: str, filetype: str) -> pd.DataFrame:
    """Load existing dataset if present; otherwise return an empty DataFrame."""
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        if filetype == 'parquet':
            df = pq.read_table(
                filepath,
                thrift_string_size_limit=1_000_000_000,
                thrift_container_size_limit=1_000_000_000,
            ).to_pandas()
        else:
            df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=['attribute', 'stock'])

        print(f"Loaded existing data from {filepath} with shape {df.shape}")
        return sort_rows(sort_columns(df)) if not df.empty else df
    except Exception as e:
        print(f"WARNING: Could not load existing file at {filepath}: {e}")
        return pd.DataFrame()


def save_dataframe(df: pd.DataFrame, filepath: str, filetype: str) -> None:
    """Persist dataset to disk as parquet or csv."""
    if filetype == 'parquet':
        df.to_parquet(filepath)
    else:
        df.to_csv(filepath, index=True)


"""
===== MAIN FUNCTION =====
"""

def main():

    print("\n=== MASSIVE DATA FETCHER ===\n")

    print(f"Using API_KEY: {API_KEY[:4]}...{API_KEY[-4:]}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Debug mode: {'\033[32mON\033[0m' if debug_mode else '\033[33mOFF\033[0m'}\n")
    print(f"Fetching data for: {', '.join(datatype_to_fetch)}\n")

    if debug_mode:
        print("⚠️  Debug mode is ON. Fetching data for a limited date range (last 10 days).")
    else:
        is_run = input("⚠️  Debug mode is OFF. Are you sure you want to proceed? This may take a long time. (y/n): ")
        if is_run.lower() != 'y':
            print("❌ Fetching cancelled by user.")
            return
        else:
            print("🚀 Starting full data fetch...\n")

    if 'ohlcv_daily' in datatype_to_fetch:

        print(f"Target output file for OHLCV data: {ohlcv_raw_filepath if not debug_mode else debug_output_filepath}")
        print("Fetching OHLCV data...")
        ohlcv_df = get_ohlcv_daily(client, start_date, end_date, debug_mode=debug_mode)

        print(f"OHLCV data saved to {ohlcv_raw_filepath}")
        print(f"Data shape: {ohlcv_df.shape}")
        print(f"Date range: {ohlcv_df.index[0]} to {ohlcv_df.index[-1]}")
        print("\nSample data:")
        print(ohlcv_df.head())

    if 'fundamentals_daily' in datatype_to_fetch:

        print(f"Target output file for fundamentals data: {fundamentals_raw_filepath if not debug_mode else fundamentals_debug_output_filepath}")
        print("Fetching fundamentals data...")
        fundamentals_df = get_fundamentals_daily(client, start_date, end_date, debug_mode=debug_mode)

        if not fundamentals_df.empty:
            print(f"Fundamentals data saved to {fundamentals_debug_output_filepath if debug_mode else fundamentals_raw_filepath}")
            print(f"Data shape: {fundamentals_df.shape}")
            print(f"Date range: {fundamentals_df.index[0]} to {fundamentals_df.index[-1]}")
            print("\nSample data:")
            print(fundamentals_df.head())
    

    # Here you would typically merge the two datasets on date and ticker
    # For example:
    # merged_data = pd.merge(ohlcv_df, fundamentals_df, on=['date', 'ticker'])
    
    # Save the merged data to CSV
    # merged_data.to_csv(ohlcv_raw_filepath, index=False)
    # print(f"Data saved to {ohlcv_raw_filepath}")

if __name__ == "__main__":
    main()
