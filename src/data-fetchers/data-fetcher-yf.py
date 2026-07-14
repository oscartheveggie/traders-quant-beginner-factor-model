import yfinance as yf
import pandas as pd
from tqdm import tqdm
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================
# Global Settings, change here
start_date = '2015-01-01'
end_date = '2025-12-31'
output_filename = 'example.csv'
interval = '1d'
# =====================================
output_filepath = os.path.join(script_dir, '..', '..', 'data', output_filename)

# placeholder stock list, replace with actual stock tickers (e.g., Russell 3000)
stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

def fetch_stock_data(tickers: list) -> pd.DataFrame:

    all_data = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Fetching stock data"):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, interval=interval)
            stock_data = stock_data.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            stock_data.columns = pd.MultiIndex.from_tuples(
                [(col[0].lower(), col[1]) for col in stock_data.columns]
            )
            all_data = pd.concat([all_data, stock_data], axis=1)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    # print(all_data.head())      # debugging line, comment out if not needed
    all_data.columns.names = ['attribute', 'stock']

    # Count rows with NaN values
    nan_counts = all_data.isna().any(axis=1).sum()
    print(f"Number of rows with at least one NaN:\t{nan_counts}/{len(all_data)}")

    # Count rows with all NaN values
    all_nan_counts = all_data.isna().all(axis=1).sum()
    print(f"Number of rows with all NaN:\t\t{all_nan_counts}/{len(all_data)}")

    if all_nan_counts != nan_counts:
        raise ValueError("There are rows with some NaN values but not all NaN values. Consider handling these cases.")
    else:
        print(f"Total working days:\t\t\t{(len(all_data)-nan_counts)/len(all_data)*365:.0f}")

    print()

    all_data.dropna(how='all', inplace=True)  # Drop rows where all values are NaN

    # Reorder columns: group by attribute, then sort stocks alphabetically
    all_data = all_data.reindex(sorted(all_data.columns, key=lambda x: (x[0], x[1])), axis=1)

    return all_data

def store_data(data: pd.DataFrame, filename: str, print_data: bool = True) -> None:
    try:
        data.to_csv(filename)
    except Exception as e:
        print(f"Error storing data to {filename}: {e}")
    else:
        print(f"Data successfully stored to {filename}")
        if print_data:
            print(data.head())

if __name__ == "__main__":

    print_data = False  # Set to True to print the fetched data

    example = fetch_stock_data(stock_list)
    rf_rate = fetch_stock_data(['^IRX']) / 100
    benchmark = fetch_stock_data(['^RUA'])
    
    store_data(benchmark, os.path.join(script_dir, '..', '..', 'data', 'benchmark.csv'), print_data=print_data)
    store_data(example, output_filepath)
    store_data(rf_rate, os.path.join(script_dir, '..', '..', 'data', 'rf_rate.csv'), print_data=print_data)
    
