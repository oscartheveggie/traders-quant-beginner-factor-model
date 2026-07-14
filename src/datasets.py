import pandas as pd
import os

DEFAULT_FILENAME = 'example.csv'

class Dataset:

    def __init__(self, 
                 filename: str = DEFAULT_FILENAME,
                 train_cv_test_split: tuple[float] = (0.7, 0.15, 0.15)
                 ):
        """
        Class to load and manage dataset for factor modeling.

        Args:
            filename: Name of the data file (CSV or Parquet) located in the 'data' directory
            train_cv_test_split: Tuple indicating the proportion of data to use for training, cross-validation, and testing (must sum to 1.0)
        
        Attribute:
            full_data: The complete dataset loaded from file
            train_data: Subset of data for training (first portion of full_data)
            val_data: Subset of data for validation (middle portion of full_data)
            test_data: Subset of data for testing (last portion of full_data)
            stocks: List of unique stock identifiers in the dataset

        Functions:
            get_individual_stock(stock, option): Return a DataFrame containing all attributes for a single stock from the specified dataset split (train/val/test)
        """

        if sum(train_cv_test_split) != 1.0:
            raise ValueError("train_cv_test_split must sum to 1.0")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_filepath = os.path.join(script_dir, '..', 'data', filename)

        # Test loading data and running factors
        file_ext = os.path.splitext(data_filepath)[1].lower()
        if file_ext == '.csv':
            self.full_data = pd.read_csv(data_filepath, header=[0,1], dtype=float, index_col=0, parse_dates=True)
        elif file_ext == '.parquet':
            self.full_data = pd.read_parquet(data_filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        print("Data loaded successfully. Shape:", self.full_data.shape)

        self.train_data = self.full_data.head(int(len(self.full_data) * train_cv_test_split[0]))
        print("Train data shape:", self.train_data.shape)
        self.val_data = self.full_data.iloc[int(len(self.full_data) * train_cv_test_split[0]) : int(len(self.full_data) * (train_cv_test_split[0] + train_cv_test_split[1]))]
        print("Validation data shape:", self.val_data.shape)
        self.test_data = self.full_data.iloc[int(len(self.full_data) * (train_cv_test_split[0] + train_cv_test_split[1])) :]
        print("Test data shape:", self.test_data.shape)

        self.stocks = self.full_data.columns.get_level_values('stock').unique()
        print("Unique stocks in dataset:", self.stocks)


    def get_individual_stock(self, stock: str, option: str) -> pd.DataFrame:
        """Return a DataFrame containing all attributes for a single stock."""
        if option == 'train':
            stock_data = self.train_data
        elif option == 'val':
            stock_data = self.val_data
        elif option == 'test':
            stock_data = self.test_data
        return stock_data.xs(stock, level='stock', axis=1, drop_level=False)

    


if __name__ == "__main__":
    dataset = Dataset()
    print(f"Train data shape: {dataset.train_data.shape}")
    print(f"Validation data shape: {dataset.val_data.shape}")
    print(f"Test data shape: {dataset.test_data.shape}")
