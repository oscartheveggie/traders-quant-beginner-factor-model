import os
import ast
import tokenize
from tqdm import tqdm
import factors
import functions as fc
import statsmodels.api as sm
import pandas as pd
import numpy as np
import cvxpy as cp
from typing import ClassVar
from datasets import Dataset
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

"""
Filename: regression.py

Main script to run multi-factor model regression. Use this file for 
regression inference, factor selection, regularization, and orthogonalization.

You can also use this file to obtain regression coefficients for both expected return
and risk models, and use those coefficients to predict expected returns and risk for
portfolio optimization.
"""

"""
===== GLOBAL VARIABLES =====
"""
factor_list = [
    'mean_reversion_0',
    'mean_reversion_3',
    'mean_reversion_4',
    'mean_reversion_5',
    'liquidity_1',
    'liquidity_2',
    'liquidity_3',
    'liquidity_test',
    'short_term_reversal_barra',
    'momentum_barra',
    'liquidity_barra',
    'quality_barra',
]
min_regression_factor_coverage = 0.5
regularization_type = 'none'  # Options: 'none', 'L1', 'L2'
optimize_regularization = False  # Set to True if you want to choose alpha on cross-validation set
data_filename = 'massive_final_all.parquet'  # Replace with actual data file in practice
rf_filename = 'rf_rate.csv'

dataset = Dataset(filename=data_filename)
script_dir = os.path.dirname(os.path.abspath(__file__))
rf_path = os.path.join(script_dir, '..', 'data', rf_filename)

# Path to store/load computed factors
computed_factors_filename = 'computed_factors.parquet'
computed_factors_path = os.path.join(script_dir, '..', 'data', computed_factors_filename)


# Load factor list if factor list is empty

def get_functions_from_file(filepath):
    with tokenize.open(filepath) as file:
        tree = ast.parse(file.read(), filename=filepath)
    
    # Extract names of top-level function definitions
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

if factor_list is None or len(factor_list) == 0:
    factor_list = get_functions_from_file(os.path.join(script_dir, 'factors.py'))

# Load risk-free rate series
def _load_risk_free_series(stock_data: pd.DataFrame) -> pd.Series:
    """Load and align the risk-free rate series from factor-model/data/rf_rate.csv."""
    rf_data = pd.read_csv(rf_path, header=[0, 1], dtype=float, index_col=0, parse_dates=True)

    if isinstance(rf_data.columns, pd.MultiIndex):
        rf_close = rf_data['close']
    else:
        rf_close = rf_data[['close']]

    rf_series = rf_close.iloc[:, 0].sort_index()
    rf_series = rf_series.reindex(stock_data.index).ffill().bfill()
    return rf_series


"""
===== CLASS DEFINITIONS =====
"""


class FactorEngine:

    def __init__(self, dataset: Dataset, factor_list: list[str]):
        self.dataset = dataset
        self.factor_list = factor_list

        print(f"Found precomputed factors at {computed_factors_path}, attempting to load...")
        if os.path.exists(computed_factors_path):
            try:
                self.all_factors, self.all_flattened_factors = self._load_computed_factors(computed_factors_path, self.factor_list)
                print("Loaded computed factors successfully.")
            except Exception as e:
                print(f"Failed to load computed factors: {e}; computing factors now...")
                self.all_factors, self.all_flattened_factors = self._get_factors(dataset.full_data, factor_list)
                try:
                    self._save_computed_factors(computed_factors_path)
                    print(f"Saved computed factors to {computed_factors_path}")
                except Exception as e2:
                    print(f"Failed to save computed factors: {e2}")
        else:
            print("No precomputed factors found — computing factors for full dataset...")
            self.all_factors, self.all_flattened_factors = self._get_factors(dataset.full_data, factor_list)
            try:
                self._save_computed_factors(computed_factors_path)
                print(f"Saved computed factors to {computed_factors_path}")
            except Exception as e:
                print(f"Failed to save computed factors: {e}")

        self._select_factors(self.factor_list)

        # Slice full factors into train/val/test splits using dataset indices
        self.train_factors = self.all_factors.reindex(dataset.train_data.index)
        print(f"Train factors shape: {getattr(self.train_factors, 'shape', None)}")
        try:
            self.train_flattened_factors = self.all_flattened_factors.loc[dataset.train_data.index]
        except Exception:
            mask = self.all_flattened_factors.index.get_level_values('timestamp').isin(dataset.train_data.index)
            self.train_flattened_factors = self.all_flattened_factors[mask]
        print(f"Train flattened factors entries: {len(self.train_flattened_factors)}")

        self.val_factors = self.all_factors.reindex(dataset.val_data.index)
        try:
            self.val_flattened_factors = self.all_flattened_factors.loc[dataset.val_data.index]
        except Exception:
            mask = self.all_flattened_factors.index.get_level_values('timestamp').isin(dataset.val_data.index)
            self.val_flattened_factors = self.all_flattened_factors[mask]

        self.test_factors = self.all_factors.reindex(dataset.test_data.index)
        try:
            self.test_flattened_factors = self.all_flattened_factors.loc[dataset.test_data.index]
        except Exception:
            mask = self.all_flattened_factors.index.get_level_values('timestamp').isin(dataset.test_data.index)
            self.test_flattened_factors = self.all_flattened_factors[mask]

        self._filter_sparse_factors(min_regression_factor_coverage)

        self.train_stocks = self.train_factors.columns.get_level_values('stock').unique()
        self.val_stocks = self.val_factors.columns.get_level_values('stock').unique()
        self.test_stocks = self.test_factors.columns.get_level_values('stock').unique()
        self.all_stocks = self.dataset.stocks
        self.intersect_stocks = self.train_stocks.intersection(self.val_stocks).intersection(self.test_stocks)


    def _get_factors(self, stock_data: pd.DataFrame, factor_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute and return raw factor values in stock-panel format for each factor."""
        flattened_rows = pd.MultiIndex.from_product([stock_data.index, stock_data['close'].columns], names=['timestamp', 'stock'])
        flattened_panel = pd.DataFrame(index=flattened_rows, columns=factor_names, dtype=float)
        flattened_panel.columns.name = 'factors'

        panel = pd.DataFrame()

        for factor_name in tqdm(factor_names):
            if not hasattr(factors, factor_name):
                raise AttributeError(f"Factor function '{factor_name}' not found in factors module.")
            factor_func = getattr(factors, factor_name)
            factor_values = factor_func(stock_data)
            panel = pd.concat([panel, factor_values], axis=1)
            print(f"Computed factor: {factor_name}")
            flattened_panel[factor_name] = pd.Series(factor_values.to_numpy().flatten(), name=factor_name, index=flattened_rows)
            print(f"Flattened factor: {factor_name}")

        return (panel, flattened_panel)

    def _select_factors(self, factor_names: list[str]) -> None:
        """Keep only requested factor columns, including when loading from cache."""
        available = list(self.all_flattened_factors.columns)
        selected = [factor for factor in factor_names if factor in available]
        missing = [factor for factor in factor_names if factor not in available]

        if missing:
            print(f"Skipping missing cached factors: {missing}")
        if not selected:
            raise ValueError("No requested factors are available.")

        self.factor_list = selected
        self.all_flattened_factors = self.all_flattened_factors.loc[:, selected]
        if not self.all_factors.empty:
            keep_columns = self.all_factors.columns.get_level_values(0).isin(selected)
            self.all_factors = self.all_factors.loc[:, keep_columns]

    def _filter_sparse_factors(self, min_coverage: float) -> None:
        """Drop factors with insufficient finite train coverage before regression."""
        coverage = (
            self.train_flattened_factors
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .mean()
        )
        selected = [factor for factor in self.factor_list if coverage.get(factor, 0.0) >= min_coverage]
        dropped = [factor for factor in self.factor_list if factor not in selected]

        if dropped:
            dropped_with_coverage = {factor: float(coverage.get(factor, 0.0)) for factor in dropped}
            print(f"Dropping sparse regression factors: {dropped_with_coverage}")
        if not selected:
            raise ValueError(f"No factors meet min_regression_factor_coverage={min_coverage}.")

        self.factor_list = selected
        self.all_flattened_factors = self.all_flattened_factors.loc[:, selected]
        self.train_flattened_factors = self.train_flattened_factors.loc[:, selected]
        self.val_flattened_factors = self.val_flattened_factors.loc[:, selected]
        self.test_flattened_factors = self.test_flattened_factors.loc[:, selected]

        for attr in ('all_factors', 'train_factors', 'val_factors', 'test_factors'):
            panel = getattr(self, attr)
            if not panel.empty:
                keep_columns = panel.columns.get_level_values(0).isin(selected)
                setattr(self, attr, panel.loc[:, keep_columns])

    def _save_computed_factors(self, path: str) -> None:
        """Save the full flattened factors to a parquet file.

        The DataFrame written has columns: ['timestamp','stock', <factor columns...>]
        """
        if getattr(self, 'all_flattened_factors', None) is None or self.all_flattened_factors.empty:
            print("No flattened factors to save.")
            return
        combined = self.all_flattened_factors.reset_index()
        combined.to_parquet(path, index=False)
        print(f"Wrote flattened factors parquet with shape {combined.shape} to {path}")

    def _load_computed_factors(self, path: str, factor_names: list[str] | None = None):
        """Load full flattened factors saved by `_save_computed_factors` and reconstruct DataFrame forms.

        Returns: (all_panel, all_flattened)
        """
        requested_columns = None
        if factor_names is not None:
            requested_columns = ['timestamp', 'stock', *factor_names]
        df = pd.read_parquet(path, columns=requested_columns)
        if 'timestamp' not in df.columns or 'stock' not in df.columns:
            raise ValueError('Invalid computed factors file format')

        factor_cols = [c for c in df.columns if c not in ('timestamp', 'stock')]
        if factor_names is not None:
            factor_cols = [factor for factor in factor_names if factor in factor_cols]
        if not factor_cols:
            raise ValueError('No requested factors found in computed factors file')
        flat = df.set_index(['timestamp', 'stock'])[factor_cols]

        # build panel: for each factor, unstack stock into columns and set attribute level to factor name
        panels = []
        for factor in factor_cols:
            pivot = flat[factor].unstack(level='stock')
            pivot.columns = pd.MultiIndex.from_product([[factor], pivot.columns], names=['attribute', 'stock'])
            panels.append(pivot)

        panel = pd.concat(panels, axis=1) if panels else pd.DataFrame()
        print(f"Loaded flattened factors parquet from {path} with {len(df)} rows and {len(factor_cols)} factor columns")
        return panel, flat


class RegressionModel:

    def __init__(self, dataset: Dataset, factor_engine: FactorEngine, options: dict = None):
        """
        Class to perform cross-sectional regression of returns on factors.
        
        This model runs regression across all stocks for each time period (cross-sectional).
        The regression coefficients become the "factor returns" for that time period.
        
        Options dict format:
        {
            "regularization": "none",  # Options: 'none', 'L1', 'L2', 'elastic_net'
            "alpha": 0.1,              # Regularization strength
            "l1_ratio": 0.5,           # For elastic_net: L1 ratio (0=Ridge, 1=Lasso)
            "cv_folds": 5,             # Number of cross-validation folds
            "tune_alpha": False,        # Whether to tune alpha via GridSearchCV
            ...
        }

        Args:
            dataset: Dataset object containing train/val/test splits
            factor_engine: FactorEngine object containing computed factor series for each split
            options: Dictionary of options for regression.

        Attributes:
            factor_returns: DataFrame of factor returns (betas) indexed by date, columns by factor
            const_terms: Series of constant terms indexed by date
            daily_results: DataFrame with daily regression statistics (R-squared, etc.)
        """
        
        self.dataset = dataset
        self.factor_engine = factor_engine
        self.options = {} if options is None else options
        
        # Parse options
        self.regularization = self.options.get('regularization', 'none')
        self.alpha = self.options.get('alpha', 0.1)
        self.l1_ratio = self.options.get('l1_ratio', 0.5)
        self.cv_folds = self.options.get('cv_folds', 5)
        self.tune_alpha = self.options.get('tune_alpha', False)
        self.holding_period_days = max(int(self.options.get('holding_period_days', 1)), 1)
        
        # Initialize storage for factor returns and constant terms
        self.factor_returns = pd.DataFrame(
            index=self.dataset.train_data.index,
            columns=factor_engine.factor_list,
            dtype=float
        )
        self.const_terms = pd.Series(index=self.dataset.train_data.index, dtype=float)
        
        self.daily_results = pd.DataFrame(
            index=self.dataset.train_data.index,
            columns=['rsquared', 'adj_rsquared', 'rmse', 'n_obs'],
            dtype=float
        )

    def _load_regression_results(self, coeffs_path: str, const_path: str, daily_path: str) -> bool:
        """Load regression results from parquet files if they all exist.
        Returns True if successfully loaded, False otherwise."""
        try:
            if not (os.path.exists(coeffs_path) and os.path.exists(const_path) and os.path.exists(daily_path)):
                return False
            self.factor_returns = pd.read_parquet(coeffs_path)
            const_df = pd.read_parquet(const_path)
            self.const_terms = const_df.iloc[:, 0]  # Extract Series from DataFrame
            self.daily_results = pd.read_parquet(daily_path)
            print(f"Loaded regression results from parquet files")
            return True
        except Exception as e:
            print(f"Failed to load regression results: {e}")
            return False

    def _get_regularized_model(self, alpha=None):
        """Create a regularized regression model based on options."""
        if alpha is None:
            alpha = self.alpha
            
        if self.regularization == 'none':
            return None  # Use OLS
        elif self.regularization == 'L2':
            return Ridge(alpha=alpha, fit_intercept=True)
        elif self.regularization == 'L1':
            return Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        elif self.regularization == 'elastic_net':
            return ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio, fit_intercept=True, max_iter=10000)
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization}")

    def _forward_returns(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Return holding-period forward close-to-close returns."""
        close = stock_data[[col for col in stock_data.columns if col[0] == 'close']]
        future_close = close.shift(-self.holding_period_days)
        return future_close.div(close).sub(1.0).replace([np.inf, -np.inf], np.nan)

    def fit_returns(self) -> None:
        """
        Perform cross-sectional regression of returns on factors for each date.
        
        For each date, regress all stock returns on their factor exposures.
        The regression coefficients are the factor returns for that date.
        """
        
        train_dates = self.dataset.train_data.index
        processed = 0
        fitted = 0
        stock_returns = self._forward_returns(self.dataset.train_data)

        for date in tqdm(train_dates, desc='Fitting regressions'):
            processed += 1
            try:
                # Get factor exposures for all stocks at this date
                X = self.factor_engine.train_flattened_factors.loc[date].values.reshape(-1, len(self.factor_engine.factor_list))
                
                y = stock_returns.loc[date].values
                
                # Remove NaN values
                valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X_valid = X[valid_idx]
                y_valid = y[valid_idx]
                
                if len(y_valid) < len(self.factor_engine.factor_list) + 1:
                    continue  # Skip if not enough observations
                
                # Fit regression model
                if self.regularization == 'none':
                    # Use OLS
                    X_with_const = sm.add_constant(X_valid)
                    model = sm.OLS(y_valid, X_with_const)
                    results = model.fit()
                    
                    self.const_terms.loc[date] = results.params[0]
                    for i, factor in enumerate(self.factor_engine.factor_list):
                        self.factor_returns.loc[date, factor] = results.params[i + 1]
                    
                    self.daily_results.loc[date, 'rsquared'] = results.rsquared
                    self.daily_results.loc[date, 'adj_rsquared'] = results.rsquared_adj
                    self.daily_results.loc[date, 'rmse'] = np.sqrt(np.sum(results.resid ** 2) / len(y_valid))
                    self.daily_results.loc[date, 'n_obs'] = len(y_valid)
                else:
                    # Use scikit-learn regularized model
                    model = self._get_regularized_model()
                    model.fit(X_valid, y_valid)
                    
                    self.const_terms.loc[date] = model.intercept_
                    for i, factor in enumerate(self.factor_engine.factor_list):
                        self.factor_returns.loc[date, factor] = model.coef_[i]
                    
                    y_pred = model.predict(X_valid)
                    ss_res = np.sum((y_valid - y_pred) ** 2)
                    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                    self.daily_results.loc[date, 'rsquared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    self.daily_results.loc[date, 'rmse'] = np.sqrt(ss_res / len(y_valid))
                    self.daily_results.loc[date, 'n_obs'] = len(y_valid)
                    
            except Exception:
                # Skip dates with insufficient data or errors
                continue

        print(f"Regression loop complete: processed={processed}, fitted={self.factor_returns.dropna(how='all').shape[0]}")

        # After fitting across dates, save regression coefficients to disk for later reuse
        # Use holding_period_days to create period-specific filenames
        try:
            coeffs_filename = f'regression_coeffs_{self.holding_period_days}d.parquet'
            coeffs_path = os.path.join(script_dir, '..', 'data', coeffs_filename)
            # ensure index is preserved (dates) and write parquet
            self.factor_returns.to_parquet(coeffs_path)
            print(f"Saved regression coefficients to {coeffs_filename}")
        except Exception as e:
            print(f"Failed to save regression coefficients: {e}")
        try:
            base_path = os.path.join(script_dir, '..', 'data')
            const_filename = f'regression_const_terms_{self.holding_period_days}d.parquet'
            daily_filename = f'regression_daily_results_{self.holding_period_days}d.parquet'
            const_path = os.path.join(base_path, const_filename)
            daily_path = os.path.join(base_path, daily_filename)
            # Save const_terms (Series -> DataFrame) and daily_results
            if getattr(self, 'const_terms', None) is not None:
                self.const_terms.to_frame('const_term').to_parquet(const_path)
                print(f"Saved constant terms to {const_filename}")
            if getattr(self, 'daily_results', None) is not None:
                self.daily_results.to_parquet(daily_path)
                print(f"Saved daily results to {daily_filename}")
        except Exception as e:
            print(f"Failed to save regression results: {e}")
        
        
    
    def tune_regularization(self) -> None:
        """
        Use cross-validation to find optimal regularization parameter (alpha).
        Only works with regularization != 'none'.
        """
        if self.regularization == 'none':
            print("No regularization specified. Skipping tuning.")
            return
        
        # For simplicity, collect all data across all dates
        X_all = self.factor_engine.train_flattened_factors.values
        
        # Get all returns (flattened across all dates and stocks)
        stock_returns_flattened = fc.ts_delta(self.dataset.train_data, window=-1)
        close_cols = [col for col in stock_returns_flattened.columns if col[0] == 'close']
        y_all = stock_returns_flattened[close_cols].values.flatten()
        
        # Match indices
        valid_idx = ~(np.isnan(X_all).any(axis=1) | np.isnan(y_all))
        X_valid = X_all[valid_idx]
        y_valid = y_all[valid_idx]
        
        # Grid search for optimal alpha
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        base_model = self._get_regularized_model(alpha=alphas[0])
        
        grid_search = GridSearchCV(
            base_model,
            {'alpha': alphas},
            cv=self.cv_folds,
            scoring='neg_mean_squared_error'
        )
        
        grid_search.fit(X_valid, y_valid)
        self.alpha = grid_search.best_params_['alpha']
        print(f"Optimal alpha found: {self.alpha}")
        print(f"CV score: {grid_search.best_score_}")


    def predict_returns(self, option: str) -> pd.DataFrame:
        """
        Predict expected returns using cross-sectional regression results.
        
        Expected return = (average factor returns) * (factor exposures) + (average constant term)

        Args:
            option: One of 'train', 'val', 'test' or 'all' to specify which dataset split to predict on

        Returns:
            DataFrame of predicted expected returns indexed by date and stocks
        """
        if option == 'train':
            data = self.dataset.train_data
            factors = self.factor_engine.train_flattened_factors
        elif option == 'val':
            data = self.dataset.val_data
            factors = self.factor_engine.val_flattened_factors
        elif option == 'test':
            data = self.dataset.test_data
            factors = self.factor_engine.test_flattened_factors
        elif option == 'all':
            data = self.dataset.full_data
            factors = self.factor_engine.all_flattened_factors
        else:
            raise ValueError("Option must be 'train', 'val', 'test' or 'all'")

        # Calculate average factor returns and constant term from training period.
        # Missing factor exposures are imputed below, so non-finite coefficients
        # should not poison the full prediction vector.
        avg_factor_returns = (
            self.factor_returns
            .replace([np.inf, -np.inf], np.nan)
            .mean()
            .reindex(self.factor_engine.factor_list)
            .fillna(0.0)
        )
        avg_const_term = self.const_terms.replace([np.inf, -np.inf], np.nan).mean()
        if not np.isfinite(avg_const_term):
            avg_const_term = 0.0

        factor_history_median = (
            factors
            .replace([np.inf, -np.inf], np.nan)
            .median()
            .reindex(self.factor_engine.factor_list)
            .fillna(0.0)
        )
        
        # Create output DataFrame with MultiIndex (date, stock)
        dates = data.index
        stocks = data['close'].columns
        multi_idx = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        predicted_returns = pd.DataFrame(index=multi_idx, columns=['predicted_return'], dtype=float)
        
        # For each date, predict returns for each stock
        for date in tqdm(dates):
            try:
                X_date = (
                    factors.loc[date]
                    .reindex(index=stocks, columns=self.factor_engine.factor_list)
                    .replace([np.inf, -np.inf], np.nan)
                )
                X_date = X_date.fillna(X_date.median()).fillna(factor_history_median).fillna(0.0)
                
                # predicted_return = (factor_exposures) * (avg_factor_returns) + avg_const_term
                pred = X_date.to_numpy() @ avg_factor_returns.values + avg_const_term
                
                for i, stock in enumerate(stocks):
                    predicted_returns.loc[(date, stock), 'predicted_return'] = pred[i]
            except Exception as e:
                # Skip dates with errors
                continue
        
        return predicted_returns


class RiskModel:

    FRENCH_49_INDUSTRIES: ClassVar[list[str]] = [
        'Agric', 'Food ', 'Soda ', 'Beer ', 'Smoke', 'Toys ', 'Fun  ', 'Books', 'Hshld', 'Clths',
        'Hlth ', 'MedEq', 'Drugs', 'Chems', 'Rubbr', 'Txtls', 'BldMt', 'Cnstr', 'Steel', 'FabPr',
        'Mach ', 'ElcEq', 'Autos', 'Aero ', 'Ships', 'Guns ', 'Gold ', 'Mines', 'Coal ', 'Oil  ',
        'Util ', 'Telcm', 'PerSv', 'BusSv', 'Hardw', 'Softw', 'Chips', 'LabEq', 'Paper', 'Boxes',
        'Trans', 'Whlsl', 'Rtail', 'Meals', 'Banks', 'Insur', 'RlEst', 'Fin  ', 'Other'
    ]

    RANGES: ClassVar[list[tuple[str, list[tuple[int, int]]]]] = [
        ('Agric', [(100, 999), (2048, 2048)]),
        ('Food ', [(2000, 2046), (2050, 2063), (2070, 2079), (2090, 2092), (2095, 2099)]),
        ('Soda ', [(2064, 2068), (2086, 2086), (2087, 2087), (2093, 2094)]),
        ('Beer ', [(2080, 2085)]),
        ('Smoke', [(2100, 2199)]),
        ('Toys ', [(3940, 3949)]),
        ('Fun  ', [(7800, 7833), (7930, 7933), (7940, 7949), (7990, 7999)]),
        ('Books', [(2700, 2732), (2740, 2749), (2770, 2771), (2780, 2799)]),
        ('Hshld', [(2047, 2047), (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2844), (3161, 3161), (3221, 3221), (3262, 3263), (3269, 3269), (3630, 3639), (3750, 3751), (3800, 3800), (3860, 3861), (3870, 3873), (3960, 3962)]),
        ('Clths', [(2300, 2390), (3020, 3021), (3100, 3111), (3130, 3131), (3140, 3151), (3963, 3965)]),
        ('Hlth ', [(8000, 8099)]),
        ('MedEq', [(3693, 3693), (3840, 3851)]),
        ('Drugs', [(2830, 2831), (2833, 2836)]),
        ('Chems', [(2800, 2829), (2845, 2845), (2850, 2859), (2870, 2899)]),
        ('Rubbr', [(3000, 3000), (3050, 3053), (3060, 3069), (3080, 3089)]),
        ('Txtls', [(2200, 2269), (2270, 2279), (2280, 2284), (2290, 2295), (2297, 2299), (2393, 2395), (2397, 2399)]),
        ('BldMt', [(800, 899), (2400, 2439), (2450, 2459), (2490, 2499), (2660, 2661), (2950, 2952), (3200, 3219), (3240, 3241), (3250, 3259), (3261, 3261), (3264, 3264), (3270, 3275), (3280, 3281), (3290, 3293), (3295, 3299), (3420, 3429), (3430, 3433), (3440, 3441), (3442, 3442), (3446, 3446), (3448, 3448), (3449, 3449), (3450, 3451), (3452, 3452), (3490, 3499), (3996, 3996)]),
        ('Cnstr', [(1500, 1511), (1520, 1529), (1530, 1539), (1540, 1549), (1600, 1699), (1700, 1799)]),
        ('Steel', [(3300, 3300), (3310, 3317), (3320, 3325), (3330, 3339), (3340, 3341), (3350, 3357), (3360, 3369), (3370, 3379), (3390, 3399)]),
        ('FabPr', [(3400, 3400), (3443, 3444), (3460, 3469), (3470, 3479)]),
        ('Mach ', [(3510, 3519), (3520, 3529), (3530, 3536), (3540, 3549), (3550, 3559), (3560, 3569), (3580, 3582), (3585, 3585), (3586, 3586), (3589, 3589), (3590, 3599)]),
        ('ElcEq', [(3600, 3600), (3610, 3613), (3620, 3629), (3640, 3644), (3645, 3645), (3646, 3646), (3648, 3649), (3660, 3660), (3690, 3690), (3691, 3692), (3699, 3699)]),
        ('Autos', [(2296, 2296), (2396, 2396), (3010, 3011), (3537, 3537), (3710, 3711), (3713, 3716), (3792, 3792), (3799, 3799)]),
        ('Aero ', [(3720, 3721), (3723, 3724), (3725, 3725), (3728, 3729)]),
        ('Ships', [(3730, 3732), (3740, 3743)]),
        ('Guns ', [(3480, 3489), (3760, 3769), (3795, 3795)]),
        ('Gold ', [(1040, 1049)]),
        ('Mines', [(1000, 1000), (1010, 1019), (1020, 1029), (1030, 1039), (1050, 1059), (1060, 1069), (1070, 1079), (1080, 1089), (1090, 1099), (1100, 1119), (1400, 1499)]),
        ('Coal ', [(1200, 1299)]),
        ('Oil  ', [(1300, 1300), (1310, 1319), (1320, 1329), (1330, 1339), (1380, 1389), (2900, 2912), (2990, 2999)]),
        ('Util ', [(4900, 4900), (4910, 4911), (4920, 4925), (4930, 4939), (4940, 4942)]),
        ('Telcm', [(4800, 4800), (4810, 4813), (4820, 4822), (4830, 4839), (4840, 4841), (4880, 4889), (4890, 4899)]),
        ('PerSv', [(7020, 7021), (7030, 7033), (7200, 7200), (7210, 7212), (7214, 7214), (7215, 7216), (7217, 7217), (7219, 7219), (7220, 7221), (7230, 7231), (7240, 7241), (7250, 7251), (7260, 7269), (7270, 7290), (7291, 7299), (7395, 7395), (7500, 7500), (7520, 7529), (7530, 7539), (7540, 7549), (7600, 7600), (7620, 7620), (7622, 7622), (7623, 7623), (7629, 7629), (7630, 7631), (7640, 7641), (7690, 7690), (7692, 7692), (7699, 7699), (8100, 8111), (8200, 8299), (8300, 8399), (8400, 8489), (8600, 8699), (8800, 8899)]),
        ('BusSv', [(2750, 2759), (3993, 3993), (7218, 7218), (7300, 7300), (7310, 7319), (7320, 7329), (7330, 7339), (7340, 7342), (7349, 7349), (7350, 7351), (7352, 7352), (7353, 7353), (7359, 7359), (7360, 7369), (7370, 7372), (7374, 7374), (7375, 7375), (7376, 7376), (7377, 7377), (7378, 7378), (7379, 7379), (7380, 7380), (7381, 7382), (7383, 7383), (7384, 7384), (7385, 7385), (7389, 7390), (7391, 7391), (7392, 7392), (7393, 7393), (7394, 7394), (7396, 7396), (7397, 7397), (7399, 7399), (7510, 7515), (8700, 8700), (8710, 8713), (8720, 8721), (8730, 8734), (8740, 8748), (8900, 8910), (8911, 8911), (8920, 8999)]),
        ('Hardw', [(3570, 3579), (3680, 3689), (3695, 3695)]),
        ('Softw', [(7373, 7373)]),
        ('Chips', [(3670, 3679)]),
        ('LabEq', [(3810, 3812), (3820, 3820), (3821, 3821), (3822, 3822), (3823, 3823), (3824, 3824), (3825, 3825), (3826, 3826), (3827, 3827), (3829, 3829), (3830, 3839)]),
        ('Paper', [(2520, 2549), (2600, 2639), (2670, 2699), (2760, 2761), (3950, 3955)]),
        ('Boxes', [(2640, 2659)]),
        ('Trans', [(4000, 4013), (4040, 4049), (4100, 4100), (4110, 4119), (4120, 4122), (4130, 4131), (4140, 4142), (4150, 4151), (4170, 4173), (4190, 4190), (4200, 4200), (4210, 4219), (4220, 4229), (4230, 4231), (4240, 4249), (4400, 4499), (4500, 4599), (4600, 4699), (4700, 4700), (4710, 4712), (4720, 4721), (4722, 4722), (4723, 4723), (4724, 4724), (4725, 4725), (4729, 4729), (4730, 4739), (4740, 4749), (4780, 4780), (4782, 4782), (4783, 4783), (4784, 4784), (4785, 4785), (4789, 4789)]),
        ('Whlsl', [(5000, 5000), (5010, 5015), (5020, 5023), (5030, 5039), (5040, 5042), (5043, 5043), (5044, 5044), (5045, 5045), (5046, 5046), (5047, 5047), (5048, 5048), (5049, 5049), (5050, 5059), (5060, 5060), (5063, 5063), (5064, 5064), (5065, 5065), (5070, 5078), (5080, 5080), (5081, 5081), (5082, 5082), (5083, 5083), (5084, 5084), (5085, 5085), (5086, 5087), (5088, 5088), (5089, 5089), (5090, 5090), (5093, 5093), (5094, 5094), (5099, 5099), (5100, 5100), (5110, 5113), (5120, 5122), (5130, 5139), (5140, 5149), (5150, 5159), (5160, 5169), (5170, 5172), (5180, 5182), (5190, 5199)]),
        ('Rtail', [(5200, 5200), (5210, 5219), (5220, 5229), (5230, 5231), (5250, 5251), (5260, 5261), (5270, 5271), (5300, 5300), (5310, 5311), (5320, 5320), (5330, 5331), (5340, 5349), (5390, 5399), (5400, 5400), (5410, 5411), (5412, 5412), (5420, 5429), (5430, 5439), (5440, 5449), (5450, 5459), (5460, 5469), (5490, 5499), (5500, 5500), (5510, 5529), (5530, 5539), (5540, 5549), (5550, 5559), (5560, 5569), (5570, 5579), (5590, 5599), (5600, 5600), (5610, 5619), (5620, 5621), (5630, 5632), (5640, 5641), (5650, 5651), (5660, 5661), (5680, 5681), (5690, 5699), (5700, 5700), (5710, 5719), (5720, 5722), (5730, 5733), (5734, 5734), (5735, 5735), (5736, 5736), (5750, 5799), (5900, 5900), (5910, 5912), (5920, 5929), (5930, 5932), (5940, 5940), (5941, 5941), (5942, 5942), (5943, 5943), (5944, 5944), (5945, 5945), (5946, 5946), (5947, 5947), (5948, 5948), (5949, 5949), (5950, 5959), (5960, 5969), (5970, 5979), (5980, 5989), (5990, 5990), (5992, 5992), (5993, 5993), (5994, 5994), (5995, 5995), (5999, 5999)]),
        ('Meals', [(5800, 5819), (5820, 5829), (5890, 5899), (7000, 7000), (7010, 7019), (7040, 7049)]),
        ('Banks', [(6000, 6000), (6010, 6019), (6020, 6020), (6021, 6021), (6022, 6022), (6023, 6024), (6025, 6025), (6026, 6026), (6027, 6027), (6028, 6029), (6030, 6036), (6040, 6059), (6060, 6062), (6080, 6082), (6090, 6099), (6100, 6100), (6110, 6111), (6112, 6113), (6120, 6129), (6130, 6139), (6140, 6149), (6150, 6159), (6160, 6169), (6170, 6179), (6190, 6199)]),
        ('Insur', [(6300, 6300), (6310, 6319), (6320, 6329), (6330, 6331), (6350, 6351), (6360, 6361), (6370, 6379), (6390, 6399), (6400, 6411)]),
        ('RlEst', [(6500, 6500), (6510, 6510), (6512, 6512), (6513, 6513), (6514, 6514), (6515, 6515), (6517, 6519), (6520, 6529), (6530, 6531), (6532, 6532), (6540, 6541), (6550, 6553), (6590, 6599), (6610, 6611)]),
        ('Fin  ', [(6200, 6299), (6700, 6700), (6710, 6719), (6720, 6722), (6723, 6723), (6724, 6724), (6725, 6725), (6726, 6726), (6730, 6733), (6790, 6791), (6792, 6792), (6793, 6793), (6794, 6794), (6795, 6795), (6798, 6798), (6799, 6799)])
    ]

    def __init__(self, dataset: Dataset, factor_engine: FactorEngine = None):
        self.dataset = dataset
        self.factor_engine = factor_engine or FactorEngine(dataset, factor_list)
    
    def _ewma_weights(self, half_life: int, length: int) -> np.ndarray:
        """
        Generates normalized exponential decay weights.
        w_t = exp(-lambda * t) where lambda = ln(2) / half_life
        """
        decay_rate = np.log(2) / half_life
        # t goes from 0 (most recent) to length-1 (oldest)
        # However, for an array representing chronology [oldest ... newest],
        # the indices for t should be reversed.
        t = np.arange(length)[::-1]
        weights = np.exp(-decay_rate * t)
        return weights / np.sum(weights)

    def _map_sic_to_french_49(self, sic: float) -> str:
        """
        Maps a 4-digit SIC code to one of the 49 Kenneth French Industry classfications.
        Uses a subset/simplification of the standard Siccodes49.txt mapping logic.
        """
        if pd.isna(sic):
            return 'Other'
        try:
            s = int(sic)
        except ValueError:
            return 'Other'
        
        for name, intervals in self.RANGES:
            for start, end in intervals:
                if start <= s <= end:
                    return name
        return 'Other'

    def estimate_covariance_matrix(
        self,
        stock_data: pd.DataFrame,
        date: pd.Timestamp = None,
        window: int = 63,
        specific_window: int = 21,
        holding_period_days: int = 1,
    ) -> pd.DataFrame:
        """
        Estimate a Barra-style Fundamental Factor Covariance Matrix (V = X F X^T + Delta).
        If date is provided, only observations up to that date are used. The factor
        covariance uses up to window return rows; specific variance uses up to
        specific_window recent residual rows.
        """
        if date is not None:
            historical_data = stock_data.loc[:date]
            if len(historical_data) > window + 1:
                historical_data = historical_data.iloc[-(window + 1):]
        else:
            historical_data = stock_data
            
        # Phase 1: Data Ingestion & Return prep
        close = historical_data['close']
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='all')
        
        stocks = close.columns
        if returns.empty:
            return pd.DataFrame(np.eye(len(stocks), dtype=float), index=stocks, columns=stocks)
        
        N = len(stocks)
        T = returns.shape[0]
        
        # 1. Regression Factors (i.e., Style Factors) computed by FactorEngine
        style_cols = self.factor_engine.factor_list
        style_exposures = {
            f: self.factor_engine.all_flattened_factors[f]
            .unstack('stock')
            .reindex(index=returns.index, columns=stocks)
            .fillna(0.0)
            for f in style_cols
        }
        
        # 2. 49 Industry Factors (One-Hot Encoded) carried forward over time
        if 'sic_code' in historical_data.columns.levels[0]:
            sic_codes = historical_data['sic_code'].reindex(index=returns.index, columns=stocks).ffill()
        else:
            sic_codes = pd.DataFrame(np.nan, index=returns.index, columns=stocks)
            
        categories = pd.CategoricalDtype(categories=self.FRENCH_49_INDUSTRIES)
        
        F_returns_list = []
        Residuals_list = []
        
        # We need the final X to reconstruct the forward-looking V matrix
        X_final = None 
        
        # Compute daily cross-sectional regressions over the window
        for t_idx, current_date in tqdm(enumerate(returns.index)):
            ret_t = returns.iloc[t_idx].fillna(0.0).values  # length N
            
            # Construct X_t for this day
            X_style_t = np.column_stack([style_exposures[f].iloc[t_idx].values for f in style_cols]) # N x len(style_cols)
            
            ind_t = sic_codes.iloc[t_idx].map(self._map_sic_to_french_49).astype(categories)
            X_ind_t = pd.get_dummies(ind_t).values # N x 49
            
            X_t = np.hstack((X_style_t, X_ind_t)) # N x (49 + len(style_cols))
            
            # Cross-sectional regression: R_t = X_t * f_t + e_t
            X_pinv_t = np.linalg.pinv(X_t) # (49 + len(style_cols)) x N
            f_t = X_pinv_t @ ret_t # (49 + len(style_cols))
            e_t = ret_t - X_t @ f_t # N
            
            F_returns_list.append(f_t)
            Residuals_list.append(e_t)
            
            if t_idx == T - 1:
                X_final = X_t
                
        # Phase 2: Covariance Mathematics (EWMA)
        F_returns = np.vstack(F_returns_list) # T x (49 + len(style_cols))
        Residuals = np.vstack(Residuals_list) # T x N
        K = 49 + len(style_cols) # 49 industries + len(style_cols) style factors
        
        # Factor Covariance (F) using 63-day half-life EWMA
        if T > 1:
            w_F = self._ewma_weights(63, T)
            # Center returns
            F_mean = np.sum(F_returns * w_F[:, None], axis=0) # K
            F_centered = F_returns - F_mean
            bias_F = 1.0 - np.sum(w_F**2)
            F_cov = (F_centered.T @ np.diag(w_F) @ F_centered) / bias_F
        else:
            F_cov = np.eye(K)
            
        # Specific Risk (Delta) using 21-day half-life EWMA on recent residuals
        residual_window = min(specific_window, T)
        Residuals_recent = Residuals[-residual_window:]
        if residual_window > 1:
            w_S = self._ewma_weights(21, residual_window)
            Res_mean = np.sum(Residuals_recent * w_S[:, None], axis=0) # N
            Res_centered = Residuals_recent - Res_mean
            bias_S = 1.0 - np.sum(w_S**2)
            S_var = np.sum((Res_centered**2) * w_S[:, None], axis=0) / bias_S
        else:
            S_var = np.ones(N) * 1e-6
            
        Delta = np.diag(S_var) # N x N
        
        # Phase 3: Assembly & Risk Prevention (Shrinkage)
        V_raw = X_final @ F_cov @ X_final.T + Delta
        
        # Conditioning Check & Ledoit-Wolf Shrinkage
        eigenvalues = np.linalg.eigvalsh(V_raw)
        eig_max = np.max(eigenvalues)
        eig_min = np.min(eigenvalues)
        
        condition_number = eig_max / eig_min if eig_min > 1e-12 else np.inf
        
        if condition_number > 10000:
            alpha = 0.15
            target_diag = np.diag(np.diag(V_raw))
            V_raw = (1 - alpha) * V_raw + alpha * target_diag
            
        holding_period_days = max(int(holding_period_days), 1)
        V_raw = V_raw * holding_period_days

        return pd.DataFrame(V_raw, index=stocks, columns=stocks)


class PortfolioOptimizer:

    def __init__(self, dataset: Dataset,
                 regression_model: RegressionModel,
                 risk_model: RiskModel,
                 options: dict = None):
        """
        Class to perform portfolio optimization using regression results.

        Options dict format:
        {
            <option_name>: <option_value>,
            "max_weight": 0.08,
            "risk_aversion": 0.5,
            "allow_short": False,   # change to True to allow short selling (negative weights)
            ...
        }

        Args:
            dataset: Dataset object containing train/val/test splits
            regression_model: RegressionModel object containing fitted regression results
            risk_model: RiskModel object containing the risk model
            options: Dictionary of options for portfolio optimization.

        Attributes:
            weights: DataFrame of optimized portfolio weights with stock columns and DateTime index
        """
        
        self.dataset = dataset
        self.regression_model = regression_model
        self.risk_model = risk_model
        self.weights = None


        # PARSING OPTIONS

        if options is None:
            options = {}

        if "max_weight" in options:
            self.max_weight = options["max_weight"] if options["max_weight"] > 0 else None
        else:
            self.max_weight = None
        
        if "risk_aversion" in options:
            self.risk_aversion = options["risk_aversion"] if options["risk_aversion"] >= 0 else 2.5
        else:
            self.risk_aversion = 2.5    # see Andrew Ang's "Asset Management: A Systematic Approach to Factor Investing"
        
        if "allow_short" in options:
            self.allow_short = options["allow_short"] if isinstance(options["allow_short"], bool) else False
        else:
            self.allow_short = False

        if "min_factor_coverage" in options:
            self.min_factor_coverage = min(max(float(options["min_factor_coverage"]), 0.0), 1.0)
        else:
            self.min_factor_coverage = 0.5

        self.holding_period_days = max(
            int(options.get('holding_period_days', getattr(self.regression_model, 'holding_period_days', 1))),
            1,
        )
        self.rebalance_interval_days = max(
            int(options.get('rebalance_interval_days', self.holding_period_days)),
            1,
        )
        self.solver = options.get('solver', 'OSQP')

        # Add more options parsing as needed




    def optimize(self, option: str) -> None:
        
        # Choose 'test', 'val', 'train' or 'all' dataset to optimize on
        if option == 'train':
            stock_data = self.dataset.train_data
        elif option == 'val':
            stock_data = self.dataset.val_data
        elif option == 'test':
            stock_data = self.dataset.test_data
        elif option == 'all':
            stock_data = self.dataset.full_data
        else:
            raise ValueError("Option must be 'train', 'val', 'test' or 'all'")
        
        if self.regression_model.factor_returns.empty:
            self.regression_model.fit_returns()
        expected_returns = self.regression_model.predict_returns(option)
        covariance = self.risk_model.estimate_covariance_matrix(
            stock_data,
            holding_period_days=self.holding_period_days,
        )
        rf_rate = _load_risk_free_series(stock_data)

        stocks = list(stock_data['close'].columns)
        weights = pd.DataFrame(index=stock_data.index, columns=stocks, dtype=float)

        sigma = (
            covariance
            .reindex(index=stocks, columns=stocks)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = 0.5 * (sigma + sigma.T)
        sigma += np.eye(len(stocks)) * 1e-8

        n_assets = len(stocks)
        w = cp.Variable(n_assets)
        excess_param = cp.Parameter(n_assets)

        objective = cp.Maximize(
            excess_param @ w - self.risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma)) / 2.0
        )

        constraints = [cp.sum(w) == 1]
        if not self.allow_short:
            constraints.append(w >= 0)
        if self.max_weight is not None:
            constraints.append(w <= self.max_weight)

        problem = cp.Problem(objective, constraints)

        first_usable_date = stock_data.index.min()
        factor_source = {
            'train': self.regression_model.factor_engine.train_flattened_factors,
            'val': self.regression_model.factor_engine.val_flattened_factors,
            'test': self.regression_model.factor_engine.test_flattened_factors,
            'all': self.regression_model.factor_engine.all_flattened_factors,
        }[option]
        if len(self.regression_model.factor_engine.factor_list) > 0 and not factor_source.empty:
            finite_factor = (
                factor_source
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .any(axis=1)
            )
            coverage_by_date = finite_factor.groupby(level='timestamp').mean()
            usable_dates = coverage_by_date[coverage_by_date >= self.min_factor_coverage].index
            if len(usable_dates) > 0:
                first_usable_date = usable_dates.min()

        rebalance_dates = stock_data.index[::self.rebalance_interval_days]

        for date in tqdm(rebalance_dates):
            if date < first_usable_date:
                continue

            # Extract predicted returns for this date
            mu = np.array([expected_returns.loc[(date, stock), 'predicted_return'] 
                          if (date, stock) in expected_returns.index else np.nan 
                          for stock in stocks], dtype=float)
            if not np.isfinite(mu).any():
                continue
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            
            rf = float(rf_rate.loc[date]) if date in rf_rate.index else float(rf_rate.iloc[-1])
            if not np.isfinite(rf):
                rf = 0.0
            excess = mu - rf
            excess = np.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)

            if np.allclose(excess, 0.0):
                row = np.repeat(1.0 / n_assets, n_assets)
                weights.loc[date] = row
                continue

            excess_param.value = excess
            try:
                problem.solve(solver=self.solver, verbose=False, warm_start=True)
            except (cp.error.SolverError, ValueError):
                try:
                    problem.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)
                except cp.error.SolverError:
                    problem.solve(solver=cp.SCS, verbose=False, warm_start=True)

            if w.value is None:
                row = np.repeat(1.0 / n_assets, n_assets)
            else:
                row = np.asarray(w.value).ravel()
                row = np.maximum(row, 0.0)
                row_sum = row.sum()
                row = row / row_sum if row_sum > 0 else np.repeat(1.0 / n_assets, n_assets)

            weights.loc[date] = row

        self.weights = weights.ffill(limit=self.rebalance_interval_days - 1)
        self.weights.to_parquet(os.path.join(script_dir, '..', 'data', f"optimized_weights_{self.holding_period_days}d.parquet"))
        return self.weights
        



"""
===== MAIN REGRESSION & PORTFOLIO OPTIMIZATION =====
"""

if __name__ == "__main__":

    if not factor_list:
        factor_list = get_functions_from_file(os.path.join(script_dir, "factors.py"))
    
    factor_engine = FactorEngine(dataset, factor_list)

    # Perform cross-sectional regression
    # Run regression for multiple holding periods
    holding_periods = [1, 5, 10, 30]
    
    for holding_period in holding_periods:
        print(f"\n{'='*60}")
        print(f"Running regression for {holding_period}-day holding period")
        print(f"{'='*60}")
        
        regression_options = {
            "regularization": regularization_type,  # 'none', 'L1', 'L2', 'elastic_net'
            "alpha": 0.1,
            "cv_folds": 5,
            "tune_alpha": optimize_regularization,
            "holding_period_days": holding_period,
        }
        regression_model = RegressionModel(dataset, factor_engine, options=regression_options)
        
        if optimize_regularization and regularization_type != 'none':
            print("Tuning regularization parameter...")
            regression_model.tune_regularization()
        
        # Check if regression results parquets exist; if so load them, otherwise compute
        coeffs_filename = f'regression_coeffs_{holding_period}d.parquet'
        const_filename = f'regression_const_terms_{holding_period}d.parquet'
        daily_filename = f'regression_daily_results_{holding_period}d.parquet'
        
        coeffs_path = os.path.join(script_dir, '..', 'data', coeffs_filename)
        const_path = os.path.join(script_dir, '..', 'data', const_filename)
        daily_path = os.path.join(script_dir, '..', 'data', daily_filename)
        
        if not regression_model._load_regression_results(coeffs_path, const_path, daily_path):
            print(f"Fitting cross-sectional regression model for {holding_period}-day holding period...")
            regression_model.fit_returns()
        else:
            print(f"Skipping regression fit for {holding_period}-day holding period; loaded from cache.")
        
        print(f"\nFactor returns by date (first 5 rows) - {holding_period}d:")
        print(regression_model.factor_returns.head(), end='\n\n')
        
        print(f"Constant terms by date (first 5 values) - {holding_period}d:")
        print(regression_model.const_terms.head(), end='\n\n')
        
        print(f"Daily regression statistics (first 5 rows) - {holding_period}d:")
        print(regression_model.daily_results.head(), end='\n\n')

    
    print("Building risk model...")
    # Perform portfolio optimization
    risk_model = RiskModel(dataset, factor_engine)
    optimizer_options = {
        "max_weight": 0.08,
        "risk_aversion": 0.5,
        "allow_short": False,
        "holding_period_days": regression_model.holding_period_days,
        "rebalance_interval_days": regression_model.holding_period_days,
    }

    print("Optimizing portfolio weights...")
    portfolio_optimizer = PortfolioOptimizer(dataset, regression_model, risk_model, options=optimizer_options)
    optimized_weights = portfolio_optimizer.optimize(option='test')
    
    print("Optimized portfolio weights (first 5 rows):")
    print(optimized_weights.head(), end='\n\n')

    optimized_weights.to_parquet(os.path.join(script_dir, '..', 'data', f"optimized_weights_{regression_model.holding_period_days}d.parquet"))
