"""
File: walk_forward_cv.py
Description: Walk-forward cross-validation for time-series models with regularization tuning.

Walk-forward CV (also called rolling-window CV) is the proper way to do time-series cross-validation:
1. Train on historical data
2. Validate on the next period
3. Move the window forward (expand or roll)
4. Repeat until end of data

This avoids look-ahead bias that standard K-fold CV would introduce in time-series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from regression import RegressionModel, FactorEngine
from datasets import Dataset
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Implements walk-forward cross-validation for time-series factor models.
    
    Supports:
    - Expanding window (train size grows over time)
    - Rolling window (fixed train size)
    - Multiple regularization types (L1, L2, Elastic Net)
    - Hyperparameter tuning via GridSearchCV
    """
    
    def __init__(self, 
                 dataset: Dataset,
                 factor_engine: FactorEngine,
                 train_size: float = 0.6,
                 val_size: float = 0.2,
                 window_type: str = 'expanding'):
        """
        Args:
            dataset: Dataset object with full_data, train_data, val_data, test_data
            factor_engine: FactorEngine object with computed factors
            train_size: Proportion of data for initial training (for rolling window)
            val_size: Proportion of data for validation in each fold
            window_type: 'expanding' or 'rolling'
        """
        self.dataset = dataset
        self.factor_engine = factor_engine
        self.train_size = train_size
        self.val_size = val_size
        self.window_type = window_type
        
        self.cv_results = []
        self.best_alpha = None
        self.best_model = None
    
    def _split_indices(self) -> List[Tuple[List, List]]:
        """Generate (train_indices, val_indices) pairs for walk-forward CV."""
        n_samples = len(self.dataset.full_data)
        splits = []
        
        if self.window_type == 'expanding':
            # Expanding window: training set grows
            initial_train_size = int(n_samples * self.train_size)
            val_period_size = int(n_samples * self.val_size)
            
            current_train_end = initial_train_size
            while current_train_end + val_period_size <= n_samples:
                train_idx = list(range(current_train_end))
                val_idx = list(range(current_train_end, current_train_end + val_period_size))
                splits.append((train_idx, val_idx))
                current_train_end += val_period_size
        
        elif self.window_type == 'rolling':
            # Rolling window: fixed training window size
            train_period_size = int(n_samples * self.train_size)
            val_period_size = int(n_samples * self.val_size)
            
            current_start = 0
            while current_start + train_period_size + val_period_size <= n_samples:
                train_idx = list(range(current_start, current_start + train_period_size))
                val_idx = list(range(current_start + train_period_size, 
                                    current_start + train_period_size + val_period_size))
                splits.append((train_idx, val_idx))
                current_start += val_period_size
        
        return splits
    
    def tune_regularization(self, 
                          regularization_type: str = 'L2',
                          alphas: List[float] = None) -> dict:
        """
        Use walk-forward CV to find optimal regularization parameter.
        
        Args:
            regularization_type: 'L1', 'L2', or 'elastic_net'
            alphas: List of alpha values to test. Defaults to [0.001, 0.01, 0.1, 1.0, 10.0]
        
        Returns:
            Dictionary with tuning results and best alpha
        """
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        splits = self._split_indices()
        fold_results = []
        
        print(f"\nStarting walk-forward CV with {len(splits)} folds ({self.window_type} window)...")
        
        for fold_num, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold_num + 1}/{len(splits)}:")
            print(f"  Train: indices 0-{max(train_idx)}, Val: indices {min(val_idx)}-{max(val_idx)}")
            
            # Get data for this fold
            train_dates = self.dataset.full_data.index[train_idx]
            val_dates = self.dataset.full_data.index[val_idx]
            
            # Collect all factor exposures and returns for training set
            X_train_list = []
            y_train_list = []
            
            for date in train_dates:
                try:
                    X_date = self.factor_engine.all_flattened_factors.loc[date].values
                    X_train_list.append(X_date)
                except:
                    continue
            
            # Get returns for training set (next day returns)
            stock_returns = self.dataset.full_data[[col for col in self.dataset.full_data.columns if col[0] == 'close']].pct_change(periods=-1)
            for date in train_dates:
                try:
                    y_date = stock_returns.loc[date].values
                    y_train_list.append(y_date)
                except:
                    continue
            
            if len(X_train_list) == 0 or len(y_train_list) == 0:
                continue
            
            X_train = np.vstack(X_train_list)
            y_train = np.hstack(y_train_list)
            
            # Remove NaN values
            valid_idx = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
            
            # Collect validation set data
            X_val_list = []
            y_val_list = []
            
            for date in val_dates:
                try:
                    X_date = self.factor_engine.all_flattened_factors.loc[date].values
                    X_val_list.append(X_date)
                except:
                    continue
            
            for date in val_dates:
                try:
                    y_date = stock_returns.loc[date].values
                    y_val_list.append(y_date)
                except:
                    continue
            
            if len(X_val_list) == 0 or len(y_val_list) == 0:
                continue
            
            X_val = np.vstack(X_val_list)
            y_val = np.hstack(y_val_list)
            
            # Remove NaN values
            valid_idx_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
            X_val = X_val[valid_idx_val]
            y_val = y_val[valid_idx_val]
            
            # Test each alpha
            fold_alphas_results = []
            for alpha in alphas:
                # Fit model
                if regularization_type == 'L2':
                    model = Ridge(alpha=alpha, fit_intercept=True)
                elif regularization_type == 'L1':
                    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
                elif regularization_type == 'elastic_net':
                    model = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, max_iter=10000)
                else:
                    raise ValueError(f"Unknown regularization type: {regularization_type}")
                
                try:
                    model.fit(X_train, y_train)
                    
                    # Evaluate on validation set
                    y_val_pred = model.predict(X_val)
                    mse = np.mean((y_val - y_val_pred) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_val - y_val_pred))
                    
                    fold_alphas_results.append({
                        'alpha': alpha,
                        'train_mse': np.mean((y_train - model.predict(X_train)) ** 2),
                        'val_mse': mse,
                        'val_rmse': rmse,
                        'val_mae': mae
                    })
                    
                    print(f"    alpha={alpha:.6f}: val_RMSE={rmse:.6f}, val_MAE={mae:.6f}")
                except Exception as e:
                    print(f"    alpha={alpha:.6f}: Failed - {str(e)}")
                    continue
            
            fold_results.append(fold_alphas_results)
        
        # Aggregate results across folds
        print("\n" + "="*60)
        print("Walk-Forward CV Results Summary")
        print("="*60)
        
        all_alpha_scores = {}
        for fold_results_list in fold_results:
            for result in fold_results_list:
                alpha = result['alpha']
                rmse = result['val_rmse']
                if alpha not in all_alpha_scores:
                    all_alpha_scores[alpha] = []
                all_alpha_scores[alpha].append(rmse)
        
        # Calculate average RMSE for each alpha
        alpha_avg_rmse = {}
        for alpha, rmses in all_alpha_scores.items():
            avg_rmse = np.mean(rmses)
            std_rmse = np.std(rmses)
            alpha_avg_rmse[alpha] = (avg_rmse, std_rmse)
            print(f"alpha={alpha:.6f}: Avg RMSE={avg_rmse:.6f} ± {std_rmse:.6f}")
        
        # Best alpha is the one with lowest average RMSE
        self.best_alpha = min(alpha_avg_rmse, key=lambda x: alpha_avg_rmse[x][0])
        best_rmse, best_std = alpha_avg_rmse[self.best_alpha]
        
        print(f"\nBest alpha: {self.best_alpha:.6f} (Avg RMSE: {best_rmse:.6f})")
        
        return {
            'best_alpha': self.best_alpha,
            'all_results': all_alpha_scores,
            'alpha_avg_rmse': alpha_avg_rmse,
            'num_folds': len(fold_results)
        }


# Example usage
if __name__ == "__main__":
    
    # Load data and factors
    dataset = Dataset(filename='example.csv')
    
    factor_list = ['mean_reversion_1', 'mean_reversion_3']
    factor_engine = FactorEngine(dataset, factor_list)
    
    # Run walk-forward CV
    wf_validator = WalkForwardValidator(
        dataset, 
        factor_engine,
        train_size=0.6,
        val_size=0.2,
        window_type='expanding'
    )
    
    print("="*60)
    print("L2 Regularization (Ridge) Tuning")
    print("="*60)
    l2_results = wf_validator.tune_regularization(
        regularization_type='L2',
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0]
    )
    
    print("\n" + "="*60)
    print("L1 Regularization (Lasso) Tuning")
    print("="*60)
    l1_results = wf_validator.tune_regularization(
        regularization_type='L1',
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0]
    )
