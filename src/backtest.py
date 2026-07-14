"""
File: backtest.py
Description: Weight-driven backtesting engine for equity portfolios.

Design goals:
1. This module does not fit regression models.
2. Input portfolio is a DataFrame of target stock weights by timestamp.
3. NaN weight means exit that stock at that timestamp.
4. Portfolio rows are checked to sum to 1 over non-NaN entries.
5. Outputs include equity curve, Sharpe, Sortino, UPI, drawdown, turnover.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_per_share: float = 0.005
    commission_min_per_order: float = 1.0
    slippage_tiers_bps: tuple[tuple[int, int, float], ...] = (
        (1, 500, 13.0),
        (501, 1500, 20.0),
        (1501, 3000, 50.0),
    )
    default_slippage_bps: float = 50.0
    short_borrow_rate_annual: float = 0.01
    periods_per_year: int = 252
    enforce_unit_sum: bool = True
    sum_tolerance: float = 1e-6
    holding_period_days: int = 1
    min_trade_weight: float = 0.0005
    min_trade_notional: float = 0.0


@dataclass
class BacktestResults:
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    target_weights: pd.DataFrame
    executed_weights: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict = field(default_factory=dict)
    config: BacktestConfig = None
    benchmark_equity: Optional[pd.Series] = None


class Backtester:
    """Backtest engine that takes target per-stock weights and computes portfolio performance."""

    def __init__(self, data: pd.DataFrame, target_weights: pd.DataFrame, **kwargs):
        self.data = data
        self.target_weights = target_weights
        self.cfg = BacktestConfig(**kwargs)

    def _slippage_bps_by_stock(self, stocks: pd.Index) -> pd.Series:
        """Assign slippage bps by 1-based stock rank position in the panel."""
        n = len(stocks)
        slippage = pd.Series(self.cfg.default_slippage_bps, index=stocks, dtype=float)
        ranks = np.arange(1, n + 1)

        for start_rank, end_rank, bps in self.cfg.slippage_tiers_bps:
            mask = (ranks >= start_rank) & (ranks <= end_rank)
            if mask.any():
                slippage.iloc[mask] = float(bps)

        return slippage

    def _prepare_weights(self, close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align and validate target weights.

        - Target rows are normalized to sum to 1 over non-NaN entries.
        - If a row sum is below 1, raise a warning and reweight that row.
        - NaN means no target position for that stock at that timestamp.
        - Fully empty rows between active target rows can hold the previous target
          for `holding_period_days`.
        - Executed weights fill NaN with 0 so missing targets become exits.
        """
        tw = self.target_weights.reindex(index=close.index, columns=close.columns)
        holding_period_days = max(int(self.cfg.holding_period_days), 1)
        if holding_period_days > 1:
            active_target_rows = tw.notna().any(axis=1)
            held_tw = tw.ffill(limit=holding_period_days - 1)
            tw = tw.where(active_target_rows, held_tw)

        active_rows = tw.notna().any(axis=1)
        row_sum = tw.sum(axis=1, skipna=True)

        if self.cfg.enforce_unit_sum:
            zero_sum = active_rows & row_sum.abs().lt(self.cfg.sum_tolerance)
            if zero_sum.any():
                bad_dates = [str(idx) for idx in tw.index[zero_sum][:5]]
                raise ValueError(
                    "Active weight rows have near-zero sum and cannot be normalized. "
                    f"Examples: {bad_dates}"
                )

            underweight = active_rows & row_sum.lt(1.0 - self.cfg.sum_tolerance)
            if underweight.any():
                bad_dates = [str(idx) for idx in tw.index[underweight][:5]]
                warnings.warn(
                    "Some active weight rows sum to less than 1. "
                    f"Reweighting to 1 for {int(underweight.sum())} rows. Examples: {bad_dates}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            needs_norm = active_rows & (row_sum.sub(1.0).abs() > self.cfg.sum_tolerance)
            if needs_norm.any():
                overweight = active_rows & row_sum.gt(1.0 + self.cfg.sum_tolerance)
                if overweight.any():
                    print(
                        f"[WEIGHTS] {int(overweight.sum())} rows sum to more than 1. "
                        "Normalizing those rows before backtest."
                    )
                tw = tw.div(row_sum.where(active_rows).replace(0.0, np.nan), axis=0)

        executed = tw.fillna(0.0)
        return tw, executed

    def _compute_metrics(self, returns: pd.Series, gross_returns: pd.Series, turnover: pd.Series) -> dict:
        ppy = self.cfg.periods_per_year
        ret = returns.dropna()
        if len(ret) == 0:
            return {}

        total_return = (1 + ret).prod() - 1
        ann_return = ret.mean() * ppy
        ann_vol = ret.std() * np.sqrt(ppy)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        downside = ret[ret < 0]
        downside_vol = downside.std() * np.sqrt(ppy) if len(downside) > 0 else np.nan
        sortino = ann_return / downside_vol if pd.notna(downside_vol) and downside_vol > 0 else np.nan

        equity = (1 + ret).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()

        ulcer = np.sqrt((drawdown**2).mean())
        upi = ann_return / ulcer if ulcer > 0 else np.nan

        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "ulcer_index": ulcer,
            "upi": upi,
            "avg_turnover": turnover.mean(),
            "gross_annual_return": gross_returns.mean() * ppy,
            "cost_drag_annual": (gross_returns.mean() - ret.mean()) * ppy,
            "n_periods": len(ret),
        }

    def _print_metrics(self, m: dict) -> None:
        if not m:
            print("[BACKTEST] No valid return observations.")
            return
        print("\n" + "-" * 72)
        print(f"{'Total Return':<28s}{m['total_return']*100:>12.2f}%")
        print(f"{'Annual Return (net)':<28s}{m['annual_return']*100:>12.2f}%")
        print(f"{'Annual Return (gross)':<28s}{m['gross_annual_return']*100:>12.2f}%")
        print(f"{'Cost Drag (annual)':<28s}{m['cost_drag_annual']*100:>12.2f}%")
        print(f"{'Annual Volatility':<28s}{m['annual_volatility']*100:>12.2f}%")
        print(f"{'Sharpe Ratio':<28s}{m['sharpe_ratio']:>12.3f}")
        print(f"{'Sortino Ratio':<28s}{m['sortino_ratio']:>12.3f}")
        print(f"{'Max Drawdown':<28s}{m['max_drawdown']*100:>12.2f}%")
        print(f"{'Ulcer Index':<28s}{m['ulcer_index']:>12.4f}")
        print(f"{'UPI':<28s}{m['upi']:>12.3f}")
        print(f"{'Avg Daily Turnover':<28s}{m['avg_turnover']:>12.3f}")
        print(f"{'Periods':<28s}{m['n_periods']:>12d}")
        print("-" * 72)

    def run(self) -> BacktestResults:
        if "close" not in self.data.columns.get_level_values(0):
            raise ValueError("Input data must include ('close', stock) columns.")

        close = self.data["close"]
        _, target_executed_weights = self._prepare_weights(close)

        fwd_ret = close.pct_change().shift(-1)
        aligned_w = target_executed_weights.reindex_like(fwd_ret).fillna(0.0)

        slippage_bps = self._slippage_bps_by_stock(close.columns)

        valid_idx = fwd_ret.dropna(how="all").index
        net_ret_values: list[float] = []
        gross_ret_values: list[float] = []
        turnover_values: list[float] = []
        executed_weight_rows: list[pd.Series] = []

        equity = float(self.cfg.initial_capital)
        prev_w = pd.Series(0.0, index=close.columns, dtype=float)
        min_trade_weight = max(float(self.cfg.min_trade_weight), 0.0)
        min_trade_notional = max(float(self.cfg.min_trade_notional), 0.0)

        for dt in valid_idx:
            price_t = close.loc[dt]
            target_w_t = aligned_w.loc[dt].fillna(0.0)
            ret_t = fwd_ret.loc[dt]

            dw = (target_w_t - prev_w).fillna(0.0)
            abs_dw = dw.abs()
            traded_notional = abs_dw * equity

            skip_trade = pd.Series(False, index=close.columns)
            if min_trade_weight > 0:
                skip_trade |= abs_dw < min_trade_weight
            if min_trade_notional > 0:
                skip_trade |= traded_notional < min_trade_notional

            w_t = target_w_t.where(~skip_trade, prev_w)
            dw = (w_t - prev_w).fillna(0.0)
            abs_dw = dw.abs()
            gross_ret_t = float((w_t * ret_t.fillna(0.0)).sum())
            turnover_t = float(abs_dw.sum())

            traded_notional = abs_dw * equity
            traded_shares = pd.Series(0.0, index=close.columns, dtype=float)
            valid_price = price_t > 0
            traded_shares.loc[valid_price] = traded_notional.loc[valid_price] / price_t.loc[valid_price]

            has_trade = traded_shares > 0
            commission = pd.Series(0.0, index=close.columns, dtype=float)
            commission.loc[has_trade] = np.maximum(
                traded_shares.loc[has_trade] * self.cfg.commission_per_share,
                self.cfg.commission_min_per_order,
            )
            commission_cost = float(commission.sum())

            slippage_cost = float((traded_notional * (slippage_bps / 10_000.0)).sum())

            short_notional = (w_t.clip(upper=0.0).abs()) * equity
            short_borrow_cost = float(short_notional.sum() * (self.cfg.short_borrow_rate_annual / self.cfg.periods_per_year))

            total_cost_dollar = commission_cost + slippage_cost + short_borrow_cost
            total_cost_return = total_cost_dollar / equity if equity > 0 else 0.0

            net_ret_t = gross_ret_t - total_cost_return
            equity *= 1.0 + net_ret_t

            net_ret_values.append(net_ret_t)
            gross_ret_values.append(gross_ret_t)
            turnover_values.append(turnover_t)
            executed_weight_rows.append(w_t)
            prev_w = w_t

        net_pnl = pd.Series(net_ret_values, index=valid_idx)
        gross_pnl = pd.Series(gross_ret_values, index=valid_idx)
        turnover = pd.Series(turnover_values, index=valid_idx)
        executed_weights = pd.DataFrame(executed_weight_rows, index=valid_idx).reindex(index=close.index, columns=close.columns)

        equity_curve = (1 + net_pnl).cumprod()
        metrics = self._compute_metrics(net_pnl, gross_pnl, turnover)
        self._print_metrics(metrics)

        return BacktestResults(
            returns=net_pnl,
            gross_returns=gross_pnl,
            turnover=turnover,
            target_weights=self.target_weights,
            executed_weights=executed_weights,
            equity_curve=equity_curve,
            metrics=metrics,
            config=self.cfg,
        )


def plot_results(results: BacktestResults, benchmark_equity: Optional[pd.Series] = None, save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(results.equity_curve.index, (results.equity_curve - 1) * 100, label="Strategy", color="#2E86AB", linewidth=1.8)
    if benchmark_equity is not None:
        # align benchmark to the equity curve index
        b_eq = benchmark_equity.reindex(results.equity_curve.index).ffill().bfill()
        axes[0].plot(b_eq.index, (b_eq - 1) * 100, label="Benchmark", color="#FF7F0E", linewidth=1.2, linestyle="--")

    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Equity Curve - Net of Costs", fontweight="bold")
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    eq = results.equity_curve
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    axes[1].fill_between(dd.index, dd, 0, color="#D62828", alpha=0.5)
    axes[1].set_title("Drawdown", fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(results.turnover.index, results.turnover, color="#6A4C93", linewidth=0.8)
    axes[2].set_title("Daily Turnover", fontweight="bold")
    axes[2].set_ylabel("Turnover")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved to {save_path}")
    plt.show()
