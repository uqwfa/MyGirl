import optuna
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Any

from backtesting.backtester import Backtester, BacktestResult
from strategy.optimizer.strategyOptimizer import StrategyOptimizer
from strategy.strategies.base import BaseStrategy


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward optimization and out-of-sample test window."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_score: float | None
    oos_result: BacktestResult


@dataclass
class WFOResult:
    """Aggregated results from a completed Walk-Forward Optimization run."""

    strategy: str
    target: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    windows: list[WalkForwardWindow]

    def __str__(self) -> str:
        lines = [
            "╔══ Walk-Forward Optimizer Result ════════════════════════════════════════════",
            f"║  Strategy          : {self.strategy}",
            f"║  Target            : {self.target}",
            f"║  WFO Windows       : {len(self.windows)}",
            f"║  Total Capital     : {self.initial_capital:>12,.2f}  →  {self.final_capital:,.2f}",
            f"║  Total Return      : {self.total_return_pct:>+11.2f}%",
            "╠══ OOS Window Breakdown ═════════════════════════════════════════════════════",
            f"║ {'#':<3} | {'Test Period':<23} | {'Ret%':>8} | {'Sharpe':>7} | {'Trades':>6} | Best Params"
        ]

        for w in self.windows:
            period = f"{w.test_start.strftime('%d.%m.%Y')} → {w.test_end.strftime('%d.%m.%Y')}"
            ret = f"{w.oos_result.total_return_pct:+.2f}%"
            sharpe = f"{w.oos_result.sharpe_ratio:.2f}" if w.oos_result.sharpe_ratio is not None else "—"
            trades = w.oos_result.num_trades

            params_str = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in w.best_params.items())

            lines.append(f"║ {w.window_id:<3} | {period:<23} | {ret:>8} | {sharpe:>7} | {trades:>6} | {params_str}")

        lines.append("╚═════════════════════════════════════════════════════════════════════════════")
        return "\n".join(lines)


class WFO:
    """Performs Walk-Forward Optimization (WFO) by rolling a train/test window over a DataFrame."""

    def __init__(self, strategy_class: type[BaseStrategy], param_space_func: Callable[[optuna.Trial], dict[str, Any]],
                 *, target: str = "sharpe_ratio", initial_capital: float = 100_000.00, fee_fixed: float = 1.0,
                 ticker: str = "unknown", min_lookback: int = 0, min_trades: int = 1, verbose: bool = True):

        self.strategy_class = strategy_class
        self.param_space_func = param_space_func
        self.target = target
        self.initial_capital = initial_capital
        self.fee_fixed = fee_fixed
        self.ticker = ticker
        self.min_lookback = min_lookback
        self.min_trades = min_trades
        self.verbose = verbose

    def run(self, df: pd.DataFrame, *, train_years: int = 4, test_years: int = 1, step_months: int = 12,
            n_trials: int = 100, maximize: bool = True, seed: int = 42) -> WFOResult:
        """"""

        df = df.copy().sort_index()
        start_date = df.index[0]
        end_date = df.index[-1]

        window_start = start_date
        current_capital = self.initial_capital
        windows: list[WalkForwardWindow] = []
        window_idx = 1

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"Starting Walk-Forward Optimization ({train_years}y train / {test_years}y test / {step_months}mo step)...")

        while True:
            train_end = window_start + pd.DateOffset(years=train_years)
            test_end = train_end + pd.DateOffset(years=test_years)

            if test_end > end_date:
                break

            train_df = df.loc[window_start:train_end]

            test_start_actual = train_end + pd.Timedelta(days=1)
            test_start_idx = df.index.searchsorted(test_start_actual)

            # buffer the start so indicators can compute
            buffer_idx = max(0, test_start_idx - self.min_lookback)
            test_end_idx = df.index.searchsorted(test_end, side="right")

            test_df = df.iloc[buffer_idx:test_end_idx]

            if len(test_df) <= self.min_lookback:
                break

            if self.verbose:
                print(f"\n[Window {window_idx}] Train: {window_start.date()} -> {train_end.date()} | "
                      f"Test: {test_start_actual.date()} -> {test_end.date()}")

            optimizer = StrategyOptimizer(
                strategy_class=self.strategy_class,
                param_space_func=self.param_space_func,
                target=self.target,
                initial_capital=self.initial_capital,
                fee_fixed=self.fee_fixed,
                ticker=self.ticker,
                min_lookback=self.min_lookback,
                min_trades=self.min_trades,
                verbose=False
            )

            opt_result = optimizer.run(train_df, n_trials=n_trials, maximize=maximize, seed=seed)

            best_strat = self.strategy_class(params=opt_result.best_params)
            oos_backtester = Backtester(
                strat=best_strat,
                initial_capital=current_capital,
                min_lookback=self.min_lookback,
                fee_fixed=self.fee_fixed,
                ticker=self.ticker
            )

            oos_result = oos_backtester.run(test_df)

            windows.append(WalkForwardWindow(
                window_id=window_idx,
                train_start=window_start,
                train_end=train_end,
                test_start=test_start_actual,
                test_end=test_end,
                best_params=opt_result.best_params,
                train_score=opt_result.best_score,
                oos_result=oos_result
            ))

            current_capital = oos_result.final_capital

            window_start += pd.DateOffset(months=step_months)
            window_idx += 1

        total_return_pct = ((current_capital / self.initial_capital) - 1.0) * 100.0

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        return WFOResult(
            strategy=self.strategy_class.__name__,
            target=self.target,
            initial_capital=self.initial_capital,
            final_capital=current_capital,
            total_return_pct=total_return_pct,
            windows=windows
        )
