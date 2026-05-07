"""
strategy/optimizer/strategyOptimizer.py
-----------------------------
"""

import optuna
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable

from backtesting.backtester import Backtester, BacktestResult
from strategy.strategies.base import BaseStrategy


@dataclass
class OptimizerResult:
    """"""

    best_params: dict[str, Any]
    best_score: float | None
    best_backtest: BacktestResult
    target: str
    n_evaluated: int
    all_runs: list[tuple[dict[str, Any], float | None]] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
        score_str = f"{self.best_score:.4f}" if self.best_score is not None else "—"

        lines = [
            "╔══ OptimizerResult ══════════════════════════════════════════",
            f"║  Target            : {self.target}",
            f"║  Best score        : {score_str}",
            f"║  Best params       : {params_str}",
            f"║  Evaluated         : {self.n_evaluated:,}",
            "╠══ Best BacktestResult ═══════════════════════════════════════",
        ]

        for line in str(self.best_backtest).splitlines()[1:-1]:  # strip box borders
            lines.append(f"║  {line.lstrip('║ ')}")

        lines.append("╚═════════════════════════════════════════════════════════════")

        return "\n".join(lines)


class StrategyOptimizer:
    """
    Optimizes a strategy's parameters over a given DataFrame using Optuna's Bayesian optimization framework.
    """

    def __init__(self, strategy_class: type[BaseStrategy], param_space_func: Callable[[optuna.Trial], dict[str, Any]],
                 *, target: str = "sharpe_ratio", initial_capital: float = 10_000.00, fee_fixed: float = 1.0,
                 ticker: str = "unknown", min_lookback: int = 0, min_trades: int = 1, verbose: bool = True):
        """Initializes a new instance of the ``StrategyOptimizer`` class."""

        self.strategy_class = strategy_class
        self.param_space_func = param_space_func
        self.target = target
        self.initial_capital = initial_capital
        self.fee_fixed = fee_fixed
        self.ticker = ticker
        self.min_lookback = min_lookback
        self.min_trades = min_trades
        self.verbose = verbose

        valid_objectives = {
            "sharpe_ratio", "calmar_ratio", "total_return_pct", "annualized_return_pct",
            "win_rate", "profit_factor", "max_drawdown_pct"
        }

        if target not in valid_objectives:
            print(f"Warning: Objective {target} is not a valid objective. Valid objectives: {valid_objectives}")

    def run(self, df: pd.DataFrame, *, n_trials: int = 100, maximize: bool = True, seed: int = 42) -> OptimizerResult:
        """"""

        direction = "maximize" if maximize else "minimize"
        study = optuna.create_study(direction=direction)

        best_run_data = {"result": None}

        def objective(trial: optuna.Trial) -> float:
            """"""

            params = self.param_space_func(trial)
            strategy_instance = self.strategy_class(params=params)
            backtester = Backtester(
                strat=strategy_instance,
                initial_capital=self.initial_capital,
                min_lookback=self.min_lookback,
                fee_fixed=self.fee_fixed,
                ticker=self.ticker
            )

            try:
                result = backtester.run(df)

                if self.min_trades > 0 and result.num_trades < self.min_trades:
                    raise optuna.TrialPruned(f"Only {result.num_trades} trades, minimum is {self.min_trades}")

                metric_val = getattr(result, self.target)

                # todo: what about a composite score? Weighted sum of multiple metrics?
                #  Or a custom metric function that can be passed in?

                if metric_val is None:
                    raise optuna.TrialPruned()

                duration_val = getattr(result, "avg_trade_duration_days")
                if duration_val is not None and duration_val < 30.0:
                    penalty = duration_val / 30.0

                    if direction == "maximize":
                        metric_val *= penalty

                    else:
                        metric_val /= penalty

                try:
                    current_best = study.best_value
                except ValueError:
                    current_best = float("-inf") if maximize else float("inf")

                is_better = (metric_val > current_best) if maximize else (metric_val < current_best)
                if is_better:
                    best_run_data["result"] = result

                return metric_val

            except ValueError as exc:
                raise optuna.TrialPruned(exc)

        print(f"Starting Optuna optimization for {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)

        if best_run_data is None:
            raise RuntimeError("All trials were pruned. No valid backtest result found.")

        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        return OptimizerResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_backtest=best_run_data["result"],
            target=self.target,
            n_evaluated=n_completed,
            all_runs=[
                (t.params, t.value)
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )


def walk_forward(optimizer: StrategyOptimizer, df: pd.DataFrame, *, train_years: int = 4, test_years: int = 1,
                 step_months: int = 6, n_trials: int = 100) -> list[BacktestResult]:
    """"""

    results = []
    start = df.index[0]
    end = df.index[-1]

    window_start = start
    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break

        train_df = df.loc[window_start:train_end]
        test_df = df.loc[train_end:test_end]

        opt_result = optimizer.run(train_df, n_trials=n_trials)
        best_strat = optimizer.strategy_class(params=opt_result.best_params)
        backtester = Backtester(strat=best_strat)
        results.append(backtester.run(test_df))

        window_start += pd.DateOffset(months=step_months)

    return results
