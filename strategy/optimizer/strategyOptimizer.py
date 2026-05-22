"""
strategy/optimizer/strategyOptimizer.py
-----------------------------
"""

import optuna
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
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
            f"╔══ OptimizerResult ══════════════════════════════════════════",
            f"║  Target            : {self.target}",
            f"║  Best score        : {score_str}",
            f"║  Best params       : {params_str}",
            f"║  Evaluated         : {self.n_evaluated:,}",
            f"╠══ Best BacktestResult ═══════════════════════════════════════",
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
                 *, initial_capital: float = 100_000.00, fee_fixed: float = 1.0,
                 ticker: str = "unknown", min_lookback: int = 0, min_trades: int = 1, verbose: bool = True):

        self.strategy_class = strategy_class
        self.param_space_func = param_space_func
        self.initial_capital = initial_capital
        self.fee_fixed = fee_fixed
        self.ticker = ticker
        self.min_lookback = min_lookback
        self.min_trades = min_trades
        self.verbose = verbose

    def run(self, df: pd.DataFrame, *, n_trials: int = 100, maximize: bool = True, start_date: date | None = None,
            end_date: date | None = None) -> OptimizerResult:
        """
        """

        direction = "maximize" if maximize else "minimize"
        study = optuna.create_study(direction=direction)

        best_run_data = {"result": None}

        def objective(trial: optuna.Trial) -> float:
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
                result = backtester.run(df, start_date=start_date, end_date=end_date)
            except ValueError as exc:
                raise optuna.TrialPruned(exc)

            if self.min_trades > 0 and result.num_trades < self.min_trades:
                raise optuna.TrialPruned(f"Only {result.num_trades} trades, minimum is {self.min_trades}")

            try:
                # calmar ratio can be between 0.0 ~ 3.0, normalize to 0.0 ~ 1.0
                score_c = (getattr(result, "calmar_ratio") / 3)
                # sharpe ratio can be between 0.0 ~ 2.0, normalize to 0.0 ~ 1.0
                score_s = (getattr(result, "sharpe_ratio") / 2)
                # total return can be between -100 (%) ~ 200 (%), normalize to 0.0 ~ 1.0
                score_r = ((getattr(result, "total_return_pct") + 100) / 3)
            except TypeError as exc:
                raise optuna.TrialPruned(exc)

            score = 0.5 * score_c + 0.25 * score_s + 0.25 * score_r

            try:
                curr_best = study.best_value
            except ValueError as exc:
                curr_best = float("-inf") if maximize else float("inf")

            is_better = (score > curr_best) if maximize else (score < curr_best)
            if is_better:
                best_run_data["result"] = result

            return score

        print(f"Starting Optuna optimization for {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)

        if best_run_data["result"] is None:
            raise RuntimeError("All trials were pruned. No valid backtest result found.")

        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        return OptimizerResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_backtest=best_run_data["result"],
            target="score",
            n_evaluated=n_completed,
            all_runs=[
                (t.params, t.value)
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )
