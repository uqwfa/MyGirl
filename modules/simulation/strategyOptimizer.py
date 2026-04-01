import pandas as pd
import optuna
import random

from modules.simulation.objects.security import Security
from modules.technical_analysis.book import BookStrategy
from modules.simulation.backtester import Backtester


class StrategyOptimizer:

    def __init__(self, securities: list[Security], start_date: pd.Timestamp, end_date: pd.Timestamp):
        self.securities = securities
        self.full_start = start_date
        self.full_end = end_date

        total_days = (end_date - start_date).days
        split_days = int(total_days * 0.8)
        self.split_date = start_date + pd.Timedelta(days=split_days)

        print(f"Full Range: {self.full_start.date()} to {self.full_end.date()}")
        print(f"In-Sample (Train): {self.full_start.date()} to {self.split_date.date()}")
        print(f"Out-of-Sample (Test): {self.split_date.date()} to {self.full_end.date()}")

    def _objective(self, trial):
        random.seed(42)

        params = {
            "bb_window": trial.suggest_int("bb_window", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.0),
            "ma_fast": trial.suggest_int("ma_fast", 2, 10),
            "ma_medium": trial.suggest_int("ma_medium", 10, 30),
            "ma_slow": trial.suggest_int("ma_slow", 31, 60),
            "ma_sell_factor": trial.suggest_float("ma_sell_factor", 0.90, 1.05),
            "drawdown_limit": trial.suggest_float("drawdown_limit", 0.70, 1.0)
        }

        strategy = BookStrategy(**params)

        try:
            backtester = Backtester(strategy, silent=True)
            equity_df, trades_df = backtester.run(
                securities=self.securities,
                start_date=self.full_start,
                end_date=self.split_date
            )

            if trades_df.empty or len(trades_df) < 5:
                return -1.0

            metrics = backtester._calc_metrics(equity_df)
            return metrics.get("sortino_ratio", -1.0)

            # return equity_df.iloc[-1]['equity']

        except Exception:
            return -1.0

    def optimize(self, n_trials: int = 1000):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials)

        best_params = study.best_params
        print("\n" + "=" * 30)
        print("OPTIMIZATION COMPLETE")
        print(f"Best In-Sample Value: {study.best_value:.2f}")
        print(f"Best Parameters: {best_params}")
        print("=" * 30)

        print("\nRunning Out-of-Sample Validation (The 20%)...")

        val_strategy = BookStrategy(**best_params)
        val_backtester = Backtester(val_strategy, silent=True)

        val_equity, val_trades = val_backtester.run(
            securities=self.securities,
            start_date=self.split_date,
            end_date=self.full_end
        )

        val_metrics = val_backtester._calc_metrics(val_equity)

        if not val_equity.empty:
            oos_return = (val_equity['equity'].iloc[-1] / val_equity['equity'].iloc[0]) - 1
            print(f"OOS Total Return: {oos_return * 100:.2f}%")
            print(f"OOS Sharpe Ratio: {val_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"OOS Sortino Ratio: {val_metrics.get('sortino_ratio', 0):.2f}")
            print(f"OOS Trades: {len(val_trades)}")
        else:
            print("OOS Error: No data found in validation range.")

        return best_params
