from datetime import date

import numpy as np
import optuna
import pandas as pd

from strategy.models import Signal, Direction
from strategy.strategies.base import BaseStrategy


class DualTrendStrategy(BaseStrategy):
    """
        Dual-Timeframe Trend Following Strategy designed for high-momentum indices.

        Params (all optional, with defaults for optimization):
            ma_macro_fast  (int, default 50)   - Fast SMA for macro trend filter
            ma_macro_slow  (int, default 200)  - Slow SMA for macro trend filter
            ema_agile_fast (int, default 9)    - Fast EMA for agile entry/exit trigger
            ema_agile_slow (int, default 21)   - Slow EMA for agile entry/exit trigger
            adx_window     (int, default 14)   - Lookback for Trend Strength
            adx_threshold  (float, default 20.0)- Minimum ADX required to enter a trade
            atr_window     (int, default 14)   - Lookback for Volatility (True Range)
            atr_multiplier (float, default 2.5)- Multiplier for the trailing stop loss
        """

    def param_space(trial: optuna.Trial):
        return {
            "ma_macro_fast": trial.suggest_int("ma_macro_fast", 25, 75),
            "ma_macro_slow": trial.suggest_int("ma_macro_slow", 100, 300),

            "ema_agile_fast": trial.suggest_int("ema_agile_fast", 4, 14),
            "ema_agile_slow": trial.suggest_int("ema_agile_slow", 11, 31),

            "rsi_period": trial.suggest_int("rsi_period", 10, 18),
            "rsi_up_threshold": trial.suggest_int("rsi_up_threshold", 20, 60),
            "rsi_down_threshold": trial.suggest_int("rsi_down_threshold", 50, 90),
            "x_prev_rsi": trial.suggest_int("x_prev_rsi", 2, 5)
        }

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_index()

        ma_macro_fast = int(self.params.get("ma_macro_fast", 50))
        ma_macro_slow = int(self.params.get("ma_macro_slow", 200))
        df["ma_macro_fast"] = df["close"].rolling(ma_macro_fast).mean()
        df["ma_macro_slow"] = df["close"].rolling(ma_macro_slow).mean()

        ema_agile_fast = int(self.params.get("ema_agile_fast", 9))
        ema_agile_slow = int(self.params.get("ema_agil_slow", 21))
        df["ema_agile_fast"] = df["close"].ewm(span=ema_agile_fast, adjust=False).mean()
        df["ema_agile_slow"] = df["close"].ewm(span=ema_agile_slow, adjust=False).mean()

        rsi_period = int(self.params.get("rsi_period", 14))
        delta = df["close"].diff()
        gains = delta.clip(lower=0)
        losses = -1 * delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))

        df["prev_rsi"] = df["rsi"].shift(1)

        x_prev_rsi = int(self.params.get("x_prev_rsi", 3))
        df["x_prev_rsi"] = df["rsi"].shift(x_prev_rsi)

        self.rsi_down_trend = False
        self.rsi_up_trend = False

        return df

    def generate_signal(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
        strategy_name = self.__class__.__name__

        if df.empty:
            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": "Empty DataFrame."}
            )

        try:
            last = df.iloc[-1]
            period_high = 0.0

            # Identify the highest high since entry for the trailing ATR stop
            if buy_date is not None:
                if 'high' in df.columns:
                    period_high = df.loc[buy_date:, "high"].max()
                else:
                    period_high = df.loc[buy_date:, "close"].max()

            return self._score(
                price=last["close"],
                current_date=df.index[-1],
                ma_macro_fast=last["ma_macro_fast"],
                ma_macro_slow=last["ma_macro_slow"],
                ema_agile_fast=last["ema_agile_fast"],
                ema_agile_slow=last["ema_agile_slow"],
                period_high=period_high,
                strategy_name=strategy_name,
                rsi_prev=last["prev_rsi"],
                rsi=last["rsi"],
                x_prev_rsi=last["x_prev_rsi"]
            )

        except Exception as exc:
            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": str(exc)}
            )

    def _score(self, *, price: float, current_date: date,
               ma_macro_fast: float, ma_macro_slow: float,
               ema_agile_fast: float, ema_agile_slow: float,
               period_high: float, strategy_name: str,
               rsi_prev: float, rsi: float, x_prev_rsi: float) -> Signal:

        long_candidates: list[tuple[float, str]] = []
        short_candidates: list[tuple[float, str]] = []

        #
        # BUY
        #
        if (not self.rsi_down_trend) or (rsi > x_prev_rsi):
            macro_bull = ma_macro_fast > ma_macro_slow
            agile_bull = ema_agile_fast > ema_agile_slow
            if macro_bull and agile_bull:
                macro_strength = (ma_macro_fast / ma_macro_slow) - 1.0
                agile_strength = (ema_agile_fast / ema_agile_slow) - 1.0
                strength = (macro_strength + agile_strength) / 2.0
                long_candidates.append((strength, f"Macro & Agile trends aligned."))
                self.rsi_up_trend = False

            rsi_up_threshold = int(self.params.get("rsi_up_threshold", 40))
            if (rsi_prev < rsi_up_threshold) & (rsi >= rsi_up_threshold):
                strength = (rsi / rsi_prev) - 1.0
                long_candidates.append((strength, f"RSI trend aligned."))
                self.rsi_up_trend = True

        #
        # SELL
        #
        if (not self.rsi_up_trend) or (rsi < x_prev_rsi):
            rsi_down_threshold = int(self.params.get("rsi_down_threshold", 60))
            if (rsi_prev > rsi_down_threshold) & (rsi <= rsi_down_threshold):
                strength = 1.0 - (rsi / rsi_prev)
                short_candidates.append((strength, f"RSI trend flipped."))
                self.rsi_down_trend = True

            if ema_agile_fast < ema_agile_slow:
                strength = 1.0 - (ema_agile_fast / ema_agile_slow)
                short_candidates.append((strength, f"Agile trends flipped."))
                self.rsi_down_trend = False

            if ma_macro_fast < ma_macro_slow:
                strength = 1.0 - (ma_macro_fast / ma_macro_slow)
                short_candidates.append((strength, f"Macro trends flipped."))

        # -- DECISION RESOLUTION --
        if long_candidates:
            best_strength, best_reason = max(long_candidates, key=lambda x: x[0])
            return Signal(
                direction=Direction.LONG,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={
                    "long_reasons": [r for r in long_candidates],
                    "short_reasons": [r for r in short_candidates]
                }
            )

        if short_candidates:
            best_strength, best_reason = max(short_candidates, key=lambda x: x[0])
            return Signal(
                direction=Direction.SHORT,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={
                    "long_reasons": [r for r in long_candidates],
                    "short_reasons": [r for r in short_candidates]
                }
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name
        )
