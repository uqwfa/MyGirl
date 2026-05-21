from datetime import date

import numpy as np
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

        high = df['high']
        low = df['low']
        adx_window = int(self.params.get("adx_window", 14))
        up_move = high - high.shift(1)  # > 0, if p_today > p_yesterday
        down_move = low.shift(1) - low  # > 0, if p_today < p_yesterday

        atr_window = int(self.params.get("atr_window", 14))
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing approximated via Exponential Moving Average
        df['atr'] = tr.ewm(alpha=1 / atr_window, adjust=False).mean()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = tr.ewm(alpha=1 / adx_window, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / adx_window, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / adx_window, adjust=False).mean() / tr_smooth)

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))  # replace 0 to avoid div by zero
        df['adx'] = dx.ewm(alpha=1 / adx_window, adjust=False).mean()

        return df

    def generate_signal(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
        strategy_name = self.__class__.__name__

        if df.empty:
            return Signal(
                direction=Direction.LONG,
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
                adx=last["adx"],
                atr=last["atr"],
                period_high=period_high,
                strategy_name=strategy_name
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
               adx: float, atr: float, period_high: float, strategy_name: str) -> Signal:

        # Abort if data is warming up (NaNs present)
        if (pd.isna(price) or pd.isna(ma_macro_fast) or pd.isna(ma_macro_slow) or
                pd.isna(ema_agile_fast) or pd.isna(ema_agile_slow) or pd.isna(adx) or pd.isna(atr)):
            return Signal(
                direction=Direction.FLAT,
                strength=None,
                date=current_date,
                strategy=strategy_name,
                metadata={"info": "Waiting for moving averages to populate."}
            )

        long_candidates: list[tuple[float, str]] = []
        short_candidates: list[tuple[float, str]] = []

        # -- BUY LOGIC --
        adx_threshold = self.params.get("adx_threshold", 20.0)

        macro_bull = ma_macro_fast > ma_macro_slow
        agile_bull = ema_agile_fast > ema_agile_slow
        strong_trend = adx > adx_threshold

        if macro_bull and agile_bull and strong_trend:
            long_candidates.append((1.0, f"Macro & Agile trends aligned. ADX ({adx:.1f}) > {adx_threshold}"))

        # -- SELL LOGIC --

        # 1. Trailing ATR Stop (Primary Exit)
        if period_high > 0:
            atr_multiplier = self.params.get("atr_multiplier", 2.5)
            trailing_stop = period_high - (atr_multiplier * atr)

            if price < trailing_stop:
                strength = 1.0 - (price / trailing_stop) if trailing_stop > 0 else 1.0
                short_candidates.append(
                    (strength, f"Price broke {atr_multiplier}x ATR trailing stop ({trailing_stop:.2f})."))

        # 2. Agile Trend Breakdown (Early Exit)
        if ema_agile_fast < ema_agile_slow:
            short_candidates.append((0.8, "Agile momentum reversed (Fast EMA < Slow EMA)."))

        # 3. Macro Regime Shift (Absolute Cutoff)
        if ma_macro_fast < ma_macro_slow:
            short_candidates.append((1.0, "Macro regime flipped to bearish. Liquidating."))

        # -- DECISION RESOLUTION --
        if long_candidates:
            best_strength, best_reason = max(long_candidates, key=lambda x: x[0])
            return Signal(
                direction=Direction.LONG,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={"reasons": [r for _, r in long_candidates], "strongest_reason": best_reason}
            )

        if short_candidates:
            best_strength, best_reason = max(short_candidates, key=lambda x: x[0])
            return Signal(
                direction=Direction.SHORT,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={"reasons": [r for _, r in short_candidates], "strongest_reason": best_reason}
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name,
            metadata={}
        )
