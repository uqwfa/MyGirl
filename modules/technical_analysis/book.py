import pandas as pd
import numpy as np

from modules.simulation.objects.strategy import BacktesterStrategy, ThresholdStrategy, PositionContext


class BookStrategy(BacktesterStrategy, ThresholdStrategy):
    def __init__(self, bb_window=20, ma_fast=4, ma_medium=9, ma_slow=18, bb_std=2, drawdown_limit=0.8, ma_sell_factor=0.96):
        super().__init__()
        self.bb_window = bb_window
        self.ma_fast = ma_fast
        self.ma_medium = ma_medium
        self.ma_slow = ma_slow
        self.bb_std = bb_std
        self.drawdown_limit = drawdown_limit
        self.ma_sell_factor = ma_sell_factor

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rolling = df['close'].rolling(window=self.bb_window)
        df['mean_20'] = rolling.mean()
        df['std_20'] = rolling.std()
        df['upper_band'] = df['mean_20'] + (self.bb_std * df['std_20'])
        df['lower_band'] = df['mean_20'] - (self.bb_std * df['std_20'])

        df['ma_4'] = df['close'].rolling(self.ma_fast).mean()
        df['ma_9'] = df['close'].rolling(self.ma_medium).mean()
        df['ma_18'] = df['close'].rolling(self.ma_slow).mean()

        cond_bb_buy = df['close'] < df['lower_band']
        cond_ma_buy = (df['ma_4'] > df['ma_9']) & (df['ma_9'] > df['ma_18'])
        df['signal_buy'] = cond_bb_buy | cond_ma_buy

        df['cond_sell_bb'] = df['close'] > df['upper_band']
        df['signal_sell_ma'] = df['ma_4'] < (self.ma_sell_factor * df['ma_18'])

        return df

    def check_buy(self, row: pd.Series) -> bool:
        return bool(row['signal_buy'])

    def check_sell(self, row: pd.Series, context: PositionContext) -> bool:
        if row['cond_sell_bb'] or row['signal_sell_ma']:
            if not row['signal_buy']:
                return True

        if row['close'] < (self.drawdown_limit * context.current_maximum_price):
            if not row['signal_buy']:
                return True

        return False

    @property
    def lookback_period(self) -> int:
        # Bollinger Bands use window 20, so we need 19 historical points
        # to calculate the 20th point hypothetically.
        return self.bb_window - 1

    def get_buy_mask(self, history: np.ndarray, hypo_prices: np.ndarray, **kwargs) -> np.ndarray:
        """
        history: np.ndarray of shape (n-1,)
        hypo_prices: np.ndarray of shape (num_steps,); default=1000
        """
        n = self.bb_window

        hist_sum = history.sum()
        means = (hist_sum + hypo_prices) / n

        hist_sq_sum = (history ** 2).sum()
        sum_sq = hist_sq_sum + (hypo_prices ** 2)

        # Var = E[X^2] - (E[X])^2
        variances = (sum_sq / n) - (means ** 2)
        stds = np.sqrt(np.maximum(0, variances))

        lower_bands = means - (self.bb_std * stds)
        cond_bb = hypo_prices < lower_bands

        ma4 = (history[-(self.ma_fast-1):].sum() + hypo_prices) / self.ma_fast
        ma9 = (history[-(self.ma_medium-1):].sum() + hypo_prices) / self.ma_medium
        ma18 = (history[-(self.ma_slow-1):].sum() + hypo_prices) / self.ma_slow

        cond_ma = (ma4 > ma9) & (ma9 > ma18)

        # OR logic
        return cond_bb | cond_ma

    def get_sell_mask(self, history: np.ndarray, hypo_prices: np.ndarray, maximum: float = 0.0, **kwargs) -> np.ndarray:
        n = self.bb_window

        hist_sum = history.sum()
        means = (hist_sum + hypo_prices) / n

        hist_sq_sum = (history ** 2).sum()
        sum_sq = hist_sq_sum + (hypo_prices ** 2)

        variance = (sum_sq / n) - (means ** 2)
        stds = np.sqrt(np.maximum(0, variance))

        upper_bands = means + (self.bb_std * stds)
        cond_bb = hypo_prices > upper_bands

        ma_4 = (history[-(self.ma_fast-1):].sum() + hypo_prices) / self.ma_fast
        ma_18 = (history[-(self.ma_slow-1):].sum() + hypo_prices) / self.ma_slow

        cond_ma_cross = ma_4 < (self.ma_sell_factor * ma_18)

        safe_max = maximum if maximum > 0 else 0.0
        cond_drawdown = hypo_prices < (self.drawdown_limit * safe_max)

        raw_sell_signal = cond_bb | cond_ma_cross | cond_drawdown

        buy_mask = self.get_buy_mask(history, hypo_prices)

        return raw_sell_signal & (~buy_mask)
