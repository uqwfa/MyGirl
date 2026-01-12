import pandas as pd
import numpy as np

from modules.simulation.objects.strategy import Strategy, PositionContext


class BookStrategy(Strategy):

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rolling = df['close'].rolling(window=20)
        df['mean_20'] = rolling.mean()
        df['std_20'] = rolling.std()
        df['upper_band'] = df['mean_20'] + (2 * df['std_20'])
        df['lower_band'] = df['mean_20'] - (2 * df['std_20'])

        df['ma_4'] = df['close'].rolling(4).mean()
        df['ma_9'] = df['close'].rolling(9).mean()
        df['ma_18'] = df['close'].rolling(18).mean()

        cond_bb_buy = df['close'] < df['lower_band']
        cond_ma_buy = (df['ma_4'] > df['ma_9']) & (df['ma_9'] > df['ma_18'])
        df['signal_buy'] = cond_bb_buy | cond_ma_buy

        df['cond_sell_bb'] = df['close'] > df['upper_band']
        df['signal_sell_ma'] = df['ma_4'] < (0.96 * df['ma_18'])

        return df

    def check_buy(self, row: pd.Series) -> bool:
        return bool(row['signal_buy'])

    def check_sell(self, row: pd.Series, context: PositionContext) -> bool:
        if row['cond_sell_bb'] or row['signal_sell_ma']:
            if not row['signal_buy']:
                return True

        if row['close'] < (0.8 * context.current_maximum_price):
            if not row['signal_buy']:
                return True

        return False

    @staticmethod
    def get_buy_mask(history: np.ndarray, hypo_prices: np.ndarray, **kwargs) -> np.ndarray:
        """
        history: np.ndarray of shape (n-1,)
        hypo_prices: np.ndarray of shape (num_steps,); default=1000
        """
        n = 20

        hist_sum = history.sum()
        means = (hist_sum + hypo_prices) / n

        hist_sq_sum = (history ** 2).sum()
        sum_sq = hist_sq_sum + (hypo_prices ** 2)

        # Var = E[X^2] - (E[X])^2
        variances = (sum_sq / n) - (means ** 2)
        stds = np.sqrt(np.maximum(0, variances))

        lower_bands = means - (2 * stds)

        cond_bb = hypo_prices < lower_bands

        sum_3 = history[-3:].sum()
        sum_8 = history[-8:].sum()
        sum_17 = history[-17:].sum()

        ma4 = (sum_3 + hypo_prices) / 4
        ma9 = (sum_8 + hypo_prices) / 9
        ma18 = (sum_17 + hypo_prices) / 18

        cond_ma = (ma4 > ma9) & (ma9 > ma18)

        # OR logic
        return cond_bb | cond_ma

    @staticmethod
    def get_sell_mask(history: np.ndarray, hypo_prices: np.ndarray, maximum: float = 0.0, **kwargs) -> bool:
        n = 20

        hist_sum = history.sum()
        means = (hist_sum + hypo_prices) / n

        hist_sq_sum = (history ** 2).sum()
        sum_sq = hist_sq_sum + (hypo_prices ** 2)

        variance = (sum_sq / n) - (means ** 2)
        stds = np.sqrt(np.maximum(0, variance))

        upper_bands = means + (2 * stds)

        cond_bb = hypo_prices > upper_bands

        sum_3 = history[-3:].sum()
        sum_17 = history[-17:].sum()

        ma_4 = (sum_3 + hypo_prices) / 4
        ma_18 = (sum_17 + hypo_prices) / 18

        cond_ma_cross = ma_4 < (0.96 * ma_18)

        safe_max = maximum if maximum > 0 else 0.0
        cond_drawdown = hypo_prices < (0.8 * safe_max)

        raw_sell_signal = cond_bb | cond_ma_cross | cond_drawdown

        buy_mask = BookStrategy.get_buy_mask(history, hypo_prices)

        return raw_sell_signal & (~buy_mask)
