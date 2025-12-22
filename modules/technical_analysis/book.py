import numpy as np


class BookStrategy:
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
    def get_sell_mask(history: np.ndarray, hypo_prices: np.ndarray, maximum: float = 0.0) -> bool:
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
