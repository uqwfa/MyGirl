from datetime import date, timedelta

import optuna

from backtesting.backtester import Backtester
from ingestion.scheduler import schedule_updates
from storage.database import init_db
from storage.models import Security, DateRange
from storage.repository import fetch_ohlcv
from strategy.optimizer.strategyWalkForwardOptimizer import WFO
from strategy.strategies.book import BookStrategy
from strategy.strategies.book_v2 import BookStrategyV2
from strategy.optimizer.strategyOptimizer import StrategyOptimizer


def book_strategy_param_space(trial: optuna.Trial):
    return {
        "bb_window": trial.suggest_int("bb_window", 10, 50),
        "bb_factor": trial.suggest_float("bb_factor", 1.0, 3.0),
        "ma_short": trial.suggest_int("ma_short", 2, 6),
        "ma_medium": trial.suggest_int("ma_medium", 6, 12),
        "ma_long": trial.suggest_int("ma_long", 14, 22),
        "sell_factor": trial.suggest_float("sell_factor", 0.85, 1.05),
        "drawdown_limit": trial.suggest_float("drawdown_limit", 0.70, 1.0)
    }


def update_last_30_days():
    today = date.today()
    d = DateRange(start=(today - timedelta(days=30)), end=today)

    tasks = [
        (
            Security(isin="US6311011026", name="NASDAQ 100 Index", ariva_id=72),
            d,
            "ArivaScraper"
        ),

        (
            Security(isin="DE0007164600", name="SAP SE", ariva_id=6),
            d,
            "ArivaScraper"
        )
    ]
    schedule_updates(tasks)


def wfo():
    print(f"\n--- Walk-Forward Optimization ---")

    today = date.today()
    d_wfo = DateRange(start=(today - timedelta(days=(10*365 + 10))), end=today)
    y_wfo = fetch_ohlcv("US6311011026", d_wfo)

    wfo = WFO(
        strategy_class=BookStrategyV2,
        param_space_func=book_strategy_param_space,
        min_lookback=50,
        initial_capital=100_000.00,
        target="sharpe_ratio",
        verbose=False
    )

    wfo_result = wfo.run(
        y_wfo,
        train_years=4,
        test_years=1,
        step_months=12,
        n_trials=200
    )

    print("\n" + str(wfo_result))

    for w in wfo_result.windows:
        print(f"\n{w.oos_result.trade_log()}")

if __name__ == "__main__":
    init_db()
    # update_last_30_days()
    wfo()

    import sys
    sys.exit(1)

    d2 = DateRange(start=date(2025, 4, 28), end=today)
    x = fetch_ohlcv("US6311011026", d2)
    b = BookStrategy()
    print(f"\nCurrent signal:\n{b.run(x, buy_date=date(2025, 4, 28))}")

    dt, levels = b.compute_price_levels(x, as_intervals=True, buy_date=date(2025, 4, 28))
    if dt is not None:
        print(f"\nPrice Levels for {dt.strftime("%d.%m.%Y")} with current price {x["close"].iloc[-1]}:\n{levels}")

    # t = Backtester(strat=b, min_lookback=20, initial_capital=100_000.00)
    # d3 = DateRange(start=(today - timedelta(days=(5*365))), end=today)
    # y = fetch_ohlcv("US6311011026", d3)
    # result = t.run(y)
    # print(result)
    # print(result.trade_log())



    optimizer = StrategyOptimizer(
        strategy_class=BookStrategy,
        param_space_func=book_strategy_param_space,
        min_lookback=50,
        initial_capital=100_000.00
    )

    d4 = DateRange(start=(today - timedelta(days=(6*365))), end=(today - timedelta(days=365)))
    y4 = fetch_ohlcv("US6311011026", d4)
    result = optimizer.run(y4, n_trials=200)
    print(result)
    print(result.best_backtest.trade_log())

    best_params = result.best_params

    # out of sample test
    d5 = DateRange(start=(today - timedelta(days=365 + 50)), end=today)
    y5 = fetch_ohlcv("US6311011026", d5)
    b5 = BookStrategy(params=best_params)
    t = Backtester(strat=b5, min_lookback=50, initial_capital=100_000.00)
    result = t.run(y5)
    print(result)

    """
    BEST RESULT TO THIS DAY:
    
    ╔══ OptimizerResult ══════════════════════════════════════════
    ║  Target            : sharpe_ratio
    ║  Best score        : 1.2746
    ║  Best params       : bb_window=36, bb_factor=2.035233625372893, ma_short=4, ma_medium=7, ma_long=14, sell_factor=0.870176047947008, drawdown_limit=0.9651853761918358
    ║  Evaluated         : 200
    ╠══ Best BacktestResult ═══════════════════════════════════════
    ║  Strategy          : BookStrategy
    ║  Ticker            : unknown
    ║  Period            : 20.07.2020 → 06.05.2025  (1751 cal. days)
    ║  Capital           :   100,000.00  →  272,829.42
    ║  Total Return      :   +172.83%
    ║  Ann. Return       :    +23.29%
    ║  Sharpe Ratio      :      1.275
    ║  Max Drawdown      :    -16.01%
    ║  Calmar Ratio      :      1.455
    ║  Win Rate          :      57.1%
    ║  Profit Factor     :       3.47
    ║  # Trades          :         35
    ║  Avg Duration      :    36.6 days
    ║  Total Fees        :      70.00
    ║  Avg Fee           :       2.00
    ╚═════════════════════════════════════════════════════════════
    #    Dir    Entry Date   Exit Date          Entry       Exit         PnL      Ret%   Days                                              Reason
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    1    long   20.07.2020   24.07.2020      10952.08   10483.13   -4222.55    -4.28%      4d                       Close below drawdown limit.
    2    long   04.08.2020   08.09.2020      11096.54   11068.26    -228.24    -0.26%     35d                       Close below drawdown limit.
    3    long   30.09.2020   19.10.2020      11418.06   11634.35  +  1728.32  +   1.89%     19d                       Close below drawdown limit.
    4    long   09.11.2020   29.01.2021      11830.39   12925.38  +  8757.92  +   9.25%     81d                       Close below drawdown limit.
    5    long   04.02.2021   22.02.2021      13560.89   13223.74   -2362.05    -2.49%     18d                       Close below drawdown limit.
    6    long   04.03.2021   04.05.2021      12464.00   13544.67  +  8643.36  +   8.67%     61d                       Close below drawdown limit.
    7    long   24.05.2021   23.08.2021      13641.75   15312.82  + 13366.56  +  12.25%     91d                                   Above upper BB.
    8    long   25.08.2021   20.09.2021      15368.92   15012.19   -2855.84    -2.32%     26d                       Close below drawdown limit.
    9    long   28.09.2021   01.12.2021      14770.30   15877.72  +  8857.36  +   7.50%     64d                       Close below drawdown limit.
    10   long   20.12.2021   05.01.2022      15627.64   15771.78  +  1151.12  +   0.92%     16d                       Close below drawdown limit.
    11   long   18.01.2022   28.01.2022      15210.76   14454.61   -6051.20    -4.97%     10d                       Close below drawdown limit.
    12   long   04.02.2022   14.02.2022      14694.35   14268.60   -3408.00    -2.90%     10d                       Close below drawdown limit.
    13   long   07.03.2022   06.04.2022      13319.38   14498.89  + 10613.59  +   8.85%     30d                       Close below drawdown limit.
    14   long   09.05.2022   18.05.2022      12187.72   11928.31   -2596.10    -2.13%      9d                       Close below drawdown limit.
    15   long   31.05.2022   09.06.2022      12642.10   12269.78   -3725.20    -2.95%      9d                       Close below drawdown limit.
    16   long   13.06.2022   28.06.2022      11288.32   11637.77  +  3841.95  +   3.09%     15d                       Close below drawdown limit.
    17   long   29.06.2022   22.08.2022      11658.26   12890.54  + 13553.08  +  10.57%     54d                       Close below drawdown limit.
    18   long   24.10.2022   02.11.2022      11430.26   10906.34   -6289.04    -4.59%      9d                       Close below drawdown limit.
    19   long   14.11.2022   07.12.2022      11700.94   11497.39   -2241.05    -1.74%     23d                       Close below drawdown limit.
    20   long   28.12.2022   10.02.2023      10679.35   12304.92  + 19504.84  +  15.22%     44d                       Close below drawdown limit.
    21   long   16.02.2023   24.02.2023      12442.48   11969.65   -5675.96    -3.80%      8d                       Close below drawdown limit.
    22   long   09.03.2023   04.08.2023      11995.88   15274.92  + 39346.48  +  27.33%    148d                       Close below drawdown limit.
    23   long   17.08.2023   21.09.2023      14715.81   14694.24    -260.84    -0.15%     35d                       Close below drawdown limit.
    24   long   26.09.2023   20.10.2023      14545.83   14560.88  +   193.65  +   0.10%     24d                       Close below drawdown limit.
    25   long   26.10.2023   04.01.2024      14109.57   16282.01  + 28239.72  +  15.40%     70d                       Close below drawdown limit.
    26   long   16.01.2024   23.04.2024      16830.71   17471.47  +  7687.12  +   3.81%     98d                       Close below drawdown limit.
    27   long   01.05.2024   17.07.2024      17318.55   19799.14  + 32245.67  +  14.32%     77d                       Close below drawdown limit.
    28   long   02.08.2024   03.09.2024      18440.85   18958.74  +  6730.57  +   2.81%     32d                       Close below drawdown limit.
    29   long   17.09.2024   06.11.2024      19432.40   20781.33  + 17534.09  +   6.94%     50d                                   Above upper BB.
    30   long   07.11.2024   19.12.2024      21101.57   21110.51  +   114.22  +   0.04%     42d                       Close below drawdown limit.
    31   long   22.01.2025   24.02.2025      21853.00   21352.08   -6013.04    -2.29%     33d                       Close below drawdown limit.
    32   long   27.02.2025   14.03.2025      20550.95   19704.64  -11004.03    -4.12%     15d                       Close below drawdown limit.
    33   long   25.03.2025   28.03.2025      20287.83   19281.40  -13085.59    -4.96%      3d                       Close below drawdown limit.
    34   long   04.04.2025   10.04.2025      17397.70   18343.57  + 13240.18  +   5.44%      6d                       Close below drawdown limit.
    35   long   24.04.2025   06.05.2025      19214.40   19791.35  +  7498.35  +   3.00%     12d                                 Simulation ended!
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ╔══ BacktestResult ═══════════════════════════════════════════
    ║  Strategy          : BookStrategy
    ║  Ticker            : unknown
    ║  Period            : 28.05.2025 → 06.05.2026  (343 cal. days)
    ║  Capital           :   100,000.00  →  129,800.24
    ║  Total Return      :    +29.80%
    ║  Ann. Return       :    +32.02%
    ║  Sharpe Ratio      :      2.238
    ║  Max Drawdown      :     -9.65%
    ║  Calmar Ratio      :      3.319
    ║  Win Rate          :      66.7%
    ║  Profit Factor     :      10.27
    ║  # Trades          :          6
    ║  Avg Duration      :    49.3 days
    ║  Total Fees        :      12.00
    ║  Avg Fee           :       2.00
    ╚═════════════════════════════════════════════════════════════
    """
