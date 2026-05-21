from datetime import date, timedelta

import optuna

from backtesting.backtester import Backtester, BacktestResult
from ingestion.scheduler import schedule_updates
from storage.database import init_db
from storage.models import Security, DateRange
from storage.repository import fetch_ohlcv
from strategy.optimizer.strategyWalkForwardOptimizer import WFO
from strategy.strategies.book import BookStrategy
from strategy.strategies.book_v2 import BookStrategyV2
from strategy.optimizer.strategyOptimizer import StrategyOptimizer
from strategy.strategies.buyandhold import BuyAndHold
from strategy.strategies.george import DualTrendStrategy
import matplotlib.pyplot as plt


def plot_backtest_comparison(results: list[BacktestResult]):

    if not results:
        print("No backtest results provided.")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, :])

    for res in results:
        # Create a label indicating strategy and ticker
        label = f"{res.strategy} ({res.ticker})"

        # Plot the equity curve (pandas Series handles the datetime index gracefully)
        ax1.plot(res.equity_curve.index, res.equity_curve.values, label=label, linewidth=2)

    ax1.set_title("Equity Curve Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Capital ($)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left")

    # ==========================================
    # Extract Metrics for Bar Charts
    # ==========================================
    # Shorten strategy names for the x-axis if needed
    labels = [f"{r.strategy}\n({r.ticker})" for r in results]

    returns = [r.total_return_pct for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]

    # Handle potentially missing Sharpe ratios
    sharpes = [r.sharpe_ratio if r.sharpe_ratio is not None else 0.0 for r in results]

    # ==========================================
    # 2. Total Return Bar Chart (Bottom Left)
    # ==========================================
    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.bar(labels, returns, color='skyblue', edgecolor='black', alpha=0.8)
    ax2.set_title("Total Return (%)", fontsize=12)
    ax2.grid(axis='y', linestyle="--", alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1)

    # Add values on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + (2 if yval >= 0 else -6),
                 f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)

    # ==========================================
    # 3. Max Drawdown Bar Chart (Bottom Middle)
    # ==========================================
    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.bar(labels, drawdowns, color='salmon', edgecolor='black', alpha=0.8)
    ax3.set_title("Max Drawdown (%)", fontsize=12)
    ax3.grid(axis='y', linestyle="--", alpha=0.7)
    ax3.axhline(0, color='black', linewidth=1)

    # Add values on bottom of bars
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval - 2,
                 f"{yval:.1f}%", ha='center', va='top', fontsize=10)

    # ==========================================
    # 4. Sharpe Ratio Bar Chart (Bottom Right)
    # ==========================================
    ax4 = fig.add_subplot(gs[1, 2])
    bars4 = ax4.bar(labels, sharpes, color='lightgreen', edgecolor='black', alpha=0.8)
    ax4.set_title("Sharpe Ratio (Ann.)", fontsize=12)
    ax4.grid(axis='y', linestyle="--", alpha=0.7)
    ax4.axhline(0, color='black', linewidth=1)

    for bar in bars4:
        yval = bar.get_height()
        offset = 0.05 if yval >= 0 else -0.15
        ax4.text(bar.get_x() + bar.get_width() / 2, yval + offset,
                 f"{yval:.2f}", ha='center', va='bottom', fontsize=10)

    # Formatting adjustments
    for ax in [ax2, ax3, ax4]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    plt.tight_layout()

    plt.show()


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

def george_strategy_param_space(trial: optuna.Trial):
    return {
        "ma_macro_fast": trial.suggest_int("ma_macro_fast", 25, 75),
        "ma_macro_slow": trial.suggest_int("ma_macro_slow", 100, 300),
        "ema_agile_fast": trial.suggest_int("ema_agile_fast", 4, 14),
        "ema_agile_slow": trial.suggest_int("ema_agile_slow", 11, 31),
        "adx_window": trial.suggest_int("adx_window", 7, 21),
        "adx_threshold": trial.suggest_float("adx_threshold", 10.0, 30.0),
        "atr_window": trial.suggest_int("atr_window", 7, 21),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.0, 4.0)

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
        train_years=2,
        test_years=1,
        step_months=12,
        n_trials=200
    )

    print("\n" + str(wfo_result))

    for w in wfo_result.windows:
        print(f"\n{w.oos_result.trade_log()}")


def wfo_2():
    print(f"\n--- Walk-Forward Optimization ---")

    today = date.today()
    d_wfo = DateRange(start=(today - timedelta(days=(9*365 + 400))), end=today)
    y_wfo = fetch_ohlcv("US6311011026", d_wfo)

    wfo = WFO(
        strategy_class=DualTrendStrategy,
        param_space_func=george_strategy_param_space,
        min_lookback=250,
        initial_capital=100_000.00,
        target="sharpe_ratio",
        verbose=False
    )

    wfo_result = wfo.run(
        y_wfo,
        train_years=2,
        test_years=1,
        step_months=12,
        n_trials=500
    )

    print("\n" + str(wfo_result))

    for w in wfo_result.windows:
        print(f"\n{w.oos_result.trade_log()}")


def dailys():
    today = date.today()
    d2 = DateRange(start=date(2025, 4, 28), end=today)
    x = fetch_ohlcv("US6311011026", d2)
    b = BookStrategy()
    print(f"\nCurrent signal:\n{b.run(x, buy_date=date(2025, 4, 28))}")

    dt, levels = b.compute_price_levels(x, as_intervals=True, buy_date=date(2025, 4, 28))
    if dt is not None:
        print(f"\nPrice Levels for {dt.strftime("%d.%m.%Y")} with current price {x["close"].iloc[-1]}:\n{levels}")

    d3 = DateRange(start=date(2020, 1, 1), end=today)
    y = fetch_ohlcv("US6311011026", d3)
    l = 200
    args = {"start_date": date(2021, 5, 21)}


    b = BookStrategy()
    t = Backtester(strat=b, min_lookback=l)
    result = t.run(y, **args)
    print(result)
    print(result.trade_log())

    b = BuyAndHold()
    t = Backtester(strat=b, min_lookback=l)
    result_2 = t.run(y, **args)
    print(result_2)
    print(result_2.trade_log())

    b = BookStrategyV2()
    t = Backtester(strat=b, min_lookback=l)
    result_3 = t.run(y, **args)
    print(result_3)
    print(result_3.trade_log())

    b = DualTrendStrategy()
    t = Backtester(strat=b, min_lookback=l)
    result_4 = t.run(y, **args)
    print(result_4)
    print(result_4.trade_log())

    plot_backtest_comparison([result, result_2, result_3, result_4])

if __name__ == "__main__":
    init_db()
    # update_last_30_days()
    # wfo_2()
    dailys()
