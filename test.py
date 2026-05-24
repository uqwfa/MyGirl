from datetime import date, timedelta

import optuna

from backtesting.backtester import Backtester, BacktestResult
from ingestion.scheduler import schedule_updates
from storage.database import init_db
from storage.models import Security, DateRange
from storage.repository import fetch_ohlcv
from strategy.optimizer.strategyWalkForwardOptimizer import WFO, WalkForwardWindow, WFOResult
from strategy.strategies.agileStrategy import SimpleAgileStrategy, simple_agile_space
from strategy.strategies.base import BaseStrategy
from strategy.strategies.book import BookStrategy
from strategy.strategies.book_v2 import BookStrategyV2
from strategy.optimizer.strategyOptimizer import StrategyOptimizer
from strategy.strategies.buyandhold import BuyAndHold
from strategy.strategies.george import DualTrendStrategy
import matplotlib.pyplot as plt
from typing import Callable, Any
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from strategy.strategies.macroStrategy import SimpleMacroStrategy, simple_macro_space
from strategy.strategies.rsiStrategy import SimpleRSIStrategy


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

def plot_wfo_comparison(
    wfo_results: list[WFOResult],
    figsize: tuple[int, int] = (18, 14),
    title: str = "Walk-Forward Optimization — Strategy Comparison",
) -> plt.Figure:
    """
    Plot a multi-panel comparison of WFO results across strategies.

    Panels:
      1. Combined OOS equity curves (stitched across windows, rebased to 100)
      2. Per-window OOS return % (grouped bar chart)
      3. Rolling Sharpe ratio (per window)
      4. Max drawdown per window (grouped bar chart)
      5. Summary metrics table (Sharpe, CAGR, MaxDD, Win Rate, Trades)

    Args:
        wfo_results: List of WFOResult objects, one per strategy.
        figsize:     Overall figure size.
        title:       Figure suptitle.

    Returns:
        The matplotlib Figure object (caller can call plt.show() or fig.savefig()).
    """
    COLORS = [
        "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
        "#FF9800", "#00BCD4", "#E91E63", "#8BC34A",
    ]
    WINDOW_ALPHA = 0.15

    # ── helpers ──────────────────────────────────────────────────────────────

    def _stitch_equity(wfo: WFOResult) -> pd.Series:
        """Concatenate per-window OOS equity curves, rebasing each to continue
        from where the previous window ended."""
        segments = []
        carry = 1.0
        for w in sorted(wfo.windows, key=lambda x: x.window_id):
            curve = w.oos_result.equity_curve.copy()
            # Normalise segment to [carry … carry * segment_return]
            normalised = carry * (curve / curve.iloc[0])
            segments.append(normalised)
            carry = float(normalised.iloc[-1])
        combined = pd.concat(segments)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        # Rebase so the whole curve starts at 100
        combined = 100.0 * combined / combined.iloc[0]
        return combined

    def _window_metric(wfo: WFOResult, attr: str) -> list[float]:
        return [
            getattr(w.oos_result, attr) or 0.0
            for w in sorted(wfo.windows, key=lambda x: x.window_id)
        ]

    def _pct_fmt(x, _):
        return f"{x:.1f}%"

    def _money_fmt(x, _):
        return f"{x:.0f}"

    # ── layout ───────────────────────────────────────────────────────────────

    fig = plt.figure(figsize=figsize, facecolor="#0f1117")
    fig.suptitle(title, fontsize=15, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.45,
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.93, bottom=0.06,
    )

    ax_equity  = fig.add_subplot(gs[0, :])       # full-width top
    ax_ret     = fig.add_subplot(gs[1, 0])
    ax_sharpe  = fig.add_subplot(gs[1, 1])
    ax_mdd     = fig.add_subplot(gs[2, 0])
    ax_table   = fig.add_subplot(gs[2, 1])

    _style_ax(ax_equity)
    _style_ax(ax_ret)
    _style_ax(ax_sharpe)
    _style_ax(ax_mdd)
    ax_table.axis("off")

    n_strats = len(wfo_results)
    max_windows = max(len(w.windows) for w in wfo_results)
    window_ids = list(range(1, max_windows + 1))
    bar_width = 0.8 / n_strats

    # ── 1. OOS equity curves ─────────────────────────────────────────────────

    for i, wfo in enumerate(wfo_results):
        color = COLORS[i % len(COLORS)]
        curve = _stitch_equity(wfo)
        ax_equity.plot(curve.index, curve.values, color=color,
                       linewidth=1.8, label=wfo.strategy, zorder=3)

        # shade individual OOS windows
        for w in wfo.windows:
            ax_equity.axvspan(
                w.test_start, w.test_end,
                alpha=WINDOW_ALPHA if i == 0 else 0,
                color=color, zorder=1,
            )

    # window boundary lines (from first strategy only)
    for w in wfo_results[0].windows:
        ax_equity.axvline(w.test_start, color="#444", linewidth=0.6,
                          linestyle="--", zorder=2)

    ax_equity.axhline(100, color="#555", linewidth=0.8, linestyle=":")
    ax_equity.set_ylabel("Equity (rebased to 100)", color="#aaa", fontsize=9)
    ax_equity.set_title("Out-of-Sample Equity Curves (stitched)", color="#ddd",
                        fontsize=10, pad=6)
    ax_equity.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_equity.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_equity.xaxis.get_majorticklabels(), rotation=30, ha="right",
             fontsize=7)
    ax_equity.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                     loc="upper left")

    # ── 2. Per-window return % ────────────────────────────────────────────────

    for i, wfo in enumerate(wfo_results):
        color = COLORS[i % len(COLORS)]
        returns = _window_metric(wfo, "total_return_pct")
        xs = [w + (i - n_strats / 2 + 0.5) * bar_width for w in window_ids[:len(returns)]]
        bars = ax_ret.bar(xs, returns, width=bar_width * 0.9,
                          color=color, alpha=0.85, label=wfo.strategy)
        for bar, val in zip(bars, returns):
            if abs(val) > 0.5:
                ax_ret.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.3 if val >= 0 else -0.8),
                    f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=6, color="white",
                )

    ax_ret.axhline(0, color="#666", linewidth=0.8)
    ax_ret.set_xticks(window_ids)
    ax_ret.set_xticklabels([f"W{w}" for w in window_ids], fontsize=8, color="#aaa")
    ax_ret.set_title("OOS Return % per Window", color="#ddd", fontsize=10, pad=6)
    ax_ret.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax_ret.set_ylabel("Return %", color="#aaa", fontsize=9)
    ax_ret.legend(fontsize=7, framealpha=0.2, labelcolor="white")

    # ── 3. Sharpe per window ──────────────────────────────────────────────────

    for i, wfo in enumerate(wfo_results):
        color = COLORS[i % len(COLORS)]
        sharpes = _window_metric(wfo, "sharpe_ratio")
        xs = [w + (i - n_strats / 2 + 0.5) * bar_width for w in window_ids[:len(sharpes)]]
        ax_sharpe.bar(xs, sharpes, width=bar_width * 0.9,
                      color=color, alpha=0.85, label=wfo.strategy)

    ax_sharpe.axhline(0, color="#666", linewidth=0.8)
    ax_sharpe.axhline(1, color="#888", linewidth=0.6, linestyle="--")
    ax_sharpe.set_xticks(window_ids)
    ax_sharpe.set_xticklabels([f"W{w}" for w in window_ids], fontsize=8, color="#aaa")
    ax_sharpe.set_title("Sharpe Ratio per Window", color="#ddd", fontsize=10, pad=6)
    ax_sharpe.set_ylabel("Sharpe", color="#aaa", fontsize=9)
    ax_sharpe.legend(fontsize=7, framealpha=0.2, labelcolor="white")

    # ── 4. Max drawdown per window ────────────────────────────────────────────

    for i, wfo in enumerate(wfo_results):
        color = COLORS[i % len(COLORS)]
        mdds = [-abs(v) for v in _window_metric(wfo, "max_drawdown_pct")]
        xs = [w + (i - n_strats / 2 + 0.5) * bar_width for w in window_ids[:len(mdds)]]
        ax_mdd.bar(xs, mdds, width=bar_width * 0.9,
                   color=color, alpha=0.85, label=wfo.strategy)

    ax_mdd.axhline(0, color="#666", linewidth=0.8)
    ax_mdd.set_xticks(window_ids)
    ax_mdd.set_xticklabels([f"W{w}" for w in window_ids], fontsize=8, color="#aaa")
    ax_mdd.set_title("Max Drawdown % per Window", color="#ddd", fontsize=10, pad=6)
    ax_mdd.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax_mdd.set_ylabel("Drawdown %", color="#aaa", fontsize=9)
    ax_mdd.legend(fontsize=7, framealpha=0.2, labelcolor="white")

    # ── 5. Summary table ──────────────────────────────────────────────────────

    col_labels = ["Strategy", "Total Ret%", "Ann. Ret%", "Sharpe", "MaxDD%",
                  "Win Rate", "Trades", "Profit F."]

    def _safe_avg(vals):
        clean = [v for v in vals if v is not None]
        return np.mean(clean) if clean else None

    def _fmt(v, pct=False, decimals=2):
        if v is None:
            return "—"
        return f"{v:.{decimals}f}{'%' if pct else ''}"

    rows = []
    for wfo in wfo_results:
        all_oos = [w.oos_result for w in wfo.windows]
        rows.append([
            wfo.strategy,
            _fmt(wfo.total_return_pct, pct=True, decimals=1),
            _fmt(_safe_avg([r.annualized_return_pct for r in all_oos]), pct=True, decimals=1),
            _fmt(_safe_avg([r.sharpe_ratio for r in all_oos]), decimals=2),
            _fmt(-abs(_safe_avg([r.max_drawdown_pct for r in all_oos]) or 0), pct=True, decimals=1),
            _fmt(_safe_avg([r.win_rate for r in all_oos if r.win_rate]), pct=True, decimals=1),
            str(sum(r.num_trades for r in all_oos)),
            _fmt(_safe_avg([r.profit_factor for r in all_oos if r.profit_factor]), decimals=2),
        ])

    tbl = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1d27" if row % 2 == 0 else "#13151e")
        cell.set_edgecolor("#333")
        cell.set_text_props(color="white" if row > 0 else "#aaa")
        if row == 0:
            cell.set_facecolor("#0d0f17")
            cell.set_text_props(fontweight="bold", color="#90CAF9")

    ax_table.set_title("Aggregate OOS Summary", color="#ddd", fontsize=10, pad=6)

    return fig


def _style_ax(ax: plt.Axes) -> None:
    """Apply dark-theme styling to an axes."""
    ax.set_facecolor("#13151e")
    ax.tick_params(colors="#888", labelsize=8)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#aaa")
    ax.grid(axis="y", color="#1e2130", linewidth=0.7, linestyle="-")


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


def dailys():
    today = date.today()
    d2 = DateRange(start=date(2025, 4, 28), end=today)
    x = fetch_ohlcv("US6311011026", d2)
    b = BookStrategy()
    print(f"\nCurrent signal:\n{b.run(x, buy_date=date(2025, 4, 28))}")

    dt, levels = b.compute_price_levels(x, as_intervals=True, buy_date=date(2025, 4, 28))
    if dt is not None:
        print(f"\nPrice Levels for {dt.strftime("%d.%m.%Y")} with current price {x["close"].iloc[-1]}:\n{levels}")


def compare_wfos(strats: list[type[BaseStrategy]], *, min_lookback: int = 300):
    """"""

    today = date.today()
    d = DateRange(start=date(2019, 1, 1), end=today)
    y = fetch_ohlcv("US6311011026", d)

    results = []
    for strat_class in strats:
        o = WFO(
            strategy_class=strat_class,
            initial_capital=100_000.00,
            fee_fixed=1.0,
            min_lookback=min_lookback,
            min_trades=1,
            verbose=False
        )

        res = o.run(
            y,
            train_years=4,
            test_years=1,
            step_months=12,
            n_trials=200,
            maximize=True
        )

        print(res)
        results.append(res)

    plot_wfo_comparison(results, title="Strategy Comparison — WFO OOS")
    plt.show()

if __name__ == "__main__":
    init_db()
    # update_last_30_days()
    # wfo_2()
    # dailys()

    compare_wfos([DualTrendStrategy, BuyAndHold, SimpleMacroStrategy, SimpleAgileStrategy, SimpleRSIStrategy])
