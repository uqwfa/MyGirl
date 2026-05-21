"""
backtesting/backtester.py
-------------------------
"""

import math
import pandas as pd
from dataclasses import dataclass, field
from datetime import date

from strategy.models import Direction
from strategy.strategies.base import BaseStrategy


@dataclass
class Trade:
    """Represents a single completed round-trip trade."""

    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    direction: Direction
    shares: float
    pnl: float
    return_pct: float
    fees: float
    reason: str

    @property
    def duration_days(self) -> int:
        try:
            return (self.exit_date - self.entry_date).days

        except TypeError:
            return 0


@dataclass
class BacktestResult:
    """Aggregated results and statistics from a completed backtest run."""

    strategy: str
    ticker: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float
    calmar_ratio: float | None
    win_rate: float | None
    num_trades: int
    profit_factor: float | None
    avg_trade_duration_days: float | None
    trades: list[Trade]
    equity_curve: pd.Series = field(compare=False, repr=False)
    total_fees: float = 0.0
    avg_fee: float = 0.0

    def __str__(self) -> str:
        def _f(v, fmt, unit=""):
            return (format(v, fmt) + unit) if v is not None else "—"

        try:
            cal_days = (self.end_date - self.start_date).days
        except TypeError:
            cal_days = "?"

        return "\n".join([
            "╔══ BacktestResult ═══════════════════════════════════════════",
            f"║  Strategy          : {self.strategy}",
            f"║  Ticker            : {self.ticker}",
            f"║  Period            : {self.start_date.strftime("%d.%m.%Y")} → {self.end_date.strftime("%d.%m.%Y")}  ({cal_days} cal. days)",
            f"║  Capital           : {self.initial_capital:>12,.2f}  →  {self.final_capital:,.2f}",
            f"║  Total Return      : {_f(self.total_return_pct, '+.2f', '%'):>10}",
            f"║  Ann. Return       : {_f(self.annualized_return_pct, '+.2f', '%'):>10}",
            f"║  Sharpe Ratio      : {_f(self.sharpe_ratio, '.3f'):>10}",
            f"║  Max Drawdown      : {_f(self.max_drawdown_pct, '.2f', '%'):>10}",
            f"║  Calmar Ratio      : {_f(self.calmar_ratio, '.3f'):>10}",
            f"║  Win Rate          : {_f(self.win_rate, '.1%') if self.win_rate is not None else '—':>10}",
            f"║  Profit Factor     : {_f(self.profit_factor, '.2f'):>10}",
            f"║  # Trades          : {_f(self.num_trades, 'd'):>10}",
            f"║  Avg Duration      : {_f(self.avg_trade_duration_days, '.1f', ' days'):>12}",
            f"║  Total Fees        : {_f(self.total_fees, '.2f'):>10}",
            f"║  Avg Fee           : {_f(self.avg_fee, '.2f'):>10}",
            "╚═════════════════════════════════════════════════════════════",
        ])

    def trade_log(self) -> str:
        """Return a formatted string listing every completed trade."""

        if not self.trades:
            return "No completed trades."

        header = f"{'#':<4} {'Dir':<5}  {'Entry Date':<12} {'Exit Date':<12}  " \
                 f"{'Entry':>10} {'Exit':>10}  {'PnL':>10}  {'Ret%':>8}  {'Days':>5}  {'Reason':>50}"
        sep = "─" * len(header)
        rows = [header, sep]

        for n, t in enumerate(self.trades, 1):
            sign = "+" if t.pnl >= 0 else ""

            rows.append(
                f"{n:<4} {t.direction.value:<5}  "
                f"{t.entry_date.strftime("%d.%m.%Y"):<12} {t.exit_date.strftime("%d.%m.%Y"):<12}  "
                f"{t.entry_price:>10.2f} {t.exit_price:>10.2f}  "
                f"{sign}{t.pnl:>9.2f}  "
                f"{sign}{t.return_pct * 100:>7.2f}%  "
                f"{t.duration_days:>5}d"
                f"{t.reason:>50}"
            )

        rows.append(sep)

        return "\n".join(rows)


class Backtester:
    """Event-driven backtester that replays a strategy over a price DataFrame."""

    def __init__(self, strat: BaseStrategy, *, initial_capital: float = 100_000.00, min_lookback: int = 0,
                 fee_fixed: float = 1.0, ticker: str = "unknown"):

        # todo: add slippage

        self.strat = strat
        self.initial_capital = initial_capital
        self.fee_fixed = fee_fixed
        self.ticker = ticker
        self.min_lookback = min_lookback

    def run(self, df: pd.DataFrame, *, start_date: date | None = None, end_date: date | None = None) -> BacktestResult:
        """
        Run the backtest over ``df`` and return a ``BacktestResult``.

        Since the prices don't change during the backtesting, the indicators can be computed only once at the
        start.

        If ``start_date`` is set to ``None``, the backtester starts at earliest available date of the input DataFrame
        with respect to the minimum lookback window. If the ``start_date`` is provided but falls within the first
        ``min_lookback`` bars of the data, a ValueError is raised since the strategy won't have enough historical data
        to compute indicators and generate signals for that date.

        If ``end_date`` is set to ``None``, the backtester runs until the latest available date in the input DataFrame.
        If ``end_date`` is provided but is beyond the last available date, a ValueError is raised.
        """

        df = df.copy().sort_index()
        n = len(df)

        if n < self.min_lookback:
            raise ValueError(f"DataFrame has {n} rows but at least {self.min_lookback} are required")

        if start_date is not None:
            start_date = pd.Timestamp(start_date)
        if end_date is not None:
            end_date = pd.Timestamp(end_date)

        # if self.min_lookback = 20, then the earliest_possible_start is at positional index 19 (the 20th row)
        earliest_possible_start: pd.Timestamp = df.index[(self.min_lookback - 1)]
        if start_date is not None and start_date < earliest_possible_start:
            raise ValueError(
                f"start_date {start_date.date().strftime("%d.%m.%Y")} is before the first available bar "
                f"({earliest_possible_start.date().strftime('%d.%m.%Y')})."
            )

        # if end_date is provided, it must not be beyond the last available bar in the data
        latest_possible_end: pd.Timestamp = df.index[-1]
        if end_date is not None and end_date > latest_possible_end:
            raise ValueError(
                f"end_date {end_date.date().strftime("%d.%m.%Y")} is beyond the last available bar "
                f"({latest_possible_end.date().strftime('%d.%m.%Y')})."
            )

        # if start_date is gap day, use next date; else use earliest possible (min_lookback - 1) index
        loop_start = (
            int(df.index.searchsorted(start_date, side="left"))
            if start_date is not None
            else (self.min_lookback - 1)
        )
        # if end_date is gap day, use prev date; else use latest possible (n - 1) index
        loop_end = (
            int(df.index.searchsorted(end_date, side="right")) - 1
            if end_date is not None
            else (n - 1)
        )

        # final sanity check to ensure loop indices are valid and in correct order
        if loop_end < loop_start:
            raise ValueError(
                f"end_date {df.index[loop_end].date().strftime('%d.%m.%Y')} is before the start_date "
                f"({df.index[loop_start].date().strftime('%d.%m.%Y')})."
            )

        # calculate the indicators once because prices don't change during backtesting
        df = self.strat.compute_indicators(df)

        closes = df["close"].to_numpy(dtype=float)
        dates = df.index.tolist()

        cash = self.initial_capital
        shares = 0.0
        entry_price = 0.0
        entry_date: date | None = None
        entry_direction = Direction.FLAT

        trades: list[Trade] = []
        eq_index = []
        eq_values: list[float] = []

        for i in range(loop_start, (loop_end + 1)):
            # up to, but not including index (i + 1)
            signal = self.strat.generate_signal(df[: i + 1], buy_date=entry_date)
            print(signal)

            if signal.direction == Direction.INVALID:
                raise ValueError(f"Strategy returned an INVALID signal at bar {i}: {signal.metadata.get('error')}")

            if shares != 0.0 and signal.direction == Direction.SHORT:
                trade, cash = self._close(
                    cash=cash, shares=shares, entry_price=entry_price, entry_date=entry_date, exit_price=closes[i],
                    exit_date=dates[i], direction=entry_direction, reason=signal.metadata.get("strongest_reason", "")
                )

                trades.append(trade)
                shares = 0.0
                entry_price = 0.0
                entry_date = None
                entry_direction = Direction.FLAT

            elif shares == 0.0 and signal.direction == Direction.LONG:
                res = self._open(cash, closes[i])

                if res is None:
                    print(f"Insufficient cash {cash:.2f} to buy at {closes[i]:.2f} — skipping bar {i}d")
                    eq_index.append(dates[i])
                    eq_values.append(cash + shares * closes[i])
                    continue

                shares, cash = res
                entry_price, entry_date = closes[i], dates[i]
                entry_direction = Direction.LONG

            eq_index.append(dates[i])
            eq_values.append(cash + shares * closes[i])

        if shares != 0.0:  # force close any open position at the end
            trade, cash = self._close(
                cash=cash, shares=shares, entry_price=entry_price, entry_date=entry_date, exit_price=closes[loop_end],
                exit_date=dates[loop_end], direction=entry_direction, reason="Simulation ended!"
            )

            trades.append(trade)
            if eq_values:
                eq_index.append(dates[loop_end])
                eq_values.append(cash)

        equity_curve = pd.Series(eq_values, index=eq_index, name="equity")

        return self._compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            final_capital=cash,
            start_date=dates[loop_start],
            end_date=dates[loop_end]
        )

    def _open(self, cash: float, price: float) -> tuple[float, float] | None:
        """Compute how many float shares to buy and return ``(shares, remaining_cash)``."""

        shares = (cash - self.fee_fixed) / price

        if shares <= 0:
            return None

        return shares, 0.0


    def _close(self, *, cash: float, shares: float, entry_price: float, entry_date: date, exit_price: float,
               exit_date: date, direction: Direction, reason: str) -> tuple[Trade, float]:
        """Settle a position and return ``(Trade,new_cash)``."""

        gross_proceeds = shares * exit_price
        exit_fee = self.fee_fixed

        net_proceeds = gross_proceeds - exit_fee
        new_cash = cash + net_proceeds

        total_fees = 2 * self.fee_fixed
        pnl = shares * (exit_price - entry_price) - total_fees

        entry_total_cost = shares * entry_price + self.fee_fixed
        return_pct = pnl / entry_total_cost

        t = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            shares=shares,
            pnl=pnl,
            return_pct=return_pct,
            fees=total_fees,
            reason=reason
        )

        return t, new_cash

    def _compute_metrics(self, *, trades: list[Trade], equity_curve: pd.Series, final_capital: float, start_date: date,
                         end_date: date) -> BacktestResult:
        """Compute all performance metrics and assemble a ``BacktestResult``."""

        total_ret = ((final_capital / self.initial_capital) - 1) * 100

        try:
            cal_days = max((end_date - start_date).days, 1)
        except TypeError:
            cal_days = 1

        # Annualized return (only meaningful for periods >= 30 days)
        ann_ret: float | None = None
        if cal_days >= 30:
            ann_ret = (
                (final_capital / self.initial_capital) ** (365.25 / cal_days) - 1
            ) * 100

        # Sharpe ratio (annualized, risk-free rate = 0)
        daily_rets = equity_curve.pct_change().dropna()
        sharpe: float | None = None
        if len(daily_rets) >= 2:
            std = float(daily_rets.std())
            if std > 1e-12:
                sharpe = float(daily_rets.mean() / std * math.sqrt(252))

        # Maximum drawdown
        rolling_peak = equity_curve.cummax()
        dd_series = (equity_curve - rolling_peak) / rolling_peak * 100.0
        max_dd = float(dd_series.min())

        # Calmar ratio
        calmar: float | None = None
        if ann_ret is not None and max_dd < -1e-12:
            calmar = ann_ret / abs(max_dd)

        # Trade-level statistics
        win_rate: float | None = None
        pf: float | None = None
        avg_dur: float | None = None
        total_fees: float = 0.0
        avg_fee: float = 0.0

        if trades:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            win_rate = len(winners) / len(trades)
            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))
            pf = (gross_profit / gross_loss) if gross_loss > 1e-12 else None
            avg_dur = sum(t.duration_days for t in trades) / len(trades)
            total_fees = sum(t.fees for t in trades)
            avg_fee = total_fees / len(trades)

        return BacktestResult(
            strategy=self.strat.__class__.__name__,
            ticker=self.ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=round(final_capital, 4),
            total_return_pct=round(total_ret, 4),
            annualized_return_pct=round(ann_ret, 4) if ann_ret is not None else None,
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            max_drawdown_pct=round(max_dd, 4),
            calmar_ratio=round(calmar, 4) if calmar is not None else None,
            win_rate=round(win_rate, 4) if win_rate is not None else None,
            num_trades=len(trades),
            profit_factor=round(pf, 4) if pf is not None else None,
            avg_trade_duration_days=round(avg_dur, 1) if avg_dur is not None else None,
            trades=trades,
            equity_curve=equity_curve,
            total_fees=total_fees,
            avg_fee=avg_fee
        )
