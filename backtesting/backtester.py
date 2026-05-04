"""
backtesting/backtester.py
-------------------------
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date

from strategy.models import Direction
from strategy.strategies.base import BaseStrategy


@dataclass
class Trade:
    """"""

    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    direction: Direction
    shares: float
    pnl: float
    return_pct: float
    fees: float

    @property
    def duration_days(self) -> int:
        try:
            return (self.exit_date - self.entry_date).days

        except TypeError:
            return 0


@dataclass
class BacktestResult:
    """"""

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
    equity_curve: pd.Series
    total_fees: float
    avg_fee: float

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
                 f"{'Entry':>10} {'Exit':>10}  {'PnL':>10}  {'Ret%':>8}  {'Days':>5}"
        sep = "─" * len(header)
        rows = [header, sep]

        for n, t in enumerate(self.trades, 1):
            sign = "+" if t.pnl >= 0 else ""

            rows.append(
                f"{n:<4} {t.direction.value:<5}  "
                f"{t.entry_date.strftime("%d.%m.%Y"):<12} {t.exit_date.strftime("%d.%m.%Y"):<12}  "
                f"{t.entry_price:>10.2f} {t.exit_price:>10.2f}  "
                f"{sign}{t.pnl:>9.2f}  "
                f"{sign}{t.return_pct:>7.2f}%  "
                f"{t.duration_days:>5}d"
            )

        rows.append(sep)

        return "\n".join(rows)


class Backtester:
    """"""

    def __init__(self, strat: BaseStrategy, *, initial_capital: float = 10_000.00, min_lookback: int = 0,
                 fee_fixed: float = 1.0, ticker: str = "unknown"):

        self.strat = strat
        self.initial_capital = initial_capital
        self.fee_fixed = fee_fixed
        self.ticker = ticker
        self.min_lookback = min_lookback

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """"""

        df = df.copy().sort_index()
        n = len(df)

        required = self.min_lookback
        if n < required:
            raise ValueError(f"DataFrame has {n} rows but {required} are required")

        closes = df["close"].to_numpy(dtype=float)
        dates = df.index.tolist()

        cash = self.initial_capital
        shares = 0.0
        entry_price = 0.0
        entry_date: date | None = None

        trades: list[Trade] = []
        eq_index = [dates[self.min_lookback]]
        eq_values = [self.initial_capital]

        for i in range(self.min_lookback, n - 1):
            signal = self.strat.run(df.iloc[:i+1])

            if signal.direction == Direction.INVALID:
                raise ValueError(f"The signal at index {i} is invalid: {signal.metadata.get('error')}")

            elif shares != 0.0 and signal.direction == Direction.SHORT:  # close position
                trade, cash = self._close(cash=cash, shares=shares, entry_price=entry_price, entry_date=entry_date,
                                          exit_price=closes[i], exit_date=dates[i])

                trades.append(trade)
                shares = 0.0

            elif shares == 0.0 and signal.direction == Direction.LONG:  # open new position
                res = self._open(cash, closes[i])
                if res is None:
                    print(f"cant buy")
                    continue

                shares, cash = res
                entry_price, entry_date = closes[i], dates[i]

            eq_index.append(dates[i])
            eq_values.append((cash + shares * closes[i]))

        if shares != 0.0:  # force close any open position at the end
            trade, cash = self._close(cash=cash, shares=shares, entry_price=entry_price, entry_date=entry_date,
                                      exit_price=closes[-1], exit_date=dates[-1])

            trades.append(trade)
            eq_values[-1] = cash

        equity_curve = pd.Series(eq_values, index=eq_index, name="equity")

        return self._compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            final_capital=cash,
            start_date=dates[self.min_lookback],
            end_date=dates[-1]
        )

    def _open(self, cash: float, price: float) -> tuple[float, float] | None:
        """"""

        raw_quantity = ((cash - self.fee_fixed) / price)
        shares = int(math.floor(raw_quantity))

        if shares <= 0:  # cant be a single share
            print(f"cash {cash} is less than one share costs {price}")
            return None

        total_cost = (price * shares) + self.fee_fixed
        remaining_cash = cash - total_cost

        return shares, remaining_cash

    def _close(self, *, cash: float, shares: float, entry_price: float, entry_date: date, exit_price: float,
               exit_date: date) -> tuple[Trade, float]:
        """"""

        gross_proceeds = shares * exit_price
        total_fees = 2 * self.fee_fixed  # fee for opening and closing
        net_proceeds = gross_proceeds - total_fees  # cash got for selling

        pnl = (shares * exit_price) - (shares * entry_price) - total_fees

        buy_costs = shares * entry_price  # price paid for buying
        return_pct = (net_proceeds / (buy_costs + total_fees)) - 1

        new_cash = cash + net_proceeds

        t = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=Direction.LONG,
            shares=shares,
            pnl=pnl,
            return_pct=return_pct,
            fees=total_fees
        )

        return t, new_cash

    def _compute_metrics(self, *, trades: list[Trade], equity_curve: pd.Series, final_capital: float, start_date: date,
                         end_date: date) -> BacktestResult:
        """"""

        total_ret = ((final_capital / self.initial_capital) - 1) * 100

        try:
            cal_days = max((end_date - start_date).days, 1)

        except TypeError:
            cal_days = 1

        # annualized return
        ann_ret: float | None = None
        if cal_days >= 30:
            ann_ret = (
                (final_capital / self.initial_capital) ** (365.25 / cal_days) - 1
            ) * 100

        # Sharpe ratio
        daily_rets = equity_curve.pct_change().dropna()
        sharpe: float | None = None
        if len(daily_rets) >= 2:
            std = float(daily_rets.std())
            if std > 1e-12:
                sharpe = float(daily_rets.mean() / std * math.sqrt(252))

        # drawdown
        rolling_peak = equity_curve.cummax()
        dd_series = (equity_curve - rolling_peak) / rolling_peak * 100.0
        max_dd = float(dd_series.min())

        # Calmer ratio
        calmar: float | None = None
        if ann_ret is not None and max_dd < -1e-12:
            calmar = ann_ret / abs(max_dd)

        # trade statistics
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
