import pandas as pd
import numpy as np
import random
from dataclasses import dataclass

from modules.simulation.objects.strategy import BacktesterStrategy, PositionContext
from modules.simulation.objects.security import Security


@dataclass
class Trade:
    isin: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float


@dataclass
class Position:
    security: Security
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    current_max_price: float
    last_price: float


class Backtester:
    def __init__(self, strategy: BacktesterStrategy, initial_capital: float = 100000.0, risk_free_rate: float = 0.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    def run(self, securities: list[Security], start_date: pd.Timestamp, end_date: pd.Timestamp):
        all_dates = set()
        sec_data_map = {}

        print("Pre-calculating indicators...")
        for sec in securities:
            raw_df = sec.data
            processed_df = self.strategy.preprocess(raw_df)

            mask = (processed_df.index >= start_date) & (processed_df.index <= end_date)
            sliced_df = processed_df.loc[mask]

            if not sliced_df.empty:
                sec_data_map[sec.isin] = sliced_df
                all_dates.update(sliced_df.index)

        sorted_dates = sorted(list(all_dates))

        print("Running simulation loop...")
        for current_date in sorted_dates:
            self._sync_position_prices(current_date, sec_data_map)
            self._process_sells(current_date, sec_data_map)
            random.shuffle(securities)  # Shuffle to avoid bias in buy order
            self._process_buys(current_date, securities, sec_data_map)
            self._record_equity(current_date)

        return self._generate_report()

    def _sync_position_prices(self, date: pd.Timestamp, data_map: dict):
        for isin, pos in self.positions.items():
            df = data_map.get(isin)

            if df is not None and date in df.index:
                current_close = df.at[date, 'close']
                pos.last_price = current_close

                if current_close > pos.current_max_price:
                    pos.current_max_price = current_close

    def _process_sells(self, date: pd.Timestamp, data_map: dict):
        for isin in list(self.positions.keys()):
            pos = self.positions[isin]
            df = data_map.get(isin)

            if df is None or date not in df.index:
                continue

            row = df.loc[date]

            context = PositionContext(
                entry_price=pos.entry_price,
                current_maximum_price=pos.current_max_price,
                days_held=(date - pos.entry_date).days
            )

            if self.strategy.check_sell(row, context):
                self._execute_sell(pos, date, row['close'])

    def _process_buys(self, date: pd.Timestamp, securities: list[Security], data_map: dict):
        max_positions = 5
        current_pos_count = len(self.positions)

        if current_pos_count >= max_positions:
            return

        # If universe is 2, slots = 2 => 50%. If universe is 10, slots = 5 => 20%
        total_slots = min(len(securities), max_positions)
        target_allocation = self._get_current_equity() / total_slots

        for sec in securities:
            if len(self.positions) >= max_positions:
                break

            if sec.isin in self.positions:
                continue

            df = data_map.get(sec.isin)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]

            if self.strategy.check_buy(row):
                buy_amount = min(target_allocation, self.cash * 0.99)

                if buy_amount > 1:
                    self._execute_buy(sec, date, row['close'], buy_amount)

    def _execute_sell(self, pos: Position, date: pd.Timestamp, price: float):
        revenue = pos.quantity * price
        self.cash += revenue

        pnl = revenue - (pos.quantity * pos.entry_price)
        ret_pct = (price / pos.entry_price) - 1

        trade = Trade(
            isin=pos.security.isin,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            return_pct=ret_pct
        )

        self.trades.append(trade)
        del self.positions[pos.security.isin]

    def _execute_buy(self, sec: Security, date: pd.Timestamp, price: float, amount: float):
        quantity = amount / price
        self.cash -= amount

        self.positions[sec.isin] = Position(
            security=sec,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            current_max_price=price,
            last_price=price
        )

    def _get_current_equity(self) -> float:
        equity = self.cash

        for pos in self.positions.values():
            equity += pos.quantity * pos.last_price

        return equity

    def _record_equity(self, date: pd.Timestamp):
        self.equity_curve.append({
            "date": date,
            "equity": self._get_current_equity(),
            "cash": self.cash
        })

    def _calc_metrics(self, df: pd.DataFrame) -> dict:
        daily_returns = df["equity"].pct_change().dropna()
        negative_daily_returns = daily_returns[daily_returns < 0]

        if len(daily_returns) > 1:
            daily_rf = self.risk_free_rate / 252
            excess_returns = daily_returns - daily_rf

            std_dev = daily_returns.std()
            neg_std_dev = negative_daily_returns.std()

            sharpe_ratio = (excess_returns.mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0
            sortino_ratio = (excess_returns.mean() / neg_std_dev) * np.sqrt(252) if neg_std_dev != 0 else 0

        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio
        }

    def _generate_report(self):
        if not self.equity_curve:
            return pd.DataFrame(), pd.DataFrame()

        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])

        total_return = (equity_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        metrics = self._calc_metrics(equity_df)

        print("-" * 30)
        print("BACKTEST FINISHED")
        print(f"Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Total Trades: {len(trades_df)}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        print("-" * 30)

        return equity_df, trades_df
