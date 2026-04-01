import pandas as pd
import numpy as np
import random
import math
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
    quantity: int
    pnl: float
    return_pct: float
    commission: float


@dataclass
class Position:
    security: Security
    quantity: int
    entry_price: float
    entry_date: pd.Timestamp
    current_max_price: float
    last_price: float
    buy_commission: float


class Backtester:
    def __init__(self,
                 strategy: BacktesterStrategy,
                 initial_capital: float = 100000.0,
                 max_positions: int = 5,
                 risk_free_rate: float = 0.0,
                 transaction_fee_pct: float = 0.0,
                 transaction_fee_fixed: float = 1.0,
                 silent: bool = False):

        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.risk_free_rate = risk_free_rate

        self.fee_pct = transaction_fee_pct
        self.fee_fixed = transaction_fee_fixed

        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

        self.silent = silent

    def run(self, securities: list[Security], start_date: pd.Timestamp, end_date: pd.Timestamp):
        all_dates = set()
        sec_data_map = {}

        if not self.silent: print("Pre-calculating indicators...")

        for sec in securities:
            # preprocess data
            processed_df = self.strategy.preprocess(sec.data)

            # filter dates early to reduce memory usage
            mask = (processed_df.index >= start_date) & (processed_df.index <= end_date)
            sliced_df = processed_df.loc[mask]

            if not sliced_df.empty:
                sec_data_map[sec.isin] = sliced_df
                all_dates.update(sliced_df.index)

        sorted_dates = sorted(list(all_dates))

        if not self.silent: print("Running simulation loop...")

        for current_date in sorted_dates:
            if self.cash < 0:
                print(f"Warning: Cash balance negative at {current_date}. Stopping simulation.")
                break

            # 1. update portfolio values first
            self._sync_position_prices(current_date, sec_data_map)

            # 2. check sells
            self._process_sells(current_date, sec_data_map)

            # 3. check buys (shuffle to ensure no bias)
            random.shuffle(securities)
            self._process_buys(current_date, securities, sec_data_map)

            # 4. record history
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
        current_pos_count = len(self.positions)
        if current_pos_count >= self.max_positions:
            return

        # allocate capital evenly based on current equity
        current_equity = self._get_current_equity()
        target_per_position = current_equity / self.max_positions

        for sec in securities:
            # stop if we already filled all available slots
            if len(self.positions) >= self.max_positions:
                break

            # skip if we already have a position in this security
            if sec.isin in self.positions:
                continue

            df = data_map.get(sec.isin)
            if df is None or date not in df.index:
                continue

            row = df.loc[date]

            if self.strategy.check_buy(row):
                # buy the target amount, but cannot exceed available cash
                buy_amount = min(target_per_position, self.cash * 0.99)

                price = row['close']
                est_fee = (price * self.fee_pct) + self.fee_fixed

                # check if we can afford at least 1 unit + fee
                if buy_amount < (price + est_fee):
                    continue

                self._execute_buy(sec, date, price, buy_amount)

    def _execute_sell(self, pos: Position, date: pd.Timestamp, price: float):
        gross_revenue = pos.quantity * price
        commission = (price * pos.quantity * self.fee_pct) + self.fee_fixed
        net_revenue = gross_revenue - commission

        self.cash += net_revenue

        total_commission = pos.buy_commission + commission

        cost_basis = pos.entry_price * pos.quantity
        pnl = net_revenue - cost_basis - pos.buy_commission
        ret_pct = (net_revenue / (cost_basis + pos.buy_commission)) - 1

        trade = Trade(
            isin=pos.security.isin,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            return_pct=ret_pct,
            commission=total_commission
        )

        self.trades.append(trade)
        del self.positions[pos.security.isin]

    def _execute_buy(self, sec: Security, date: pd.Timestamp, price: float, amount: float):
        # calculate how many whole units we can buy with the allocated amount after fees
        raw_quantity = (amount - self.fee_fixed) / (price * (1 + self.fee_pct))
        quantity = int(math.floor(raw_quantity))

        if quantity <= 0:
            return

        # calculate total cost including fees
        commission = (price * quantity * self.fee_pct) + self.fee_fixed
        total_cost = (price * quantity) + commission

        # safety check
        if total_cost > self.cash:
            quantity -= 1

            if quantity <= 0:
                return

            commission = (price * quantity * self.fee_pct) + self.fee_fixed
            total_cost = (price * quantity) + commission

        self.cash -= total_cost

        self.positions[sec.isin] = Position(
            security=sec,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            current_max_price=price,
            last_price=price,
            buy_commission=commission
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

        # todo: add open positions to trades_df with NaNs for exit data
        for pos in self.positions.values():
            open_trade = Trade(
                isin=pos.security.isin,
                entry_date=pos.entry_date,
                exit_date=pd.NaT,
                entry_price=pos.entry_price,
                exit_price=np.nan,
                quantity=pos.quantity,
                pnl=np.nan,
                return_pct=np.nan,
                commission=pos.buy_commission
            )

            self.trades.append(open_trade)

        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])

        total_return = (equity_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        metrics = self._calc_metrics(equity_df)

        if not self.silent:
            print("-" * 30)
            print("BACKTEST FINISHED")
            print(f"Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
            print(f"Total Return: {total_return * 100:.2f}%")
            print(f"Total Trades: {len(trades_df)}")
            print(f"Avg trade time: {(trades_df['exit_date'] - trades_df['entry_date']).dt.days.mean():.2f} days")
            print(f"Commission Paid: {trades_df['commission'].sum():.2f}")
            print(f"Avg Commission: {trades_df['commission'].mean():.2f}")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

            print("-" * 30)

        return equity_df, trades_df
