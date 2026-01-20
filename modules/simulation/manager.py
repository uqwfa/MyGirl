import pandas as pd
import sqlite3
from functools import partial
from pathlib import Path

from modules.simulation.objects.security import Security


class SecurityManager:

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_securities(self, isins: list[str]) -> list[Security]:
        if not isins:
            return []

        query = """
            SELECT id, isin, name, linked_security_id
            FROM securities
            WHERE isin IN ({})
        """.format(", ".join("?" for _ in isins))

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=isins)

        except sqlite3.Error as e:
            print(f"Error fetching: {e}")
            return []

        securities = []
        for _, row in df.iterrows():
            loader = partial(self._fetch_prices, db_path=self.db_path)

            linked_sec = None
            if pd.notna(row["linked_security_id"]):
                linked_id = int(row["linked_security_id"])
                linked_sec = self._get_security_by_id(linked_id)

            sec = Security(
                id=int(row["id"]),
                isin=row["isin"],
                name=row["name"],
                loader=loader,
                linked_security=linked_sec
            )

            securities.append(sec)

        return securities

    def _get_security_by_id(self, sec_id: int) -> Security | None:
        query = "SELECT id, isin, name FROM securities WHERE id = ?"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(sec_id,))

                if df.empty:
                    return None

                row = df.iloc[0]
                loader = partial(self._fetch_prices, db_path=self.db_path)

                return Security(
                    id=int(row["id"]),
                    isin=row["isin"],
                    name=row["name"],
                    loader=loader,
                    linked_security=None
                )

        except sqlite3.Error:
            return None

    @staticmethod
    def _fetch_prices(id: int, db_path: Path) -> pd.DataFrame:
        query = """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE security_id = ?
        """

        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(id,))

            if df.empty:
                return pd.DataFrame()

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            return df

        except sqlite3.Error as e:
            print(f"Error fetching prices for security ID {id}: {e}")
            return pd.DataFrame()
