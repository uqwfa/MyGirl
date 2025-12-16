import pandas as pd
import datetime
import sqlite3
from pathlib import Path

from modules.data.proxy import Proxy


class Utils:

    @staticmethod
    def get_proxies(db_path: Path, isins: list[str]) -> list[Proxy]:
        if not isins:
            return []

        query = """
            SELECT s.id, s.isin, s.name, s.exchange_id, p_max.last_date
            FROM securities s
            LEFT JOIN (
                SELECT security_id, exchange_id, MAX(date) AS last_date
                FROM prices
                GROUP BY security_id, exchange_id
            ) p_max ON s.id = p_max.security_id AND s.exchange_id = p_max.exchange_id
            WHERE s.isin IN ({})
        """.format(", ".join("?" for _ in isins))

        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn, params=isins, parse_dates=["last_date"])

        except sqlite3.Error as e:
            print(f"Error fetching proxies: {e}")
            return []

        return [
            Proxy(
                id=row["id"],
                isin=row["isin"],
                name=row["name"],
                exchange_id=row["exchange_id"],
                last_date=row["last_date"]
            )
            for _, row in df.iterrows()
        ]

    @staticmethod
    def store_prices(conn: sqlite3.Connection, sec_id: int, ex_id: int, df: pd.DataFrame) -> None:
        if df.empty:
            return

        store_df = df.reset_index().copy()
        store_df['security_id'] = sec_id
        store_df['exchange_id'] = ex_id

        cols = ['security_id', 'exchange_id', 'date', 'open', 'high', 'low', 'close', 'volume']

        store_df['date'] = store_df['date'].astype(str)
        records = store_df[cols].to_numpy().tolist()

        try:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT OR REPLACE INTO prices 
                        (security_id, exchange_id, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                records
            )

            cursor.execute(
                """UPDATE securities SET last_updated = ? WHERE id = ?""",
                (datetime.datetime.now(), sec_id)
            )
            conn.commit()

        except sqlite3.Error as e:
            print(f"DB Error storing prices for sec_id {sec_id}: {e}")
            conn.rollback()
