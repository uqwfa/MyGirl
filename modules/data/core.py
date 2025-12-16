import pandas as pd
import datetime
import requests
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modules.data.scraper import Scraper
from modules.data.utils import Utils
from modules.data.proxy import Proxy


class DataCore:

    _start_date = datetime.date(1990, 1, 1)
    _relative_batch_size = 0.1

    def __init__(self, config: dict):
        self.db_path = Path(config['database']['path'])
        self.max_workers = int(config['data']['max_workers'])
        self.retry_attempts = int(config['data']['retry_attempts'])
        self.backoff_factor = float(config['data']['backoff_factor'])

    def update_data(self, args: dict[str, list[int]]) -> bool:
        """
        Example args: {"US6311011026": [72, 123], "US5949181045": [72]}
        """

        print("Starting data update...")

        isins = list(args.keys())
        proxies = Utils.get_proxies(self.db_path, isins)

        if not proxies:
            print("No proxies found to process.")
            return True

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._process_proxy, proxy): proxy for proxy in proxies}

                for future in as_completed(futures):
                    proxy = futures[future]

                    try:
                        result = future.result()
                        status = "Success" if result else "Skipped/Failed"
                        print(f"[{status}] ISIN {proxy.isin} (Ex: {proxy.exchange_id})")

                    except Exception as exc:
                        print(f"Error processing ISIN {proxy.isin}: {exc}")

        except Exception as e:
            print(f"Data update failed: {e}")
            return False

        print("Data update process finished.")
        return True

    def _retry_func(self, func, *args, **kwargs):
        for att in range(1, self.retry_attempts + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                if att == self.retry_attempts:
                    raise e

                sleep_time = self.backoff_factor * (2 ** (att - 1))
                time.sleep(sleep_time)

        return None

    def _process_proxy(self, proxy: Proxy) -> bool:
        last_date = proxy.last_data.date() if not pd.isna(proxy.last_date) else None

        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })

            if last_date is None:
                min_date = Scraper.min_date_available(session, proxy.isin, proxy.exchange_id)

                if min_date is None:
                    return False

                start = max(min_date, self._start_date)

            else:
                is_recent = (datetime.date.today() - last_date).days <= 22
                start = datetime.date.min if is_recent else last_date

            with sqlite3.connect(self.db_path, timeout=30) as conn:
                self._fetch_and_store(session, proxy, conn, start)

        return True

    def _fetch_and_store(self, session: requests.Session, proxy: Proxy, conn: sqlite3.Connection, start_date: datetime.date):
        if start_date == datetime.date.min:
            df = self._retry_func(Scraper.fetch, session, proxy.isin, proxy.exchange_id, 0, 0)

            if not df.empty:
                Utils.store_prices(conn, proxy.id, proxy.exchange_id, df)

            return

        end_date = datetime.date.today()
        year, month = start_date.year, start_date.month

        buffer = []

        while (year, month) <= (end_date.year, end_date.month):
            df = self._retry_func(Scraper.fetch, session, proxy.isin, proxy.exchange_id, year, month)

            if not df.empty:
                buffer.append(df)

            month += 1
            if month > 12:
                if buffer:
                    full_df = pd.concat(buffer)
                    Utils.store_prices(conn, proxy.id, proxy.exchange_id, full_df)

                month = 1
                year += 1

        if buffer:
            full_df = pd.concat(buffer)
            Utils.store_prices(conn, proxy.id, proxy.exchange_id, full_df)
