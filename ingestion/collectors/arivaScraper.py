"""
ingestion/collectors/arivaScraper.py
------------------------------------
"""

import calendar
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from datetime import datetime, date
from itertools import chain

from ingestion.collectors.baseCollector import Collector
from storage.models import DateRange


class ArivaScraper(Collector):
    """
    An implementation of the Collector base class.
    Uses the ariva.de webpage to scrape OHLCV data.
    """

    _BASE_URL = "https://www.ariva.de/{isin}/kurse/historische-kurse"
    _PARAMS = {
        "go": 1,
        "boerse_id": None,
        "month": "",
        "clean_split": 1,
        "clean_bezug": 1
    }
    _TABLE_SELECTOR = "div#pageHistoricQuotes.quoteContent table.line"
    _FLOAT_REGEX = re.compile(r'^\s*([\d.,]+)\s*([A-Za-z.]*)\s*')
    _UNIT_MULTIPLIERS = {"Mrd": 1_000_000_000, "M": 1_000_000}

    def fetch(self, isin: str, date_range: DateRange, ariva_id: int = None, **kwargs) -> pd.DataFrame:
        """Fetch OHLCV data for a given ISIN over the supplied DateRange."""

        if ariva_id is None:
            print(f"Ariva ID not provided for {isin}, skipping.")
            return pd.DataFrame()

        with requests.Session() as session:
            monthly_rows = [
                ArivaScraper._scrape_month(session, isin, ariva_id, year, month)
                for year, month in ArivaScraper._get_months(date_range)
            ]

        all_rows = list(chain.from_iterable(monthly_rows))

        if not all_rows:
            print(f"No data returned for {isin} in range {date_range}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).set_index("date").sort_index()
        df = df[df.index.to_series().between(date_range.start, date_range.end)]

        if df.empty:
            print(f"No data returned for {isin} in range {date_range}.")

        return df

    @staticmethod
    def _scrape_month(session: requests.Session, isin: str, ariva_id: int, year: int, month: int) -> list[dict]:
        """Scrape one calendar month of OHLCV rows. Returns an empty list on any failure."""

        url, params = ArivaScraper._build_url(isin, ariva_id, year, month)

        try:
            soup = ArivaScraper._make_soup(session, url, params)
            table = ArivaScraper._find_table(soup)
            return ArivaScraper._parse_table(table)

        except requests.exceptions.RequestException as e:
            print(f"Network error fetching {isin} {year}-{month:02d}: {e}")

        except ValueError as e:
            print(f"Parse error for {isin} {year}-{month:02d}: {e}")

        except Exception as e:
            print(f"Unexpected error for {isin} {year}-{month:02d}: {e}")

        return []

    @staticmethod
    def _make_soup(session: requests.Session, url: str, params: dict) -> BeautifulSoup:
        """Perform the HTTP request and return a parsed BeautifulSoup object."""

        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    @staticmethod
    def _find_table(soup: BeautifulSoup) -> Tag:
        """Locate the OHLCV table in the page, raising ValueError if absent."""

        table = soup.select_one(ArivaScraper._TABLE_SELECTOR)

        if not table:
            raise ValueError("Data table not found in page.")

        return table

    @staticmethod
    def _parse_table(table: Tag) -> list[dict]:
        """Parse all data rows from the OHLCV table."""

        data = []

        for row in table.find_all("tr", class_="arrow0"):
            cols = row.find_all("td")

            parsed_date = ArivaScraper._parse_date(cols[0])
            if parsed_date is None:
                print(f"Skipping row with unparseable date: {cols[0].get_text(strip=True)!r}")
                continue

            data.append({
                "date": parsed_date,
                "open": ArivaScraper._parse_float(cols[1]),
                "high": ArivaScraper._parse_float(cols[2]),
                "low": ArivaScraper._parse_float(cols[3]),
                "close": ArivaScraper._parse_float(cols[4]),
                "volume": ArivaScraper._parse_float(cols[-1])
            })

        return data

    @staticmethod
    def _parse_date(date_tag: Tag) -> date | None:
        """Parse a date string (DD.MM.YY) from a table cell."""

        try:
            return datetime.strptime(date_tag.get_text(strip=True), "%d.%m.%y").date()

        except ValueError:
            return None

    @staticmethod
    def _parse_float(float_tag: Tag) -> float | None:
        """Parse a localized float string from a table cell."""

        float_str = float_tag.get_text(strip=True)

        if not float_str or float_str == "-":
            return None

        match = ArivaScraper._FLOAT_REGEX.match(float_str)
        if not match:
            return None

        value_str, unit = match.groups()
        cleaned_value_str = value_str.replace(".", "").replace(",", ".")

        try:
            return float(cleaned_value_str) * ArivaScraper._UNIT_MULTIPLIERS.get(unit, 1)

        except ValueError:
            return None

    @staticmethod
    def _get_months(date_range: DateRange) -> list[tuple[int, int]]:
        """Return a list of (year, month) tuples covering the full DateRange."""

        months = []
        year, month = date_range.start.year, date_range.start.month
        end_year, end_month = date_range.end.year, date_range.end.month

        while (year, month) <= (end_year, end_month):
            months.append((year, month))
            if month == 12:
                year, month = year + 1, 1
            else:
                month += 1

        return months

    @staticmethod
    def _build_url(isin: str, ariva_id: int, year: int, month: int) -> tuple[str, dict]:
        """Construct the request URL and parameter dict for a given month."""

        url = ArivaScraper._BASE_URL.format(isin=isin)

        params = ArivaScraper._PARAMS.copy()
        params["boerse_id"] = ariva_id
        params["month"] = ArivaScraper._create_month_param(year, month)

        return url, params

    @staticmethod
    def _create_month_param(year: int, month: int) -> str:
        """Format the 'month' query parameter expected by Ariva (YYYY-MM-DD of last day)."""

        if year == 0 and month == 0:
            return ""

        _, last_day = calendar.monthrange(year, month)

        return f"{year:04d}-{month:02d}-{last_day:02d}"
