import pandas as pd
import calendar
import datetime
import requests
import math
import re
from bs4 import BeautifulSoup
from bs4.element import Tag


class Scraper:

    _base_url = "https://www.ariva.de/{isin}/kurse/historische-kurse"
    _params = {
        "go": 1,
        "boerse_id": None,
        "month": "",
        "clean_split": 1,
        "clean_bezug": 1
    }
    _table_selector = "div#pageHistoricQuotes.quoteContent table.line"
    _float_regex = re.compile(r'^\s*([\d.,]+)\s*([A-Za-z.]*)\s*')
    _unit_multipliers = {"Mrd": 1_000_000_000, "M": 1_000_000}

    @staticmethod
    def fetch(session: requests.Session, isin: str, exchange_id: int, year: int, month: int) -> pd.DataFrame:
        url, params = Scraper._build_url(isin, exchange_id, year, month)
        soup = Scraper._make_soup(session, url, params)

        if not soup:
            return pd.DataFrame()

        try:
            table = Scraper._find_table(soup, isin, exchange_id)
            data = Scraper._parse_table(table)

        except ValueError:
            return pd.DataFrame()

        except Exception as e:
            print(f"Error parsing data for ISIN {isin}: {e}")
            return pd.DataFrame()

        return data

    @staticmethod
    def min_date_available(session: requests.Session, isin: str, exchange_id: int) -> datetime.date | None:
        url, params = Scraper._build_url(isin, exchange_id, 0, 0)
        soup = Scraper._make_soup(session, url, params)

        if soup is None:
            return None

        dropdown = soup.select_one("select[name='month']")
        if not dropdown:
            return None

        options = dropdown.find_all("option")
        if not options:
            return None

        dateString = options[-1].get("value")
        try:
            return datetime.datetime.strptime(dateString, "%Y-%m-%d").date()

        except ValueError:
            return None

    @staticmethod
    def _make_soup(session: requests.Session, url: str, params: dict) -> BeautifulSoup | None:
        try:
            resp = session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")

        except requests.exceptions.RequestException:
            return None

    @staticmethod
    def _find_table(soup: BeautifulSoup, isin: str, exchange_id: int) -> Tag:
        table = soup.select_one(Scraper._table_selector)

        if not table:
            raise ValueError(f"Data table not found.")

        return table

    @staticmethod
    def _parse_table(table: Tag) -> pd.DataFrame:
        data = []

        rows = table.find_all("tr", class_="arrow0")

        for row in rows:
            quotes = row.find_all("td")

            try:
                data.append({
                    "date": Scraper._parse_date(quotes[0]),
                    "open": Scraper._parse_float(quotes[1]),
                    "high": Scraper._parse_float(quotes[2]),
                    "low": Scraper._parse_float(quotes[3]),
                    "close": Scraper._parse_float(quotes[4]),
                    "volume": Scraper._parse_float(quotes[-1])
                })

            except ValueError:
                continue

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data).set_index("date")

    @staticmethod
    def _parse_date(tag: Tag) -> datetime.date:
        date_str = tag.get_text(strip=True)
        return datetime.datetime.strptime(date_str, "%d.%m.%y").date()

    @staticmethod
    def _parse_float(tag: Tag) -> float:
        float_str = tag.get_text(strip=True)

        if not float_str or float_str == "-":
            return math.nan

        match = Scraper._float_regex.match(float_str)
        if not match:
            return math.nan

        value_str, unit = match.groups()
        cleaned_value_str = value_str.replace(".", "").replace(",", ".")

        return float(cleaned_value_str) * Scraper._unit_multipliers.get(unit, 1)

    @staticmethod
    def _build_url(isin: str, exchange_id: int, year: int, month: int) -> tuple[str, dict]:
        url = Scraper._base_url.format(isin=isin)

        params = Scraper._params.copy()
        params["boerse_id"] = exchange_id
        params["month"] = Scraper._create_month_param(year, month)

        return url, params

    @staticmethod
    def _create_month_param(year: int, month: int) -> str:
        if year == 0 and month == 0:
            return ""

        _, lastDay = calendar.monthrange(year, month)

        return f"{year:04d}-{month:02d}-{lastDay:02d}"
