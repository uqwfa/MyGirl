import pandas as pd
import yaml
from pathlib import Path

from modules.data.core import DataCore


def load_config():
    config_path = Path("config.yaml")

    if not config_path.exists():
        print("config.yaml not found!")
        exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_seed_database(config: dict):
    db_path = Path(config['database']['path'])

    securities = [
        ("US6311011026", "NASDAQ 100", 72),
        ("DE0007164600", "SAP SE", 6)
    ]

    exchanges = [
        (6, "XETRA", "EUR"),
        (72, "NASDAQ Indizes", "PNT")
    ]

    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.executemany("INSERT OR IGNORE INTO securities (isin, name, exchange_id) VALUES (?, ?, ?);", securities)
        cursor.executemany("INSERT OR IGNORE INTO exchanges (id, name, currency) VALUES (?, ?, ?);", exchanges)


def load_active_trades():
    """
    load the "active_trades.json" file and return the data as a dictionary
    """

    import json
    active_trades_path = Path("active_trades.json")
    if not active_trades_path.exists():
        return {}

    with open(active_trades_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    path = Path(config['database']['path'])

    # init and seed db
    # from modules.database.database import init_database
    # init_database(config)
    # test_seed_database(config)

    # update data
    data_core = DataCore(config)
    data_core.update_data({"US6311011026": [72], "DE0007164600": [6]})

    active_trades = load_active_trades()

    from modules.simulation.manager import SecurityManager
    manager = SecurityManager(path)

    from modules.technical_analysis.core import TechnicalAnalysisCore
    t_core = TechnicalAnalysisCore()

    from modules.simulation.objects.order import Order

    for t in active_trades:
        isin, buy_date = t.get("isin"), t.get("buy_date")

        sec = manager.get_securities([isin])[0]

        order = Order(sec, 0.0, pd.Timestamp(t.get("buy_date")))
        stats = t_core.get_stats([order])[order]
        print(stats)


if __name__ == "__main__":
    main()
