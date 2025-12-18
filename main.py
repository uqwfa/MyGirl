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


def main():
    config = load_config()
    path = Path(config['database']['path'])

    # from modules.database.database import init_database
    # init_database(config)
    # test_seed_database(config)

    #data_core = DataCore(config)
    #data_core.update_data({"US6311011026": [72], "DE0007164600": [6]})

    from modules.simulation.manager import SecurityManager
    manager = SecurityManager(path)
    secs = manager.get_securities(["US6311011026"])

    from modules.technical_analysis.core import TechnicalAnalysisCore
    t_core = TechnicalAnalysisCore()
    stats = t_core.get_stats(secs)
    print(stats)


if __name__ == "__main__":
    main()
