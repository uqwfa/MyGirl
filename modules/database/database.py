import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS securities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    isin TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    exchange_id INTEGER NOT NULL,
    last_updated DATE,
    FOREIGN KEY (exchange_id) REFERENCES exchanges(id)
);
    
CREATE TABLE IF NOT EXISTS exchanges (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    currency TEXT
);
    
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER,
    exchange_id INTEGER,
    date DATE NOT NULL,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL NOT NULL,
    volume INTEGER,
    FOREIGN KEY (security_id) REFERENCES securities(id) ON DELETE CASCADE,
    FOREIGN KEY (exchange_id) REFERENCES exchanges(id) ON DELETE CASCADE,
    UNIQUE (security_id, exchange_id, date)
);
"""


def init_database(config: dict) -> bool:
    db_path = Path(config['database']['path'])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(SCHEMA)

        return True

    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        return False
