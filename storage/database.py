"""
storage/database.py
-------------------
Database tables:
- securities: ISIN, name, ariva_id
- ohlcv: security_isin, date, open, high, low, close, volume
"""

import sqlite3
from config import DB_PATH
from contextlib import contextmanager
from typing import Generator
from pathlib import Path


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS securities (
    isin        TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,
    ariva_id    INTEGER NOT NULL
);
    
CREATE TABLE IF NOT EXISTS ohlcv (
    security_isin   TEXT    NOT NULL,
    date            DATE    NOT NULL,
    open            DECIMAL,
    high            DECIMAL,
    low             DECIMAL,
    close           DECIMAL NOT NULL,
    volume          INTEGER,
    PRIMARY KEY (security_isin, date),
    FOREIGN KEY (security_isin) REFERENCES securities (isin) ON DELETE CASCADE
);
    
CREATE INDEX IF NOT EXISTS idx_ohlcv_isin ON ohlcv (security_isin);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv (date);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    print(f"Initializing database at {db_path}")

    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)

    print("Database initialized successfully.")


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
