import sqlite3
from datetime import datetime
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "birdwatcher.db")

def get_connection():
    return sqlite3.connect(DB_FILE)

def initialize_db():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            species TEXT,
            confidence REAL,
            status TEXT CHECK(status IN ('accepted', 'review', 'not_a_bird')) NOT NULL,
            classified BOOLEAN NOT NULL DEFAULT 0
        )
        """)
        conn.commit()

def add_visit(filename, timestamp, species, confidence, status, classified=False):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO visits
        (filename, timestamp, species, confidence, status, classified)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, timestamp, species, confidence, status, int(classified)))
        conn.commit()

def update_status(filename, new_status):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("UPDATE visits SET status = ? WHERE filename = ?", (new_status, filename))
        conn.commit()

def delete_visit(filename):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM visits WHERE filename = ?", (filename,))
        conn.commit()

def get_visits_by_status(status):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM visits WHERE status = ? ORDER BY timestamp DESC", (status,))
        return c.fetchall()
