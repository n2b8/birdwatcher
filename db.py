# db.py
import sqlite3
from datetime import datetime

DB_FILE = "birdwatcher.db"

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
            motion_score INTEGER,
            status TEXT CHECK(status IN ('accepted', 'review', 'not_a_bird')) NOT NULL
        )
        """)
        conn.commit()

def add_visit(filename, timestamp, species, confidence, motion_score, status):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO visits
        (filename, timestamp, species, confidence, motion_score, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, timestamp, species, confidence, motion_score, status))
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

def get_not_a_bird_count():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM visits WHERE status = 'not_a_bird'")
        return c.fetchone()[0]
