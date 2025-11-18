# file: database.py

import sqlite3
import os
import logging

CANONICAL_EVENT_COLUMNS = {
    "id", "vehicle_identification_number", "user_id",
    "time", "status", "img_drowsiness", "img_path"
}

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self._ensure_schema()
            logging.info(f"Database initialized at {db_path}")
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            self.conn = None

    def _ensure_schema(self):
        if not self.conn:
            return
        cur = self.conn.cursor()

        # Create users (kept minimal for compatibility; not required for events)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT UNIQUE,
                ear_threshold REAL
            )
        """)

        # Create events if not exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='drowsiness_events'")
        exists = cur.fetchone() is not None

        if not exists:
            logging.info("[SCHEMA] Creating drowsiness_events (canonical)")
            cur.executescript("""
            CREATE TABLE drowsiness_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_identification_number TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                img_drowsiness BLOB,
                img_path TEXT
            );
            CREATE INDEX idx_events_time ON drowsiness_events(time);
            CREATE INDEX idx_events_user ON drowsiness_events(user_id);
            """)
            self.conn.commit()
            return

        # If exists, verify columns; rebuild if needed
        cur.execute("PRAGMA table_info(drowsiness_events)")
        cols = {row[1] for row in cur.fetchall()}
        if CANONICAL_EVENT_COLUMNS.issubset(cols) and len(cols) == len(CANONICAL_EVENT_COLUMNS):
            logging.info("[SCHEMA] drowsiness_events already canonical")
            return

        logging.info("[SCHEMA] Migrating drowsiness_events to canonical schema (preserve rows)")
        cur.executescript("""
        PRAGMA foreign_keys=off;

        CREATE TABLE drowsiness_events_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_identification_number TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            img_drowsiness BLOB,
            img_path TEXT
        );

        INSERT INTO drowsiness_events_new (vehicle_identification_number, user_id, time, status, img_drowsiness, img_path)
        SELECT
            COALESCE(vehicle_identification_number, 'UNKNOWN'),
            COALESCE(user_id, 0),
            -- Prefer existing 'time' if present; else now
            COALESCE(time, datetime('now','localtime')),
            LOWER(COALESCE(status, 'drowsy')),
            img_drowsiness,  -- copies existing bytes or text as-is
            img_path
        FROM drowsiness_events;

        DROP TABLE drowsiness_events;
        ALTER TABLE drowsiness_events_new RENAME TO drowsiness_events;

        CREATE INDEX IF NOT EXISTS idx_events_time ON drowsiness_events(time);
        CREATE INDEX IF NOT EXISTS idx_events_user ON drowsiness_events(user_id);

        PRAGMA foreign_keys=on;
        """)
        self.conn.commit()

    def get_connection(self):
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")