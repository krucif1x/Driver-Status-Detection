import sqlite3
import logging
from typing import Optional, Iterable, Any

class DrowsinessEventRepository:
    """
    Persistence for canonical drowsiness events.
    Columns: vin, user_id, time, status, img_drowsiness (BLOB), img_path, duration, value
    """
    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection

    def add_event_row(
        self,
        vehicle_identification_number: str,
        user_id: int,
        time: str,
        status: str,
        img_drowsiness: Optional[bytes],
        img_path: Optional[str],
        duration: float = 0.0,  # <--- Added
        value: float = 0.0      # <--- Added
    ) -> int:
        cur = self.conn.cursor()
        try:
            # Try inserting with new columns
            cur.execute(
                """
                INSERT INTO drowsiness_events
                (vehicle_identification_number, user_id, time, status, img_drowsiness, img_path, duration, value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (vehicle_identification_number, user_id, time, status, img_drowsiness, img_path, duration, value),
            )
        except sqlite3.OperationalError:
            # Fallback for old schema (prevents crash if DB isn't migrated)
            logging.warning("Database missing columns (duration/value). Saving simplified event.")
            cur.execute(
                """
                INSERT INTO drowsiness_events
                (vehicle_identification_number, user_id, time, status, img_drowsiness, img_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (vehicle_identification_number, user_id, time, status, img_drowsiness, img_path),
            )
            
        self.conn.commit()
        event_id = int(cur.lastrowid)
        logging.info(
            f"Event saved: id={event_id} type={status} dur={duration:.1f}s val={value:.2f}"
        )
        return event_id

    def _fetchall(self, query: str, params: Iterable[Any] = ()) -> list[tuple]:
        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()

    def get_all_events(self) -> list[tuple]:
        return self._fetchall(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, img_path, duration, value
            FROM drowsiness_events
            ORDER BY time DESC
            """
        )

    def get_events_by_user(self, user_id: int) -> list[tuple]:
        return self._fetchall(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, img_path, duration, value
            FROM drowsiness_events
            WHERE user_id = ?
            ORDER BY time DESC
            """,
            (user_id,),
        )

    def get_events_by_vehicle(self, vin: str) -> list[tuple]:
        return self._fetchall(
            """
            SELECT id, vehicle_identification_number, user_id, time, status, img_path, duration, value
            FROM drowsiness_events
            WHERE vehicle_identification_number = ?
            ORDER BY time DESC
            """,
            (vin,),
        )