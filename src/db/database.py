import sqlite3
import os
from typing import Optional, List, Dict, Any

DB_PATH = os.environ.get("DB_PATH", "data/memory.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def get_connection():
    # Make sure the data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    with open(SCHEMA_PATH, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()

def log_event(object_name: str, event_type: str, zone_id: Optional[str] = None, confidence: Optional[float] = None, frame_path: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (object_name, zone_id, event_type, confidence, frame_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (object_name, zone_id, event_type, confidence, frame_path))
    conn.commit()
    conn.close()

def get_last_known_location(object_name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()

    # Get the last event for this object
    cursor.execute('''
        SELECT * FROM events
        WHERE object_name = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (object_name,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None

def add_zone(zone_id: str, description: str, x1: int, y1: int, x2: int, y2: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO zones (id, description, x1, y1, x2, y2)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (zone_id, description, x1, y1, x2, y2))
    conn.commit()
    conn.close()
