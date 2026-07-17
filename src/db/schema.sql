CREATE TABLE IF NOT EXISTS zones (
    id TEXT PRIMARY KEY,
    description TEXT,
    x1 INTEGER,
    y1 INTEGER,
    x2 INTEGER,
    y2 INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_name TEXT NOT NULL,
    zone_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL, -- 'appeared', 'disappeared', 'moved'
    confidence REAL,
    frame_path TEXT,
    FOREIGN KEY(zone_id) REFERENCES zones(id)
);

CREATE INDEX IF NOT EXISTS idx_events_object ON events(object_name);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
