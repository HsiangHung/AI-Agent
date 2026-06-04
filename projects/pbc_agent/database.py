"""
database.py — SQLite schema, upsert, and query helpers.
Tables:
  pbc_state_data   — one row per (state, field), stores value + metadata
  run_log          — tracks each agent run for auditability
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH

log = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pbc_state_data (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                state       TEXT    NOT NULL,
                state_abbr  TEXT    NOT NULL,
                field       TEXT    NOT NULL,
                value_json  TEXT,           -- JSON-serialized value
                confidence  TEXT,           -- high/medium/low/estimated/failed
                source      TEXT,
                notes       TEXT,
                updated_at  TEXT    NOT NULL,
                UNIQUE(state, field)        -- upsert key
            );

            CREATE TABLE IF NOT EXISTS run_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT    NOT NULL,
                state       TEXT,
                field       TEXT,
                status      TEXT,           -- ok/error/skipped
                message     TEXT,
                ts          TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_state ON pbc_state_data(state);
            CREATE INDEX IF NOT EXISTS idx_field  ON pbc_state_data(field);
        """)
    conn.close()
    log.info("Database initialized at %s", DB_PATH)


def upsert_field(state: str, state_abbr: str, field: str, extraction: dict, run_id: str):
    """Insert or replace a single field result."""
    conn = get_connection()
    now  = datetime.now(timezone.utc).isoformat()
    value_json = json.dumps(extraction.get("value"))
    with conn:
        conn.execute("""
            INSERT INTO pbc_state_data
                (state, state_abbr, field, value_json, confidence, source, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(state, field) DO UPDATE SET
                value_json = excluded.value_json,
                confidence = excluded.confidence,
                source     = excluded.source,
                notes      = excluded.notes,
                updated_at = excluded.updated_at
        """, (
            state, state_abbr, field,
            value_json,
            extraction.get("confidence"),
            extraction.get("source"),
            extraction.get("notes"),
            now,
        ))
        conn.execute("""
            INSERT INTO run_log (run_id, state, field, status, message, ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, state, field, "ok", None, now))
    conn.close()


def log_error(state: str, field: str, message: str, run_id: str):
    conn = get_connection()
    now  = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute("""
            INSERT INTO run_log (run_id, state, field, status, message, ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, state, field, "error", message, now))
    conn.close()


def get_state_data(state: str) -> list[dict]:
    """Return all fields for one state as a list of dicts."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM pbc_state_data WHERE state = ? ORDER BY field", (state,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def export_wide_table() -> list[dict]:
    """
    Return a wide-format list (one dict per state) suitable for pandas/Excel export.
    Each dict has state + one key per field containing the raw value.
    """
    conn = get_connection()
    rows = conn.execute("SELECT * FROM pbc_state_data ORDER BY state, field").fetchall()
    conn.close()

    from collections import defaultdict
    table: dict[str, dict] = defaultdict(dict)

    for r in rows:
        state = r["state"]
        if "state" not in table[state]:
            table[state]["state"]      = r["state"]
            table[state]["state_abbr"] = r["state_abbr"]
        try:
            table[state][r["field"]]              = json.loads(r["value_json"])
            table[state][r["field"] + "_conf"]    = r["confidence"]
            table[state][r["field"] + "_source"]  = r["source"]
        except Exception:
            table[state][r["field"]] = r["value_json"]

    return list(table.values())
