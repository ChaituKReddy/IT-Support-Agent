"""SQLite persistence layer: schema, seeding and connection management.

The database is created and seeded on first use, so an evaluator can clone the
repository and run the application without any manual data preparation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    department    TEXT NOT NULL,
    location      TEXT NOT NULL,
    manager       TEXT,
    device_id     TEXT,
    vpn_enabled   INTEGER NOT NULL DEFAULT 1,
    mfa_enrolled  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id     TEXT PRIMARY KEY,
    employee_id   TEXT NOT NULL REFERENCES employees(employee_id),
    category      TEXT NOT NULL,
    priority      TEXT NOT NULL,
    status        TEXT NOT NULL,
    subject       TEXT NOT NULL,
    description   TEXT NOT NULL,
    assigned_to   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    resolution    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tickets_employee ON tickets(employee_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status   ON tickets(status);

CREATE TABLE IF NOT EXISTS kb_articles (
    article_id    TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    category      TEXT NOT NULL,
    tags          TEXT NOT NULL,
    content       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_status (
    service       TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    region        TEXT NOT NULL,
    message       TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

_INIT_LOCK = threading.Lock()
_INITIALISED: set[str] = set()


def _read_seed(seed_dir: Path, name: str) -> list[dict]:
    path = seed_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Seed file missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def seed_database(conn: sqlite3.Connection, seed_dir: Path) -> None:
    """Insert sample rows into any table that is currently empty.

    Seeding is idempotent: tables that already hold rows are left untouched so
    that tickets created by the agent survive a restart.
    """
    if conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
        conn.executemany(
            """INSERT INTO employees
               (employee_id, name, email, department, location, manager,
                device_id, vpn_enabled, mfa_enrolled)
               VALUES (:employee_id, :name, :email, :department, :location,
                       :manager, :device_id, :vpn_enabled, :mfa_enrolled)""",
            _read_seed(seed_dir, "employees"),
        )

    if conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
        conn.executemany(
            """INSERT INTO tickets
               (ticket_id, employee_id, category, priority, status, subject,
                description, assigned_to, created_at, updated_at, resolution)
               VALUES (:ticket_id, :employee_id, :category, :priority, :status,
                       :subject, :description, :assigned_to, :created_at,
                       :updated_at, :resolution)""",
            _read_seed(seed_dir, "tickets"),
        )

    if conn.execute("SELECT COUNT(*) FROM kb_articles").fetchone()[0] == 0:
        rows = [
            {**article, "tags": ", ".join(article["tags"])}
            for article in _read_seed(seed_dir, "knowledge_base")
        ]
        conn.executemany(
            """INSERT INTO kb_articles (article_id, title, category, tags, content)
               VALUES (:article_id, :title, :category, :tags, :content)""",
            rows,
        )

    if conn.execute("SELECT COUNT(*) FROM system_status").fetchone()[0] == 0:
        conn.executemany(
            """INSERT INTO system_status (service, status, region, message, updated_at)
               VALUES (:service, :status, :region, :message, :updated_at)""",
            _read_seed(seed_dir, "system_status"),
        )

    conn.commit()


def initialise(settings: Settings | None = None) -> Path:
    """Create and seed the database once per process for a given path."""
    settings = settings or get_settings()
    db_file = settings.database_file
    key = str(db_file)

    with _INIT_LOCK:
        if key in _INITIALISED:
            return db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_file)
        try:
            create_schema(conn)
            seed_database(conn, settings.seed_directory)
        finally:
            conn.close()
        _INITIALISED.add(key)
    return db_file


@contextmanager
def connection(settings: Settings | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a row-dict SQLite connection with the schema guaranteed to exist."""
    settings = settings or get_settings()
    db_file = initialise(settings)
    conn = sqlite3.connect(db_file, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def reset_initialisation_cache() -> None:
    """Forget which databases were initialised. Used by the test-suite."""
    with _INIT_LOCK:
        _INITIALISED.clear()
