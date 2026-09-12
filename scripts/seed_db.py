"""Create and seed the local SQLite database.

Usage:
    python scripts/seed_db.py          # create if missing, keep existing rows
    python scripts/seed_db.py --force  # delete the database and rebuild it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from it_support import db  # noqa: E402
from it_support.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the IT support database.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete the existing database file before seeding",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.force and settings.database_file.exists():
        settings.database_file.unlink()
        db.reset_initialisation_cache()
        print(f"Deleted {settings.database_file}")

    path = db.initialise(settings)
    with db.connection(settings) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("employees", "tickets", "kb_articles", "system_status")
        }
    print(f"Database ready at {path}")
    for table, count in counts.items():
        print(f"  {table:<14} {count:>3} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
