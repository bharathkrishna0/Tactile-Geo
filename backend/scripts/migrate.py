"""Apply backend/migrations/*.sql to DATABASE_URL in filename order.

Every migration is idempotent (``create ... if not exists``), so re-running is
safe. Applied filenames are recorded in ``schema_migrations``.

    DATABASE_URL=postgresql://... python scripts/migrate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def apply_migrations(dsn: str) -> list[str]:
    applied: list[str] = []
    with psycopg.connect(dsn, prepare_threshold=None) as connection:
        connection.execute(
            "create table if not exists schema_migrations (name text primary key, applied_at timestamptz not null default now())"
        )
        done = {row[0] for row in connection.execute("select name from schema_migrations").fetchall()}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in done:
                continue
            connection.execute(path.read_text())
            connection.execute("insert into schema_migrations (name) values (%s)", (path.name,))
            applied.append(path.name)
    return applied


def main() -> int:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 1
    applied = apply_migrations(dsn)
    print(f"Applied {len(applied)} migration(s): {', '.join(applied) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
