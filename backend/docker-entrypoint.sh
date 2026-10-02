#!/bin/sh
set -e
# Migrations are idempotent; applying them on boot keeps a fresh database usable.
if [ -n "$DATABASE_URL" ] && [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    python scripts/migrate.py
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WEB_CONCURRENCY:-1}" --proxy-headers
