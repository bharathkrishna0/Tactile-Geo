"""Durable session store on Postgres (Supabase or any Postgres 14+).

Implements the ``SessionRepository`` interface from ``session_store.py``. Each
call opens a short connection, which keeps the store safe to share between
request handlers, background tasks and separate worker processes without a
pool library. Apply ``migrations/*.sql`` (``scripts/migrate.py``) first.
"""
from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import SESSION_TTL_SECONDS
from app.models.session import ConversionSession
from app.services.object_storage import ObjectStorage

class PostgresSessionStore:
    def __init__(self, dsn: str, storage: ObjectStorage | None = None, ttl_seconds: float = SESSION_TTL_SECONDS) -> None:
        self._dsn = dsn
        self._storage = storage
        self._ttl_seconds = ttl_seconds

    def _connect(self) -> psycopg.Connection:
        # prepare_threshold=None keeps the store usable behind PgBouncer-style
        # transaction poolers, which do not support server-side prepared statements.
        return psycopg.connect(self._dsn, row_factory=dict_row, prepare_threshold=None)

    def add(self, session: ConversionSession) -> None:
        self.prune()
        expires_at = session.created_at + timedelta(seconds=self._ttl_seconds) if self._ttl_seconds > 0 else None
        with self._connect() as connection:
            connection.execute(
                """
                insert into sessions (id, original_filename, source_storage_key, source_local_path, created_at, expires_at)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (id) do update set
                    original_filename = excluded.original_filename,
                    source_storage_key = excluded.source_storage_key,
                    source_local_path = excluded.source_local_path,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    updated_at = now()
                """,
                (
                    session.session_id,
                    session.original_filename,
                    session.source_storage_key,
                    str(session.original_image_path),
                    session.created_at,
                    expires_at,
                ),
            )
            self._insert_event(connection, session.session_id, "session_created", {"filename": session.original_filename})
        self.save(session)

    def get(self, session_id: str) -> ConversionSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "select * from sessions where id = %s and (expires_at is null or expires_at > now())",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row else None

    def save(self, session: ConversionSession) -> None:
        documents = {
            "processing_params": session.processing_params,
            "detected_shapes": session.detected_shapes,
            "detected_labels": session.detected_labels,
            "quality_report": session.quality_report,
            "semantic_geometry": session.semantic_geometry,
            "simplified_geometry": session.simplified_geometry,
            "qa_report": session.qa_report,
        }
        with self._connect() as connection:
            connection.execute(
                """
                update sessions set
                    processing_params = %(processing_params)s,
                    detected_shapes = %(detected_shapes)s,
                    detected_labels = %(detected_labels)s,
                    quality_report = %(quality_report)s,
                    semantic_geometry = %(semantic_geometry)s,
                    simplified_geometry = %(simplified_geometry)s,
                    qa_report = %(qa_report)s,
                    preview_svg = %(preview_svg)s,
                    tactile_svg = %(tactile_svg)s,
                    updated_at = now()
                where id = %(id)s
                """,
                {
                    **{name: None if value is None else Jsonb(value) for name, value in documents.items()},
                    "preview_svg": session.preview_svg,
                    "tactile_svg": session.tactile_svg,
                    "id": session.session_id,
                },
            )

    def record_run(self, session_id: str, kind: str, params: dict, summary: dict, duration_ms: int | None) -> None:
        with self._connect() as connection:
            connection.execute(
                "insert into processing_runs (session_id, kind, params, summary, duration_ms) values (%s, %s, %s, %s, %s)",
                (session_id, kind, Jsonb(params), Jsonb(summary), duration_ms),
            )

    def record_edits(self, session_id: str, edits: list[dict]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.executemany(
                "insert into teacher_edits (session_id, element_id, edit) values (%s, %s, %s)",
                [(session_id, str(edit.get("element_id")), Jsonb(edit)) for edit in edits],
            )

    def record_event(self, session_id: str, event_type: str, payload: dict | None = None) -> None:
        with self._connect() as connection:
            self._insert_event(connection, session_id, event_type, payload or {})

    def events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return connection.execute(
                "select event_type, payload, created_at from audit_events where session_id = %s order by id",
                (session_id,),
            ).fetchall()

    def prune(self) -> int:
        """Delete expired sessions (history cascades) and their stored images."""
        with self._connect() as connection:
            rows = connection.execute(
                "delete from sessions where expires_at is not null and expires_at <= now() returning source_storage_key"
            ).fetchall()
        keys = [row["source_storage_key"] for row in rows if row["source_storage_key"]]
        if keys and self._storage is not None:
            self._storage.delete(keys)
        return len(rows)

    @staticmethod
    def _insert_event(connection: psycopg.Connection, session_id: str, event_type: str, payload: dict) -> None:
        connection.execute(
            "insert into audit_events (session_id, event_type, payload) values (%s, %s, %s)",
            (session_id, event_type, Jsonb(payload)),
        )


def _session_from_row(row: dict[str, Any]) -> ConversionSession:
    created_at = row["created_at"]
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return ConversionSession(
        session_id=str(row["id"]),
        original_filename=row["original_filename"],
        original_image_path=Path(row["source_local_path"] or ""),
        source_storage_key=row["source_storage_key"],
        created_at=created_at,
        processing_params=row["processing_params"] or {},
        detected_shapes=row["detected_shapes"] or [],
        detected_labels=row["detected_labels"] or [],
        preview_svg=row["preview_svg"],
        tactile_svg=row["tactile_svg"],
        quality_report=row["quality_report"],
        semantic_geometry=row["semantic_geometry"],
        simplified_geometry=row["simplified_geometry"],
        qa_report=row["qa_report"],
    )
