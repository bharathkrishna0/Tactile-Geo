"""Durable session store, migrations, TTL and RLS against a real Postgres.

Skipped unless TEST_DATABASE_URL points at a disposable database, e.g.

    docker run -d -p 54329:5432 -e POSTGRES_PASSWORD=postgres postgres:16-alpine
    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:54329/postgres pytest tests/test_postgres_store.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")

from app.models.session import ConversionSession  # noqa: E402
from app.services.object_storage import LocalObjectStorage  # noqa: E402
from app.services.postgres_store import PostgresSessionStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from migrate import apply_migrations  # noqa: E402

DSN = os.getenv("TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")

# Minimal stand-in for Supabase's auth schema so the RLS policies are created
# and can be exercised exactly as Supabase evaluates them.
SUPABASE_AUTH_STUB = """
create schema if not exists auth;
create or replace function auth.uid() returns uuid language sql stable as
$$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
do $$ begin
    if not exists (select 1 from pg_roles where rolname = 'authenticated') then
        create role authenticated nologin;
    end if;
end $$;
grant usage on schema public to authenticated;
"""


@pytest.fixture(scope="module")
def dsn():
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute(
            "drop table if exists model_b_cache, model_b_jobs, audit_events, teacher_edits, processing_runs, sessions, projects, profiles, schema_migrations cascade"
        )
        connection.execute(SUPABASE_AUTH_STUB)
    apply_migrations(DSN)
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute("grant select on all tables in schema public to authenticated")
    return DSN


def _session(created_at=None, key=None) -> ConversionSession:
    session_id = str(uuid4())
    return ConversionSession(
        session_id=session_id,
        original_filename="triangle.png",
        original_image_path=Path(f"/tmp/{session_id}.png"),
        source_storage_key=key,
        created_at=created_at or datetime.now(timezone.utc),
    )


def test_migrations_are_idempotent(dsn):
    assert apply_migrations(dsn) == []


def test_session_round_trips_and_survives_a_new_store_instance(dsn):
    store = PostgresSessionStore(dsn, ttl_seconds=3600)
    session = _session(key="sessions/x/source.png")
    store.add(session)
    session.processing_params = {"edge_sensitivity": 50}
    session.semantic_geometry = {"elements": [{"id": "el_0", "type": "triangle"}], "image_width": 400}
    session.qa_report = {"passes": True, "score_0_100": 98, "issues": []}
    session.tactile_svg = "<svg/>"
    store.save(session)

    restarted = PostgresSessionStore(dsn, ttl_seconds=3600)
    loaded = restarted.get(session.session_id)
    assert loaded is not None
    assert loaded.source_storage_key == "sessions/x/source.png"
    assert loaded.semantic_geometry == session.semantic_geometry
    assert loaded.qa_report["score_0_100"] == 98
    assert loaded.tactile_svg == "<svg/>"
    assert loaded.processing_params == {"edge_sensitivity": 50}


def test_history_is_recorded(dsn):
    store = PostgresSessionStore(dsn, ttl_seconds=3600)
    session = _session()
    store.add(session)
    store.record_run(session.session_id, "process", {"edge_sensitivity": 50}, {"qa_passes": True}, 1200)
    store.record_edits(session.session_id, [{"element_id": "el_0", "operation": "delete"}])
    store.record_event(session.session_id, "teacher_edit", {"element_ids": ["el_0"]})
    assert [e["event_type"] for e in store.events(session.session_id)] == ["session_created", "teacher_edit"]
    with psycopg.connect(dsn) as connection:
        runs = connection.execute("select kind, duration_ms from processing_runs where session_id = %s", (session.session_id,)).fetchall()
        edits = connection.execute("select element_id from teacher_edits where session_id = %s", (session.session_id,)).fetchall()
    assert runs == [("process", 1200)]
    assert edits == [("el_0",)]


def test_expired_sessions_are_deleted_with_history_and_stored_image(dsn, tmp_path):
    storage = LocalObjectStorage(tmp_path)
    storage.put("sessions/old/source.png", b"x", "image/png")
    store = PostgresSessionStore(dsn, storage=storage, ttl_seconds=60)
    old = _session(created_at=datetime.now(timezone.utc) - timedelta(seconds=120), key="sessions/old/source.png")
    store.add(old)
    assert store.get(old.session_id) is None
    store.prune()
    assert not storage.path_for("sessions/old/source.png").exists()
    with psycopg.connect(dsn) as connection:
        remaining = connection.execute("select count(*) from audit_events where session_id = %s", (old.session_id,)).fetchone()
    assert remaining == (0,)


def test_rls_limits_a_teacher_to_their_own_sessions(dsn):
    teacher_a, teacher_b = str(uuid4()), str(uuid4())
    store = PostgresSessionStore(dsn, ttl_seconds=3600)
    mine, theirs = _session(), _session()
    store.add(mine)
    store.add(theirs)
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("insert into profiles (id) values (%s), (%s)", (teacher_a, teacher_b))
        connection.execute("update sessions set owner_id = %s where id = %s", (teacher_a, mine.session_id))
        connection.execute("update sessions set owner_id = %s where id = %s", (teacher_b, theirs.session_id))

    with psycopg.connect(dsn) as connection:
        connection.execute("set local role authenticated")
        connection.execute("select set_config('request.jwt.claim.sub', %s, true)", (teacher_a,))
        visible = {str(row[0]) for row in connection.execute("select id from sessions").fetchall()}
        events = {str(row[0]) for row in connection.execute("select distinct session_id from audit_events").fetchall()}
    assert visible == {mine.session_id}
    assert events == {mine.session_id}

    with psycopg.connect(dsn) as connection:
        connection.execute("set local role authenticated")
        anonymous = connection.execute("select count(*) from sessions").fetchone()
    assert anonymous == (0,)
