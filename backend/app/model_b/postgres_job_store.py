"""Durable Model B job store on Postgres.

Same interface as `ModelBJobStore`. Every state transition is a single
conditional ``update ... where status = ...``, so claiming, cancelling and
completing stay atomic across threads, worker processes and machines: two
workers can never both claim a job, and a cancel can never be overwritten by a
late claim.
"""
from __future__ import annotations

from datetime import timezone
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.model_b.result_codec import result_from_dict, result_to_dict
from app.models.model_b_job import JobStatus, ModelBJob, ModelBJobError
from app.models.model_b_result import ModelBResult
from app.services.model_b_store import INTERRUPTED_ERROR

_ACTIVE = ("queued", "running")
_TERMINAL = ("completed", "failed", "cancelled")


class PostgresModelBJobStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row, prepare_threshold=None)

    def create(self, session_id: str, model: str, provider: str = "", cache_key: str = "") -> ModelBJob:
        with self._connect() as connection:
            row = connection.execute(
                """
                insert into model_b_jobs (id, session_id, model, provider, cache_key)
                values (%s, %s, %s, %s, %s) returning *
                """,
                (str(uuid4()), session_id, model, provider, cache_key),
            ).fetchone()
        return _job_from_row(row)

    def get(self, job_id: str) -> ModelBJob | None:
        with self._connect() as connection:
            row = connection.execute("select * from model_b_jobs where id = %s", (job_id,)).fetchone()
        return _job_from_row(row) if row else None

    def list_for_session(self, session_id: str) -> list[ModelBJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "select * from model_b_jobs where session_id = %s order by created_at desc", (session_id,),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def has_active_job(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "select exists(select 1 from model_b_jobs where session_id = %s and status = any(%s)) as active",
                (session_id, list(_ACTIVE)),
            ).fetchone()
        return bool(row["active"])

    def active_job_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "select count(*) as n from model_b_jobs where status = any(%s)", (list(_ACTIVE),),
            ).fetchone()
        return int(row["n"])

    def cancel(self, job_id: str) -> ModelBJob | None:
        return self._transition(
            "update model_b_jobs set status = 'cancelled', completed_at = now() where id = %s and status = 'queued' returning *",
            (job_id,),
        )

    def begin(self, job_id: str) -> ModelBJob | None:
        return self._transition(
            "update model_b_jobs set status = 'running', started_at = now(), attempts = attempts + 1 "
            "where id = %s and status = 'queued' returning *",
            (job_id,),
        )

    def complete(self, job_id: str, result: ModelBResult, cache_hit: bool = False) -> ModelBJob | None:
        return self._transition(
            "update model_b_jobs set status = 'completed', completed_at = now(), result = %s, cache_hit = %s "
            "where id = %s and status <> all(%s) returning *",
            (Jsonb(result_to_dict(result)), cache_hit, job_id, list(_TERMINAL)),
        )

    def fail(self, job_id: str, error: ModelBJobError) -> ModelBJob | None:
        return self._transition(
            "update model_b_jobs set status = 'failed', completed_at = now(), error = %s "
            "where id = %s and status <> all(%s) returning *",
            (Jsonb(_error_to_dict(error)), job_id, list(_TERMINAL)),
        )

    def recover_interrupted(self, stale_after_s: float) -> int:
        """Fail queued or running jobs older than ``stale_after_s``.

        A job outlives its worker when the process restarts or crashes mid-call.
        Failing it as retryable lets the teacher request it again instead of
        polling a job no worker will ever finish.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                update model_b_jobs set status = 'failed', completed_at = now(), error = %s
                where status = any(%s) and coalesce(started_at, created_at) < now() - make_interval(secs => %s)
                returning id
                """,
                (Jsonb(_error_to_dict(INTERRUPTED_ERROR)), list(_ACTIVE), stale_after_s),
            ).fetchall()
        return len(rows)

    def cache_get(self, cache_key: str) -> ModelBResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "update model_b_cache set hits = hits + 1, last_used_at = now() where cache_key = %s returning result",
                (cache_key,),
            ).fetchone()
        return result_from_dict(row["result"]) if row else None

    def cache_put(self, cache_key: str, result: ModelBResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into model_b_cache (cache_key, result, resolved_model) values (%s, %s, %s)
                on conflict (cache_key) do update set result = excluded.result, last_used_at = now()
                """,
                (cache_key, Jsonb(result_to_dict(result)), result.resolved_model),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("delete from model_b_jobs")
            connection.execute("delete from model_b_cache")

    def _transition(self, sql: str, params: tuple[Any, ...]) -> ModelBJob | None:
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return _job_from_row(row) if row else None


def _error_to_dict(error: ModelBJobError) -> dict[str, Any]:
    return {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
        "retry_after_s": error.retry_after_s,
    }


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


def _job_from_row(row: dict[str, Any]) -> ModelBJob:
    return ModelBJob(
        job_id=row["id"],
        session_id=row["session_id"],
        status=JobStatus(row["status"]),
        created_at=_utc(row["created_at"]),
        started_at=_utc(row["started_at"]),
        completed_at=_utc(row["completed_at"]),
        result=result_from_dict(row["result"]) if row["result"] else None,
        error=ModelBJobError(**row["error"]) if row["error"] else None,
        model=row["model"],
        provider=row["provider"],
        cache_key=row["cache_key"],
        cache_hit=row["cache_hit"],
    )
