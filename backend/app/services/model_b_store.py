"""In-memory Model B job store.

Mirrors `session_store.py`: a process-local dict with a clear note that it is
not deployment-grade. Two additions, both because Model B introduces
concurrency that Milestone 1 never had:

* a `threading.Lock`, since a job worker mutates its record from a threadpool
  worker while a request handler reads it;
* cancellation, so a teacher who realises they uploaded the wrong page does not
  pay for an analysis nobody will read.

Everything here is per-process, so jobs do not survive a restart and do not
work behind more than one worker. With DATABASE_URL set, `model_b_job_store` is
the Postgres implementation in `app/model_b/postgres_job_store.py` instead, which keeps
the same interface and the same atomic claim semantics as SQL row updates.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.core.config import DATABASE_URL
from app.models.model_b_job import JobStatus, ModelBJob, ModelBJobError
from app.models.model_b_result import ModelBResult

# Results are the largest thing Model B stores, and holding a full analysis for
# every job forever is a slow memory leak. Bounded by count, oldest first.
MAX_RETAINED_JOBS = 200
MAX_CACHED_RESULTS = 200


class ModelBJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, ModelBJob] = {}
        self._cache: OrderedDict[str, ModelBResult] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, session_id: str, model: str, provider: str = "", cache_key: str = "") -> ModelBJob:
        job = ModelBJob(
            job_id=str(uuid4()),
            session_id=session_id,
            model=model,
            provider=provider,
            cache_key=cache_key,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._evict_locked()
        return job

    def get(self, job_id: str) -> ModelBJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_session(self, session_id: str) -> list[ModelBJob]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.session_id == session_id]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def has_active_job(self, session_id: str) -> bool:
        with self._lock:
            return any(
                job.session_id == session_id and not job.status.is_terminal
                for job in self._jobs.values()
            )

    def active_job_count(self) -> int:
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            )

    def cancel(self, job_id: str) -> ModelBJob | None:
        """Cancel a job that has not started.

        Returns None when the job is missing, already finished, or already
        running. A running provider call cannot be interrupted without a
        provider-side abort, so cancelling one is reported as not-cancellable
        rather than pretending to have stopped work that is still billing.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                return None
            job.mark_cancelled()
            return job

    def begin(self, job_id: str) -> ModelBJob | None:
        """Atomically claim a queued job for the worker.

        Returns the job only if it is still QUEUED, otherwise None. Checking
        the status and setting RUNNING as two separate steps leaves a window in
        which `cancel()` slips in between: the cancel is then overwritten by
        `mark_running()`, the teacher believes they stopped the work, and the
        provider call is spent anyway. One lock acquisition closes that window.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                return None
            job.mark_running()
            return job

    def complete(self, job_id: str, result: ModelBResult, cache_hit: bool = False) -> ModelBJob | None:
        """Record a result unless the job already reached a terminal state."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status.is_terminal:
                return None
            job.cache_hit = cache_hit
            job.mark_completed(result)
            return job

    def fail(self, job_id: str, error: ModelBJobError) -> ModelBJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status.is_terminal:
                return None
            job.mark_failed(error)
            return job

    def recover_interrupted(self, stale_after_s: float) -> int:
        """Fail jobs whose worker is gone, so a poller is never left waiting.

        In one process a job only stalls if its worker thread died; the
        Postgres store uses the same rule to recover jobs orphaned by a restart.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_s)
        recovered = 0
        with self._lock:
            for job in self._jobs.values():
                if not job.status.is_terminal and (job.started_at or job.created_at) < cutoff:
                    job.mark_failed(INTERRUPTED_ERROR)
                    recovered += 1
        return recovered

    def cache_get(self, cache_key: str) -> ModelBResult | None:
        with self._lock:
            result = self._cache.get(cache_key)
            if result is not None:
                self._cache.move_to_end(cache_key)
            return result

    def cache_put(self, cache_key: str, result: ModelBResult) -> None:
        with self._lock:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > MAX_CACHED_RESULTS:
                self._cache.popitem(last=False)

    def _evict_locked(self) -> None:
        if len(self._jobs) <= MAX_RETAINED_JOBS:
            return
        # Evict terminal jobs first, oldest first among them, so a burst of
        # queued work is never the thing that gets dropped. The `not is_terminal`
        # negation is what puts finished jobs at the front of the ascending sort.
        ordered = sorted(
            self._jobs.values(),
            key=lambda job: (not job.status.is_terminal, job.created_at),
        )
        for job in ordered[: len(self._jobs) - MAX_RETAINED_JOBS]:
            self._jobs.pop(job.job_id, None)

    def clear(self) -> None:
        """Test helper. Also the seam for a future store swap."""
        with self._lock:
            self._jobs.clear()
            self._cache.clear()


INTERRUPTED_ERROR = ModelBJobError(
    code="interrupted",
    message="The analysis was interrupted before it finished. You can request it again.",
    retryable=True,
)


def build_model_b_job_store():
    if DATABASE_URL:
        from app.model_b.postgres_job_store import PostgresModelBJobStore

        return PostgresModelBJobStore(DATABASE_URL)
    return ModelBJobStore()


model_b_job_store = build_model_b_job_store()
