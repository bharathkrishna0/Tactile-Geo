"""Model B job record.

A job exists because a vision-language model call takes seconds, not
milliseconds. The teacher gets a `queued` job immediately and polls; the analysis
never blocks a request handler, and a failure can be inspected after the fact
instead of vanishing into a timed-out response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


@dataclass
class ModelBJobError:
    """A job failure.

    Named distinctly from `app.model_b.errors.ModelBError`. Both modules are
    imported into the same API module, and two different classes sharing a name
    means one import silently shadows the other.

    `retry_after_s` is present only for rate limiting, where the provider told us
    how long to wait. Absent means "no specific wait is known", which is
    different from "do not retry".
    """

    code: str
    message: str
    retryable: bool = False
    retry_after_s: float | None = None


@dataclass
class ModelBJob:
    job_id: str
    session_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    #: Holds a `ModelBResult` once completed. The annotation is `Any` rather
    #: than `dict` because that is what is actually stored; the API layer guards
    #: on `isinstance` before projecting it onto the wire schema.
    result: Any = None
    error: ModelBJobError | None = None
    #: Model identifier this job was submitted to. May be a router; the model
    #: that actually served it is recorded on the result.
    model: str = ""
    #: Gateway the job was submitted to, e.g. `openrouter`.
    provider: str = ""

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, result: Any) -> None:
        self.status = JobStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error: ModelBJobError) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc)

    def mark_cancelled(self) -> None:
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
