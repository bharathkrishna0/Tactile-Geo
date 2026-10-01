import type { ModelBJob, ModelBJobStatusValue } from './modelBTypes'

/**
 * Polling policy for a Model B job.
 *
 * Exported as plain functions rather than living inside the component so the
 * timing rules are testable and so a wrong delay cannot hide in a `useEffect`.
 */

export const TERMINAL_STATUSES: ReadonlySet<ModelBJobStatusValue> = new Set([
  'completed',
  'failed',
  'cancelled',
])

export function isTerminal(status: ModelBJobStatusValue): boolean {
  return TERMINAL_STATUSES.has(status)
}

export interface PollTiming {
  /** Delay before the first poll. Short: the job often finishes in a second or two. */
  initialDelayMs: number
  /** Steady-state interval. */
  intervalMs: number
  /** Backoff ceiling, so a long analysis does not hammer the server. */
  maxDelayMs: number
  /** Give up after this long. Beyond it the teacher is better served by a retry. */
  timeoutMs: number
}

export const DEFAULT_POLL_TIMING: PollTiming = {
  initialDelayMs: 600,
  intervalMs: 1200,
  maxDelayMs: 4000,
  timeoutMs: 90_000,
}

/**
 * Delay before the poll at `attempt` (zero-based).
 *
 * Grows geometrically and then saturates. The alternative, a fixed interval,
 * either hammers the backend for a five-second job or feels broken for a
 * one-minute one.
 */
export function nextDelayMs(attempt: number, timing: PollTiming = DEFAULT_POLL_TIMING): number {
  if (attempt <= 0) return timing.initialDelayMs
  const grown = timing.intervalMs * 1.5 ** Math.min(attempt, 6)
  return Math.min(Math.round(grown), timing.maxDelayMs)
}

export function hasTimedOut(elapsedMs: number, timing: PollTiming = DEFAULT_POLL_TIMING): boolean {
  return elapsedMs >= timing.timeoutMs
}

/**
 * Whether the UI should offer a retry button.
 *
 * Only for errors where retrying can actually help. Offering "Try again" on a
 * bad API key just teaches a teacher to click a button that will never work.
 */
export function canRetry(job: Pick<ModelBJob, 'status' | 'error'>): boolean {
  if (job.status !== 'failed') return false
  if (!job.error) return false
  return job.error.retryable
}

/**
 * How long the provider asked us to wait, in seconds, or null if it did not say.
 *
 * The free tier rate-limits per account rather than per key, so the wait can be
 * long enough that "Try again" without a number reads as a broken button. The
 * provider's own figure is preferred over a guess.
 */
export function retryAfterSeconds(job: Pick<ModelBJob, 'error'>): number | null {
  const value = job.error?.retry_after_s
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return null
  return value
}

/**
 * A one-line status suitable for a live region.
 *
 * Announced politely rather than assertively: a progress update is not an
 * emergency, and an assertive region would interrupt whatever the teacher is
 * reading with a screen reader.
 */
export function describeJob(job: Pick<ModelBJob, 'status' | 'error'>): string {
  switch (job.status) {
    case 'queued':
      return 'Model B analysis queued.'
    case 'running':
      return 'Model B analysis running.'
    case 'completed':
      return 'Model B analysis complete.'
    case 'cancelled':
      return 'Model B analysis cancelled.'
    case 'failed':
      return `Model B analysis failed. ${job.error?.message ?? ''}`.trim()
    default:
      return 'Model B status unknown.'
  }
}
