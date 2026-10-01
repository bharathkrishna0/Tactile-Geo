import { describe, expect, it } from 'vitest'
import {
  DEFAULT_POLL_TIMING,
  canRetry,
  describeJob,
  hasTimedOut,
  isTerminal,
  nextDelayMs,
  retryAfterSeconds,
} from './modelBPoll'
import type { ModelBJobStatusValue } from './modelBTypes'

describe('isTerminal', () => {
  it.each<ModelBJobStatusValue>(['completed', 'failed', 'cancelled'])('%s is terminal', status => {
    expect(isTerminal(status)).toBe(true)
  })

  it.each<ModelBJobStatusValue>(['queued', 'running'])('%s is not terminal', status => {
    expect(isTerminal(status)).toBe(false)
  })
})

describe('nextDelayMs', () => {
  it('uses the initial delay for the first poll', () => {
    expect(nextDelayMs(0)).toBe(DEFAULT_POLL_TIMING.initialDelayMs)
  })

  it('grows with successive attempts', () => {
    expect(nextDelayMs(1)).toBeGreaterThan(nextDelayMs(0))
    expect(nextDelayMs(3)).toBeGreaterThan(nextDelayMs(1))
  })

  it('saturates at the ceiling', () => {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      expect(nextDelayMs(attempt)).toBeLessThanOrEqual(DEFAULT_POLL_TIMING.maxDelayMs)
    }
    expect(nextDelayMs(40)).toBe(DEFAULT_POLL_TIMING.maxDelayMs)
  })

  it('never returns a non-positive delay', () => {
    for (let attempt = -3; attempt < 20; attempt += 1) {
      expect(nextDelayMs(attempt)).toBeGreaterThan(0)
    }
  })
})

describe('hasTimedOut', () => {
  it('is false before the budget is spent', () => {
    expect(hasTimedOut(0)).toBe(false)
    expect(hasTimedOut(DEFAULT_POLL_TIMING.timeoutMs - 1)).toBe(false)
  })

  it('is true at and after the budget', () => {
    expect(hasTimedOut(DEFAULT_POLL_TIMING.timeoutMs)).toBe(true)
    expect(hasTimedOut(DEFAULT_POLL_TIMING.timeoutMs + 1)).toBe(true)
  })
})

describe('canRetry', () => {
  it('allows retry for a retryable failure', () => {
    expect(canRetry({ status: 'failed', error: { code: 'x', message: 'm', retryable: true } })).toBe(true)
  })

  it('refuses retry for a configuration failure', () => {
    expect(canRetry({ status: 'failed', error: { code: 'x', message: 'm', retryable: false } })).toBe(false)
  })

  it('refuses retry when there is no error object', () => {
    expect(canRetry({ status: 'failed', error: null })).toBe(false)
  })

  it('refuses retry for a job that did not fail', () => {
    expect(canRetry({ status: 'completed', error: { code: 'x', message: 'm', retryable: true } })).toBe(false)
  })
})

describe('describeJob', () => {
  it('describes each status', () => {
    expect(describeJob({ status: 'queued', error: null })).toMatch(/queued/i)
    expect(describeJob({ status: 'running', error: null })).toMatch(/running/i)
    expect(describeJob({ status: 'completed', error: null })).toMatch(/complete/i)
    expect(describeJob({ status: 'cancelled', error: null })).toMatch(/cancelled/i)
  })

  it('includes the failure message', () => {
    expect(
      describeJob({ status: 'failed', error: { code: 'x', message: 'Upstream down.', retryable: true } }),
    ).toContain('Upstream down.')
  })

  it('does not leave a dangling space when there is no message', () => {
    expect(describeJob({ status: 'failed', error: null })).toBe('Model B analysis failed.')
  })
})

describe('retryAfterSeconds', () => {
  it('returns the provider figure when given', () => {
    expect(retryAfterSeconds({ error: { code: 'x', message: 'y', retryable: true, retry_after_s: 42 } })).toBe(42)
  })

  it('is null when the provider said nothing', () => {
    expect(retryAfterSeconds({ error: { code: 'x', message: 'y', retryable: true } })).toBeNull()
    expect(retryAfterSeconds({ error: null })).toBeNull()
  })

  it('rejects a nonsensical wait rather than showing it', () => {
    // A 0 or negative value would render as "wait about 0s", which reads as
    // "retry immediately" and is the opposite of what a rate limit means.
    expect(retryAfterSeconds({ error: { code: 'x', message: 'y', retryable: true, retry_after_s: 0 } })).toBeNull()
    expect(retryAfterSeconds({ error: { code: 'x', message: 'y', retryable: true, retry_after_s: -5 } })).toBeNull()
    expect(retryAfterSeconds({ error: { code: 'x', message: 'y', retryable: true, retry_after_s: Number.NaN } })).toBeNull()
  })
})
