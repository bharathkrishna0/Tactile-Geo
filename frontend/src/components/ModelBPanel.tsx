import { useCallback, useEffect, useRef, useState } from 'react'
import {
  cancelModelBJob,
  fetchModelBJob,
  fetchModelBFusion,
  requestModelB,
} from '../lib/modelBApi'
import {
  DEFAULT_POLL_TIMING,
  canRetry,
  describeJob,
  hasTimedOut,
  isTerminal,
  nextDelayMs,
  retryAfterSeconds,
} from '../lib/modelBPoll'
import type {
  ModelBAvailability,
  ModelBFusionReport,
  ModelBJob,
} from '../lib/modelBTypes'

interface ModelBPanelProps {
  sessionId: string
  availability: ModelBAvailability | null
  /** False until Model A has produced a result; there is nothing to compare against. */
  modelAReady: boolean
  api: {
    request: typeof requestModelB
    poll: typeof fetchModelBJob
    cancel: typeof cancelModelBJob
    fusion: typeof fetchModelBFusion
  }
}

/**
 * On-demand advisory panel for Model B.
 *
 * Three properties this component is built around:
 *
 * 1. It never auto-runs. Model B costs money and takes seconds, so the teacher
 *    asks for it. Nothing is requested on upload.
 * 2. It never writes to Model A. Every suggestion is displayed as a
 *    suggestion; there is no accept button wired to a mutation, because
 *    accepting would mean fabricating geometry from a bounding box.
 * 3. It is honest when it is off, busy, failed, or partial. A silent panel that
 *    shows nothing is indistinguishable from one that found nothing.
 */
export default function ModelBPanel({
  sessionId,
  availability,
  modelAReady,
  api,
}: ModelBPanelProps) {
  const [job, setJob] = useState<ModelBJob | null>(null)
  const [fusion, setFusion] = useState<ModelBFusionReport | null>(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)
  const pollTimer = useRef<number | null>(null)
  const startedAt = useRef<number>(0)

  const clearTimer = useCallback(() => {
    if (pollTimer.current !== null) {
      window.clearTimeout(pollTimer.current)
      pollTimer.current = null
    }
  }, [])

  // A new session invalidates every piece of Model B state. Without this, a
  // panel that survives a re-upload would show the previous page's findings.
  useEffect(() => {
    setJob(null)
    setFusion(null)
    setError('')
    clearTimer()
  }, [sessionId, clearTimer])

  useEffect(() => clearTimer, [clearTimer])

  const loadFusion = useCallback(
    async (currentJob: ModelBJob) => {
      try {
        setFusion(await api.fusion(sessionId, currentJob.job_id))
      } catch (fusionError) {
        // A missing comparison is a degraded state, not a failed analysis.
        setError(
          fusionError instanceof Error
            ? fusionError.message
            : 'The comparison with the Model A result is not available yet.',
        )
      }
    },
    [api, sessionId],
  )

  const poll = useCallback(
    (jobId: string, attempt: number) => {
      pollTimer.current = window.setTimeout(async () => {
        if (hasTimedOut(Date.now() - startedAt.current, DEFAULT_POLL_TIMING)) {
          setError('The Model B analysis is taking too long. Try again in a moment.')
          return
        }
        try {
          const updated = await api.poll(sessionId, jobId)
          setJob(updated)
          if (isTerminal(updated.status)) {
            if (updated.status === 'completed') void loadFusion(updated)
            return
          }
          poll(jobId, attempt + 1)
        } catch (pollError) {
          setError(
            pollError instanceof Error ? pollError.message : 'Lost contact with the Model B job.',
          )
        }
      }, nextDelayMs(attempt, DEFAULT_POLL_TIMING))
    },
    [api, sessionId, loadFusion],
  )

  const start = useCallback(async () => {
    setError('')
    setFusion(null)
    setStarting(true)
    try {
      const created = await api.request(sessionId)
      setJob(created)
      startedAt.current = Date.now()
      if (isTerminal(created.status)) {
        // A quick analysis can be finished before the create response is even
        // serialised. The comparison is the whole point of asking, so the
        // already-terminal path must fetch it just like the polling path does.
        if (created.status === 'completed') void loadFusion(created)
      } else {
        poll(created.job_id, 0)
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : 'Could not start the analysis.',
      )
    } finally {
      setStarting(false)
    }
  }, [api, sessionId, poll, loadFusion])

  const cancel = useCallback(async () => {
    if (!job) return
    clearTimer()
    try {
      setJob(await api.cancel(sessionId, job.job_id))
    } catch (cancelError) {
      setError(
        cancelError instanceof Error ? cancelError.message : 'Could not cancel the analysis.',
      )
    }
  }, [api, sessionId, job, clearTimer])

  if (availability && !availability.available) {
    return (
      <section className="model-b-panel" aria-labelledby="model-b-heading">
        <h2 id="model-b-heading">Model B advisory check</h2>
        <p className="model-b-note">{availability.reason ?? 'Model B is unavailable.'}</p>
        {availability.model && (
          <p className="model-b-note">Configured model: {availability.model}</p>
        )}
        <p className="model-b-note">
          The Model A result above is unaffected and remains the embossed output.
        </p>
      </section>
    )
  }

  const busy = starting || (job !== null && !isTerminal(job.status))

  return (
    <section className="model-b-panel" aria-labelledby="model-b-heading">
      <h2 id="model-b-heading">Model B advisory check</h2>
      <p className="model-b-note">
        Optional second opinion that looks for relationships the geometry pipeline
        cannot detect, such as right-angle markers, tangencies, and handwriting.
        It never changes the output above.
      </p>

      {!modelAReady && (
        <p className="model-b-note">
          Run the Model A analysis first so the two can be compared.
        </p>
      )}

      <button
        type="button"
        className="model-b-action"
        onClick={start}
        disabled={busy || !modelAReady}
      >
        {busy ? 'Analysing…' : 'Run Model B check'}
      </button>

      {busy && job && (
        <button type="button" className="model-b-cancel" onClick={cancel}>
          Cancel
        </button>
      )}

      {job && (
        <p className="model-b-status" role="status" aria-live="polite">
          {describeJob(job)}
        </p>
      )}

      {job?.result?.truncated === true && (
        <p className="model-b-warning">
          This response was cut off, so it may be missing findings.
        </p>
      )}

      {error && (
        <p className="model-b-error" role="alert">
          {error}
        </p>
      )}

      {job?.error && !canRetry(job) && (
        <p className="model-b-note">
          Retrying will not help until this is fixed. The Model A result is still usable.
        </p>
      )}

      {job?.error && retryAfterSeconds(job) !== null && (
        <p className="model-b-note">
          The provider asked to wait about {Math.ceil(retryAfterSeconds(job) ?? 0)}s before
          trying again.
        </p>
      )}

      {job?.status === 'completed' && job.result && (
        <ModelBFindings result={job.result} />
      )}

      {fusion && <FusionSummary fusion={fusion} />}
    </section>
  )
}

function ModelBFindings({ result }: { result: NonNullable<ModelBJob['result']> }) {
  return (
    <div className="model-b-findings">
      <p className="model-b-note">
        {result.diagram_description} <strong>({result.diagram_kind})</strong>
      </p>

      {result.diagram_relations.length > 0 && (
        <>
          <h3>Relationships found</h3>
          <ul>
            {result.diagram_relations.map(relation => (
              <li key={relation.id}>
                <strong>{relation.kind.replace(/_/g, ' ')}</strong>: {relation.statement}
                <span className="model-b-evidence"> — {relation.evidence}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {result.uncertainties.length > 0 && (
        <>
          <h3>Flagged for review</h3>
          <ul>
            {result.uncertainties.map(uncertainty => (
              <li key={uncertainty.id}>
                <strong>{uncertainty.kind.replace(/_/g, ' ')}</strong>: {uncertainty.note}
              </li>
            ))}
          </ul>
        </>
      )}

      {result.text_items.filter(item => item.needs_review).length > 0 && (
        <>
          <h3>Text needing a human eye</h3>
          <ul>
            {result.text_items
              .filter(item => item.needs_review)
              .map(item => (
                <li key={item.id}>
                  {item.text ?? 'Could not read this text'} — {item.evidence}
                </li>
              ))}
          </ul>
        </>
      )}

      <p className="model-b-note">
        {result.mapped_entity_count} region{result.mapped_entity_count === 1 ? '' : 's'} match a
        geometry type; {result.unmapped_entity_count} do not and are listed as advisory only.
      </p>

      <ModelBAttribution result={result} />
    </div>
  )
}

/**
 * Which model actually answered.
 *
 * Shown because a router may serve each request from a different upstream, so
 * "openrouter/free" is not a statement about what read the page. Without this a
 * teacher comparing two runs has no way to tell whether a difference came from
 * the worksheet or from a different model.
 */
function ModelBAttribution({ result }: { result: NonNullable<ModelBJob['result']> }) {
  const served = result.resolved_model || result.requested_model
  if (!served) return null

  // Only worth calling out when the configured id is not the model that ran.
  const routed = Boolean(result.requested_model) && result.requested_model !== served

  return (
    <p className="model-b-note">
      {routed
        ? `Read by ${served}${result.upstream_provider ? ` via ${result.upstream_provider}` : ''}, routed from the configured ${result.requested_model}.`
        : `Read by ${served}${result.upstream_provider ? ` via ${result.upstream_provider}` : ''}.`}
    </p>
  )
}

function FusionSummary({ fusion }: { fusion: ModelBFusionReport }) {
  const { summary } = fusion
  return (
    <div className="model-b-fusion">
      <h3>Comparison with the Model A result</h3>
      <ul>
        <li>{summary.agreements} region{summary.agreements === 1 ? '' : 's'} matched</li>
        <li>{summary.candidate_additions} possible addition{summary.candidate_additions === 1 ? '' : 's'}</li>
        <li>{summary.disagreements} disagreement{summary.disagreements === 1 ? '' : 's'}</li>
      </ul>

      {fusion.candidate_additions.length > 0 && (
        <>
          <h4>Model B found something Model A did not</h4>
          <ul>
            {fusion.candidate_additions.map(addition => (
              <li key={addition.model_b_id}>
                {addition.suggested_type ?? 'Unrecognised shape'}{' '}
                <span className="model-b-evidence">
                  — review the image here before changing anything.
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      {fusion.disagreements.length > 0 && (
        <>
          <h4>Disagreements worth checking</h4>
          <ul>
            {fusion.disagreements.map(item => (
              <li key={`${item.kind}-${item.model_b_id}-${item.model_a_id ?? ''}`}>
                {item.detail}
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="model-b-note">
        Nothing here has been applied. Model A remains the embossed output until you
        correct it yourself in the geometry view.
      </p>
    </div>
  )
}
