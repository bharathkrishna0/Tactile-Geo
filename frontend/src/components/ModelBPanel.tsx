import { useCallback, useEffect, useRef, useState } from 'react'
import {
  applyModelBFindings,
  cancelModelBJob,
  fetchModelBJob,
  fetchModelBFusion,
  recordModelBDecision,
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
import type { AnalysisResult } from '../types'
import type {
  ModelBApplyResult,
  ModelBAvailability,
  ModelBDecision,
  ModelBDecisionValue,
  ModelBEntityReview,
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
    decide: typeof recordModelBDecision
    apply: typeof applyModelBFindings
  }
  /** Receives the session regenerated from Semantic Geometry v2. */
  onApplied?: (session: AnalysisResult) => void
}

/**
 * On-demand advisory panel for Model B.
 *
 * Three properties this component is built around:
 *
 * 1. It never auto-runs. Model B costs money and takes seconds, so the teacher
 *    asks for it. Nothing is requested on upload.
 * 2. Nothing changes the tactile output until the teacher says so. The review
 *    queue records accept / reject / defer; "Apply accepted findings" then
 *    builds Semantic Geometry v2 on the server and regenerates simplification,
 *    Braille, QA and the SVG. Coordinates always stay Model A's, so a finding
 *    with no Model A geometry behind it is reported as not applied rather than
 *    turned into a shape drawn from a bounding box.
 * 3. It is honest when it is off, busy, failed, or partial. A silent panel that
 *    shows nothing is indistinguishable from one that found nothing.
 */
export default function ModelBPanel({
  sessionId,
  availability,
  modelAReady,
  api,
  onApplied,
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
        It changes the output above only for findings you accept and apply.
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

      {job?.cache_hit === true && (
        <p className="model-b-note">
          Reused the earlier analysis of this identical image; no new provider call was made.
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

      {fusion && job && (
        <FusionSummary
          fusion={fusion}
          onDecide={(modelBId, decision) => api.decide(sessionId, job.job_id, modelBId, decision)}
          onApply={async () => {
            const applied = await api.apply(sessionId, job.job_id)
            onApplied?.(applied.session)
            return applied
          }}
        />
      )}
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

function FusionSummary({
  fusion,
  onDecide,
  onApply,
}: {
  fusion: ModelBFusionReport
  onDecide: (modelBId: string, decision: ModelBDecisionValue) => Promise<ModelBDecision>
  onApply: () => Promise<ModelBApplyResult>
}) {
  const { summary } = fusion
  return (
    <div className="model-b-fusion">
      <h3>Comparison with the Model A result</h3>
      <ul>
        <li>{summary.agreements} region{summary.agreements === 1 ? '' : 's'} matched</li>
        <li>{summary.candidate_additions} possible addition{summary.candidate_additions === 1 ? '' : 's'}</li>
        <li>{summary.disagreements} disagreement{summary.disagreements === 1 ? '' : 's'}</li>
      </ul>

      <ReviewQueue
        reviews={(fusion.entity_reviews ?? []).filter(review => review.requires_review)}
        initialDecisions={fusion.decisions ?? {}}
        onDecide={onDecide}
        onApply={onApply}
      />

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
        Nothing here has been applied unless you accept it and choose Apply. Model A
        stays the source of every coordinate in the embossed output.
      </p>
    </div>
  )
}

const DECISION_LABELS: Record<ModelBDecisionValue, string> = {
  accept: 'Accept',
  reject: 'Reject',
  defer: 'Decide later',
}

const CORRESPONDENCE_LABELS: Record<ModelBEntityReview['correspondence'], string> = {
  strong: 'matches a Model A element',
  weak: 'partly overlaps a Model A element',
  none: 'no Model A element',
}

/**
 * Findings a teacher should decide on, one at a time.
 *
 * Decisions are recorded in the session's audit history. Applying is a separate,
 * explicit step so a single click never silently changes the embossed output.
 */
function ReviewQueue({
  reviews,
  initialDecisions,
  onDecide,
  onApply,
}: {
  reviews: ModelBEntityReview[]
  initialDecisions: Record<string, ModelBDecision>
  onDecide: (modelBId: string, decision: ModelBDecisionValue) => Promise<ModelBDecision>
  onApply: () => Promise<ModelBApplyResult>
}) {
  const [decisions, setDecisions] = useState<Record<string, ModelBDecisionValue>>(() =>
    Object.fromEntries(Object.entries(initialDecisions).map(([id, d]) => [id, d.decision])),
  )
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [applying, setApplying] = useState(false)
  const [lastApply, setLastApply] = useState<ModelBApplyResult | null>(null)

  if (reviews.length === 0) return null

  const apply = async () => {
    setApplying(true)
    setError('')
    try {
      setLastApply(await onApply())
    } catch (applyError) {
      setError(applyError instanceof Error ? applyError.message : 'Could not apply the accepted findings.')
    } finally {
      setApplying(false)
    }
  }

  const hasAccepted = Object.values(decisions).includes('accept')

  const decide = async (modelBId: string, decision: ModelBDecisionValue) => {
    setPending(modelBId)
    setError('')
    try {
      const recorded = await onDecide(modelBId, decision)
      setDecisions(current => ({ ...current, [modelBId]: recorded.decision }))
    } catch (decideError) {
      setError(decideError instanceof Error ? decideError.message : 'Could not record the decision.')
    } finally {
      setPending(null)
    }
  }

  const remaining = reviews.filter(review => !decisions[review.model_b_id] || decisions[review.model_b_id] === 'defer').length

  return (
    <section className="model-b-review" aria-labelledby="model-b-review-heading">
      <h4 id="model-b-review-heading">Review queue</h4>
      <p className="model-b-note" role="status" aria-live="polite">
        {remaining === 0
          ? 'Every finding has a decision.'
          : `${remaining} of ${reviews.length} finding${reviews.length === 1 ? '' : 's'} still need a decision.`}
      </p>
      <ol className="model-b-review-list">
        {reviews.map(review => {
          const current = decisions[review.model_b_id]
          const name = `${review.model_b_kind.replace(/_/g, ' ')} ${review.model_b_id}`
          return (
            <li key={review.model_b_id} className="model-b-review-item">
              <p>
                <strong>{name}</strong>: {review.reason}{' '}
                <span className="model-b-evidence">
                  ({CORRESPONDENCE_LABELS[review.correspondence]}
                  {review.model_a_id ? ` ${review.model_a_id}` : ''})
                </span>
              </p>
              <div role="group" aria-label={`Decision for ${name}`} className="model-b-decisions">
                {(Object.keys(DECISION_LABELS) as ModelBDecisionValue[]).map(value => (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={current === value}
                    disabled={pending === review.model_b_id}
                    onClick={() => void decide(review.model_b_id, value)}
                  >
                    {DECISION_LABELS[value]}
                  </button>
                ))}
              </div>
            </li>
          )
        })}
      </ol>
      {error && (
        <p className="model-b-error" role="alert">
          {error}
        </p>
      )}
      <p className="model-b-note">
        Accepted findings change the tactile output only after you apply them, and only where Model
        A detected the geometry: a type correction, a label Model A read but did not attach, or a
        confirmation. Re-apply after changing a decision to withdraw it.
      </p>
      {(hasAccepted || lastApply) && (
        <button type="button" className="model-b-action" onClick={() => void apply()} disabled={applying || pending !== null}>
          {applying ? 'Applying…' : 'Apply accepted findings to the tactile output'}
        </button>
      )}
      {lastApply && <ApplySummary result={lastApply} />}
    </section>
  )
}

function ApplySummary({ result }: { result: ModelBApplyResult }) {
  const applied = result.outcomes.filter(outcome => outcome.applied)
  const skipped = result.outcomes.filter(outcome => !outcome.applied)
  return (
    <div className="model-b-apply" role="status" aria-live="polite">
      <p>
        Semantic Geometry v2: {applied.length} finding{applied.length === 1 ? '' : 's'} applied
        {result.reverted.length > 0 ? `, ${result.reverted.length} withdrawn` : ''}. The tactile
        output and QA were regenerated.
      </p>
      {applied.length > 0 && (
        <ul>
          {applied.map(outcome => <li key={outcome.model_b_id}>{outcome.model_b_id}: {outcome.detail}</li>)}
        </ul>
      )}
      {skipped.length > 0 && (
        <>
          <p>Not applied:</p>
          <ul>
            {skipped.map(outcome => <li key={outcome.model_b_id}>{outcome.model_b_id}: {outcome.detail}</li>)}
          </ul>
        </>
      )}
    </div>
  )
}
