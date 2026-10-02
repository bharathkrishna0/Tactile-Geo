import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'
import { describe, expect, it, vi } from 'vitest'
import ModelBPanel from '../components/ModelBPanel'
import type {
  ModelBApplyResult,
  ModelBAvailability,
  ModelBAnalysis,
  ModelBFusionReport,
  ModelBJob,
} from './modelBTypes'

const available: ModelBAvailability = {
  available: true,
  enabled: true,
  reason: null,
  provider: 'openrouter',
  model: 'openrouter/free',
}

function makeJob(overrides: Partial<ModelBJob> = {}): ModelBJob {
  return {
    job_id: 'job-1',
    session_id: 's1',
    status: 'queued',
    created_at: '2026-01-01T00:00:00Z',
    started_at: null,
    completed_at: null,
    result: null,
    error: null,
    ...overrides,
  }
}

function makeResult(overrides: Partial<ModelBAnalysis> = {}): ModelBAnalysis {
  return {
    schema_version: 'model_b.diag.v1',
    diagram_kind: 'geometric_construction',
    diagram_description: 'A labelled triangle with an inscribed circle.',
    image_width: 1000,
    image_height: 800,
    prepared_width: 1568,
    prepared_height: 1568,
    entities: [],
    relationships: [],
    diagram_relations: [
      {
        id: 'b_d1',
        kind: 'tangent',
        subject_ids: ['b_e1', 'b_e2'],
        statement: 'The circle touches side BC at exactly one point.',
        confidence: 0.7,
        confidence_level: 'medium',
        needs_review: false,
        detection_kind: 'observed',
        evidence: 'the curve meets the side at a single contact point',
        advisory_only: true,
      },
    ],
    text_items: [
      {
        id: 'b_t1',
        text: null,
        role: 'measurement',
        region: { norm: [0, 0, 1, 1], x: 0, y: 0, width: 10, height: 10, advisory_only: true },
        confidence: 0.1,
        confidence_level: 'low',
        needs_review: true,
        evidence: 'ambiguous middle character',
        advisory_only: true,
      },
    ],
    uncertainties: [
      {
        id: 'b_u1',
        subject_ids: ['b_e3'],
        kind: 'occluded_geometry',
        note: 'A scrawl crosses the apex.',
        severity: 'worth_review',
      },
    ],
    image_readable: true,
    image_quality_issues: ['handwriting'],
    truncated: false,
    finish_reason: 'STOP',
    mapped_entity_count: 2,
    unmapped_entity_count: 1,
    review_entity_count: 1,
    provider: 'openrouter',
    requested_model: 'openrouter/free',
    resolved_model: 'qwen/qwen3.8-27b',
    upstream_provider: 'novita',
    request_id: 'gen-fake',
    usage: { total_tokens: 150 },
    validation_warnings: [],
    advisory_only: true,
    ...overrides,
  }
}

function makeFusion(overrides: Partial<ModelBFusionReport> = {}): ModelBFusionReport {
  return {
    summary: { agreements: 2, candidate_additions: 1, disagreements: 0, diagram_relations: 1, text_notes: 1, uncertainties: 1 },
    agreements: [],
    candidate_additions: [
      {
        model_b_id: 'b_e1',
        suggested_type: 'triangle',
        region: [10, 20, 300, 200],
        reason: 'Model B found a region Model A did not report.',
        requires_teacher_approval: true,
        advisory_only: true,
      },
    ],
    disagreements: [],
    diagram_relations: [],
    text_notes: [],
    uncertainties: [],
    advisory_only: true,
    ...overrides,
  }
}

type ModelBApi = React.ComponentProps<typeof ModelBPanel>['api']

function makeApplyResult(): ModelBApplyResult {
  return {
    session: { session_id: 's1', tactile_svg: '<svg />', semantic_geometry: null } as unknown as ModelBApplyResult['session'],
    outcomes: [
      { model_b_id: 'b_e2', applied: true, change: 'set_type', model_a_id: 'el_3', detail: 'Element el_3 is now a circle, keeping Model A\'s coordinates.' },
      { model_b_id: 'b_e1', applied: false, change: null, model_a_id: null, detail: 'Model A has no geometry here, so nothing can be embossed from this finding.' },
    ],
    reverted: [],
  }
}

function makeApi(overrides: Partial<ModelBApi> = {}): ModelBApi {
  return {
    request: vi.fn().mockResolvedValue(makeJob()),
    poll: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    cancel: vi.fn().mockResolvedValue(makeJob({ status: 'cancelled' })),
    fusion: vi.fn().mockResolvedValue(makeFusion()),
    decide: vi.fn().mockImplementation(async (_s: string, jobId: string, modelBId: string, decision: string) => ({
      job_id: jobId,
      model_b_id: modelBId,
      decision,
      note: null,
      decided_at: '2026-01-01T00:00:00Z',
      applied_to_geometry: false,
    })),
    apply: vi.fn().mockResolvedValue(makeApplyResult()),
    ...overrides,
  } as ModelBApi
}

describe('ModelBPanel availability', () => {
  it('explains when Model B is switched off and reassures about Model A', () => {
    render(
      <ModelBPanel
        sessionId="s1"
        availability={{
          available: false,
          enabled: false,
          reason: 'Model B is turned off on this server.',
          provider: 'openrouter',
          model: 'openrouter/free',
        }}
        modelAReady
        api={makeApi()}
      />,
    )
    expect(screen.getByText(/turned off/i)).toBeInTheDocument()
    expect(screen.getByText(/Model A result above is unaffected/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Run Model B/i })).not.toBeInTheDocument()
  })

  it('explains a missing key without offering a retry', () => {
    render(
      <ModelBPanel
        sessionId="s1"
        availability={{
          available: false,
          enabled: true,
          reason: 'Model B is enabled but no API key is configured.',
          provider: 'openrouter',
          model: 'openrouter/free',
        }}
        modelAReady
        api={makeApi()}
      />,
    )
    expect(screen.getByText(/no API key/i)).toBeInTheDocument()
  })
})

describe('ModelBPanel request flow', () => {
  it('does not request anything on mount', () => {
    const api = makeApi()
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    expect(api.request).not.toHaveBeenCalled()
    expect(api.poll).not.toHaveBeenCalled()
  })

  it('is disabled until Model A has run', () => {
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady={false} api={makeApi()} />)
    expect(screen.getByRole('button', { name: /Run Model B check/i })).toBeDisabled()
    expect(screen.getByText(/Run the Model A analysis first/i)).toBeInTheDocument()
  })

  it('requests on demand when the teacher asks', async () => {
    const user = userEvent.setup()
    const api = makeApi()
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(api.request).toHaveBeenCalledWith('s1')
  })

  it('surfaces a request failure', async () => {
    const user = userEvent.setup()
    const api = makeApi({ request: vi.fn().mockRejectedValue(new Error('A Model B analysis is already running.')) })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('already running')
  })
})

describe('ModelBPanel findings', () => {
  it('shows a completed analysis with its relationships', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/touches side BC at exactly one point/i)).toBeInTheDocument()
    expect(screen.getByText(/A scrawl crosses the apex/i)).toBeInTheDocument()
  })

  it('never invents unreadable text', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/Could not read this text/i)).toBeInTheDocument()
  })

  it('states plainly that nothing was applied', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/Nothing here has been applied/i)).toBeInTheDocument()
  })

  it('has no control that could apply a suggestion to Model A', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    await screen.findByText(/Comparison with the Model A result/i)
    // Accepting a region would mean inventing geometry from a bounding box, so
    // no such affordance may exist.
    expect(screen.queryByRole('button', { name: /accept|apply|add to geometry/i })).not.toBeInTheDocument()
  })

  it('warns when the response was truncated', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({ status: 'completed', result: makeResult({ truncated: true }) }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/may be missing findings/i)).toBeInTheDocument()
  })
})

describe('ModelBPanel failures', () => {
  it('tells the teacher when retrying will not help', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({
          status: 'failed',
          error: { code: 'model_b_misconfigured', message: 'Check OPENROUTER_API_KEY.', retryable: false },
        }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/Retrying will not help/i)).toBeInTheDocument()
  })

  it('reports a failed job status politely to screen readers', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({ status: 'failed', error: { code: 'x', message: 'Upstream down.', retryable: true } }),
      ),
    })
    const { container } = render(
      <ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />,
    )
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    const status = await screen.findByRole('status')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveTextContent(/failed/i)
    expect(container).toBeTruthy()
  })
})

describe('ModelBPanel session changes', () => {
  it('clears previous results when the session changes', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    })
    const { rerender } = render(
      <ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />,
    )
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    await screen.findByText(/touches side BC at exactly one point/i)

    rerender(<ModelBPanel sessionId="s2" availability={available} modelAReady api={api} />)

    await waitFor(() => {
      expect(screen.queryByText(/touches side BC at exactly one point/i)).not.toBeInTheDocument()
    })
  })
})

/**
 * Assert no axe violations.
 *
 * Asserting on `results.violations` rather than using a `toHaveNoViolations`
 * matcher: `@types/jest-axe` augments Jest's `Matchers`, not Vitest's, so that
 * matcher neither typechecks here nor registers at runtime. Calling `axe()`
 * without asserting anything would pass unconditionally and prove nothing.
 */
async function expectNoAxeViolations(container: HTMLElement) {
  const results = await axe(container)
  const summary = results.violations.map(
    violation => `${violation.id}: ${violation.help} (${violation.nodes.length} node(s))`,
  )
  expect(results.violations, `axe violations:\n${summary.join('\n')}`).toEqual([])
}

describe('ModelBPanel accessibility', () => {
  it('has no detectable accessibility violations when idle', async () => {
    const { container } = render(
      <ModelBPanel sessionId="s1" availability={available} modelAReady api={makeApi()} />,
    )
    await expectNoAxeViolations(container)
  })

  it('has no detectable accessibility violations when unavailable', async () => {
    const { container } = render(
      <ModelBPanel
        sessionId="s1"
        availability={{
          available: false,
          enabled: false,
          reason: 'Model B is turned off.',
          provider: 'openrouter',
          model: 'openrouter/free',
        }}
        modelAReady
        api={makeApi()}
      />,
    )
    await expectNoAxeViolations(container)
  })

  it('has no detectable accessibility violations after a completed analysis', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
    })
    const { container } = render(
      <ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />,
    )
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    await screen.findByText(/Comparison with the Model A result/i)
    await expectNoAxeViolations(container)
  })
})

describe('ModelBPanel attribution', () => {
  it('names the model that actually read the page, not the configured router', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({
          status: 'completed',
          result: makeResult({ requested_model: 'openrouter/free', resolved_model: 'qwen/qwen3.8-27b' }),
        }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/Read by qwen\/qwen3\.8-27b via novita/)).toBeInTheDocument()
  })

  it('flags that a router served the request when the configured id is not the model', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({
          status: 'completed',
          result: makeResult({ requested_model: 'openrouter/free', resolved_model: 'qwen/qwen3.8-27b' }),
        }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/routed from the configured openrouter\/free/)).toBeInTheDocument()
  })

  it('does not claim routing when a specific model was configured', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({
          status: 'completed',
          result: makeResult({
            requested_model: 'qwen/qwen3.8-27b',
            resolved_model: 'qwen/qwen3.8-27b',
            upstream_provider: '',
          }),
        }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/Read by qwen\/qwen3\.8-27b\./)).toBeInTheDocument()
    expect(screen.queryByText(/routed from/i)).not.toBeInTheDocument()
  })

  it('shows the configured model when Model B is unavailable, so setup is diagnosable', () => {
    render(
      <ModelBPanel
        sessionId="s1"
        availability={{ available: false, enabled: false, reason: 'Off.', provider: 'openrouter', model: 'openrouter/free' }}
        modelAReady
        api={makeApi()}
      />,
    )
    expect(screen.getByText(/Configured model: openrouter\/free/)).toBeInTheDocument()
  })
})

describe('ModelBPanel rate limits', () => {
  it('reports the wait the provider asked for', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({
          status: 'failed',
          error: {
            code: 'model_b_rate_limited',
            message: 'Rate limited by the provider.',
            retryable: true,
            retry_after_s: 42,
          },
        }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/asked to wait about 42s/)).toBeInTheDocument()
  })

  it('does not claim a wait when the provider did not give one', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(
        makeJob({
          status: 'failed',
          error: { code: 'model_b_timeout', message: 'Timed out.', retryable: true },
        }),
      ),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    await screen.findByText(/Timed out/)
    expect(screen.queryByText(/asked to wait/)).not.toBeInTheDocument()
  })
})

const reviews: NonNullable<ModelBFusionReport['entity_reviews']> = [
  {
    model_b_id: 'b_e1',
    model_b_kind: 'triangle',
    model_a_id: null,
    correspondence: 'none',
    adds_semantics: true,
    contradicts_model_a: false,
    requires_review: true,
    reason: 'Only Model B reports this.',
    advisory_only: true,
  },
  {
    model_b_id: 'b_e2',
    model_b_kind: 'circle',
    model_a_id: 'el_3',
    correspondence: 'strong',
    adds_semantics: false,
    contradicts_model_a: true,
    requires_review: true,
    reason: 'Model A reads a ellipse; Model B reads a circle.',
    advisory_only: true,
  },
  {
    model_b_id: 'b_e3',
    model_b_kind: 'point',
    model_a_id: 'el_4',
    correspondence: 'strong',
    adds_semantics: false,
    contradicts_model_a: false,
    requires_review: false,
    reason: 'Both analyses agree.',
    advisory_only: true,
  },
]

async function openQueue(api: ModelBApi) {
  const user = userEvent.setup()
  const view = render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
  await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
  await screen.findByRole('heading', { name: /Review queue/i })
  return { user, ...view }
}

describe('ModelBPanel review queue', () => {
  const completed = () =>
    makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
      fusion: vi.fn().mockResolvedValue(makeFusion({ entity_reviews: reviews })),
    })

  it('lists only findings that need a teacher decision', async () => {
    await openQueue(completed())
    expect(screen.getByRole('group', { name: /Decision for triangle b_e1/i })).toBeInTheDocument()
    expect(screen.getByRole('group', { name: /Decision for circle b_e2/i })).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: /b_e3/ })).not.toBeInTheDocument()
    expect(screen.getByText(/2 of 2 findings still need a decision/)).toBeInTheDocument()
  })

  it('records a decision and marks it pressed without touching Model A', async () => {
    const api = completed()
    const { user } = await openQueue(api)
    const group = screen.getByRole('group', { name: /Decision for circle b_e2/i })
    const reject = within(group).getByRole('button', { name: 'Reject' })
    await user.click(reject)
    expect(api.decide).toHaveBeenCalledWith('s1', 'job-1', 'b_e2', 'reject')
    await waitFor(() => expect(reject).toHaveAttribute('aria-pressed', 'true'))
    expect(screen.getByText(/1 of 2 findings still need a decision/)).toBeInTheDocument()
    expect(screen.getByText(/change the tactile output only after you apply them/)).toBeInTheDocument()
  })

  it('restores decisions already recorded on the server', async () => {
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
      fusion: vi.fn().mockResolvedValue(
        makeFusion({
          entity_reviews: reviews,
          decisions: {
            b_e1: { job_id: 'job-1', model_b_id: 'b_e1', decision: 'accept', note: null, decided_at: '2026-01-01T00:00:00Z', applied_to_geometry: false },
          },
        }),
      ),
    })
    await openQueue(api)
    const group = screen.getByRole('group', { name: /Decision for triangle b_e1/i })
    expect(within(group).getByRole('button', { name: 'Accept' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('is operable from the keyboard', async () => {
    const api = completed()
    const { user } = await openQueue(api)
    const accept = within(screen.getByRole('group', { name: /Decision for triangle b_e1/i })).getByRole('button', { name: 'Accept' })
    accept.focus()
    await user.keyboard('{Enter}')
    expect(api.decide).toHaveBeenCalledWith('s1', 'job-1', 'b_e1', 'accept')
  })

  it('surfaces a failed save as an alert', async () => {
    const api = completed()
    api.decide = vi.fn().mockRejectedValue(new Error('Could not record the decision.'))
    const { user } = await openQueue(api)
    await user.click(within(screen.getByRole('group', { name: /b_e1/ })).getByRole('button', { name: 'Reject' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Could not record/)
  })

  it('has no detectable accessibility violations', async () => {
    const { container } = await openQueue(completed())
    await expectNoAxeViolations(container)
  })
})

describe('ModelBPanel applying accepted findings', () => {
  const completed = () =>
    makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult() })),
      fusion: vi.fn().mockResolvedValue(makeFusion({ entity_reviews: reviews })),
    })

  it('offers no apply control until a finding is accepted', async () => {
    const api = completed()
    const { user } = await openQueue(api)
    expect(screen.queryByRole('button', { name: /Apply accepted findings/i })).not.toBeInTheDocument()
    await user.click(within(screen.getByRole('group', { name: /b_e2/ })).getByRole('button', { name: 'Reject' }))
    expect(screen.queryByRole('button', { name: /Apply accepted findings/i })).not.toBeInTheDocument()
    await user.click(within(screen.getByRole('group', { name: /b_e2/ })).getByRole('button', { name: 'Accept' }))
    expect(await screen.findByRole('button', { name: /Apply accepted findings/i })).toBeInTheDocument()
  })

  it('applies accepted findings, hands back the regenerated session, and reports each outcome', async () => {
    const api = completed()
    const onApplied = vi.fn()
    const user = userEvent.setup()
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} onApplied={onApplied} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    await screen.findByRole('heading', { name: /Review queue/i })
    await user.click(within(screen.getByRole('group', { name: /b_e2/ })).getByRole('button', { name: 'Accept' }))
    await user.click(await screen.findByRole('button', { name: /Apply accepted findings/i }))
    expect(api.apply).toHaveBeenCalledWith('s1', 'job-1')
    expect(onApplied).toHaveBeenCalledWith(makeApplyResult().session)
    expect(await screen.findByText(/Semantic Geometry v2: 1 finding applied/)).toBeInTheDocument()
    expect(screen.getByText(/el_3 is now a circle/)).toBeInTheDocument()
    expect(screen.getByText(/nothing can be embossed from this finding/)).toBeInTheDocument()
  })

  it('surfaces a failed apply as an alert', async () => {
    const api = completed()
    api.apply = vi.fn().mockRejectedValue(new Error('Could not apply the accepted findings.'))
    const { user } = await openQueue(api)
    await user.click(within(screen.getByRole('group', { name: /b_e2/ })).getByRole('button', { name: 'Accept' }))
    await user.click(await screen.findByRole('button', { name: /Apply accepted findings/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Could not apply/)
  })

  it('has no detectable accessibility violations after applying', async () => {
    const api = completed()
    const { user, container } = await openQueue(api)
    await user.click(within(screen.getByRole('group', { name: /b_e2/ })).getByRole('button', { name: 'Accept' }))
    await user.click(await screen.findByRole('button', { name: /Apply accepted findings/i }))
    await screen.findByText(/Semantic Geometry v2/)
    await expectNoAxeViolations(container)
  })
})

describe('ModelBPanel cache reuse', () => {
  it('says when an identical earlier analysis was reused', async () => {
    const user = userEvent.setup()
    const api = makeApi({
      request: vi.fn().mockResolvedValue(makeJob({ status: 'completed', result: makeResult(), cache_hit: true })),
    })
    render(<ModelBPanel sessionId="s1" availability={available} modelAReady api={api} />)
    await user.click(screen.getByRole('button', { name: /Run Model B check/i }))
    expect(await screen.findByText(/Reused the earlier analysis/)).toBeInTheDocument()
  })
})
