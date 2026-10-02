import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { ModelBAvailability } from './modelBTypes'

const uploadImage = vi.fn()
const processSession = vi.fn()

const fetchModelBAvailability = vi.fn()
const requestModelB = vi.fn()
const fetchModelBJob = vi.fn()
const cancelModelBJob = vi.fn()
const fetchModelBFusion = vi.fn()

vi.mock('../api/client', () => ({
  uploadImage: (...args: unknown[]) => uploadImage(...args),
  processSession: (...args: unknown[]) => processSession(...args),
  patchElement: vi.fn(),
  batchEdit: vi.fn(),
}))

vi.mock('./modelBApi', () => ({
  fetchModelBAvailability: (...args: unknown[]) => fetchModelBAvailability(...args),
  requestModelB: (...args: unknown[]) => requestModelB(...args),
  fetchModelBJob: (...args: unknown[]) => fetchModelBJob(...args),
  cancelModelBJob: (...args: unknown[]) => cancelModelBJob(...args),
  fetchModelBFusion: (...args: unknown[]) => fetchModelBFusion(...args),
  recordModelBDecision: vi.fn(),
  applyModelBFindings: vi.fn(),
}))

const available: ModelBAvailability = {
  available: true,
  enabled: true,
  reason: null,
  provider: 'openrouter',
  model: 'openrouter/free',
}

function analysisResult() {
  return {
    session_id: 'sess-1',
    image_width: 1200,
    image_height: 900,
    semantic: { elements: [], element_count: 0, low_confidence_count: 0, relationships: [], explanations: [] },
    quality_report: { passes_gate: true, issues: [] },
    simplified_geometry: { elements: [], explanations: [] },
    tactile_svg: '<svg xmlns="http://www.w3.org/2000/svg" data-testid="tactile" />',
    qa_report: { overall_score: 100, passes: true, issues: [], element_checks: 0 },
  } as never
}

const PNG = new File([new Uint8Array([137, 80, 78, 71])], 'worksheet.png', { type: 'image/png' })

async function runModelA(user: ReturnType<typeof userEvent.setup>) {
  uploadImage.mockResolvedValue({ session_id: 'sess-1' })
  processSession.mockResolvedValue(analysisResult())
  await user.upload(screen.getByLabelText(/choose a worksheet image/i), PNG)
  await user.click(screen.getByRole('button', { name: /create tactile preview/i }))
  await screen.findByRole('button', { name: /export print-ready svg/i })
}

describe('Model B integration in App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // jsdom does not implement object URLs; App builds a preview URL on upload.
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:mock'),
      revokeObjectURL: vi.fn(),
    })
    fetchModelBAvailability.mockResolvedValue(available)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('checks availability on mount without starting any analysis', async () => {
    const user = userEvent.setup()
    render(<App />)
    await runModelA(user)
    expect(await screen.findByRole('button', { name: /Run Model B check/i })).toBeInTheDocument()
    // Availability is a cheap GET. The analysis itself must wait for the teacher.
    expect(fetchModelBAvailability).toHaveBeenCalledTimes(1)
    expect(requestModelB).not.toHaveBeenCalled()
  })

  it('still renders the Model A flow when Model B is unavailable', async () => {
    fetchModelBAvailability.mockResolvedValue({
      available: false,
      enabled: false,
      reason: 'Model B is turned off on this server.',
      provider: 'openrouter',
      model: 'openrouter/free',
    })
    const user = userEvent.setup()
    render(<App />)
    await runModelA(user)
    expect(await screen.findByRole('button', { name: /export print-ready svg/i })).toBeInTheDocument()
    expect(screen.getByText(/Model A result above is unaffected/i)).toBeInTheDocument()
  })

  it('offers the Model B action only once a session exists to send it', async () => {
    const user = userEvent.setup()
    render(<App />)

    // Model B needs a session id, so it must not be offered before Model A runs.
    expect(screen.queryByRole('button', { name: /Run Model B check/i })).not.toBeInTheDocument()

    await runModelA(user)

    expect(await screen.findByRole('button', { name: /Run Model B check/i })).toBeEnabled()
  })

  it('sends the Model A session id when the teacher asks for a second opinion', async () => {
    requestModelB.mockResolvedValue({
      job_id: 'job-9',
      session_id: 'sess-1',
      status: 'queued',
      created_at: '2026-01-01T00:00:00Z',
      started_at: null,
      completed_at: null,
      result: null,
      error: null,
    })
    const user = userEvent.setup()
    render(<App />)
    await runModelA(user)
    await user.click(await screen.findByRole('button', { name: /Run Model B check/i }))
    expect(requestModelB).toHaveBeenCalledWith('sess-1')
  })

  it('survives an availability check that fails outright', async () => {
    fetchModelBAvailability.mockRejectedValue(new Error('offline'))
    const user = userEvent.setup()
    render(<App />)
    await runModelA(user)
    expect(await screen.findByRole('button', { name: /export print-ready svg/i })).toBeInTheDocument()
  })
})
