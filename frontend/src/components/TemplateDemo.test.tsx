import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import metadata from '../../public/demo/tactile-template/metadata.json'
import { TEMPLATE_STAGES, templateAsset } from '../demo/templateDemo'

const uploadImage = vi.fn()
const processSession = vi.fn()
const patchElement = vi.fn()
const batchEdit = vi.fn()
const fetchModelBAvailability = vi.fn()
const requestModelB = vi.fn()
const fetchModelBJob = vi.fn()
const cancelModelBJob = vi.fn()
const fetchModelBFusion = vi.fn()
const recordModelBDecision = vi.fn()
const applyModelBFindings = vi.fn()

vi.mock('../api/client', () => ({
  uploadImage: (...args: unknown[]) => uploadImage(...args),
  processSession: (...args: unknown[]) => processSession(...args),
  patchElement: (...args: unknown[]) => patchElement(...args),
  batchEdit: (...args: unknown[]) => batchEdit(...args),
}))

vi.mock('../lib/modelBApi', () => ({
  fetchModelBAvailability: (...args: unknown[]) => fetchModelBAvailability(...args),
  requestModelB: (...args: unknown[]) => requestModelB(...args),
  fetchModelBJob: (...args: unknown[]) => fetchModelBJob(...args),
  cancelModelBJob: (...args: unknown[]) => cancelModelBJob(...args),
  fetchModelBFusion: (...args: unknown[]) => fetchModelBFusion(...args),
  recordModelBDecision: (...args: unknown[]) => recordModelBDecision(...args),
  applyModelBFindings: (...args: unknown[]) => applyModelBFindings(...args),
}))

const fetchMock = vi.fn((input: RequestInfo | URL) => {
  const url = String(input)
  if (url === templateAsset('metadata.json')) {
    return Promise.resolve(new Response(JSON.stringify(metadata), { status: 200, headers: { 'Content-Type': 'application/json' } }))
  }
  return Promise.resolve(new Response('{}', { status: 404 }))
})

const liveApis = [
  uploadImage, processSession, patchElement, batchEdit,
  requestModelB, fetchModelBJob, cancelModelBJob, fetchModelBFusion, recordModelBDecision, applyModelBFindings,
]

async function openTemplate() {
  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByRole('button', { name: 'Template (Demo)' }))
  await screen.findByText(metadata.model_a.element_count.toString(), { exact: false }).catch(() => undefined)
  await screen.findByRole('img', { name: /original pie chart titled/i })
  return user
}

function currentTab() {
  return screen.getByRole('tab', { selected: true })
}

describe('Template (Demo) mode', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn(() => 'blob:mock'), revokeObjectURL: vi.fn() })
    fetchModelBAvailability.mockResolvedValue({ available: false, enabled: false, reason: 'off', provider: '', model: '' })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('renders the Template button next to the live upload, labelled as live processing', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: 'Template (Demo)' })).toBeInTheDocument()
    expect(screen.getByText('LIVE PROCESSING')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Choose image' })).toBeInTheDocument()
  })

  it('opens a clearly labelled precomputed demo on the Original stage', async () => {
    await openTemplate()
    expect(screen.getByText('TEMPLATE DEMO')).toBeInTheDocument()
    expect(screen.getByText('Precomputed successful example')).toBeInTheDocument()
    expect(screen.queryByText('LIVE PROCESSING')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Choose image' })).not.toBeInTheDocument()
    expect(currentTab()).toHaveTextContent('Original')
    expect(screen.getByRole('heading', { name: 'Original Diagram' })).toBeInTheDocument()
    expect(screen.getByText('The source visual diagram provided to TactileGeo.')).toBeInTheDocument()
  })

  it('walks every stage with Next and back with Previous', async () => {
    const user = await openTemplate()
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    for (let index = 1; index < TEMPLATE_STAGES.length; index += 1) {
      await user.click(screen.getByRole('button', { name: `Next: ${TEMPLATE_STAGES[index].label}` }))
      expect(currentTab()).toHaveTextContent(TEMPLATE_STAGES[index].label)
      expect(screen.getByRole('heading', { level: 3, name: TEMPLATE_STAGES[index].title })).toBeInTheDocument()
      expect(screen.getByText(`Stage ${index + 1} of ${TEMPLATE_STAGES.length}`)).toBeInTheDocument()
    }
    expect(screen.getByRole('button', { name: 'Done' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Previous' }))
    expect(currentTab()).toHaveTextContent('Braille')
  })

  it('shows Model A counts and the OCR text from the stored metadata', async () => {
    const user = await openTemplate()
    await user.click(screen.getByRole('tab', { name: /model a/i }))
    expect(screen.getByText('MODEL A — GEOMETRY ANALYSIS')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: `Model A geometry overlay with ${metadata.model_a.element_count} elements` })).toHaveAttribute('src', templateAsset('model-a.svg'))
    for (const label of metadata.model_a.labels) expect(screen.getAllByText(label.text).length).toBeGreaterThan(0)
  })

  it('presents Model B as advisory semantics that do not draw the SVG', async () => {
    const user = await openTemplate()
    await user.click(screen.getByRole('tab', { name: /model b/i }))
    expect(screen.getByText('Advisory AI')).toBeInTheDocument()
    expect(screen.getByText(metadata.model_b.description)).toBeInTheDocument()
    expect(screen.getByText(/Model B provides semantic interpretation\. Final geometry is generated by the deterministic tactile pipeline\./)).toBeInTheDocument()
    expect(screen.getByText(metadata.provenance.model_b.model)).toBeInTheDocument()
    expect(screen.queryByText(/%/)).not.toBeInTheDocument()
  })

  it('shows fusion results without moving any coordinate', async () => {
    const user = await openTemplate()
    await user.click(screen.getByRole('tab', { name: /fusion/i }))
    const table = screen.getByRole('table', { name: /model b findings/i })
    expect(within(table).getAllByRole('row')).toHaveLength(metadata.fusion.reviews.length + 1)
    expect(within(table).getAllByText('Confirmed')).toHaveLength(metadata.fusion.applied)
    expect(screen.getByText('Coordinates changed').previousSibling).toHaveTextContent('0')
  })

  it('lists the stored Liblouis Braille for each label', async () => {
    const user = await openTemplate()
    await user.click(screen.getByRole('tab', { name: /braille/i }))
    for (const label of metadata.braille.labels) {
      expect(screen.getByLabelText(`Braille for ${label.text}`)).toHaveTextContent(label.braille)
    }
  })

  it('renders the final SVG with preview, fullscreen and export', async () => {
    const user = await openTemplate()
    await user.click(screen.getByRole('tab', { name: /final svg/i }))
    expect(screen.getByRole('img', { name: /final print-ready tactile svg/i })).toHaveAttribute('src', templateAsset('final.svg'))
    expect(screen.getByRole('link', { name: 'Preview SVG' })).toHaveAttribute('href', templateAsset('final.svg'))
    expect(screen.getByRole('link', { name: 'Export SVG' })).toHaveAttribute('download', 'tactile-template-pie-chart.svg')
    expect(screen.getByText('QA passed')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'View Fullscreen' }))
    const dialog = screen.getByRole('dialog', { name: /fullscreen/i })
    expect(within(dialog).getByRole('button', { name: 'Close fullscreen' })).toHaveFocus()
    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View Fullscreen' })).toHaveFocus()
  })

  it('moves between stages with the arrow, Home and End keys', async () => {
    const user = await openTemplate()
    currentTab().focus()
    await user.keyboard('{ArrowRight}')
    expect(currentTab()).toHaveTextContent('Model A')
    expect(currentTab()).toHaveFocus()
    await user.keyboard('{End}')
    expect(currentTab()).toHaveTextContent('Final SVG')
    await user.keyboard('{ArrowLeft}')
    expect(currentTab()).toHaveTextContent('Braille')
    await user.keyboard('{Home}')
    expect(currentTab()).toHaveTextContent('Original')
  })

  it('exits back to the live upload workflow and restores focus', async () => {
    const user = await openTemplate()
    await user.click(screen.getByRole('button', { name: 'Exit Template' }))
    expect(screen.queryByText('TEMPLATE DEMO')).not.toBeInTheDocument()
    expect(screen.getByText('LIVE PROCESSING')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Choose image' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Template (Demo)' })).toHaveFocus()
  })

  it('keeps an existing live session when entering and leaving the demo', async () => {
    uploadImage.mockResolvedValue({ session_id: 'sess-1' })
    processSession.mockResolvedValue({
      session_id: 'sess-1', image_width: 10, image_height: 10,
      semantic: { elements: [], element_count: 0, low_confidence_count: 0, relationships: [], explanations: [] },
      quality_report: { passes_gate: true, issues: [] },
      simplified_geometry: { elements: [], explanations: [] },
      tactile_svg: '<svg xmlns="http://www.w3.org/2000/svg"/>',
      qa_report: { overall_score: 100, score_0_100: 100, passes: true, issues: [], element_checks: 1 },
    })
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByLabelText(/choose a worksheet image/i), new File([new Uint8Array([137, 80])], 'sheet.png', { type: 'image/png' }))
    await user.click(screen.getByRole('button', { name: /create tactile preview/i }))
    await screen.findByRole('button', { name: /export print-ready svg/i })

    await user.click(screen.getByRole('button', { name: 'Template (Demo)' }))
    expect(screen.queryByRole('button', { name: /export print-ready svg/i })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Exit Template' }))
    expect(screen.getByRole('button', { name: /export print-ready svg/i })).toBeInTheDocument()
    expect(screen.getByText('sheet.png')).toBeInTheDocument()
    expect(processSession).toHaveBeenCalledTimes(1)
  })

  it('makes no backend or AI call while the demo is open', async () => {
    const user = userEvent.setup()
    render(<App />)
    const availabilityChecks = fetchModelBAvailability.mock.calls.length
    fetchMock.mockClear()

    await user.click(screen.getByRole('button', { name: 'Template (Demo)' }))
    await screen.findByRole('img', { name: /original pie chart titled/i })
    for (let index = 1; index < TEMPLATE_STAGES.length; index += 1) {
      await user.click(screen.getByRole('button', { name: `Next: ${TEMPLATE_STAGES[index].label}` }))
    }
    await user.click(screen.getByRole('button', { name: 'View Fullscreen' }))

    for (const api of liveApis) expect(api).not.toHaveBeenCalled()
    expect(fetchModelBAvailability).toHaveBeenCalledTimes(availabilityChecks)
    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    expect(urls).toEqual([templateAsset('metadata.json')])
    expect(urls.some(url => url.includes('/api/'))).toBe(false)
  })

  it('shows an error instead of fake data when the template files are missing', async () => {
    fetchMock.mockImplementationOnce(() => Promise.resolve(new Response('', { status: 404 })))
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Template (Demo)' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The template files could not be loaded.')
  })

  it('has no detectable accessibility violations on any stage', async () => {
    const user = await openTemplate()
    for (let index = 0; index < TEMPLATE_STAGES.length; index += 1) {
      if (index > 0) await user.click(screen.getByRole('button', { name: `Next: ${TEMPLATE_STAGES[index].label}` }))
      const results = await axe(document.body)
      expect(results.violations.map(violation => `${TEMPLATE_STAGES[index].id}: ${violation.id}`)).toEqual([])
    }
  })
})
