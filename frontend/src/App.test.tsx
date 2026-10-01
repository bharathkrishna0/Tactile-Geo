import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { AnalysisResult, QAIssue } from './types'

const uploadImage = vi.fn()
const processSession = vi.fn()

vi.mock('./api/client', () => ({
  uploadImage: (...args: unknown[]) => uploadImage(...args),
  processSession: (...args: unknown[]) => processSession(...args),
  patchElement: vi.fn(),
  batchEdit: vi.fn(),
}))

function issue(overrides: Partial<QAIssue> & Pick<QAIssue, 'check'>): QAIssue {
  return { severity: 'warning', message: `${overrides.check} message`, ...overrides }
}

function result(issues: QAIssue[], svg = '<svg xmlns="http://www.w3.org/2000/svg" data-testid="tactile" />'): AnalysisResult {
  return {
    session_id: 'sess-1',
    image_width: 1200,
    image_height: 900,
    semantic: { elements: [], element_count: 0, low_confidence_count: 0, relationships: [], explanations: [] },
    quality_report: { passes_gate: true, issues: [] },
    simplified_geometry: { elements: [], explanations: [] },
    tactile_svg: svg,
    qa_report: {
      overall_score: 100,
      score_0_100: issues.length ? 40 : 100,
      passes: !issues.some(i => i.severity === 'error'),
      issues,
      element_checks: 1,
    },
  } as unknown as AnalysisResult
}

const PNG = new File([new Uint8Array([137, 80, 78, 71])], 'worksheet.png', { type: 'image/png' })

async function runAnalysis(result: AnalysisResult) {
  uploadImage.mockResolvedValue({ session_id: 'sess-1' })
  processSession.mockResolvedValue(result)
  const user = userEvent.setup()
  render(<App />)
  await user.upload(screen.getByLabelText(/choose a worksheet image/i), PNG)
  await user.click(screen.getByRole('button', { name: /create tactile preview/i }))
  await screen.findByRole('button', { name: /export print-ready svg/i })
  return user
}

describe('print-ready export interlock', () => {
  beforeEach(() => {
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:mock'),
      revokeObjectURL: vi.fn(),
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it('enables export when QA reports no issues', async () => {
    await runAnalysis(result([]))
    expect(screen.getByRole('button', { name: /export print-ready svg/i })).toBeEnabled()
  })

  it('enables export when only non-blocking warnings are present', async () => {
    await runAnalysis(result([issue({ check: 'text_too_small', severity: 'warning' })]))
    expect(screen.getByRole('button', { name: /export print-ready svg/i })).toBeEnabled()
  })

  it('disables export when a blocking QA error is present', async () => {
    await runAnalysis(result([issue({ check: 'braille_on_line', severity: 'error' })]))
    const button = screen.getByRole('button', { name: /export print-ready svg/i })
    expect(button).toBeDisabled()
    expect(button).toHaveAttribute('aria-describedby', 'export-blocked-reason')
  })

  it('disables export when a blocking check arrives with downgraded severity', async () => {
    await runAnalysis(result([issue({ check: 'feature_below_minimum_size', severity: 'warning' })]))
    expect(screen.getByRole('button', { name: /export print-ready svg/i })).toBeDisabled()
  })

  it('explains what is blocking and offers no "export anyway" bypass', async () => {
    await runAnalysis(
      result([
        issue({ check: 'braille_on_line', severity: 'error', message: 'Braille marker overlaps a line.' }),
        issue({ check: 'stroke_width_out_of_bounds', severity: 'error', message: 'Stroke is below 0.6 mm.' }),
      ]),
    )
    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent(/print-ready export blocked/i)
    expect(alert).toHaveTextContent('Braille marker overlaps a line.')
    expect(alert).toHaveTextContent('Stroke is below 0.6 mm.')
    expect(screen.queryByRole('button', { name: /export anyway/i })).toBeNull()
  })

  it('refuses to download even if the disabled handler is invoked directly', async () => {
    await runAnalysis(result([issue({ check: 'element_outside_printable_area', severity: 'error' })]))
    // The upload preview also calls createObjectURL, so isolate the export click.
    const { createObjectURL } = URL as unknown as { createObjectURL: ReturnType<typeof vi.fn> }
    createObjectURL.mockClear()
    const button = screen.getByRole('button', { name: /export print-ready svg/i })
    // Bypass the disabled attribute the way a scripted caller or a stale
    // attribute would, and confirm the guard still refuses.
    button.removeAttribute('disabled')
    await userEvent.click(button)
    await waitFor(() => expect(createObjectURL).not.toHaveBeenCalled())
  })

  it('downloads the SVG when the gate is clear', async () => {
    await runAnalysis(result([]))
    const { createObjectURL } = URL as unknown as { createObjectURL: ReturnType<typeof vi.fn> }
    createObjectURL.mockClear()
    await userEvent.click(screen.getByRole('button', { name: /export print-ready svg/i }))
    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(1))
    const blob = createObjectURL.mock.calls[0][0] as Blob
    expect(blob.type).toContain('image/svg+xml')
  })
})

describe('accessibility', () => {
  it('has no detectable accessibility violations in the uploaded state', async () => {
    const { container } = render(<App />)
    // `axe` resolves with a result object; awaiting it alone asserts nothing.
    // This must be a real assertion, or the test passes while the app
    // regresses. Asserting on `violations` directly (rather than the optional
    // `toHaveNoViolations` matcher) keeps the failure output readable and needs
    // no global matcher registration.
    const results = await axe(container)
    expect(results.violations).toEqual([])
  })
})
