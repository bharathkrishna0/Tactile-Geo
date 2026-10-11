import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { DesktopConfig } from './lib/desktop'

const shell = vi.hoisted(() => ({
  backendExitStatus: vi.fn(),
  openLogsFolder: vi.fn(),
  saveSvg: vi.fn(),
  modelBKeyStatus: vi.fn(),
  setModelBKey: vi.fn(),
  clearModelBKey: vi.fn(),
}))
vi.mock('./lib/desktopShell', () => ({ desktopShell: shell }))

const uploadImage = vi.fn()
const processSession = vi.fn()
vi.mock('./api/client', () => ({
  uploadImage: (...args: unknown[]) => uploadImage(...args),
  processSession: (...args: unknown[]) => processSession(...args),
  patchElement: vi.fn(),
  batchEdit: vi.fn(),
}))

const config: DesktopConfig = { apiBase: 'http://127.0.0.1:49152', logsDir: 'C:\\Users\\t\\AppData\\Local\\org.tactilegeo.desktop\\logs', version: '0.1.0', startupError: null }

function healthAfter(failures: number) {
  let calls = 0
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.endsWith('/api/health')) {
      calls += 1
      if (calls <= failures) throw new TypeError('connection refused')
      return new Response('{"status":"ok"}', { status: 200 })
    }
    return new Response('{"available":false,"enabled":false,"reason":"off","provider":"","model":""}', { status: 200 })
  })
}

describe('desktop mode', () => {
  beforeEach(() => {
    shell.backendExitStatus.mockResolvedValue(null)
    shell.modelBKeyStatus.mockResolvedValue(false)
    vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn(() => 'blob:mock'), revokeObjectURL: vi.fn() })
  })
  afterEach(() => {
    delete window.__TACTILEGEO_DESKTOP__
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('keeps live processing disabled until the local engine is healthy, and hides the download button', async () => {
    window.__TACTILEGEO_DESKTOP__ = config
    const fetchMock = healthAfter(1)
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    expect(screen.getByRole('button', { name: 'Choose image' })).toBeDisabled()
    expect(screen.getByText(/starting the local tactilegeo engine/i)).toBeInTheDocument()
    expect(screen.queryByText('Download Desktop App')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /template \(demo\)/i })).toBeEnabled()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Choose image' })).toBeEnabled(), { timeout: 3000 })
    expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:49152/api/health')
    expect(screen.queryByText(/starting the local tactilegeo engine/i)).not.toBeInTheDocument()
  })

  it('shows a useful error when the engine process exits during startup', async () => {
    window.__TACTILEGEO_DESKTOP__ = config
    vi.stubGlobal('fetch', healthAfter(Number.POSITIVE_INFINITY))
    shell.backendExitStatus.mockResolvedValue('exit code: 1')
    const user = userEvent.setup()
    render(<App />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The local engine stopped during startup (exit code: 1).')
    expect(alert).toHaveTextContent(config.logsDir)
    expect(screen.getByRole('button', { name: 'Choose image' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /open logs folder/i }))
    expect(shell.openLogsFolder).toHaveBeenCalled()
  })

  it('reports a shell startup error without polling', async () => {
    window.__TACTILEGEO_DESKTOP__ = { ...config, apiBase: '', startupError: 'The TactileGeo engine is missing: C:\\x\\backend\\tactilegeo-backend.exe' }
    const fetchMock = healthAfter(0)
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('engine is missing')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('exports through the native save dialog', async () => {
    window.__TACTILEGEO_DESKTOP__ = config
    vi.stubGlobal('fetch', healthAfter(0))
    shell.saveSvg.mockResolvedValue('C:\\Users\\t\\Documents\\triangle.svg')
    uploadImage.mockResolvedValue({ session_id: 'sess-1' })
    processSession.mockResolvedValue({
      session_id: 'sess-1',
      semantic_geometry: { elements: [], element_count: 0, low_confidence_count: 0, relationships: [], explanations: [] },
      quality_report: { passes_gate: true, issues: [] },
      simplified_geometry: { elements: [], explanations: [] },
      tactile_svg: '<svg xmlns="http://www.w3.org/2000/svg"/>',
      qa_report: { overall_score: 1, score_0_100: 100, passes: true, issues: [], element_checks: 1 },
    })
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Choose image' })).toBeEnabled())
    await user.upload(screen.getByLabelText(/choose a worksheet image/i), new File([new Uint8Array([137, 80, 78, 71])], 'w.png', { type: 'image/png' }))
    await user.click(screen.getByRole('button', { name: /create tactile preview/i }))
    await user.click(await screen.findByRole('button', { name: /export print-ready svg/i }))
    expect(shell.saveSvg).toHaveBeenCalledWith('<svg xmlns="http://www.w3.org/2000/svg"/>', 'tactile-sess-1.svg')
    expect(await screen.findByText('Saved C:\\Users\\t\\Documents\\triangle.svg')).toBeInTheDocument()
  })

  it('saves the Model B key through the shell, never in the page', async () => {
    window.__TACTILEGEO_DESKTOP__ = config
    vi.stubGlobal('fetch', healthAfter(0))
    shell.setModelBKey.mockResolvedValue(undefined)
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: /model b key \(not set\)/i }))
    await user.type(screen.getByLabelText('OpenRouter API key'), 'sk-or-test')
    await user.click(screen.getByRole('button', { name: 'Save key' }))
    expect(shell.setModelBKey).toHaveBeenCalledWith('sk-or-test')
    expect(await screen.findByText(/restart tactilegeo to use model b/i)).toBeInTheDocument()
    expect(window.localStorage.length).toBe(0)
  })
})
