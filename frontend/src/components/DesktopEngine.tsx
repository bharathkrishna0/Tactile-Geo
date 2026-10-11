import { useEffect, useState } from 'react'
import type { DesktopConfig } from '../lib/desktop'
import type { DesktopShell } from '../lib/desktopShell'

export type EngineState = 'starting' | 'ready' | 'failed'

const POLL_MS = 1000
const STARTUP_LIMIT_MS = 180_000

/**
 * Waits for the local engine's /api/health before live processing is enabled.
 * Fails early when the shell reports the engine process has exited.
 */
export function useEngineState(config: DesktopConfig | null, shell: DesktopShell): { state: EngineState; error: string } {
  const [state, setState] = useState<EngineState>(config ? (config.startupError ? 'failed' : 'starting') : 'ready')
  const [error, setError] = useState(config?.startupError ?? '')

  useEffect(() => {
    if (!config || config.startupError) return
    let active = true
    const started = Date.now()
    async function poll() {
      while (active) {
        try {
          const response = await fetch(`${config!.apiBase}/api/health`)
          if (response.ok) { if (active) setState('ready'); return }
        } catch { /* not listening yet */ }
        const exited = await shell.backendExitStatus().catch(() => null)
        if (exited) { if (active) { setError(`The local engine stopped during startup (${exited}).`); setState('failed') } return }
        if (Date.now() - started > STARTUP_LIMIT_MS) { if (active) { setError('The local engine did not respond within 3 minutes.'); setState('failed') } return }
        await new Promise(resolve => setTimeout(resolve, POLL_MS))
      }
    }
    void poll()
    return () => { active = false }
  }, [config, shell])

  return { state, error }
}

export function EngineBanner({ state, error, config, shell }: { state: EngineState; error: string; config: DesktopConfig; shell: DesktopShell }) {
  if (state === 'ready') return null
  if (state === 'starting') {
    return <p className="status" role="status">Starting the local TactileGeo engine… The first start can take up to a minute. Template (Demo) works now.</p>
  }
  return (
    <div className="export-warning" role="alert">
      <h3>The local engine could not start</h3>
      <p>{error} Live processing is unavailable; Template (Demo) still works. Logs are in {config.logsDir}.</p>
      <div className="export-warning-actions">
        <button type="button" onClick={() => void shell.openLogsFolder()}>Open logs folder</button>
      </div>
    </div>
  )
}

/** Stores the teacher's own OpenRouter key in Windows Credential Manager via the shell. */
export function ModelBKeySettings({ shell }: { shell: DesktopShell }) {
  const [open, setOpen] = useState(false)
  const [stored, setStored] = useState<boolean | null>(null)
  const [key, setKey] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => { shell.modelBKeyStatus().then(setStored).catch(() => setStored(false)) }, [shell])

  async function save() {
    try {
      await shell.setModelBKey(key)
      setKey(''); setStored(true)
      setMessage('Key saved in Windows Credential Manager. Restart TactileGeo to use Model B.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  async function remove() {
    try {
      await shell.clearModelBKey()
      setStored(false)
      setMessage('Key removed. Restart TactileGeo to switch Model B off.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <div className="desktop-download">
      <button type="button" className="template-button" aria-expanded={open} onClick={() => setOpen(value => !value)}>
        Model B key {stored ? '(saved)' : '(not set)'}
      </button>
      {open && (
        <form className="model-b-key" onSubmit={event => { event.preventDefault(); void save() }}>
          <small>Optional. Model B needs internet and your own OpenRouter API key. Model A works offline without it.</small>
          <label>
            OpenRouter API key
            <input type="password" autoComplete="off" value={key} onChange={event => setKey(event.target.value)} />
          </label>
          <div className="export-warning-actions">
            <button type="submit" disabled={!key.trim()}>Save key</button>
            {stored && <button type="button" className="muted" onClick={() => void remove()}>Remove key</button>}
          </div>
        </form>
      )}
      {message && <small role="status">{message}</small>}
    </div>
  )
}
