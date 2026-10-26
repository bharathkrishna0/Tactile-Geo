import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react'
import { batchEdit, patchElement, processSession, uploadImage } from './api/client'
import AiGeometryView from './components/AiGeometryView'
import ElementList from './components/ElementList'
import Inspector from './components/Inspector'
import ModelBPanel from './components/ModelBPanel'
import OriginalView from './components/OriginalView'
import TactileOutputView from './components/TactileOutputView'
import TemplateDemo from './components/TemplateDemo'
import DesktopDownload from './components/DesktopDownload'
import { EngineBanner, ModelBKeySettings, useEngineState } from './components/DesktopEngine'
import { desktopConfig } from './lib/desktop'
import { desktopShell } from './lib/desktopShell'
import ViewTabs from './components/ViewTabs'
import { exportBlockReason } from './lib/exportGate'
import {
  applyModelBFindings,
  cancelModelBJob,
  fetchModelBAvailability,
  fetchModelBJob,
  fetchModelBFusion,
  recordModelBDecision,
  requestModelB,
} from './lib/modelBApi'
import type { ModelBAvailability } from './lib/modelBTypes'
import type { AnalysisResult, DetectedElement, ElementEdit, ViewId } from './types'

const acceptedTypes = ['image/png', 'image/jpeg']

/**
 * Model B client functions, passed in as a prop so the panel is testable without
 * module mocking and so the whole feature can be stubbed in one place.
 */
const modelBApi = {
  request: requestModelB,
  poll: fetchModelBJob,
  cancel: cancelModelBJob,
  fusion: fetchModelBFusion,
  decide: recordModelBDecision,
  apply: applyModelBFindings,
}

function readinessScore(result: AnalysisResult | null): number {
  if (!result?.qa_report) return 0
  return result.qa_report.score_0_100 ?? Math.round(result.qa_report.overall_score * 100)
}

function readyMessage(result: AnalysisResult): string {
  const count = result.semantic_geometry?.element_count ?? 0
  const figures = result.figures ?? []
  const figure = result.processing_params?.figure
  const where = figure != null && figures[figure] ? ` in figure ${figures[figure].number} of ${figures.length}` : ''
  return `Ready: ${count} elements detected${where}. Review and correct as needed.`
}

function defaultSelection(result: AnalysisResult | null): string | null {
  const elements = result?.semantic_geometry?.elements ?? []
  if (elements.length === 0) return null
  const review = elements.find(el => el.needs_review)
  return (review ?? elements[0]).id
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const templateButtonRef = useRef<HTMLButtonElement>(null)
  const [templateMode, setTemplateMode] = useState(false)
  const returningFromTemplate = useRef(false)
  const [file, setFile] = useState<File | null>(null)
  const [fileUrl, setFileUrl] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<ViewId>('ai')
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [modelBAvailability, setModelBAvailability] = useState<ModelBAvailability | null>(null)
  const [desktop] = useState(desktopConfig)
  const engine = useEngineState(desktop, desktopShell)
  const engineReady = engine.state === 'ready'

  // Availability is fetched once on mount and never blocks the Model A flow.
  // A backend that has Model B switched off, or that is down entirely, must not
  // stand between a teacher and the deterministic result.
  useEffect(() => {
    if (!engineReady) return
    let active = true
    fetchModelBAvailability()
      .then(value => { if (active) setModelBAvailability(value) })
      .catch(() => { if (active) setModelBAvailability({ available: false, enabled: false, reason: 'The Model B check is unavailable on this server.', provider: '', model: '' }) })
    return () => { active = false }
  }, [engineReady])

  const selectedElement: DetectedElement | null = analysis?.semantic_geometry?.elements.find(el => el.id === selectedId) ?? null
  const semantic = analysis?.semantic_geometry
  const relationships = semantic?.relationships ?? []

  function chooseFile(selected: File | undefined) {
    setError(''); setStatus(''); setAnalysis(null); setSelectedId(null); setActiveView('ai')
    if (fileUrl) URL.revokeObjectURL(fileUrl)
    if (!selected) { setFile(null); setFileUrl(null); return }
    if (!acceptedTypes.includes(selected.type)) { setFile(null); setFileUrl(null); setError('Choose a PNG, JPG, or JPEG image.'); return }
    if (selected.size > 10 * 1024 * 1024) { setFile(null); setFileUrl(null); setError('Choose an image smaller than 10 MB.'); return }
    setFile(selected)
    setFileUrl(URL.createObjectURL(selected))
  }

  async function runUpload() {
    if (!file) return
    try {
      setError(''); setBusy(true); setStatus('Uploading worksheet…')
      const session = await uploadImage(file)
      setStatus('Analyzing geometry…')
      const result = await processSession(session.session_id)
      setAnalysis(result)
      setSelectedId(defaultSelection(result))
      setStatus(readyMessage(result))
    } catch (requestError) {
      setStatus('')
      setError(requestError instanceof Error ? requestError.message : 'An unexpected error occurred.')
    } finally {
      setBusy(false)
    }
  }

  async function chooseFigure(value: string) {
    if (!analysis?.session_id) return
    try {
      setError(''); setBusy(true)
      setStatus(value === 'page' ? 'Analyzing the whole page…' : `Analyzing figure ${Number(value) + 1}…`)
      const next = await processSession(analysis.session_id, value === 'page' ? { whole_page: true } : { figure: Number(value) })
      setAnalysis(next)
      setSelectedId(defaultSelection(next))
      setStatus(readyMessage(next))
    } catch (requestError) {
      setStatus('')
      setError(requestError instanceof Error ? requestError.message : 'That figure could not be analysed.')
    } finally {
      setBusy(false)
    }
  }

  async function handleEdit(elementId: string, edit: ElementEdit) {
    if (!analysis || !analysis.session_id) return
    try {
      setError(''); setBusy(true); setStatus('Applying correction and regenerating…')
      const next = await patchElement(analysis.session_id, elementId, edit)
      setAnalysis(next)
      const remaining = next.semantic_geometry?.elements.map(el => el.id) ?? []
      setSelectedId(remaining.includes(elementId) ? elementId : defaultSelection(next))
      setStatus('Correction applied. Tactile output and QA updated.')
    } catch (requestError) {
      setStatus('')
      setError(requestError instanceof Error ? requestError.message : 'That correction could not be applied.')
    } finally {
      setBusy(false)
    }
  }

  function exportSvg() {
    const svg = analysis?.tactile_svg
    if (!svg) return
    // Defence in depth: the button is disabled when export is blocked, but the
    // download must also be refused here so no code path can emit a file the QA
    // gate rejected. There is deliberately no "export anyway" bypass.
    if (exportBlock) {
      setError(exportBlock.message)
      return
    }
    setError('')
    const figureNumber = currentFigure ? `-figure-${currentFigure.number}` : ''
    const fileName = `tactile-${analysis.session_id || 'diagram'}${figureNumber}.svg`
    if (desktop) {
      desktopShell.saveSvg(svg, fileName)
        .then(path => setStatus(path ? `Saved ${path}` : 'Export cancelled.'))
        .catch(saveError => setError(saveError instanceof Error ? saveError.message : String(saveError)))
      return
    }
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    link.click()
    URL.revokeObjectURL(url)
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) { chooseFile(event.target.files?.[0]) }
  function onDrop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); chooseFile(event.dataTransfer.files[0]) }

  const qualityPasses = analysis?.quality_report?.passes_gate ?? true
  const reviewCount = semantic?.low_confidence_count ?? 0
  const score = readinessScore(analysis)
  const exportBlock = exportBlockReason(analysis?.qa_report)
  const figures = analysis?.figures ?? []
  const figureIndex = analysis?.processing_params?.figure
  const currentFigure = figureIndex != null ? figures[figureIndex] ?? null : null
  const explanations = [
    ...(semantic?.explanations ?? []),
    ...(analysis?.simplified_geometry?.explanations ?? []).map(message => ({ stage: 'simplification', message })),
  ]

  function exitTemplate() {
    returningFromTemplate.current = true
    setTemplateMode(false)
  }

  useEffect(() => {
    if (!templateMode && returningFromTemplate.current) {
      returningFromTemplate.current = false
      templateButtonRef.current?.focus()
    }
  }, [templateMode])

  return (
    <main className="page"><section className="content">
      <p className="eyebrow">TACTILEGEO</p>
      <h1>Turn a worksheet into a tactile diagram.</h1>
      <p className="intro">Upload a clear PNG or JPG. TactileGeo extracts the geometry, shows its confidence, lets a teacher correct it, and produces a print-ready tactile SVG.</p>

      {templateMode ? (
        <TemplateDemo onExit={exitTemplate} />
      ) : (<>
      <div className="mode-bar">
        <span className="mode-badge live">LIVE PROCESSING</span>
        <button type="button" ref={templateButtonRef} className="template-button" onClick={() => setTemplateMode(true)}>
          Template (Demo)
        </button>
        {desktop ? <ModelBKeySettings shell={desktopShell} /> : <DesktopDownload />}
      </div>
      {desktop && <EngineBanner state={engine.state} error={engine.error} config={desktop} shell={desktopShell} />}

      <input ref={inputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg" aria-label="Choose a worksheet image" onChange={onFileChange} />
      <div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
        <p>Drag a worksheet image here</p>
        <span>or</span>
        <button type="button" onClick={() => inputRef.current?.click()} disabled={busy || !engineReady}>Choose image</button>
        <small>Supports PNG, JPG, and JPEG up to 10 MB.</small>
      </div>

      {file && (
        <div className="selected-file">
          <span>{file.name}</span>
          <button type="button" onClick={runUpload} disabled={busy || !engineReady}>{analysis ? 'Re-run analysis' : 'Create tactile preview'}</button>
        </div>
      )}
      {status && <p className="status" role="status">{status}</p>}
      {error && <p className="error" role="alert">{error}</p>}

      {fileUrl && file && analysis && (
        <div className="workspace">
          <div className="pipeline-summary" aria-label="Pipeline summary">
            <div className="summary-item"><span className="summary-value">{qualityPasses ? 'OK' : 'Review'}</span><span className="summary-label">Image quality</span></div>
            <div className="summary-item"><span className="summary-value">{semantic?.element_count ?? 0}</span><span className="summary-label">Elements</span></div>
            <div className="summary-item"><span className={reviewCount > 0 ? 'summary-value attention' : 'summary-value'}>{reviewCount}</span><span className="summary-label">To review</span></div>
            <div className="summary-item"><span className="summary-value score">{score}</span><span className="summary-label">Readiness</span></div>
            <button
              type="button"
              className="export-button"
              disabled={!analysis.tactile_svg || exportBlock !== null}
              aria-describedby={exportBlock ? 'export-blocked-reason' : undefined}
              onClick={exportSvg}
            >
              Export print-ready SVG
            </button>
          </div>

          {exportBlock && (
            <div className="export-warning" role="alert" id="export-blocked-reason">
              <h3>Print-ready export blocked</h3>
              <p>{exportBlock.message}</p>
              <ul>{exportBlock.issues.map(issue => <li key={`${issue.check}-${issue.message}`}>{issue.message}</li>)}</ul>
              <div className="export-warning-actions">
                <button type="button" onClick={() => setActiveView('tactile')}>Show blocking issues</button>
                <button type="button" className="muted" onClick={() => setActiveView('ai')}>Go to geometry view</button>
              </div>
            </div>
          )}

          {figures.length >= 2 && (
            <div className="figure-picker">
              <label htmlFor="figure-select">This page has {figures.length} separate figures. Tactile sheet for:</label>
              <select
                id="figure-select"
                value={currentFigure ? String(figureIndex) : 'page'}
                disabled={busy}
                onChange={event => chooseFigure(event.target.value)}
              >
                {figures.map((figure, index) => (
                  <option key={figure.number} value={String(index)}>Figure {figure.number}</option>
                ))}
                <option value="page">Whole page (all figures on one sheet)</option>
              </select>
            </div>
          )}

          <ViewTabs active={activeView} onSelect={setActiveView} />

          <div className="stage-layout">
            <div className="stage-main">
              {activeView === 'original' && (
                <OriginalView src={fileUrl} fileName={file.name} quality={analysis.quality_report} />
              )}
              {activeView === 'ai' && semantic && (
                <AiGeometryView
                  src={fileUrl}
                  fileName={file.name}
                  elements={semantic.elements}
                  width={semantic.image_width || analysis.quality_report?.image_width || 0}
                  height={semantic.image_height || analysis.quality_report?.image_height || 0}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  figure={currentFigure}
                />
              )}
              {activeView === 'tactile' && (
                <TactileOutputView
                  tactileSvg={analysis.tactile_svg ?? ''}
                  qa={analysis.qa_report}
                  simplified={analysis.simplified_geometry}
                  explanations={explanations}
                />
              )}
            </div>

            {(activeView === 'ai' || activeView === 'tactile') && (
              <div className="stage-side">
                <section className="panel-card">
                  <h3>Detected elements</h3>
                  <ElementList elements={semantic?.elements ?? []} selectedId={selectedId} onSelect={setSelectedId} />
                </section>
                <Inspector element={selectedElement} elements={semantic?.elements ?? []} relationships={relationships} onEdit={handleEdit} busy={busy} />
              </div>
            )}
          </div>

          <ModelBPanel
            sessionId={analysis.session_id}
            availability={modelBAvailability}
            modelAReady={Boolean(analysis.session_id)}
            api={modelBApi}
            onApplied={next => {
              setAnalysis(next)
              setSelectedId(current => (next.semantic_geometry?.elements.some(el => el.id === current) ? current : defaultSelection(next)))
              setStatus('Accepted findings applied. Tactile output and QA updated.')
            }}
          />
        </div>
      )}
      </>)}
    </section></main>
  )
}