import { ChangeEvent, DragEvent, useRef, useState } from 'react'
import { batchEdit, patchElement, processSession, uploadImage } from './api/client'
import AiGeometryView from './components/AiGeometryView'
import ElementList from './components/ElementList'
import Inspector from './components/Inspector'
import OriginalView from './components/OriginalView'
import TactileOutputView from './components/TactileOutputView'
import ViewTabs from './components/ViewTabs'
import type { AnalysisResult, DetectedElement, ElementEdit, QAIssue, ViewId } from './types'

const acceptedTypes = ['image/png', 'image/jpeg']

function readinessScore(result: AnalysisResult | null): number {
  if (!result?.qa_report) return 0
  return result.qa_report.score_0_100 ?? Math.round(result.qa_report.overall_score * 100)
}

function defaultSelection(result: AnalysisResult | null): string | null {
  const elements = result?.semantic_geometry?.elements ?? []
  if (elements.length === 0) return null
  const review = elements.find(el => el.needs_review)
  return (review ?? elements[0]).id
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [fileUrl, setFileUrl] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<ViewId>('ai')
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [exportWarning, setExportWarning] = useState<QAIssue[] | null>(null)

  const selectedElement: DetectedElement | null = analysis?.semantic_geometry?.elements.find(el => el.id === selectedId) ?? null
  const semantic = analysis?.semantic_geometry
  const relationships = semantic?.relationships ?? []

  function chooseFile(selected: File | undefined) {
    setError(''); setExportWarning(null); setAnalysis(null); setSelectedId(null); setActiveView('ai')
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
      setError(''); setExportWarning(null); setBusy(true); setStatus('Uploading worksheet…')
      const session = await uploadImage(file)
      setStatus('Analyzing geometry…')
      const result = await processSession(session.session_id)
      setAnalysis(result)
      setSelectedId(defaultSelection(result))
      setStatus(`Ready: ${result.semantic_geometry?.element_count ?? 0} elements detected. Review and correct as needed.`)
    } catch (requestError) {
      setStatus('')
      setError(requestError instanceof Error ? requestError.message : 'An unexpected error occurred.')
    } finally {
      setBusy(false)
    }
  }

  async function handleEdit(elementId: string, edit: ElementEdit) {
    if (!analysis || !analysis.session_id) return
    try {
      setError(''); setExportWarning(null); setBusy(true); setStatus('Applying correction and regenerating…')
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
    const critical = analysis?.qa_report?.issues.filter(issue => issue.severity === 'error') ?? []
    if (critical.length > 0 && exportWarning === null) { setExportWarning(critical); return }
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `tactile-${analysis.session_id || 'diagram'}.svg`
    link.click()
    URL.revokeObjectURL(url)
    setExportWarning(null)
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) { chooseFile(event.target.files?.[0]) }
  function onDrop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); chooseFile(event.dataTransfer.files[0]) }

  const qualityPasses = analysis?.quality_report?.passes_gate ?? true
  const reviewCount = semantic?.low_confidence_count ?? 0
  const score = readinessScore(analysis)
  const explanations = [
    ...(semantic?.explanations ?? []),
    ...(analysis?.simplified_geometry?.explanations ?? []).map(message => ({ stage: 'simplification', message })),
  ]

  return (
    <main className="page"><section className="content">
      <p className="eyebrow">TACTILEGEO</p>
      <h1>Turn a worksheet into a tactile diagram.</h1>
      <p className="intro">Upload a clear PNG or JPG. TactileGeo extracts the geometry, shows its confidence, lets a teacher correct it, and produces a print-ready tactile SVG.</p>

      <input ref={inputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg" onChange={onFileChange} />
      <div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
        <p>Drag a worksheet image here</p>
        <span>or</span>
        <button type="button" onClick={() => inputRef.current?.click()} disabled={busy}>Choose image</button>
        <small>Supports PNG, JPG, and JPEG up to 10 MB.</small>
      </div>

      {file && (
        <div className="selected-file">
          <span>{file.name}</span>
          <button type="button" onClick={runUpload} disabled={busy}>{analysis ? 'Re-run analysis' : 'Create tactile preview'}</button>
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
            <button type="button" className="export-button" disabled={!analysis.tactile_svg} onClick={exportSvg}>Export print-ready SVG</button>
          </div>

          {exportWarning && (
            <div className="export-warning" role="alert">
              <h3>Before you export</h3>
              <p>{exportWarning.length} critical QA {exportWarning.length === 1 ? 'violation remains' : 'violations remain'}:</p>
              <ul>{exportWarning.map(issue => <li key={issue.message}>{issue.message}</li>)}</ul>
              <div className="export-warning-actions">
                <button type="button" onClick={exportSvg}>Export anyway</button>
                <button type="button" className="muted" onClick={() => setExportWarning(null)}>Go back</button>
              </div>
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
        </div>
      )}
    </section></main>
  )
}