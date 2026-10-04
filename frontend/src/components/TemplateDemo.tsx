import { KeyboardEvent, ReactNode, RefObject, useEffect, useRef, useState } from 'react'
import {
  TEMPLATE_STAGES,
  TemplateFusionReview,
  TemplateMetadata,
  humanize,
  loadTemplateMetadata,
  templateAsset,
} from '../demo/templateDemo'

interface TemplateDemoProps {
  onExit: () => void
  loadMetadata?: () => Promise<TemplateMetadata>
}

const EXPORT_NAME = 'tactile-template-pie-chart.svg'

function Stat({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div className="template-stat">
      <span className="template-stat-value">{value}</span>
      <span className="template-stat-label">{label}</span>
    </div>
  )
}

function CountList({ counts, limit }: { counts: Record<string, number>; limit?: number }) {
  const entries = Object.entries(counts).slice(0, limit)
  return (
    <ul className="template-counts">
      {entries.map(([key, count]) => (
        <li key={key}><span>{humanize(key)}</span><strong>{count}</strong></li>
      ))}
    </ul>
  )
}

function Overlay({ overlay, alt, overlayAlt }: { overlay: 'model-a.svg' | 'model-b.svg' | 'fusion.svg'; alt: string; overlayAlt: string }) {
  return (
    <div className="template-overlay">
      <img src={templateAsset('original.png')} alt={alt} className="template-image dimmed" />
      <img src={templateAsset(overlay)} alt={overlayAlt} className="template-overlay-layer" />
    </div>
  )
}

function fusionOutcome(review: TemplateFusionReview): { label: string; tone: 'good' | 'neutral' | 'warn' } {
  if (review.applied) return { label: 'Confirmed', tone: 'good' }
  if (!review.model_a_id) return { label: 'Model B only: not embossed', tone: 'warn' }
  if (review.model_a_type === 'text_label') return { label: 'Kept as Model A label', tone: 'neutral' }
  return { label: 'Type conflict: left for teacher', tone: 'warn' }
}

export default function TemplateDemo({ onExit, loadMetadata = loadTemplateMetadata }: TemplateDemoProps) {
  const [metadata, setMetadata] = useState<TemplateMetadata | null>(null)
  const [error, setError] = useState('')
  const [index, setIndex] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([])
  const closeRef = useRef<HTMLButtonElement | null>(null)
  const fullscreenButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    let active = true
    loadMetadata()
      .then(value => { if (active) setMetadata(value) })
      .catch(() => { if (active) setError('The template files could not be loaded.') })
    return () => { active = false }
  }, [loadMetadata])

  useEffect(() => {
    if (fullscreen) closeRef.current?.focus()
  }, [fullscreen])

  const stage = TEMPLATE_STAGES[index]
  const last = TEMPLATE_STAGES.length - 1

  function goTo(next: number, focusTab = false) {
    const clamped = Math.max(0, Math.min(last, next))
    setIndex(clamped)
    if (focusTab) tabRefs.current[clamped]?.focus()
  }

  function onTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const keys: Record<string, number> = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: last }
    if (event.key in keys) {
      event.preventDefault()
      goTo(keys[event.key], true)
    }
  }

  function closeFullscreen() {
    setFullscreen(false)
    fullscreenButtonRef.current?.focus()
  }

  return (
    <section className="template-demo" aria-labelledby="template-title">
      <header className="template-header">
        <div>
          <p className="template-badges">
            <span className="mode-badge template">TEMPLATE DEMO</span>
            <span className="template-precomputed">Precomputed successful example</span>
          </p>
          <h2 id="template-title">{metadata?.title ?? 'Pie chart'}: image to tactile SVG</h2>
          <p className="muted">Every stage below was computed ahead of time by the real pipeline. Nothing is sent to the server or to an AI service while you browse it.</p>
        </div>
        <button type="button" className="secondary" onClick={onExit}>Exit Template</button>
      </header>

      <div className="template-stepper" role="tablist" aria-label="Template demo stages">
        {TEMPLATE_STAGES.map((item, position) => (
          <button
            key={item.id}
            ref={element => { tabRefs.current[position] = element }}
            type="button"
            role="tab"
            id={`template-tab-${item.id}`}
            aria-selected={position === index}
            aria-controls="template-panel"
            tabIndex={position === index ? 0 : -1}
            className={[
              'template-step',
              position === index ? 'active' : '',
              position < index ? 'done' : '',
            ].join(' ').trim()}
            onClick={() => goTo(position)}
            onKeyDown={onTabKeyDown}
          >
            <span className="template-step-number" aria-hidden="true">{position + 1}</span>
            <span className="template-step-label">{item.label}</span>
          </button>
        ))}
      </div>

      <div
        className="template-panel"
        role="tabpanel"
        id="template-panel"
        aria-labelledby={`template-tab-${stage.id}`}
      >
        <div className="template-panel-heading">
          <p className="template-progress">Stage {index + 1} of {TEMPLATE_STAGES.length}</p>
          <h3>{stage.title}</h3>
          <p className="template-description">{stage.description}</p>
        </div>

        {error && <p className="error" role="alert">{error}</p>}
        {!metadata && !error && <p className="status" role="status">Loading the template…</p>}
        {metadata && (
          <div className="template-body">
            <StageContent
              stageId={stage.id}
              metadata={metadata}
              fullscreenButtonRef={fullscreenButtonRef}
              onFullscreen={() => setFullscreen(true)}
            />
          </div>
        )}
      </div>

      <nav className="template-nav" aria-label="Template demo navigation">
        <button type="button" className="secondary" onClick={() => goTo(index - 1)} disabled={index === 0}>Previous</button>
        <span className="template-nav-hint">Tip: use the arrow keys on the stage tabs.</span>
        <button type="button" onClick={() => goTo(index + 1)} disabled={index === last}>
          {index === last ? 'Done' : `Next: ${TEMPLATE_STAGES[index + 1].label}`}
        </button>
      </nav>

      {fullscreen && (
        <div
          className="template-fullscreen"
          role="dialog"
          aria-modal="true"
          aria-label="Final tactile SVG, fullscreen"
          onKeyDown={event => { if (event.key === 'Escape') closeFullscreen() }}
        >
          <div className="template-fullscreen-bar">
            <span>Final tactile SVG · {metadata?.final.page}</span>
            <button type="button" ref={closeRef} onClick={closeFullscreen}>Close fullscreen</button>
          </div>
          <img src={templateAsset('final.svg')} alt="Final tactile SVG of the pie chart with Braille labels" />
        </div>
      )}
    </section>
  )
}

interface StageContentProps {
  stageId: (typeof TEMPLATE_STAGES)[number]['id']
  metadata: TemplateMetadata
  fullscreenButtonRef: RefObject<HTMLButtonElement | null>
  onFullscreen: () => void
}

function StageContent({ stageId, metadata, fullscreenButtonRef, onFullscreen }: StageContentProps) {
  const { model_a: modelA, model_b: modelB, fusion, simplification, braille, qa, final } = metadata

  if (stageId === 'original') {
    return (
      <>
        <figure className="template-visual">
          <img src={templateAsset('original.png')} alt="Original pie chart titled Favourite Fruits with Apple, Mango and Banana sectors and a legend" className="template-image" />
        </figure>
        <aside className="template-side">
          <section className="template-card">
            <h4>Source image</h4>
            <div className="template-stats">
              <Stat value={`${metadata.image.width}×${metadata.image.height}`} label="Pixels" />
              <Stat value="3" label="Sectors" />
            </div>
            <p className="muted">A printed pie chart: plain, hatched and dotted sectors, a legend and a title. It is shown unmodified.</p>
          </section>
        </aside>
      </>
    )
  }

  if (stageId === 'model-a') {
    return (
      <>
        <figure className="template-visual">
          <Overlay overlay="model-a.svg" alt="Original pie chart, dimmed" overlayAlt={`Model A geometry overlay with ${modelA.element_count} elements`} />
          <figcaption className="template-legend">
            <span className="swatch circle" />Circle <span className="swatch line" />Line <span className="swatch rect" />Rectangle <span className="swatch point" />Point <span className="swatch label" />Text
          </figcaption>
        </figure>
        <aside className="template-side">
          <section className="template-card">
            <h4 className="template-card-eyebrow">MODEL A — GEOMETRY ANALYSIS</h4>
            <div className="template-stats">
              <Stat value={modelA.element_count} label="Elements" />
              <Stat value={modelA.relationship_count} label="Relationships" />
              <Stat value={modelA.labels.length} label="Text labels" />
            </div>
            <CountList counts={modelA.by_type} />
          </section>
          <section className="template-card">
            <h4>Text read by OCR</h4>
            <ul className="template-chips">
              {modelA.labels.map(label => <li key={label.id}>{label.text}</li>)}
            </ul>
          </section>
          <p className="template-note">Model A is the geometry authority. The hatching and title strokes are also traced as line segments; simplification and QA deal with them next.</p>
        </aside>
      </>
    )
  }

  if (stageId === 'model-b') {
    const textIds = new Set(modelB.entities.filter(entity => entity.kind === 'unknown').map(entity => entity.id))
    const meaningful = modelB.relationships.filter(rel => rel.kind === 'labels' && !textIds.has(rel.from_id))
    return (
      <>
        <figure className="template-visual">
          <Overlay overlay="model-b.svg" alt="Original pie chart, dimmed" overlayAlt={`Model B regions: ${modelB.entities.length} advisory findings`} />
          <figcaption className="muted">Dashed boxes are Model B regions. They are not used as coordinates.</figcaption>
        </figure>
        <aside className="template-side">
          <section className="template-card">
            <p className="template-card-top">
              <span className="mode-badge advisory">Advisory AI</span>
              <span className="muted">{metadata.provenance.model_b.model}</span>
            </p>
            <h4>What the diagram shows</h4>
            <p>{modelB.description}</p>
          </section>
          <section className="template-card">
            <h4>Meaning Model B adds</h4>
            <ul className="template-list">
              {meaningful.map(rel => <li key={`${rel.from_id}-${rel.to_id}`}>{rel.evidence}</li>)}
            </ul>
            <ul className="template-chips">
              {modelB.text_items.map(item => <li key={item.text}>{item.text} <span>· {humanize(item.role)}</span></li>)}
            </ul>
          </section>
          {modelB.uncertainties.length > 0 && (
            <section className="template-card">
              <h4>Uncertainty it reported</h4>
              <ul className="template-list">
                {modelB.uncertainties.map(item => <li key={item.kind}>{item.note}</li>)}
              </ul>
            </section>
          )}
          <p className="template-note">Model B provides semantic interpretation. Final geometry is generated by the deterministic tactile pipeline. {metadata.provenance.model_b.note}</p>
        </aside>
      </>
    )
  }

  if (stageId === 'fusion') {
    return (
      <>
        <figure className="template-visual">
          <div className="template-flow" aria-label="Fusion inputs and output">
            <span className="template-flow-node">Model A<small>precise geometry</small></span>
            <span className="template-flow-plus" aria-hidden="true">+</span>
            <span className="template-flow-node advisory">Model B<small>semantic findings</small></span>
            <span className="template-flow-plus" aria-hidden="true">→</span>
            <span className="template-flow-node fused">Semantic Geometry v2<small>fused, teacher-approved</small></span>
          </div>
          <Overlay overlay="fusion.svg" alt="Original pie chart, dimmed" overlayAlt={`Fused geometry: ${fusion.applied} Model A elements confirmed by Model B`} />
          <figcaption className="muted">Green: Model A elements confirmed by an accepted Model B finding. Grey: unchanged Model A geometry.</figcaption>
        </figure>
        <aside className="template-side">
          <section className="template-card">
            <div className="template-stats">
              <Stat value={fusion.applied} label="Confirmed" />
              <Stat value={fusion.reviews.length - fusion.applied} label="Not applied" />
              <Stat value={fusion.coordinates_changed} label="Coordinates changed" />
            </div>
            <table className="template-table">
              <caption className="visually-hidden">Model B findings and how fusion handled them</caption>
              <thead><tr><th scope="col">Model B</th><th scope="col">Model A</th><th scope="col">Result</th></tr></thead>
              <tbody>
                {fusion.reviews.map(review => {
                  const outcome = fusionOutcome(review)
                  return (
                    <tr key={review.model_b_id}>
                      <td>{humanize(review.model_b_kind === 'unknown' ? 'text' : review.model_b_kind)}</td>
                      <td>{review.model_a_type ? humanize(review.model_a_type) : '—'}</td>
                      <td><span className={`template-outcome ${outcome.tone}`}>{outcome.label}</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </section>
          <p className="template-note">{metadata.provenance.teacher} Confirmed elements are protected from simplification.</p>
        </aside>
      </>
    )
  }

  if (stageId === 'simplified') {
    const merges = simplification.actions.merged_collinear ?? 0
    return (
      <>
        <figure className="template-visual">
          <div className="template-compare">
            <div>
              <p className="template-compare-label">Visual</p>
              <img src={templateAsset('original.png')} alt="Original visual pie chart" className="template-image" />
            </div>
            <div>
              <p className="template-compare-label">Tactile geometry</p>
              <img src={templateAsset('tactile.svg')} alt="Simplified tactile geometry of the pie chart, before Braille" className="template-image paper" />
            </div>
          </div>
        </figure>
        <aside className="template-side">
          <section className="template-card">
            <div className="template-stats">
              <Stat value={`${simplification.before} → ${simplification.after}`} label="Features" />
              <Stat value={simplification.merged} label="Segments merged" />
              <Stat value={simplification.removed} label="Removed as noise" />
            </div>
            <ul className="template-list checks">
              <li>Geometry consolidated: {simplification.merged} broken segments joined into {merges} lines.</li>
              <li>No shading or fills: only raised lines and points remain.</li>
              <li>Text kept aside for Braille ({braille.labels.length} labels).</li>
              <li>Lines set to {final.stroke_width_pt} pt for touch.</li>
            </ul>
          </section>
          <details className="template-card">
            <summary>What was merged</summary>
            <ul className="template-list">
              {simplification.explanations.map(item => <li key={item}>{item}</li>)}
            </ul>
          </details>
        </aside>
      </>
    )
  }

  if (stageId === 'braille') {
    return (
      <>
        <figure className="template-visual">
          <img src={templateAsset('final.svg')} alt="Tactile pie chart with Braille labels embossed as dots" className="template-image paper" />
        </figure>
        <aside className="template-side">
          <section className="template-card">
            <h4>Braille labels</h4>
            <ol className="template-braille">
              {braille.labels.map(label => (
                <li key={label.id}>
                  <span className="template-braille-text">{label.text}</span>
                  <span className="template-braille-cells" aria-label={`Braille for ${label.text}`}>{label.braille}</span>
                  <span className="template-braille-pos">{label.position_mm[0]} mm, {label.position_mm[1]} mm</span>
                </li>
              ))}
            </ol>
            <p className="muted">Listed in reading order, top to bottom. Translated with Liblouis ({braille.table}, UEB grade 2) and embossed at standard cell size.</p>
          </section>
        </aside>
      </>
    )
  }

  return (
    <>
      <figure className="template-visual">
        <img src={templateAsset('final.svg')} alt="Final print-ready tactile SVG of the pie chart" className="template-image paper" />
        <div className="template-actions">
          <a className="button-link secondary" href={templateAsset('final.svg')} target="_blank" rel="noreferrer">Preview SVG</a>
          <button type="button" className="secondary" ref={fullscreenButtonRef} onClick={onFullscreen}>View Fullscreen</button>
          <a className="button-link" href={templateAsset('final.svg')} download={EXPORT_NAME}>Export SVG</a>
        </div>
      </figure>
      <aside className="template-side">
        <section className="template-card">
          <p className="template-card-top">
            <span className={qa.passes ? 'mode-badge pass' : 'mode-badge fail'}>{qa.passes ? 'QA passed' : 'QA blocked'}</span>
          </p>
          <div className="template-stats">
            <Stat value={qa.score} label="Readiness" />
            <Stat value={final.feature_count} label="Raised features" />
            <Stat value={final.label_count} label="Braille labels" />
          </div>
          <ul className="template-list checks">
            <li>{final.page} vector, {final.width_mm} × {final.height_mm} mm</li>
            <li>{final.stroke_width_pt} pt raised lines</li>
            <li>Braille as embossed dots, not a font</li>
          </ul>
        </section>
        <section className="template-card">
          <h4>QA notes for the teacher</h4>
          <CountList counts={qa.issues_by_check} />
          <p className="muted">These are warnings, not blockers. Export is blocked only by QA errors ({qa.errors.length} here).</p>
        </section>
      </aside>
    </>
  )
}
