import { CSSProperties, useMemo } from 'react'
import { sanitizeSvg } from '../lib/svgSafety'
import type { QAReport, SimplifiedGeometry, TransformationExplanation } from '../types'

interface TactileOutputViewProps {
  tactileSvg: string
  qa: QAReport | null | undefined
  simplified: SimplifiedGeometry | null | undefined
  explanations: TransformationExplanation[]
}

export default function TactileOutputView({ tactileSvg, qa, simplified, explanations }: TactileOutputViewProps) {
  const score = qa ? qa.score_0_100 : 0
  const issues = qa?.issues ?? []
  const critical = issues.filter(issue => issue.severity === 'error')
  const warnings = issues.filter(issue => issue.severity === 'warning')
  const notes = issues.filter(issue => issue.severity === 'info')
  const safeSvg = useMemo(() => sanitizeSvg(tactileSvg), [tactileSvg])
  const noGeometry = issues.some(issue => issue.check === 'no_tactile_geometry')

  return (
    <div className="view-panel" role="tabpanel" id="panel-tactile" aria-labelledby="tab-tactile">
      <section className="readiness-card">
        <div className="score-ring" style={{ '--score': `${Math.max(0, Math.min(100, score))}` } as CSSProperties}>
          <span className="score-value">{score}</span>
        </div>
        <div className="readiness-copy">
          <h3>Readiness score</h3>
          <p>{qa ? (qa.passes ? 'Passed — no critical violations.' : `${critical.length} critical ${critical.length === 1 ? 'violation' : 'violations'} need attention.`) : 'No QA data yet.'}</p>
        </div>
      </section>

      <section className="tactile-canvas">
        <h3>Tactile preview</h3>
        <p className="muted">Sized for an A4 sheet: thick raised lines, standard-size Braille, {simplified?.elements.length ?? 0} features after simplification.</p>
        {noGeometry ? (
          <div className="empty-state" role="status">
            <p><strong>No lines or shapes were found in this image.</strong></p>
            <p>Try a brighter, sharper photo taken straight above the page, or raise the edge sensitivity and process again.</p>
          </div>
        ) : (
          <div className="tactile-svg" dangerouslySetInnerHTML={{ __html: safeSvg }} />
        )}
      </section>

      {critical.length > 0 && (
        <section className="qa-critical" role="alert">
          <h3>Critical QA violations</h3>
          {critical.map(issue => <p key={`${issue.check}-${issue.message}`}><strong>Error:</strong> {issue.message}</p>)}
        </section>
      )}
      {warnings.length > 0 && (
        <section className="qa-issues">
          <h3>QA warnings</h3>
          {warnings.map(issue => <p key={`${issue.check}-${issue.message}`}><strong>Warning:</strong> {issue.message}</p>)}
        </section>
      )}
      {notes.length > 0 && (
        <section className="qa-issues quiet">
          <h3>QA notes</h3>
          {notes.map(issue => <p key={`${issue.check}-${issue.message}`}>{issue.message}</p>)}
        </section>
      )}

      {explanations.length > 0 && (
        <section className="explanations">
          <h3>How the tactile diagram was created</h3>
          <ul>{explanations.map((explanation, index) => <li key={`${explanation.stage}-${index}`}>{explanation.message}</li>)}</ul>
        </section>
      )}
    </div>
  )
}