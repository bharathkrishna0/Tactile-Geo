import type { QualityReport } from '../types'

interface OriginalViewProps {
  src: string
  fileName: string
  quality: QualityReport | null | undefined
}

export default function OriginalView({ src, fileName, quality }: OriginalViewProps) {
  const issues = quality?.issues ?? []
  const errors = issues.filter(issue => issue.severity === 'error')
  const warnings = issues.filter(issue => issue.severity === 'warning')
  const infos = issues.filter(issue => issue.severity === 'info')
  const passes = quality ? quality.passes_gate : true

  return (
    <div className="view-panel" role="tabpanel" id="panel-original" aria-labelledby="tab-original">
      <img className="original-image" src={src} alt={`Uploaded worksheet image: ${fileName}`} />
      <dl className="meta-list">
        <div><dt>File</dt><dd>{fileName}</dd></div>
        {quality && (
          <>
            <div><dt>Dimensions</dt><dd>{quality.image_width} × {quality.image_height} px</dd></div>
            <div><dt>Quality</dt><dd><span className={passes ? 'status-badge pass' : 'status-badge fail'}>{passes ? 'Looks good' : 'Needs review'}</span></dd></div>
          </>
        )}
      </dl>
      {errors.length > 0 && (
        <section className="issue-block error-block" role="alert">
          <h3>Quality problems</h3>
          {errors.map(issue => <p key={issue.check}><strong>Error:</strong> {issue.message}</p>)}
        </section>
      )}
      {warnings.length > 0 && (
        <section className="issue-block">
          <h3>Quality warnings</h3>
          {warnings.map(issue => <p key={issue.check}><strong>Warning:</strong> {issue.message}</p>)}
        </section>
      )}
      {infos.length > 0 && (
        <section className="issue-block quiet">
          <h3>Notes</h3>
          {infos.map(issue => <p key={issue.check}>{issue.message}</p>)}
        </section>
      )}
      {!quality && <p className="muted">Quality details appear after processing.</p>}
    </div>
  )
}