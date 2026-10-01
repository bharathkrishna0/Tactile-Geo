import { ChangeEvent, KeyboardEvent } from 'react'
import type { DetectedElement, ElementEdit, Relationship } from '../types'

const GEOMETRY_TYPES = [
  'point', 'line_segment', 'ray', 'circle', 'ellipse', 'arc', 'triangle', 'rectangle', 'polygon', 'angle', 'axes', 'arrow',
]

const TYPE_LABEL: Record<string, string> = {
  point: 'Point', line_segment: 'Segment', ray: 'Ray', circle: 'Circle', ellipse: 'Ellipse', arc: 'Arc',
  triangle: 'Triangle', rectangle: 'Rectangle', polygon: 'Polygon', angle: 'Angle',
  axes: 'Axes', arrow: 'Arrow', text_label: 'Label',
}

interface InspectorProps {
  element: DetectedElement | null
  elements: DetectedElement[]
  relationships: Relationship[]
  onEdit: (elementId: string, edit: ElementEdit) => void
  busy: boolean
}

export default function Inspector({ element, elements, relationships, onEdit, busy }: InspectorProps) {
  if (!element) {
    return (
      <aside className="inspector empty">
        <h3>Element inspector</h3>
        <p className="muted">Select an element to review its type, confidence, label and relationships.</p>
      </aside>
    )
  }

  const geometryOptions = elements.filter(el => el.type !== 'text_label')
  const labelOptions = elements.filter(el => el.type === 'text_label')
  const isLabel = element.type === 'text_label'
  const relevantRelationships = relationships.filter(rel => rel.element_ids.includes(element.id))
  const originalText = element.geometry.text as string | undefined ?? ''

  return (
    <aside className="inspector" aria-label={`Inspector for ${element.id}`}>
      <h3>Element inspector</h3>
      <dl className="inspector-meta">
        <div><dt>Element</dt><dd>{element.id}</dd></div>
        <div><dt>Detected as</dt><dd>{TYPE_LABEL[element.type] ?? element.type}</dd></div>
        <div><dt>Confidence</dt><dd><span className={`confidence-badge ${element.confidence_level}`}>{element.confidence_level}</span> ({Math.round(element.confidence * 100)}%)</dd></div>
        {element.provenance && <div className="inspector-provenance"><dt>Why</dt><dd>{element.provenance}</dd></div>}
      </dl>

      <label className="field" htmlFor="inspector-type">
        <span>Semantic type</span>
        <select
          id="inspector-type"
          value={element.type}
          disabled={busy}
          onChange={(event: ChangeEvent<HTMLSelectElement>) => onEdit(element.id, { action: 'set_type', type: event.target.value })}
        >
          {GEOMETRY_TYPES.map(type => (
            <option key={type} value={type}>{TYPE_LABEL[type]}</option>
          ))}
        </select>
      </label>

      {isLabel && (
        <label className="field" htmlFor="inspector-text">
          <span>Label text</span>
          <input
            id="inspector-text"
            key={element.id}
            type="text"
            defaultValue={originalText}
            onBlur={event => {
              const value = event.currentTarget.value
              if (value !== originalText) onEdit(element.id, { action: 'set_label', text: value })
            }}
            onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
              if (event.key !== 'Enter') return
              const value = event.currentTarget.value
              if (value !== originalText) onEdit(element.id, { action: 'set_label', text: value })
            }}
          />
        </label>
      )}

      {isLabel ? (
        <label className="field" htmlFor="inspector-assoc2">
          <span>Attach to geometry</span>
          <select
            id="inspector-assoc2"
            value={element.associated_label_id ?? ''}
            disabled={busy}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => {
              const value = event.target.value
              onEdit(element.id, { action: 'set_association', association_target_id: value || undefined })
            }}
          >
            <option value="">None</option>
            {geometryOptions.map(geo => (
              <option key={geo.id} value={geo.id}>{geo.id} · {TYPE_LABEL[geo.type] ?? geo.type}</option>
            ))}
          </select>
        </label>
      ) : (
        <label className="field" htmlFor="inspector-assoc">
          <span>Associated label</span>
          <select
            id="inspector-assoc"
            value={element.associated_label_id ?? ''}
            disabled={busy}
            onChange={(event: ChangeEvent<HTMLSelectElement>) => {
              const value = event.target.value
              onEdit(element.id, { action: 'set_association', association_label_id: value || undefined })
            }}
          >
            <option value="">None</option>
            {labelOptions.map(label => (
              <option key={label.id} value={label.id}>{label.id} · {String(label.geometry.text ?? '')}</option>
            ))}
          </select>
        </label>
      )}

      {!isLabel && (
        <fieldset className="nudge-group" disabled={busy}>
          <legend>Move element</legend>
          <div className="nudge-grid">
            <span />
            <button type="button" aria-label="Move up" onClick={() => onEdit(element.id, { action: 'nudge', offset: [0, -10] })}>↑</button>
            <span />
            <button type="button" aria-label="Move left" onClick={() => onEdit(element.id, { action: 'nudge', offset: [-10, 0] })}>←</button>
            <button type="button" aria-disabled="true" className="nudge-home" aria-label="Keep position">·</button>
            <button type="button" aria-label="Move right" onClick={() => onEdit(element.id, { action: 'nudge', offset: [10, 0] })}>→</button>
            <span />
            <button type="button" aria-label="Move down" onClick={() => onEdit(element.id, { action: 'nudge', offset: [0, 10] })}>↓</button>
            <span />
          </div>
        </fieldset>
      )}

      <div className="inspector-actions">
        <button type="button" disabled={busy} onClick={() => onEdit(element.id, { action: 'set_confidence', confidence: 1.0 })}>Accept detection</button>
        <button type="button" disabled={busy} onClick={() => onEdit(element.id, { action: 'set_confidence', confidence: 0.2 })}>Mark uncertain</button>
        <button type="button" className="danger" disabled={busy} onClick={() => onEdit(element.id, { action: 'delete' })}>Delete element</button>
      </div>

      {element.semantic_properties && Object.keys(element.semantic_properties).length > 0 && (
        <section className="properties">
          <h4>Properties</h4>
          <dl>
            {Object.entries(element.semantic_properties).map(([property, value]) => (
              <div key={property}><dt>{property.replace(/_/g, ' ')}</dt><dd>{String(value)}</dd></div>
            ))}
          </dl>
        </section>
      )}

      {relevantRelationships.length > 0 && (
        <section className="properties">
          <h4>Relationships</h4>
          <ul>
            {relevantRelationships.map(rel => (
              <li key={rel.id}><strong>{rel.type.replace(/_/g, ' ')}</strong> — {rel.explanation ?? rel.element_ids.join(', ')}</li>
            ))}
          </ul>
        </section>
      )}
    </aside>
  )
}