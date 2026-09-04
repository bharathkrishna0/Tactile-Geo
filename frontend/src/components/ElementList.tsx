import type { DetectedElement } from '../types'

const TYPE_LABEL: Record<string, string> = {
  point: 'Point',
  line_segment: 'Segment',
  ray: 'Ray',
  circle: 'Circle',
  arc: 'Arc',
  triangle: 'Triangle',
  rectangle: 'Rectangle',
  polygon: 'Polygon',
  angle: 'Angle',
  axes: 'Axes',
  arrow: 'Arrow',
  text_label: 'Label',
}

function typeLabel(type: string): string {
  return TYPE_LABEL[type] ?? type.replace(/_/g, ' ')
}

interface ElementListProps {
  elements: DetectedElement[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export default function ElementList({ elements, selectedId, onSelect }: ElementListProps) {
  if (elements.length === 0) {
    return <div className="element-list empty"><p className="muted">No geometry detected.</p></div>
  }
  return (
    <ul className="element-list" aria-label="Detected elements">
      {elements.map(element => (
        <li key={element.id}>
          <button
            type="button"
            className="element-row"
            aria-pressed={element.id === selectedId}
            onClick={() => onSelect(element.id)}
          >
            <span className="el-id">{element.id}</span>
            <span className="el-type">{typeLabel(element.type)}</span>
            <span className={`confidence-badge ${element.confidence_level}`}>{element.confidence_level}</span>
            {element.needs_review && <span className="review-badge" aria-label="needs review">!</span>}
          </button>
        </li>
      ))}
    </ul>
  )
}