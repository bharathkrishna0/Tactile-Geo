import { KeyboardEvent, MouseEvent } from 'react'
import type { DetectedElement } from '../types'

interface AiGeometryViewProps {
  src: string
  fileName: string
  elements: DetectedElement[]
  width: number
  height: number
  selectedId: string | null
  onSelect: (id: string) => void
}

const asNumber = (value: unknown): number => typeof value === 'number' ? value : Number(value ?? 0)
const coords = (value: unknown): number[] => (Array.isArray(value) ? value : []).map(asNumber)
const asPoints = (value: unknown): number[][] => (Array.isArray(value) ? value : []).map(coords)

function renderGeometry(element: DetectedElement) {
  const geo = element.geometry
  if (element.type === 'text_label') {
    const position = coords(geo.position)
    if (position.length === 0) return null
    return <rect x={position[0] - 4} y={position[1] - 4} width="8" height="8" rx="2" className="layer label" />
  }
  const points = asPoints(geo.points)
  if (points.length >= 3) {
    return <polygon points={points.map(coord => coord.join(',')).join(' ')} className="layer" />
  }
  if (points.length === 2) {
    return <polyline points={points.map(coord => coord.join(',')).join(' ')} className="layer" />
  }
  const start = coords(geo.start)
  const end = coords(geo.end)
  if (start.length >= 2 && end.length >= 2) {
    return <line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} className="layer" />
  }
  const center = coords(geo.center)
  const radius = geo.radius
  if (center.length >= 2) {
    return <circle cx={center[0]} cy={center[1]} r={asNumber(radius)} className="layer" />
  }
  const position = coords(geo.position)
  if (position.length >= 2) {
    return <circle cx={position[0]} cy={position[1]} r="4" className="layer point" />
  }
  const vertex = coords(geo.vertex)
  const arms = asPoints(geo.arms)
  if (vertex.length >= 2 && arms.length >= 2) {
    return (
      <g className="layer">
        <line x1={vertex[0]} y1={vertex[1]} x2={arms[0][0]} y2={arms[0][1]} />
        <line x1={vertex[0]} y1={vertex[1]} x2={arms[1][0]} y2={arms[1][1]} />
      </g>
    )
  }
  return null
}

export default function AiGeometryView({ src, fileName, elements, width, height, selectedId, onSelect }: AiGeometryViewProps) {
  if (width <= 0 || height <= 0) return null
  return (
    <div className="view-panel overlay-stage" role="tabpanel" id="panel-ai" aria-labelledby="tab-ai">
      <img className="overlay-image" src={src} alt={`Uploaded worksheet image: ${fileName}`} />
      <svg
        className="overlay-svg"
        viewBox={`0 0 ${width} ${height}`}
        aria-hidden="true"
        onMouseUp={(event: MouseEvent<SVGSVGElement>) => {
          const target = event.target as SVGGraphicsElement | null
          const id = target?.getAttribute?.('data-element-id')
          if (id) onSelect(id)
        }}
      >
        {elements.map(element => (
          <g
            key={element.id}
            data-element-id={element.id}
            className={(element.id === selectedId ? 'el selected' : 'el') + (element.needs_review ? ' review' : '')}
            tabIndex={0}
            role="button"
            aria-label={`${element.id}, ${element.type.replace(/_/g, ' ')}, confidence ${element.confidence_level}`}
            onClick={() => onSelect(element.id)}
            onKeyDown={(event: KeyboardEvent<SVGGElement>) => {
              if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(element.id) }
            }}
          >
            {renderGeometry(element)}
            <title>{`${element.id} · ${element.type.replace(/_/g, ' ')} · ${element.confidence_level}`}</title>
          </g>
        ))}
      </svg>
    </div>
  )
}