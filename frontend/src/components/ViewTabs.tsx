import { KeyboardEvent } from 'react'
import type { ViewId } from '../types'

interface ViewTab {
  id: ViewId
  label: string
  hint: string
}

const TABS: ViewTab[] = [
  { id: 'original', label: 'Original', hint: 'The uploaded worksheet image and its quality.' },
  { id: 'ai', label: 'AI Geometry', hint: 'Detected shapes with confidence; select to correct.' },
  { id: 'tactile', label: 'Tactile Output', hint: 'Print-ready tactile representation and QA.' },
]

interface ViewTabsProps {
  active: ViewId
  onSelect: (view: ViewId) => void
}

function moveFocus(current: ViewId, direction: -1 | 1): ViewId {
  const index = TABS.findIndex(tab => tab.id === current)
  const next = (index + direction + TABS.length) % TABS.length
  return TABS[next].id
}

export default function ViewTabs({ active, onSelect }: ViewTabsProps) {
  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, tab: ViewTab) {
    if (event.key === 'ArrowRight') { event.preventDefault(); onSelect(moveFocus(tab.id, 1)) }
    if (event.key === 'ArrowLeft') { event.preventDefault(); onSelect(moveFocus(tab.id, -1)) }
  }

  return (
    <div className="view-tabs" role="tablist" aria-label="Conversion pipeline stage">
      {TABS.map(tab => (
        <button
          key={tab.id}
          role="tab"
          id={`tab-${tab.id}`}
          aria-selected={active === tab.id}
          aria-controls={`panel-${tab.id}`}
          tabIndex={active === tab.id ? 0 : -1}
          className={active === tab.id ? 'view-tab active' : 'view-tab'}
          onClick={() => onSelect(tab.id)}
          onKeyDown={(event) => onKeyDown(event, tab)}
        >
          <span className="tab-label">{tab.label}</span>
          <span className="tab-hint">{tab.hint}</span>
        </button>
      ))}
    </div>
  )
}