import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import TactileOutputView from './TactileOutputView'
import type { QAReport } from '../types'

function qa(checks: string[]): QAReport {
  return {
    overall_score: 0,
    score_0_100: checks.length ? 39 : 100,
    passes: checks.length === 0,
    issues: checks.map(check => ({ check, severity: 'error', message: `${check} message` })),
    element_checks: 0,
  } as unknown as QAReport
}

describe('TactileOutputView', () => {
  it('explains what to do when no geometry was found instead of showing a blank sheet', () => {
    const { container } = render(
      <TactileOutputView tactileSvg='<svg xmlns="http://www.w3.org/2000/svg"/>' qa={qa(['no_tactile_geometry'])} simplified={null} explanations={[]} />,
    )
    expect(screen.getByRole('status')).toHaveTextContent(/no lines or shapes were found/i)
    expect(container.querySelector('.tactile-svg')).toBeNull()
  })

  it('strips script from the injected tactile SVG', () => {
    const { container } = render(
      <TactileOutputView
        tactileSvg='<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(2)</script><line x1="0" y1="0" x2="5" y2="5"/></svg>'
        qa={qa([])}
        simplified={null}
        explanations={[]}
      />,
    )
    const preview = container.querySelector('.tactile-svg')!
    expect(preview.querySelector('line')).not.toBeNull()
    expect(preview.innerHTML).not.toMatch(/script|onload/i)
  })
})
