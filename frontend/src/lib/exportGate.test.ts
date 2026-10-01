import { describe, expect, it } from 'vitest'
import { BLOCKING_QA_CHECKS, blockingIssues, exportBlockReason, isBlockingIssue } from './exportGate'
import type { QAIssue, QAReport } from '../types'

function issue(overrides: Partial<QAIssue> & Pick<QAIssue, 'check'>): QAIssue {
  return { severity: 'warning', message: `${overrides.check} message`, ...overrides }
}

function report(issues: QAIssue[]): QAReport {
  return {
    overall_score: issues.length === 0 ? 100 : 50,
    score_0_100: issues.length === 0 ? 100 : 50,
    passes: !issues.some(i => i.severity === 'error'),
    issues,
    element_checks: issues.length,
  }
}

describe('BLOCKING_QA_CHECKS', () => {
  it('is exactly the four checks agreed with the backend policy', () => {
    expect([...BLOCKING_QA_CHECKS].sort()).toEqual([
      'braille_on_line',
      'element_outside_printable_area',
      'feature_below_minimum_size',
      'stroke_width_out_of_bounds',
    ])
  })
})

describe('isBlockingIssue', () => {
  it.each([...BLOCKING_QA_CHECKS])('blocks on %s even when severity is downgraded', check => {
    expect(isBlockingIssue(issue({ check, severity: 'warning' }))).toBe(true)
    expect(isBlockingIssue(issue({ check, severity: 'info' }))).toBe(true)
  })

  it('blocks on any error severity, including unknown checks', () => {
    expect(isBlockingIssue(issue({ check: 'some_future_check', severity: 'error' }))).toBe(true)
  })

  it('does not block on warning/info for non-blocking checks', () => {
    expect(isBlockingIssue(issue({ check: 'text_too_small', severity: 'warning' }))).toBe(false)
    expect(isBlockingIssue(issue({ check: 'text_too_small', severity: 'info' }))).toBe(false)
  })
})

describe('exportBlockReason', () => {
  it('allows export when there is no QA report', () => {
    expect(exportBlockReason(null)).toBeNull()
    expect(exportBlockReason(undefined)).toBeNull()
  })

  it('allows export when the report has no issues', () => {
    expect(exportBlockReason(report([]))).toBeNull()
  })

  it('allows export when only warnings and notes are present', () => {
    const qa = report([
      issue({ check: 'text_too_small', severity: 'warning' }),
      issue({ check: 'label_overlap', severity: 'info' }),
    ])
    expect(blockingIssues(qa)).toEqual([])
    expect(exportBlockReason(qa)).toBeNull()
  })

  it('blocks export and reports every blocking issue', () => {
    const qa = report([
      issue({ check: 'braille_on_line', severity: 'error', message: 'Braille marker overlaps a line.' }),
      issue({ check: 'feature_below_minimum_size', severity: 'error', message: 'Feature is 2.1 mm.' }),
      issue({ check: 'text_too_small', severity: 'warning' }),
    ])
    const block = exportBlockReason(qa)
    expect(block).not.toBeNull()
    expect(block!.issues).toHaveLength(2)
    expect(block!.issues.map(i => i.check)).toEqual(['braille_on_line', 'feature_below_minimum_size'])
    expect(block!.message).toContain('2 blocking QA violations')
  })

  it('uses singular wording for a single violation', () => {
    const block = exportBlockReason(report([issue({ check: 'stroke_width_out_of_bounds', severity: 'error' })]))
    expect(block!.message).toContain('1 blocking QA violation.')
  })
})
