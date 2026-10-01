import type { QAIssue, QAReport } from '../types'

/**
 * The QA checks that make tactile output physically unreadable, and therefore
 * block print-ready export. Mirrors `BLOCKING_CHECKS` in
 * `backend/app/services/tactile_qa.py`.
 */
export const BLOCKING_QA_CHECKS = [
  'stroke_width_out_of_bounds',
  'braille_on_line',
  'element_outside_printable_area',
  'feature_below_minimum_size',
  'no_tactile_geometry',
] as const

const blockingChecks: ReadonlySet<string> = new Set(BLOCKING_QA_CHECKS)

/**
 * An issue blocks export when the backend marked it as an error, or when it names
 * one of the blocking checks. Checking both means the gate still holds if the
 * frontend is served by an older backend that reported these as warnings.
 */
export function isBlockingIssue(issue: QAIssue): boolean {
  return issue.severity === 'error' || blockingChecks.has(issue.check)
}

export function blockingIssues(qa: QAReport | null | undefined): QAIssue[] {
  if (!qa) return []
  return (qa.issues ?? []).filter(isBlockingIssue)
}

export interface ExportBlock {
  issues: QAIssue[]
  message: string
}

/** Human-readable reason export is unavailable, or null when export is allowed. */
export function exportBlockReason(qa: QAReport | null | undefined): ExportBlock | null {
  const issues = blockingIssues(qa)
  if (issues.length === 0) return null
  const count = issues.length
  const noun = count === 1 ? 'violation' : 'violations'
  return {
    issues,
    message: `Print-ready export is blocked by ${count} blocking QA ${noun}. Correct the flagged elements in the geometry view, then export again.`,
  }
}
