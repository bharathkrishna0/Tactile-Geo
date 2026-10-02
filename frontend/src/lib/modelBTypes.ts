/**
 * Model B wire types.
 *
 * Mirrors `backend/app/schemas/model_b.py`. Two conventions are deliberate:
 *
 * - Every advisory payload carries a literal `advisory_only: true`. The type
 *   makes it impossible to build one of these objects without the flag, so the
 *   "this cannot change the embossed output" guarantee is visible in the type
 *   system rather than only in a comment.
 * - Confidence is a float, but the UI labels it as a band. The backend maps
 *   certain/likely/uncertain/unreadable to midpoints of non-overlapping ranges,
 *   so treating it as a probability would overstate what is known.
 */

import type { AnalysisResult } from '../types'

export type ConfidenceLevel = 'high' | 'medium' | 'low'

export type ModelBJobStatusValue = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface ModelBAvailability {
  available: boolean
  enabled: boolean
  reason: string | null
  /** Gateway name, e.g. `openrouter`. Never a credential. */
  provider: string
  /**
   * Configured model *or router*. A router such as `openrouter/free` may
   * serve each request from a different upstream, so the model on a result can
   * legitimately differ from this. The name is shown so a teacher reporting
   * "it said something odd" can say which configuration produced it.
   */
  model: string
}

export interface ModelBBoundingRegion {
  /** [x_min, y_min, x_max, y_max], normalized against the prepared frame. */
  norm: [number, number, number, number]
  /** Pixel rect in the original image frame, matching Model A's (x, y, w, h). */
  x: number
  y: number
  width: number
  height: number
  advisory_only: true
}

export interface ModelBEntity {
  id: string
  kind: string
  /** null when Model A has no equivalent type. Never coerced. */
  geometry_type: string | null
  region: ModelBBoundingRegion
  confidence: number
  confidence_level: ConfidenceLevel
  needs_review: boolean
  detection_kind: 'observed' | 'inferred'
  evidence: string
  occluded: boolean
  label: string | null
  mapped: boolean
  advisory_only: true
}

export interface ModelBRelationship {
  id: string
  kind: string
  relationship_type: string | null
  from_id: string
  to_id: string
  confidence: number
  confidence_level: ConfidenceLevel
  needs_review: boolean
  detection_kind: 'observed' | 'inferred'
  evidence: string
  mapped: boolean
  advisory_only: true
}

export interface ModelBDiagramRelation {
  id: string
  kind: string
  subject_ids: string[]
  statement: string
  confidence: number
  confidence_level: ConfidenceLevel
  needs_review: boolean
  detection_kind: 'observed' | 'inferred'
  evidence: string
  advisory_only: true
}

export interface ModelBTextItem {
  id: string
  /** null when any character was uncertain. Never reconstructed. */
  text: string | null
  role: string
  region: ModelBBoundingRegion
  confidence: number
  confidence_level: ConfidenceLevel
  needs_review: boolean
  evidence: string
  advisory_only: true
}

export interface ModelBUncertainty {
  id: string
  subject_ids: string[]
  kind: string
  note: string
  severity: 'blocking_review' | 'worth_review' | 'informational'
}

export interface ModelBAnalysis {
  schema_version: string
  diagram_kind: string
  diagram_description: string
  image_width: number
  image_height: number
  prepared_width: number
  prepared_height: number
  entities: ModelBEntity[]
  relationships: ModelBRelationship[]
  diagram_relations: ModelBDiagramRelation[]
  text_items: ModelBTextItem[]
  uncertainties: ModelBUncertainty[]
  image_readable: boolean
  image_quality_issues: string[]
  truncated: boolean
  finish_reason: string
  mapped_entity_count: number
  unmapped_entity_count: number
  review_entity_count: number
  /** Gateway that served the request, e.g. `openrouter`. */
  provider: string
  /** Model identifier from configuration. May be a router. */
  requested_model: string
  /**
   * The model that actually served this result. Differs from
   * `requested_model` under a router, which is the default on the free tier, so
   * a result is always attributable to a real model rather than to a route.
   */
  resolved_model: string
  /** Upstream vendor behind the gateway, when the gateway reported one. */
  upstream_provider: string
  /** Gateway-assigned id for the call, for correlating a report. */
  request_id: string
  /** Token accounting as reported. Absent keys mean "not reported". */
  usage: Record<string, number>
  validation_warnings: string[]
  advisory_only: true
}

export interface ModelBError {
  code: string
  message: string
  retryable: boolean
  /** Present only when the provider told us how long to wait. */
  retry_after_s?: number | null
}

export interface ModelBJob {
  job_id: string
  session_id: string
  status: ModelBJobStatusValue
  created_at: string
  started_at: string | null
  completed_at: string | null
  result: ModelBAnalysis | null
  error: ModelBError | null
  /** Gateway and configured model, known before the call returns. */
  provider?: string
  model?: string
  /** True when an identical earlier analysis was reused without a provider call. */
  cache_hit?: boolean
}

export interface ModelBAgreementHint {
  model_b_id: string
  model_a_id: string
  model_b_kind: string
  model_a_type: string
  iou: number
  verdict: 'agrees' | 'weak_overlap' | 'type_mismatch'
  advisory_only: true
}

export interface ModelBCandidateAddition {
  model_b_id: string
  suggested_type: string | null
  region: [number, number, number, number]
  reason: string
  requires_teacher_approval: true
  advisory_only: true
}

export interface ModelBDisagreement {
  kind: 'type_mismatch' | 'model_a_only'
  model_b_id: string
  model_a_id: string | null
  detail: string
  advisory_only: true
}

export interface ModelBEntityReview {
  model_b_id: string
  model_b_kind: string
  model_a_id: string | null
  correspondence: 'strong' | 'weak' | 'none'
  adds_semantics: boolean
  contradicts_model_a: boolean
  requires_review: boolean
  reason: string
  advisory_only: true
}

export type ModelBDecisionValue = 'accept' | 'reject' | 'defer'

/** A teacher's recorded decision. Recording alone never changes the session. */
export interface ModelBDecision {
  job_id: string
  model_b_id: string
  decision: ModelBDecisionValue
  note: string | null
  decided_at: string
  applied_to_geometry: false
}

export interface ModelBFusionReport {
  summary: Record<string, number>
  agreements: ModelBAgreementHint[]
  /** Absent from older servers; treat as empty. */
  entity_reviews?: ModelBEntityReview[]
  decisions?: Record<string, ModelBDecision>
  candidate_additions: ModelBCandidateAddition[]
  disagreements: ModelBDisagreement[]
  diagram_relations: Array<{
    id: string
    kind: string
    statement: string
    evidence: string
    subject_ids: string[]
    confidence: number
    needs_review: boolean
    detection_kind: 'observed' | 'inferred'
  }>
  text_notes: Array<{
    id: string
    text: string | null
    role: string
    bbox: [number, number, number, number]
    needs_review: boolean
    evidence: string
  }>
  uncertainties: Array<{
    id: string
    kind: string
    note: string
    severity: 'blocking_review' | 'worth_review' | 'informational'
    subject_ids: string[]
  }>
  advisory_only: true
}

/** What happened to one accepted finding when Semantic Geometry v2 was built. */
export interface ModelBApplyOutcome {
  model_b_id: string
  applied: boolean
  change: 'set_type' | 'attach_label' | 'confirm' | null
  model_a_id: string | null
  detail: string
}

/**
 * The session regenerated from Model A geometry plus accepted findings.
 * Coordinates are always Model A's; QA and the tactile SVG are rebuilt.
 */
export interface ModelBApplyResult {
  session: AnalysisResult
  outcomes: ModelBApplyOutcome[]
  reverted: string[]
}
