export type Severity = 'info' | 'warning' | 'error'

export interface QualityIssue {
  check: string
  severity: Severity
  message: string
  value?: number | null
  threshold?: number | null
}

export interface QualityReport {
  passes_gate: boolean
  issues: QualityIssue[]
  image_width: number
  image_height: number
}

export interface DetectedElement {
  id: string
  type: string
  geometry: Record<string, unknown>
  confidence: number
  confidence_level: string
  needs_review: boolean
  source: string
  bbox?: number[] | null
  semantic_properties?: Record<string, unknown>
  associated_label_id?: string | null
}

export interface Relationship {
  id: string
  type: string
  element_ids: string[]
  confidence: number
  confidence_level?: string
  needs_review?: boolean
  explanation?: string | null
  properties?: Record<string, unknown>
}

export interface TransformationExplanation {
  stage: string
  element_id?: string | null
  message: string
}

export interface SemanticGeometry {
  elements: DetectedElement[]
  relationships: Relationship[]
  image_width: number
  image_height: number
  element_count: number
  low_confidence_count: number
  review_flags: string[]
  explanations: TransformationExplanation[]
}

export interface SimplificationAction {
  element_id: string
  action: string
  detail: string
}

export interface SimplifiedGeometry {
  elements: DetectedElement[]
  relationships: Relationship[]
  actions: SimplificationAction[]
  removed_count: number
  merged_count: number
  explanations: string[]
}

export interface QAIssue {
  check: string
  severity: Severity
  message: string
  element_id?: string | null
}

export interface QAReport {
  overall_score: number
  passes: boolean
  issues: QAIssue[]
  element_checks: number
  score_0_100: number
}

export interface DetectedShape {
  type: string
  points: number[][]
}

export interface DetectedLabel {
  text: string
  braille: string
  bbox: number[][]
  confidence: number
  position: number[]
}

export interface AnalysisResult {
  session_id: string
  preview_svg: string
  detected_shapes: DetectedShape[]
  detected_labels: DetectedLabel[]
  processing_params: { edge_sensitivity: number }
  quality_report?: QualityReport | null
  semantic_geometry?: SemanticGeometry | null
  simplified_geometry?: SimplifiedGeometry | null
  qa_report?: QAReport | null
  tactile_svg?: string | null
  detail?: string
}

export interface SessionCreated {
  session_id: string
  filename: string
  created_at: string
}

export type EditAction =
  | 'set_type'
  | 'set_label'
  | 'set_association'
  | 'set_confidence'
  | 'nudge'
  | 'delete'

export interface ElementEdit {
  action: EditAction
  type?: string
  text?: string
  confidence?: number
  association_label_id?: string
  association_target_id?: string
  offset?: number[]
}

export interface BatchItem extends ElementEdit {
  element_id: string
}

export type ViewId = 'original' | 'ai' | 'tactile'