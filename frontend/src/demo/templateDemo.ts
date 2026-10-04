/**
 * Static data for Template (Demo) mode.
 *
 * Every file lives in `public/demo/tactile-template/` and is produced by
 * `backend/scripts/build_template_demo.py`. Template mode only reads these
 * same-origin static files; it never calls the backend or any AI service.
 */

export const TEMPLATE_ASSET_BASE = `${import.meta.env.BASE_URL}demo/tactile-template/`

export type TemplateAsset = 'original.png' | 'model-a.svg' | 'model-b.svg' | 'fusion.svg' | 'tactile.svg' | 'final.svg' | 'metadata.json'

export function templateAsset(name: TemplateAsset): string {
  return `${TEMPLATE_ASSET_BASE}${name}`
}

export interface TemplateFusionReview {
  model_b_id: string
  model_b_kind: string
  model_a_id: string | null
  model_a_type: string | null
  correspondence: string
  decision: string
  applied: boolean
  detail: string
}

export interface TemplateMetadata {
  version: number
  title: string
  description: string
  is_precomputed_demo: true
  provenance: {
    generator: string
    model_a: string
    model_b: { model: string; provider: string; elapsed_s: number; note: string }
    teacher: string
  }
  image: { width: number; height: number }
  model_a: {
    element_count: number
    by_type: Record<string, number>
    relationship_count: number
    relationships_by_type: Record<string, number>
    labels: { id: string; text: string }[]
    review_flags: number
  }
  model_b: {
    diagram_kind: string
    description: string
    entities: { id: string; kind: string; confidence_level: string; evidence: string }[]
    text_items: { text: string; role: string }[]
    relationships: { kind: string; from_id: string; to_id: string; evidence: string }[]
    uncertainties: { kind: string; severity: string; note: string }[]
  }
  fusion: {
    summary: Record<string, number>
    reviews: TemplateFusionReview[]
    applied: number
    not_applied: number
    coordinates_changed: number
  }
  simplification: {
    before: number
    after: number
    removed: number
    merged: number
    actions: Record<string, number>
    explanations: string[]
  }
  braille: {
    table: string
    labels: { id: string; text: string; braille: string; position_mm: [number, number] }[]
  }
  qa: { passes: boolean; score: number; errors: string[]; issues_by_check: Record<string, number> }
  final: {
    page: string
    width_mm: number
    height_mm: number
    stroke_width_pt: number
    feature_count: number
    label_count: number
  }
}

export async function loadTemplateMetadata(): Promise<TemplateMetadata> {
  const response = await fetch(templateAsset('metadata.json'))
  if (!response.ok) throw new Error('The template files could not be loaded.')
  return (await response.json()) as TemplateMetadata
}

export type TemplateStageId = 'original' | 'model-a' | 'model-b' | 'fusion' | 'simplified' | 'braille' | 'final'

export interface TemplateStage {
  id: TemplateStageId
  label: string
  title: string
  description: string
}

export const TEMPLATE_STAGES: TemplateStage[] = [
  { id: 'original', label: 'Original', title: 'Original Diagram', description: 'The source visual diagram provided to TactileGeo.' },
  { id: 'model-a', label: 'Model A', title: 'Model A: Geometry Analysis', description: 'Deterministic computer vision finds the shapes, lines and text, and owns every coordinate.' },
  { id: 'model-b', label: 'Model B', title: 'Model B: Semantic Interpretation', description: 'A vision-language model describes what the diagram means. Its findings are suggestions only.' },
  { id: 'fusion', label: 'Fusion', title: 'Fusion', description: 'Model B findings are paired with Model A geometry. Accepted findings add meaning; coordinates stay Model A’s.' },
  { id: 'simplified', label: 'Simplified', title: 'Tactile Simplification', description: 'Visual detail is simplified while preserving essential geometry and spatial relationships.' },
  { id: 'braille', label: 'Braille', title: 'Braille Layout', description: 'Braille is positioned as part of the tactile layout rather than simply overlaying text onto the image.' },
  { id: 'final', label: 'Final SVG', title: 'Final Tactile SVG', description: 'A print-ready vector file for swell paper or an embosser.' },
]

export function humanize(value: string): string {
  return value.replace(/_/g, ' ')
}
