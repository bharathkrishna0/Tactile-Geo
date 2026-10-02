import type {
  ModelBAvailability,
  ModelBDecision,
  ModelBDecisionValue,
  ModelBFusionReport,
  ModelBJob,
} from './modelBTypes'
import { API_BASE } from './apiBase'

/**
 * Model B API client.
 *
 * Kept separate from `api/client.ts` because Model B is optional: the Model A
 * path must not import Model B types, and a build of the app with Model B
 * disabled should be able to tree-shake this whole file away.
 *
 * The key never appears here. It is a backend secret; the browser only ever
 * learns whether Model B is usable, via `/api/model-b/status`.
 */

async function readJson(response: Response): Promise<Record<string, unknown>> {
  return response.json().catch(() => ({}))
}

function errorFrom(payload: Record<string, unknown>, fallback: string): Error {
  const detail = payload.detail
  return new Error(typeof detail === 'string' ? detail : fallback)
}

export async function fetchModelBAvailability(): Promise<ModelBAvailability> {
  const response = await fetch(`${API_BASE}/api/model-b/status`)
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Could not check Model B availability.')
  return payload as unknown as ModelBAvailability
}

export async function requestModelB(sessionId: string): Promise<ModelBJob> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/model-b`, { method: 'POST' })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Could not start the Model B analysis.')
  return payload as unknown as ModelBJob
}

export async function fetchModelBJob(sessionId: string, jobId: string): Promise<ModelBJob> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/model-b/${jobId}`)
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Could not read the Model B job.')
  return payload as unknown as ModelBJob
}

export async function cancelModelBJob(sessionId: string, jobId: string): Promise<ModelBJob> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/model-b/${jobId}`, {
    method: 'DELETE',
  })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Could not cancel the Model B job.')
  return payload as unknown as ModelBJob
}

export async function fetchModelBFusion(
  sessionId: string,
  jobId: string,
): Promise<ModelBFusionReport> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/model-b/${jobId}/fusion`)
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Could not compare the two analyses.')
  return payload as unknown as ModelBFusionReport
}

export async function recordModelBDecision(
  sessionId: string,
  jobId: string,
  modelBId: string,
  decision: ModelBDecisionValue,
): Promise<ModelBDecision> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/model-b/${jobId}/decisions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_b_id: modelBId, decision }),
  })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Could not record the decision.')
  return payload as unknown as ModelBDecision
}
