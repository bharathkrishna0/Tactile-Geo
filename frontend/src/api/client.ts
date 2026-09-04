import type { AnalysisResult, BatchItem, ElementEdit, SessionCreated } from '../types'

async function readJson(response: Response): Promise<Record<string, unknown>> {
  const payload = await response.json().catch(() => ({}))
  return payload
}

function errorFrom(payload: Record<string, unknown>, fallback: string): Error {
  const detail = payload.detail
  return new Error(typeof detail === 'string' ? detail : fallback)
}

export async function uploadImage(file: File): Promise<SessionCreated> {
  const form = new FormData()
  form.append('image', file)
  const response = await fetch('/api/sessions', { method: 'POST', body: form })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Upload failed.')
  return payload as unknown as SessionCreated
}

export async function processSession(sessionId: string): Promise<AnalysisResult> {
  const response = await fetch(`/api/sessions/${sessionId}/process`, { method: 'POST' })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Processing failed.')
  return payload as unknown as AnalysisResult
}

export async function patchElement(sessionId: string, elementId: string, edit: ElementEdit): Promise<AnalysisResult> {
  const response = await fetch(`/api/sessions/${sessionId}/elements/${encodeURIComponent(elementId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(edit),
  })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'That correction could not be applied.')
  return payload as unknown as AnalysisResult
}

export async function batchEdit(sessionId: string, edits: BatchItem[]): Promise<AnalysisResult> {
  const response = await fetch(`/api/sessions/${sessionId}/elements/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ edits }),
  })
  const payload = await readJson(response)
  if (!response.ok) throw errorFrom(payload, 'Those corrections could not be applied.')
  return payload as unknown as AnalysisResult
}