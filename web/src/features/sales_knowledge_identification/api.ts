import type { DocumentPackage, IdentificationResult, KnowledgeModule } from './types'
import { resolveApiBaseUrl } from '../../core/api-base'

export const apiBaseUrl = resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL)

export class IdentificationApiError extends Error {
  readonly status: number
  readonly endpoint: string

  constructor(message: string, status: number, endpoint: string) {
    super(message)
    this.name = 'IdentificationApiError'
    this.status = status
    this.endpoint = endpoint
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : '网络请求失败'
    throw new IdentificationApiError(`无法连接识别 API：${message}`, 0, `${apiBaseUrl}${path}`)
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // The status line is enough when the server did not return JSON.
    }
    throw new IdentificationApiError(detail, response.status, `${apiBaseUrl}${path}`)
  }

  return (await response.json()) as T
}

export function getDocumentPackage(documentPackageId: string): Promise<DocumentPackage> {
  return request<DocumentPackage>(
    `/sales-knowledge-identification/document-packages/${encodeURIComponent(documentPackageId)}`,
  )
}

export function getIdentificationCatalog(): Promise<{
  version: string
  modules: KnowledgeModule[]
}> {
  return request('/sales-knowledge-identification/catalog')
}

export function runIdentification(documentPackageId: string): Promise<IdentificationResult> {
  return request<IdentificationResult>('/sales-knowledge-identification/runs', {
    method: 'POST',
    body: JSON.stringify({ documentPackageId }),
  })
}

export function listIdentificationRuns(
  documentPackageId: string,
  limit = 5,
): Promise<IdentificationResult[]> {
  const query = new URLSearchParams({ documentPackageId, limit: String(limit) })
  return request<IdentificationResult[]>(`/sales-knowledge-identification/runs?${query}`)
}

export function getIdentificationRun(runId: string): Promise<IdentificationResult> {
  return request<IdentificationResult>(
    `/sales-knowledge-identification/runs/${encodeURIComponent(runId)}`,
  )
}

export function getIdentificationEvaluation(
  documentPackageId: string,
): Promise<{ documentPackageId: string; markdown: string }> {
  return request(
    `/sales-knowledge-identification/evaluations/${encodeURIComponent(documentPackageId)}`,
  )
}

export const defaultDocumentPackageId = import.meta.env.VITE_IDENTIFICATION_DOCUMENT_PACKAGE_ID ?? ''
