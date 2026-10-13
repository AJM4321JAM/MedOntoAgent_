export interface LibraryDocument {
  document_id: string
  title: string
  publisher: string
  clinical_domain: string
  document_type: string
  source_authority: string
  version: string
  approval_status: string
  index_status: string
  publication_date: string | null
  effective_date: string | null
  registered_at: string
  indexed_at: string | null
  source_url: string | null
  supersedes_document_id: string | null
  superseded_by_document_id: string | null
}
export interface LibraryResponse {
  registry_state: 'available' | 'missing'
  total_documents: number
  documents: LibraryDocument[]
}
export interface OperationsResponse {
  service: string
  environment: string
  status: 'running'
  app_version: string
  demo_llm_timeout_seconds: number
  demo_provider_attempt_limit: number
  telemetry_configured: boolean
  historical_metrics_available: boolean
  components: { label: string; state: 'available' | 'configured' | 'not_checked' | 'not_available'; explanation: string }[]
}
async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { Accept: 'application/json' } })
  if (!response.ok) throw new Error(`Unable to retrieve workspace information (HTTP ${response.status}).`)
  return await response.json() as T
}
export const getLibrary = () => getJson<LibraryResponse>('/api/demo/library')
export const getOperations = () => getJson<OperationsResponse>('/api/demo/operations')
