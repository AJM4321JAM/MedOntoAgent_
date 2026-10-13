
export type ExecutionMode =
  | 'MEDONTO'
  | 'BASELINE_ROUTED'
  | 'BASELINE_FALLBACK'

export interface Citation {
  chunk_id: string
  document_id: string
  title: string
  section_title?: string | null
  page_number?: number | null
  source_url?: string | null
}

export interface ClinicalResponse {
  query: string
  answer: string
  status: string
  execution_mode: ExecutionMode
  citations: Citation[]
  ontology_constraints: Record<string, unknown>[]
  retrieval: {
    result_count: number
    used_ontology_filter: boolean
    retrieval_relaxed: boolean
  }
}

export interface ExecutionEvent {
  sequence: number
  node: string
  elapsed_since_previous_update_ms: number
}

export interface DemoQueryResult {
  response: ClinicalResponse
  execution_trace: ExecutionEvent[]
}

export async function runClinicalQuery(
  question: string,
): Promise<DemoQueryResult> {
  const response = await fetch('/api/demo/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: question }),
  })

  if (!response.ok) {
    if (response.status === 503) {
      throw new Error('The clinical service is temporarily unavailable.')
    }

    if (response.status === 422) {
      throw new Error('Please enter a valid clinical question.')
    }

    throw new Error(
      `The request could not be completed (HTTP ${response.status}).`,
    )
  }

  return (await response.json()) as DemoQueryResult
}
