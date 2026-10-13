import { useEffect, useMemo, useState } from 'react'
import { Activity, BookOpen, ExternalLink, RefreshCw, Search, ShieldCheck } from 'lucide-react'
import { getLibrary, getOperations, type LibraryResponse, type OperationsResponse } from './workspaceApi'
import './WorkspacePages.css'

function StateMessage({ message }: { message: string }) {
  return <div className="workspace-message" role="status">{message}</div>
}

export function KnowledgeLibrary() {
  const [data, setData] = useState<LibraryResponse | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('ALL')
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let active = true
    getLibrary().then((value) => { if (active) { setData(value); setError('') } })
      .catch(() => { if (active) setError('Unable to load registered documents. Check the local backend.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [refreshKey])

  const documents = useMemo(() => (data?.documents ?? []).filter((doc) => {
    const needle = search.trim().toLowerCase()
    return (status === 'ALL' || doc.approval_status === status) &&
      (!needle || [doc.title, doc.publisher, doc.document_id, doc.clinical_domain].some((part) => part.toLowerCase().includes(needle)))
  }), [data, search, status])
  const approved = data?.documents.filter((doc) => doc.approval_status === 'APPROVED').length ?? 0
  const indexed = data?.documents.filter((doc) => doc.index_status === 'INDEXED').length ?? 0
  return <section className="workspace-page">
    <div className="eyebrow"><BookOpen size={16} /> GOVERNED CLINICAL SOURCES</div>
    <h1>Knowledge Library</h1>
    <p className="workspace-lead">Read-only document governance, provenance and indexing state from the local registry.</p>
    <div className="workspace-toolbar"><span>Document registry</span><button onClick={() => { setLoading(true); setRefreshKey((n) => n + 1) }} disabled={loading} type="button"><RefreshCw size={15} /> Refresh</button></div>
    {loading && <StateMessage message="Loading registered sources…" />}
    {error && <div className="workspace-error" role="alert">{error}</div>}
    {!loading && data && <>
      <div className="workspace-stats"><div><small>Registered</small><strong>{data.total_documents}</strong></div><div><small>Approved</small><strong>{approved}</strong></div><div><small>Indexed</small><strong>{indexed}</strong></div></div>
      {data.registry_state === 'missing' && <StateMessage message="No local document registry file exists. This is not a verified empty clinical corpus." />}
      <div className="workspace-filters"><label><Search size={15} /><input aria-label="Search documents" placeholder="Search by document or publisher" value={search} onChange={(event) => setSearch(event.target.value)} /></label><select aria-label="Approval status" value={status} onChange={(event) => setStatus(event.target.value)}><option value="ALL">All approval states</option><option value="APPROVED">Approved</option><option value="PENDING">Pending</option><option value="SUPERSEDED">Superseded</option><option value="REJECTED">Rejected</option></select></div>
      <div className="workspace-document-list">{documents.map((doc) => <article className="workspace-document" key={doc.document_id}>
        <div className="workspace-document-title"><h2>{doc.title}</h2><span className="workspace-pill">{doc.approval_status}</span><span className="workspace-pill muted">{doc.index_status}</span></div>
        <div className="workspace-document-meta"><span>{doc.publisher}</span><span>{doc.source_authority}</span><span>Version {doc.version}</span><span>{doc.clinical_domain}</span></div>
        <details><summary>View provenance</summary><dl><dt>Document ID</dt><dd>{doc.document_id}</dd><dt>Document type</dt><dd>{doc.document_type}</dd><dt>Published</dt><dd>{doc.publication_date ?? 'Not recorded'}</dd><dt>Effective</dt><dd>{doc.effective_date ?? 'Not recorded'}</dd><dt>Indexed at</dt><dd>{doc.indexed_at ?? 'Not recorded'}</dd><dt>Supersedes</dt><dd>{doc.supersedes_document_id ?? 'None'}</dd></dl>{doc.source_url?.startsWith('https://') && <a href={doc.source_url} target="_blank" rel="noreferrer">Source link <ExternalLink size={13} /></a>}</details>
      </article>)}</div>
      {documents.length === 0 && <StateMessage message={data.total_documents === 0 ? 'No documents are registered in this local registry.' : 'No documents match the current filters.'} />}
      <p className="workspace-caveat"><ShieldCheck size={14} /> Registry approval and indexing states do not independently verify clinical accuracy or current vector-store availability.</p>
    </>}
  </section>
}

export function SystemOperations() {
  const [data, setData] = useState<OperationsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)
  useEffect(() => {
    let active = true
    getOperations().then((value) => { if (active) { setData(value); setError('') } })
      .catch(() => { if (active) setError('Unable to load operations information. Check the local backend.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [refreshKey])
  return <section className="workspace-page">
    <div className="eyebrow"><Activity size={16} /> LOCAL OBSERVABILITY</div><h1>System Operations</h1>
    <p className="workspace-lead">Authenticated configuration and application response status. External providers are not probed by this view.</p>
    <div className="workspace-toolbar"><span>Operational snapshot</span><button type="button" disabled={loading} onClick={() => { setLoading(true); setRefreshKey((n) => n + 1) }}><RefreshCw size={15} /> Refresh</button></div>
    {loading && <StateMessage message="Loading local status…" />}
    {error && <div className="workspace-error" role="alert">{error}</div>}
    {!loading && data && <>
      <div className="workspace-stats"><div><small>Local API</small><strong>{data.status}</strong></div><div><small>Demo timeout / call</small><strong>{data.demo_llm_timeout_seconds}s</strong></div><div><small>Provider attempt limit</small><strong>{data.demo_provider_attempt_limit}</strong></div></div>
      <div className="workspace-settings"><span>Application</span><strong>{data.service}</strong><span>Environment</span><strong>{data.environment}</strong><span>Version</span><strong>{data.app_version}</strong><span>Telemetry export</span><strong>{data.telemetry_configured ? 'Configured' : 'Disabled'}</strong><span>Historical metrics</span><strong>{data.historical_metrics_available ? 'Available' : 'Not exposed by this API'}</strong></div>
      <h2 className="workspace-section-title">Components and verification scope</h2>
      <div className="workspace-component-list">{data.components.map((component) => <div key={component.label} className="workspace-component"><div><strong>{component.label}</strong><p>{component.explanation}</p></div><span className="workspace-pill muted">{component.state.replaceAll('_', ' ')}</span></div>)}</div>
      <p className="workspace-caveat"><ShieldCheck size={14} /> “Running” only confirms this API responds. It does not assert Qdrant/Gemini health, successful clinical validation, or historical request statistics.</p>
    </>}
  </section>
}
