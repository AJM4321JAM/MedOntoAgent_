import { useMemo, useState } from 'react'
import { Activity, ArrowRight, BookOpen, CheckCircle2, Clipboard, Download, FileSearch, GitBranch, Layers3, Search, ShieldAlert, Sparkles } from 'lucide-react'
import type { DemoQueryResult } from './api'
import './ResearchExperience.css'

const presets = [
  { label: 'Medication safety', question: 'What do the available guidelines say about ACE inhibitors during pregnancy?', note: 'Review contraindications and cited guidance' },
  { label: 'First-line treatment', question: 'What do the available guidelines say about labetalol?', note: 'Compare recommendations and source coverage' },
  { label: 'Management pathways', question: 'What guidance applies to hypertension management during pregnancy?', note: 'Explore available clinical guidance' },
]

function saveAudit(result: DemoQueryResult) {
  const content = {
    exported_at: new Date().toISOString(),
    purpose: 'MedOntoAgent academic research demonstration',
    disclaimer: 'Not validated for clinical decision-making. VERIFIED describes application checks, not clinical correctness.',
    ...result,
  }
  const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `medonto-research-audit-${new Date().toISOString().slice(0, 10)}.json`
  document.body.append(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

export function ResearchWelcome({ onSelect, hasResult }: { onSelect: (question: string) => void; hasResult: boolean }) {
  return <section className="rx-welcome" aria-label="Research workflow overview">
    <div className="rx-welcome-main">
      <span className="rx-kicker"><Sparkles size={14} /> RESEARCH WORKSPACE · EXPLORE WITH EVIDENCE</span>
      <h2>From a clinical question to an auditable evidence trail.</h2>
      <p>Ask, trace, inspect and export. Every completed answer is connected to retrieved guideline citations and an actual LangGraph execution record.</p>
      <div className="rx-flow" aria-label="Research steps">
        <span><Search size={15} /> Ask</span><ArrowRight size={14} /><span><Layers3 size={15} /> Ground</span><ArrowRight size={14} /><span><GitBranch size={15} /> Audit</span><ArrowRight size={14} /><span><FileSearch size={15} /> Review</span>
      </div>
    </div>
    <div className="rx-welcome-aside">
      <div className="rx-aside-label">RESEARCH STATUS</div>
      <strong>{hasResult ? 'Evidence ready to review' : 'Ready for your first question'}</strong>
      <p>Local research prototype</p>
      <div className="rx-scope"><ShieldAlert size={16} /> Not a clinical decision-support product</div>
    </div>
    <div className="rx-presets">
      <div className="rx-presets-title"><BookOpen size={17} /> Explore a research scenario <span>Click to populate the question field</span></div>
      <div className="rx-preset-grid">
        {presets.map((preset) => <button type="button" key={preset.label} onClick={() => onSelect(preset.question)} className="rx-preset">
          <strong>{preset.label}</strong><span>{preset.note}</span><ArrowRight size={16} />
        </button>)}
      </div>
    </div>
  </section>
}

export function ResearchReview({ result, onQuestion }: { result: DemoQueryResult; onQuestion: (question: string) => void }) {
  const [filter, setFilter] = useState('')
  const [openId, setOpenId] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const citations = result.response.citations
  const selected = useMemo(() => citations.filter((citation) =>
    [citation.title, citation.section_title ?? '', citation.document_id].join(' ').toLowerCase().includes(filter.toLowerCase())
  ), [citations, filter])
  const documents = new Set(citations.map((citation) => citation.document_id)).size
  const totalMs = result.execution_trace.reduce((sum, event) => sum + Math.max(0, event.elapsed_since_previous_update_ms), 0)
  const isVerified = result.response.status === 'VERIFIED'

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(result.response.answer)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  return <section className="rx-review" aria-label="Research review and export">
    <div className="rx-review-heading">
      <div><span className="rx-kicker"><Activity size={14} /> COMPLETED RESEARCH EXECUTION</span><h2>Evidence and verification overview</h2><p>Interactive inspection of the results returned by this execution.</p></div>
      <div className="rx-actions">
        <button type="button" onClick={copyAnswer}><Clipboard size={15} />{copied ? 'Copied' : 'Copy answer'}</button>
        <button type="button" className="rx-primary" onClick={() => saveAudit(result)}><Download size={15} /> Export audit JSON</button>
      </div>
    </div>
    <div className="rx-metrics">
      <div><span>Outcome</span><strong className={isVerified ? 'rx-good' : ''}>{result.response.status}</strong><small>Application workflow status</small></div>
      <div><span>Evidence sources</span><strong>{documents}</strong><small>Distinct cited documents</small></div>
      <div><span>Citation records</span><strong>{citations.length}</strong><small>Returned by the backend</small></div>
      <div><span>Graph elapsed</span><strong>{(totalMs / 1000).toFixed(2)}s</strong><small>Stream update intervals</small></div>
    </div>
    <div className="rx-review-grid">
      <div className="rx-review-panel">
        <div className="rx-panel-head"><h3><FileSearch size={18} /> Evidence explorer</h3><span>{selected.length} of {citations.length} citations</span></div>
        <label className="rx-search"><Search size={16} /><input aria-label="Search citations" placeholder="Filter by source, section or document ID" value={filter} onChange={(event) => setFilter(event.target.value)} /></label>
        {selected.length === 0 ? <p className="rx-muted">No matching citations. {citations.length === 0 ? 'No sources were returned for this outcome.' : 'Try another search.'}</p> :
          <div className="rx-sources">{selected.map((citation) => <div className="rx-source" key={citation.chunk_id}>
            <button type="button" aria-expanded={openId === citation.chunk_id} onClick={() => setOpenId(openId === citation.chunk_id ? null : citation.chunk_id)}>
              <span className="rx-source-icon"><BookOpen size={16} /></span>
              <span className="rx-source-name"><strong>{citation.title}</strong><small>{citation.section_title || 'Unspecified section'} · {citation.page_number == null ? 'Page unspecified' : `Page ${citation.page_number}`}</small></span>
              <ArrowRight className={openId === citation.chunk_id ? 'rx-rotated' : ''} size={16} />
            </button>
            {openId === citation.chunk_id && <div className="rx-source-details"><div><b>Document</b><span>{citation.document_id}</span></div><div><b>Evidence chunk</b><span>{citation.chunk_id}</span></div>
              {citation.source_url?.startsWith('https://') && <a href={citation.source_url} target="_blank" rel="noreferrer">Open published source ↗</a>}
              <p>Document provenance only. Source excerpts are not included in this public API response.</p>
            </div>}
          </div>)}</div>}
      </div>
      <div className="rx-review-panel rx-explain">
        <div className="rx-panel-head"><h3><CheckCircle2 size={18} /> Research interpretation</h3></div>
        <div className="rx-explain-step"><span>01</span><div><strong>Evidence retrieval</strong><p>{result.response.retrieval.result_count} result(s) returned. Ontology filtering: {result.response.retrieval.used_ontology_filter ? 'used' : 'not used'}{result.response.retrieval.retrieval_relaxed ? '. Retrieval was relaxed.' : '.'}</p></div></div>
        <div className="rx-explain-step"><span>02</span><div><strong>Ontology constraints</strong><p>{result.response.ontology_constraints.length} publicly eligible constraint(s) included in the response.</p></div></div>
        <div className="rx-explain-step"><span>03</span><div><strong>Execution record</strong><p>{result.execution_trace.length} completed graph update(s). Mode: {result.response.execution_mode.replaceAll('_', ' ')}.</p></div></div>
        <div className="rx-caution"><ShieldAlert size={18}/><p><strong>Interpretation boundary</strong> "VERIFIED" means the configured software checks completed. It does not establish clinical accuracy, completeness or superiority over baseline RAG.</p></div>
        <button type="button" className="rx-next" onClick={() => onQuestion('What guidance applies to hypertension management during pregnancy?')}>Explore another guideline question <ArrowRight size={15}/></button>
      </div>
    </div>
  </section>
}
