import { useEffect, useState } from 'react'
import { getLibrary, type LibraryDocument } from './workspaceApi'
import { Activity, ArrowRight, BookOpen, CheckCircle2, ChevronDown, Copy, Download, FileSearch, GitBranch, History, ShieldAlert, Sparkles } from 'lucide-react'
import type { DemoQueryResult } from './api'
import './ResearchWorkspaceV3.css'
import './ResearchInvestigationV6.css'
import './EvidencePassagesV7.css'

type Props = {
  onSave: () => void
  saveDisabled: boolean
  notebookCount: number
  onNotebook: () => void
  question: string
  setQuestion: (question: string) => void
  result: DemoQueryResult | null
  setResult: (result: DemoQueryResult | null) => void
  loading: boolean
  error: string | null
  setError: (error: string | null) => void
  onRun: () => void
  history: DemoQueryResult[]
  onClearHistory: () => void
}

const prompts = [
  { label: 'Medication safety', question: 'What do the available guidelines say about ACE inhibitors during pregnancy?' },
  { label: 'Labetalol guidance', question: 'What do the available guidelines say about labetalol?' },
  { label: 'Pregnancy hypertension', question: 'What guidance applies to hypertension management during pregnancy?' },
]

function formatElapsed(milliseconds: number) {
  return milliseconds < 1000 ? `${milliseconds.toFixed(0)} ms` : `${(milliseconds / 1000).toFixed(2)} s`
}

function safeSourceUrl(url: string | null | undefined) {
  if (!url) return null
  try { return new URL(url).protocol === 'https:' ? url : null } catch { return null }
}

function download(name: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function researchReport(result: DemoQueryResult) {
  const response = result.response
  return [
    '# MedOntoAgent · Research Evidence Report',
    '',
    `Generated (local time): ${new Date().toLocaleString()}`,
    '',
    '## Research question', response.query, '',
    '## Research response', response.answer, '',
    '## Execution and limitations',
    `- Application status: ${response.status}`,
    `- Execution mode: ${response.execution_mode}`,
    `- Retrieved results: ${response.retrieval.result_count}`,
    `- Ontology filtering used: ${response.retrieval.used_ontology_filter ? 'yes' : 'no'}`,
    `- Retrieval relaxed: ${response.retrieval.retrieval_relaxed ? 'yes' : 'no'}`,
    '',
    '## Cited evidence',
    ...(response.citations.length ? response.citations.flatMap((c, i) => [
      `${i + 1}. ${c.title}`,
      `   - Section: ${c.section_title || 'Not specified'}`,
      `   - Page: ${c.page_number ?? 'Not specified'}`,
      `   - Document ID: ${c.document_id}`,
      `   - Evidence ID: ${c.chunk_id}`,
      `   - Source: ${safeSourceUrl(c.source_url) || 'Not supplied'}`,
    ]) : ['No citations returned for this execution.']),
    '', '## Ontology constraints (application output)',
    ...(response.ontology_constraints.length ? response.ontology_constraints.map((c) => `- ${JSON.stringify(c)}`) : ['None returned.']),
    '', '## Graph execution trace',
    ...(result.execution_trace.length ? result.execution_trace.map((e) => `${e.sequence}. ${e.node}: ${formatElapsed(e.elapsed_since_previous_update_ms)}`) : ['No completed graph updates returned.']),
    '',
    '> Research prototype only. Not validated for independent clinical use. VERIFIED is a software workflow outcome, not proof of clinical correctness. A report records the output of one execution, not a systematic review.',
    '',
  ].join('\n')
}

const stageNames: Record<string, string> = {
  initialize: 'Initialize', map_concepts: 'Map concepts', build_constraints: 'Build ontology constraints',
  retrieve: 'Retrieve evidence', generate: 'Generate answer', validate: 'Validate response',
  retry: 'Retry decision', verified: 'Verification completed', abstain: 'Abstained', baseline_routed: 'Baseline routing',
}

export function ResearchWorkspaceV3({ onSave, saveDisabled, notebookCount, onNotebook, question, setQuestion, result, setResult, loading, error, setError, onRun, history, onClearHistory }: Props) {
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null)
  const [copyState, setCopyState] = useState<'idle'|'copied'|'unavailable'>('idle')
  const [showTechnical, setShowTechnical] = useState(false)
  const [registry, setRegistry] = useState<LibraryDocument[]>([])
  const [sourceAction, setSourceAction] = useState('')
  const [showRawOntology, setShowRawOntology] = useState(false)
  useEffect(() => { let active = true; getLibrary().then(v => { if(active) setRegistry(v.documents) }).catch(() => { /* Citation metadata remains available without registry */ }); return () => { active = false } }, [])

  const citations = result?.response.citations ?? []
  const citedPassages = ((result as (DemoQueryResult & { cited_passages?: Array<{chunk_id:string,document_id:string,text:string,truncated:boolean}> }) | null)?.cited_passages ?? [])
  const visible = citations
  const graphMs = result?.execution_trace.reduce((sum, item) => sum + Math.max(0, item.elapsed_since_previous_update_ms), 0) ?? 0
  const hasAnswer = result && result.response.status !== 'ABSTAINED' && result.response.answer.trim().length > 0
  const status = result?.response.status ?? (loading ? 'RUNNING' : 'READY')
  const sourceTotal = new Set(citations.map((c) => c.document_id)).size

  function chooseQuestion(value: string) {
    if (loading) return
    setQuestion(value)
    setResult(null)
    setError(null)
    setSelectedCitation(null)
    document.getElementById('clinical-question')?.focus()
  }
  function refineQuestion() {
    if (!result || loading) return
    chooseQuestion(result.response.query)
  }
  async function copyCitation(c: DemoQueryResult['response']['citations'][number]) {
    const doc = registry.find(d => d.document_id === c.document_id)
    const lines = [c.title, doc?.publisher || 'Publisher not recorded', c.section_title ? `Section: ${c.section_title}` : '', c.page_number == null ? '' : `Page: ${c.page_number}`, doc?.publication_date ? `Published: ${doc.publication_date}` : '', safeSourceUrl(c.source_url) || safeSourceUrl(doc?.source_url) || '', `Document ID: ${c.document_id}`].filter(Boolean)
    try { await navigator.clipboard.writeText(lines.join(' · ')); setSourceAction(`Citation copied: ${c.chunk_id}`) }
    catch { setSourceAction('Clipboard unavailable; use the source details to record this citation.') }
  }
  function restore(item: DemoQueryResult) {
    if (loading) return
    setQuestion(item.response.query)
    setResult(item)
    setError(null)
    setSelectedCitation(null)
  }
  async function copyAnswer() {
    if (!result) return
    try { await navigator.clipboard.writeText(result.response.answer); setCopyState('copied') }
    catch { setCopyState('unavailable') }
  }

  return <section className="v3-root" aria-label="Research workspace">
    <div className="v3-heading">
      <div><div className="v3-kicker"><Sparkles size={15}/> CLINICAL EVIDENCE RESEARCH</div>
        <h1>Research Workspace</h1><p>Ask questions. Inspect the cited guidelines. Review what the system could and could not establish.</p></div>
      <div className="v3-context"><ShieldAlert size={17}/><span>Research prototype<br/><strong>Not for clinical decisions</strong></span></div>
    </div>
    <div className="v3-columns">
      <aside className="v3-side" aria-label="Research navigation">
        <div className="v3-side-block"><div className="v3-side-title"><BookOpen size={16}/> START A RESEARCH QUESTION</div>
          {prompts.map((p) => <button className="v3-suggestion" type="button" key={p.label} onClick={() => chooseQuestion(p.question)} disabled={loading}>
            <span>{p.label}</span><ArrowRight size={14}/></button>)}
        </div>
        <div className="v3-side-block"><div className="v3-side-title"><History size={16}/> THIS SESSION <span className="v3-count">{history.length}</span></div>
          {history.length === 0 ? <p className="v3-subtle">Completed questions appear here. Session-only; cleared when this page reloads.</p> : history.map((item, index) =>
            <button type="button" className={`v3-history ${result === item ? 'v3-history-selected' : ''}`} key={index} onClick={() => restore(item)} disabled={loading}>
              <span>{item.response.query}</span><small>{item.response.status.replaceAll('_', ' ')} · Open without rerunning</small>
            </button>)}
          <button type="button" className="v3-clear" onClick={onNotebook}>Open Research Notebook ({notebookCount})</button>
          {history.length > 0 && <button type="button" className="v3-clear" onClick={onClearHistory}>Clear session list</button>}
        </div>
      </aside>
      <div className="v3-main">
        <section className="v3-card v3-question-card">
          <div className="v3-card-heading"><div><span className="v3-step">01</span><h2>Ask a research question</h2></div><span className="v3-subtle">Guideline corpus · evidence search</span></div>
          <label className="v3-field-label" htmlFor="clinical-question">QUESTION</label>
          <textarea id="clinical-question" rows={3} maxLength={4000} value={question} placeholder="What do the available guidelines say about ACE inhibitors during pregnancy?" onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); onRun() } }} onChange={(e) => { setQuestion(e.target.value); setResult(null); setError(null) }}/>
          <div className="v3-question-footer"><span>{question.length}/4000 characters</span><button className="v3-primary" type="button" disabled={loading || question.trim().length < 3} onClick={onRun}>{loading ? 'Analyzing evidence…' : 'Analyze evidence'} <ArrowRight size={16}/></button></div>
        </section>
        <section className="v3-card v3-answer-card" aria-label="Research findings">
          <div className="v3-card-heading"><div><span className="v3-step">02</span><h2>Research findings</h2></div><span className={`v3-status ${status === 'VERIFIED' ? 'v3-status-ok' : status === 'ABSTAINED' ? 'v3-status-warn' : ''}`}>{status === 'VERIFIED' ? 'Software checks passed' : status === 'ABSTAINED' ? 'No supported answer' : status.replaceAll('_', ' ')}</span></div>
          {loading ? <div className="v3-blank"><Activity size={27}/><h3>Research in progress</h3><p>The backend is retrieving and checking evidence. The response will appear when this request finishes.</p></div> : error ? <div className="v3-error" role="alert"><ShieldAlert size={22}/><div><h3>Analysis unavailable</h3><p>{error}</p><p>No findings are shown because the request did not complete.</p></div></div> : !result ? <div className="v3-blank"><FileSearch size={30}/><h3>Ready to investigate</h3><p>Choose a research topic or ask your own question. Evidence, citations, and the recorded execution will appear here.</p></div> : <>
            <div className="v3-answer-context"><span>{result.response.execution_mode.replaceAll('_', ' ')}</span><span>{sourceTotal} cited document(s)</span><span>{citations.length} citation(s)</span></div>
            {hasAnswer ? <div className="v3-answer-text">{result.response.answer}</div> : <div className="v3-abstain"><ShieldAlert size={24}/><div><strong>No sufficiently supported answer was produced</strong><p>{result.response.answer || 'The workflow did not return an answer.'}</p></div></div>}
            <p className="v4-status-explanation">{result.response.status === 'VERIFIED' ? 'Software checks passed. This is not independent clinical validation.' : result.response.status === 'ABSTAINED' ? 'No supported answer was produced. This does not prove the corpus contains no relevant information.' : `Workflow status: ${result.response.status.replaceAll('_', ' ')}.`} {result.response.execution_mode === 'MEDONTO' ? 'Ontology-assisted execution was recorded.' : result.response.execution_mode === 'BASELINE_ROUTED' ? 'The request was routed to conventional baseline processing.' : 'Consult the execution audit for the recorded route.'}</p>
          <div className="v3-answer-footer"><span><ShieldAlert size={15}/> Software verification is not independent medical validation.</span><div className="v3-inline-actions"><button type="button" disabled={saveDisabled} onClick={onSave}>{saveDisabled ? 'Saved to Notebook' : 'Save to Notebook'}</button><button type="button" onClick={refineQuestion} disabled={loading}>Edit and re-run</button><button type="button" onClick={copyAnswer}><Copy size={15}/>{copyState === 'copied' ? 'Copied' : copyState === 'unavailable' ? 'Copy unavailable' : 'Copy'}</button><button type="button" onClick={() => download('medonto-research-report.md', researchReport(result), 'text/markdown;charset=utf-8')}><Download size={15}/> Research report</button></div></div>
          </>}
        </section>
        {result && <section className="v3-card v3-summary-card" aria-label="Research summary">
          <div className="v3-card-heading"><div><span className="v3-step">03</span><h2>Evidence and limitations</h2></div></div>
          <div className="v3-facts"><div><span>Retrieved</span><strong>{result.response.retrieval.result_count}</strong><small>Evidence results</small></div><div><span>Cited sources</span><strong>{sourceTotal}</strong><small>Unique documents</small></div><div><span>Ontology filter</span><strong>{result.response.retrieval.used_ontology_filter ? 'Used' : 'Not used'}</strong><small>Actual routing metadata</small></div></div>
          <div className="v6-evidence-reading"><h3>What this execution establishes</h3><p>{sourceTotal === 0 ? 'No documents were cited by this result.' : sourceTotal === 1 ? 'One document was cited. This is not a cross-guideline comparison.' : `${sourceTotal} different documents were cited. Their presence does not establish agreement.`} {result.response.retrieval.result_count} retrieval result(s) were reported; retrieval is not proof that every statement is supported.</p>{result.response.status === 'ABSTAINED' && <button type="button" onClick={refineQuestion}>Refine this unanswered question</button>}</div>
          <p className="v3-subtle">{result.response.retrieval.retrieval_relaxed ? 'Retrieval constraints were relaxed during this execution. ' : ''}The corpus may not include every relevant guideline. Absence of a citation does not establish absence of medical evidence.</p>
        </section>}
        <section className="v3-card v3-technical">
          <button type="button" className="v3-technical-toggle" aria-expanded={showTechnical} onClick={() => setShowTechnical(!showTechnical)}><span><GitBranch size={19}/><strong>How was this result produced?</strong><small>Technical transparency · ontology and execution audit</small></span><ChevronDown className={showTechnical ? 'v3-rotated' : ''} size={20}/></button>
          {showTechnical && <div className="v3-technical-content">{!result ? <p className="v3-subtle">Run a question to inspect the real execution trace.</p> : <>
            <div className="v3-facts"><div><span>Graph updates</span><strong>{result.execution_trace.length}</strong></div><div><span>Trace interval total</span><strong>{formatElapsed(graphMs)}</strong></div><div><span>Constraints</span><strong>{result.response.ontology_constraints.length}</strong></div></div>
            <h3>Completed graph stages</h3><ol className="v3-trace">{result.execution_trace.map((e) => <li key={`${e.sequence}-${e.node}`}><span>{stageNames[e.node] || e.node.replaceAll('_', ' ')}</span><time>{formatElapsed(e.elapsed_since_previous_update_ms)}</time></li>)}</ol>
            {result.execution_trace.length === 0 && <p className="v3-subtle">No completed graph update trace was returned.</p>}
            <h3>Ontology information</h3><p className="v3-subtle">{result.response.ontology_constraints.length ? `${result.response.ontology_constraints.length} structured constraint(s) were returned by the application. They describe search restrictions, not a clinical endorsement.` : 'No public control-eligible constraints were returned for this execution.'}</p>{result.response.ontology_constraints.length > 0 && <button type="button" onClick={() => setShowRawOntology(v => !v)} aria-expanded={showRawOntology}>{showRawOntology ? 'Hide structured constraints' : 'Inspect structured constraints'}</button>}{showRawOntology && result.response.ontology_constraints.map((c,i) => <pre key={i} className="v3-code">{JSON.stringify(c,null,2)}</pre>)}
            <p className="v3-subtle">Durations are intervals between streamed updates; baseline work may occur outside this trace. VERIFIED describes implemented software checks only.</p>
            <button type="button" onClick={() => download('medonto-execution-audit.json', JSON.stringify({ disclaimer: 'Research prototype; not clinically validated.', ...result }, null, 2), 'application/json')}><Download size={15}/> Download technical audit JSON</button>
          </>}</div>}
        </section>
      </div>
      <aside className="v3-evidence" aria-label="Supporting evidence">
        <div className="v3-card v3-evidence-card"><div className="v3-card-heading"><div><span className="v3-step">EVIDENCE</span><h2>Supporting sources</h2></div><span className="v3-count">{citations.length}</span></div>
          <p className="v3-subtle">Inspect returned citations, copy their reference information, or open the original published guideline. These are document-level references, not claim-to-citation proofs.</p>{sourceAction && <p className="v6-source-feedback" role="status">{sourceAction}</p>}
          
          {!result ? <div className="v3-evidence-empty"><BookOpen size={27}/><p>Evidence will appear after a research question completes.</p></div> : visible.length === 0 ? <p className="v3-subtle">{citations.length === 0 ? 'No source citations returned for this outcome.' : 'No citations returned.'}</p> : <div className="v3-evidence-list">{visible.map((c) => <article className="v3-source" key={c.chunk_id}>
            <button type="button" aria-expanded={selectedCitation === c.chunk_id} onClick={() => setSelectedCitation(selectedCitation === c.chunk_id ? null : c.chunk_id)}><BookOpen size={18}/><span><strong>{c.title}</strong><small>{c.section_title || 'Section not supplied'} · {c.page_number == null ? 'Page not supplied' : `Page ${c.page_number}`}</small></span><ChevronDown size={16}/></button>
            {selectedCitation === c.chunk_id && <div className="v3-source-details"><p><strong>Document ID</strong><span>{c.document_id}</span></p><p><strong>Chunk ID</strong><span>{c.chunk_id}</span></p>{registry.find(d => d.document_id === c.document_id) && <><p><strong>Publisher</strong><span>{registry.find(d => d.document_id === c.document_id)?.publisher}</span></p><p><strong>Published</strong><span>{registry.find(d => d.document_id === c.document_id)?.publication_date || 'Not recorded'}</span></p><p><strong>Version</strong><span>{registry.find(d => d.document_id === c.document_id)?.version || 'Not recorded'}</span></p><p><strong>Registry status</strong><span>{registry.find(d => d.document_id === c.document_id)?.approval_status} / {registry.find(d => d.document_id === c.document_id)?.index_status}</span></p></>}<div className="v6-source-actions"><button type="button" onClick={() => copyCitation(c)}><Copy size={14}/> Copy citation</button>{(safeSourceUrl(c.source_url) || safeSourceUrl(registry.find(d => d.document_id === c.document_id)?.source_url)) && <a href={(safeSourceUrl(c.source_url) || safeSourceUrl(registry.find(d => d.document_id === c.document_id)?.source_url))!} target="_blank" rel="noreferrer">Open original guideline ↗</a>}</div>{(() => { const p = citedPassages.find(p => p.chunk_id === c.chunk_id && p.document_id === c.document_id); return p ? <section className="v7-passage"><h4>Retrieved passage from this execution</h4><p>{p.text}</p>{p.truncated && <small>Excerpt truncated at 1,200 characters.</small>}<small>Retrieved text is not proof that the passage supports every claim in the answer.</small></section> : <small>No eligible passage was released for this citation. Open the original guideline to investigate.</small> })()}</div>}
          </article>)}</div>}
        </div>
        <div className="v3-card v3-trust-card"><div className="v3-trust-heading"><CheckCircle2 size={17}/><strong>What you can verify</strong></div><p>Read the published source, inspect its provenance in Guideline Library, and review the software execution record.</p><p className="v3-subtle">This application does not establish clinical completeness or treatment safety.</p></div>
      </aside>
    </div>
  </section>
}
