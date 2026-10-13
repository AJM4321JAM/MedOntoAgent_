import { Activity, BookOpen, FileCheck2, GitBranch, Layers, ShieldCheck } from 'lucide-react'
import { SystemOperations } from './WorkspacePages'
import type { DemoQueryResult } from './api'
import './ResearchFinishV8.css'

type Props = { lastResult: DemoQueryResult | null; onResearch: () => void; onLibrary: () => void }
const capabilities = [
  ['Guideline evidence', 'Browse the registered documents, source provenance and indexing eligibility.', 'Guideline Library'],
  ['Retrieval and citations', 'Inspect the retrieved result count, actual citations and permitted passages from a completed query.', 'Research Workspace → Supporting sources'],
  ['Ontology assistance', 'View returned ontology constraints and the ontology-filter flag. This does not imply clinical correctness.', 'Research Workspace → How was this result produced?'],
  ['Agentic orchestration', 'Inspect only the execution stages recorded by the backend audit.', 'Research Workspace → Execution audit'],
  ['Software validation', 'Review the returned terminal status. VERIFIED means software checks passed, not medical validation.', 'Research findings'],
  ['Abstention and fallback', 'Observe actual execution modes and unsupported-answer handling; do not infer unreported routing reasons.', 'Research findings'],
  ['Governed documents', 'Review document approvals, versions and provenance using registry-supplied metadata.', 'Guideline Library'],
] as const
export function TechnicalInspectionV8({ lastResult, onResearch, onLibrary }: Props) {
  const response = lastResult?.response
  return <div className="v4-page v8-inspection">
    <div className="v4-pagehead"><div><span className="v4-eyebrow">ADVANCED / EXAMINER</span><h1>Technical Inspection</h1><p>A map of implemented capabilities and where to verify them in this running application.</p></div></div>
    <div className="v8-intro"><ShieldCheck size={20}/><div><strong>Evidence-based demonstration</strong><p>These are locations to inspect implementation artifacts, not an assertion of independent medical validation or proven superiority to a baseline.</p></div></div>
    <h2>Capability guide</h2><div className="v8-cap-grid">{capabilities.map(([name,description,location],i) => <article className="v8-cap" key={name}><div className="v8-cap-num">{String(i+1).padStart(2,'0')}</div><h3>{name}</h3><p>{description}</p><small>Inspect in: {location}</small></article>)}</div>
    <section className="v8-inspection-section"><h2>Last result in this browser session</h2>{response ? <><div className="v8-statusline"><span>Software status: <strong>{response.status}</strong></span><span>Execution mode: <strong>{response.execution_mode}</strong></span><span>Cited documents: <strong>{new Set(response.citations.map(c=>c.document_id)).size}</strong></span></div><p>Open this result in Research Workspace to see the complete returned ontology details, citations and execution audit. This summary contains no invented trace data.</p><button type="button" className="v4-action" onClick={onResearch}><GitBranch size={15}/> View result audit</button></> : <><p>No result is currently loaded. Execute or reopen a research question to inspect a real audit.</p><button type="button" className="v4-action" onClick={onResearch}><BookOpen size={15}/> Open Research Workspace</button></>} <button type="button" className="v4-action" onClick={onLibrary}><Layers size={15}/> Inspect governed guidelines</button></section>
    <section className="v8-inspection-section"><h2><Activity size={18}/> System operations</h2><p>These diagnostics report only the checks the application actually performs. They are not a live historical monitoring system.</p><SystemOperations/></section>
    <section className="v8-inspection-section"><h2><FileCheck2 size={18}/> Demonstration order</h2><p>Run a guideline query → inspect its named citation and original passage → examine the actual ontology and execution audit → save and annotate in Notebook → browse governance information in Library → review these technical diagnostics.</p></section>
  </div>
}
