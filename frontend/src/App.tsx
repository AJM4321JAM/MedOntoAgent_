
import { useState } from 'react'
import {
  Activity,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  GitBranch,
  Layers3,
  LockKeyhole,
  Search,
  ShieldCheck,
  Stethoscope,
  Workflow,
} from 'lucide-react'
import { runClinicalQuery, type DemoQueryResult } from './api'
import './App.css'

type Page = 'workbench' | 'library' | 'operations'
type DetailTab = 'evidence' | 'audit'

const exampleQuestions = [
  'What guidance applies to hypertension management during pregnancy?',
  'What considerations apply to ACE inhibitors during pregnancy?',
  'What do the available guidelines say about labetalol?',
]

function App() {
  const [page, setPage] = useState<Page>('workbench')
  const [question, setQuestion] = useState('')
  const [detailTab, setDetailTab] = useState<DetailTab>('evidence')
  const [result, setResult] = useState<DemoQueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleRun() {
    if (loading || question.trim().length < 3) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await runClinicalQuery(question.trim())
      setResult(data)
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : 'The request could not be completed.'
      )
    } finally {
      setLoading(false)
    }
  }

  const navigation = [
    { id: 'workbench' as const, name: 'Clinical Workbench', icon: Stethoscope },
    { id: 'library' as const, name: 'Knowledge Library', icon: BookOpen },
    { id: 'operations' as const, name: 'System Operations', icon: Activity },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Layers3 size={23} /></div>
          <div>
            <div className="brand-name">MedOntoAgent</div>
            <div className="brand-subtitle">Clinical intelligence</div>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="navigation" aria-label="Main navigation">
          {navigation.map(({ id, name, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => setPage(id)}
            >
              <Icon size={19} />
              <span>{name}</span>
              {page === id && <span className="nav-indicator" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-info">
            <ShieldCheck size={18} />
            <div>
              <strong>Research prototype</strong>
              <span>Not for clinical decision-making</span>
            </div>
          </div>
          <div className="sidebar-version">MedOntoAgent · Demonstration</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>
            <strong>
              {page === 'workbench'
                ? 'Clinical Workbench'
                : page === 'library'
                  ? 'Knowledge Library'
                  : 'System Operations'}
            </strong>
          </div>
          <span className="connection-status">
            <span className="status-dot" />
            Local API connected
          </span>
        </header>

        <div className="content">
          {page === 'workbench' ? (
            <>
              <section className="page-heading">
                <div className="eyebrow">
                  <Workflow size={15} /> ONTOLOGY-GROUNDED EVIDENCE
                </div>
                <h1>Clinical Evidence Workbench</h1>
                <p>
                  Explore guideline evidence through ontology-constrained retrieval,
                  grounded generation and explicit verification.
                </p>
              </section>

              <section className="question-card">
                <div className="section-heading">
                  <div className="section-icon"><Search size={19} /></div>
                  <div>
                    <h2>Ask a clinical question</h2>
                    <p>Use a synthetic scenario or a general guideline question.</p>
                  </div>
                </div>

                <label htmlFor="clinical-question" className="field-label">
                  CLINICAL QUESTION
                </label>
                <textarea
                  id="clinical-question"
                  value={question}
                  onChange={(event) => {
                    setQuestion(event.target.value)
                    setResult(null)
                    setError(null)
                  }}
                  placeholder="Enter a question about the available clinical guidelines..."
                  rows={4}
                  maxLength={4000}
                />

                <div className="question-actions">
                  <span>{question.length} / 4000 characters</span>
                  <button
                    className="run-button"
                    type="button"
                    disabled={loading || question.trim().length < 3}
                    onClick={handleRun}
                  >
                    <LockKeyhole size={16} />
                    {loading ? 'Running agent...' : 'Run Agent'}
                    <ArrowRight size={16} />
                  </button>
                </div>
                <div className="connection-note">
                  Local demonstration mode. Each run executes the real backend and may use configured AI services.
                </div>

                <div className="examples">
                  <div className="examples-label">TRY A QUESTION</div>
                  <div className="example-list">
                    {exampleQuestions.map((example) => (
                      <button
                        type="button"
                        key={example}
                        onClick={() => {
                          setQuestion(example)
                          setResult(null)
                          setError(null)
                        }}
                      >
                        {example}
                        <ArrowRight size={15} />
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              <div className="result-grid">
                <section className="panel">
                  <div className="panel-header">
                    <div className="panel-title">
                      <FileText size={19} />
                      <h2>Clinical response</h2>
                    </div>
                    <span className="neutral-tag">
                      {loading ? 'Running' : result?.response.status ?? 'Not run'}
                    </span>
                  </div>
                  {error ? (
                    <div className="result-content" role="alert">
                      <h3>Request unsuccessful</h3>
                      <p>{error}</p>
                    </div>
                  ) : result ? (
                    <div className="result-content">
                      <div className="result-metadata">
                        <span>Mode: {result.response.execution_mode}</span>
                        <span>Status: {result.response.status}</span>
                      </div>
                      <p className="clinical-answer">{result.response.answer}</p>
                      <p className="result-disclaimer">
                        Verification indicates implemented evidence checks,
                        not a guarantee of clinical completeness.
                      </p>
                    </div>
                  ) : (
                    <div className="empty-state response-empty">
                      <div className="empty-icon"><Stethoscope size={28} /></div>
                      <h3>{loading ? 'Executing clinical workflow' : 'Ready for an evidence-grounded response'}</h3>
                      <p>
                        {loading
                          ? 'Please wait while MedOntoAgent processes the question.'
                          : 'Submit a question to see the actual backend response.'}
                      </p>
                    </div>
                  )}
                </section>

                <section className="panel">
                  <div className="tab-header">
                    <button
                      className={detailTab === 'evidence' ? 'selected' : ''}
                      onClick={() => setDetailTab('evidence')}
                    >
                      <BookOpen size={17} /> Evidence
                    </button>
                    <button
                      className={detailTab === 'audit' ? 'selected' : ''}
                      onClick={() => setDetailTab('audit')}
                    >
                      <GitBranch size={17} /> Decision Audit
                    </button>
                  </div>

                  {result ? (
                    <div className="result-content">
                      {detailTab === 'audit' ? (
                        <>
                          <h3>Actual LangGraph execution</h3>
                          {result.execution_trace.map((event) => (
                            <div className="audit-event" key={event.sequence}>
                              <strong>{event.sequence}. {event.node}</strong>
                              <span>{event.elapsed_since_previous_update_ms.toFixed(1)} ms</span>
                            </div>
                          ))}
                          <p className="result-disclaimer">
                            Timings are measured between streamed node updates,
                            not isolated node execution durations.
                            Baseline execution occurs outside this graph timeline.
                          </p>
                        </>
                      ) : (
                        <>
                          <h3>Evidence citations</h3>
                          {result.response.citations.length === 0 ? (
                            <p>No citations were returned for this outcome.</p>
                          ) : (
                            result.response.citations.map((citation) => (
                              <div className="citation-item" key={citation.chunk_id}>
                                <strong>{citation.title}</strong>
                                <p>{citation.section_title || 'Section not specified'}</p>
                                <span>
                                  {citation.page_number != null
                                    ? `Page ${citation.page_number}`
                                    : 'Page not specified'}
                                </span>
                                {citation.source_url?.startsWith('https://') && (
                                  <a href={citation.source_url} target="_blank" rel="noreferrer">
                                    View source
                                  </a>
                                )}
                              </div>
                            ))
                          )}
                          <h3 className="constraints-heading">Ontology constraints</h3>
                          {result.response.ontology_constraints.length === 0 ? (
                            <p>No public ontology constraints returned for this outcome.</p>
                          ) : (
                            result.response.ontology_constraints.map((constraint, index) => (
                              <pre className="constraint-item" key={index}>
                                {JSON.stringify(constraint, null, 2)}
                              </pre>
                            ))
                          )}
                          <p className="result-disclaimer">
                            Retrieved results: {result.response.retrieval.result_count}.
                            Ontology filter used: {result.response.retrieval.used_ontology_filter ? 'Yes' : 'No'}.
                          </p>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="empty-state details-empty">
                      <div className="empty-icon">
                        {detailTab === 'evidence'
                          ? <BookOpen size={25} />
                          : <GitBranch size={25} />}
                      </div>
                      <h3>
                        {detailTab === 'evidence'
                          ? 'Evidence will appear here'
                          : 'Execution timeline will appear here'}
                      </h3>
                      <p>Run a question to display real backend data.</p>
                    </div>
                  )}
                </section>
              </div>

              <section className="pipeline-card">
                <div className="pipeline-heading">
                  <div>
                    <h2>MedOntoAgent verification pipeline</h2>
                    <p>Architecture overview, not a live execution trace</p>
                  </div>
                  <span className="neutral-tag">Reference</span>
                </div>
                <div className="pipeline-steps">
                  {[
                    { icon: Search, title: 'Concept mapping', detail: 'Identify relevant concepts' },
                    { icon: Layers3, title: 'Ontology rules', detail: 'Build eligible constraints' },
                    { icon: Database, title: 'Retrieval', detail: 'Find guideline evidence' },
                    { icon: CheckCircle2, title: 'Validation', detail: 'Check supported claims' },
                  ].map(({ icon: Icon, title, detail }, index) => (
                    <div className="pipeline-step" key={title}>
                      <div className="step-number">0{index + 1}</div>
                      <Icon size={20} />
                      <strong>{title}</strong>
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <section className="secondary-page">
              <div className="eyebrow">
                {page === 'library' ? <BookOpen size={16} /> : <Activity size={16} />}
                MEDONTOAGENT WORKSPACE
              </div>
              <h1>{page === 'library' ? 'Knowledge Library' : 'System Operations'}</h1>
              <p>
                {page === 'library'
                  ? 'A read-only view of the registered guideline documents, provenance and approval status.'
                  : 'A view of system health and diagnostic information reported by the backend.'}
              </p>
              <div className="secondary-placeholder">
                {page === 'library'
                  ? <BookOpen size={34} />
                  : <Clock3 size={34} />}
                <h2>Awaiting backend integration</h2>
                <p>
                  This section will display actual application data.
                  No sample documents or operational statistics are being fabricated.
                </p>
              </div>
            </section>
          )}

          <footer className="footer">
            <ShieldCheck size={15} />
            For research and demonstration only. Not validated for independent clinical use.
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
