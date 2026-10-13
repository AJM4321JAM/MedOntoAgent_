
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
            API not connected
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
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Enter a question about the available clinical guidelines..."
                  rows={4}
                  maxLength={4000}
                />

                <div className="question-actions">
                  <span>{question.length} / 4000 characters</span>
                  <button className="run-button" type="button" disabled>
                    <LockKeyhole size={16} />
                    Run Agent
                    <ArrowRight size={16} />
                  </button>
                </div>
                <div className="connection-note">
                  Live execution will be enabled after the secure backend connection is configured.
                </div>

                <div className="examples">
                  <div className="examples-label">TRY A QUESTION</div>
                  <div className="example-list">
                    {exampleQuestions.map((example) => (
                      <button
                        type="button"
                        key={example}
                        onClick={() => setQuestion(example)}
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
                    <span className="neutral-tag">Not run</span>
                  </div>
                  <div className="empty-state response-empty">
                    <div className="empty-icon"><Stethoscope size={28} /></div>
                    <h3>Ready for an evidence-grounded response</h3>
                    <p>
                      Once connected, answers and their verification status will
                      appear here directly from MedOntoAgent.
                    </p>
                  </div>
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
                    <p>
                      {detailTab === 'evidence'
                        ? 'Source citations and retrieved evidence details will come from the clinical API.'
                        : 'Actual executed LangGraph nodes, retries and timing will come from the protected audit API.'}
                    </p>
                  </div>
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
