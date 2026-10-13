import { useEffect, useState } from 'react'
import { Activity, BookOpen, Layers3, Moon, ShieldCheck, Stethoscope, Sun, NotebookPen } from 'lucide-react'
import { runClinicalQuery, type DemoQueryResult } from './api'
import { KnowledgeLibrary, SystemOperations } from './WorkspacePages'
import { ResearchWorkspaceV3 } from './ResearchWorkspaceV3'
import './App.css'
import './Theme.css'
import { ResearchNotebook } from './ResearchNotebook'
import { readNotebook, writeNotebook, type NotebookEntry } from './notebookStorage'
import './ResearchV4.css'

type Page = 'workbench' | 'library' | 'notebook' | 'operations'

function App() {
  const [page, setPage] = useState<Page>('workbench')
  const [themeMode, setThemeMode] = useState<'dark' | 'light'>(() => {
    try {
      const saved = window.localStorage.getItem('medonto-theme')
      if (saved === 'light' || saved === 'dark') return saved
    } catch { /* Browser storage may be disabled. */ }
    return 'dark'
  })
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeMode)
    document.documentElement.style.colorScheme = themeMode
    try { window.localStorage.setItem('medonto-theme', themeMode) } catch { /* Browser storage may be disabled. */ }
  }, [themeMode])
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<DemoQueryResult | null>(null)
  const [researchHistory, setResearchHistory] = useState<DemoQueryResult[]>([])
  const [notebook, setNotebook] = useState<NotebookEntry[]>(readNotebook)
  const [storageError, setStorageError] = useState(false)
  function updateNotebook(items: NotebookEntry[]) { setNotebook(items); setStorageError(!writeNotebook(items)) }
  function saveCurrent() {
    if (!result) return
    const id = `${Date.now()}-${Math.floor(Math.random()*100000)}`
    updateNotebook([{ id, result, savedAt: new Date().toISOString(), starred: false, notes: '' }, ...notebook].slice(0, 60))
  }
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
      setResearchHistory((items) => [data, ...items].slice(0, 8))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The request could not be completed.')
    } finally {
      setLoading(false)
    }
  }

  const navigation = [
    { id: 'workbench' as const, name: 'Research Workspace', icon: Stethoscope },
    { id: 'library' as const, name: 'Guideline Library', icon: BookOpen },
    { id: 'notebook' as const, name: 'Research Notebook', icon: NotebookPen },
    { id: 'operations' as const, name: 'Technical Inspection', icon: Activity },
  ]

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon"><Layers3 size={23}/></div><div><div className="brand-name">MedOntoAgent</div><div className="brand-subtitle">Clinical evidence research</div></div></div>
      <div className="sidebar-label">RESEARCH</div>
      <nav className="navigation" aria-label="Main navigation">
        {navigation.map(({ id, name, icon: Icon }) => <button type="button" key={id} className={`nav-item ${page === id ? 'active' : ''}`} onClick={() => setPage(id)}><Icon size={19}/><span>{name}</span>{page === id && <span className="nav-indicator"/>}</button>)}
      </nav>
      <div className="sidebar-bottom"><div className="sidebar-info"><ShieldCheck size={18}/><div><strong>Research prototype</strong><span>Not for clinical decision-making</span></div></div><div className="sidebar-version">MedOntoAgent · Demonstration</div></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span><strong>{navigation.find((item) => item.id === page)?.name}</strong></div><div className="topbar-actions">
        <button type="button" className="theme-toggle" onClick={() => setThemeMode((mode) => mode === 'dark' ? 'light' : 'dark')} aria-label={themeMode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>{themeMode === 'dark' ? <Sun size={16}/> : <Moon size={16}/>}<span>{themeMode === 'dark' ? 'Light mode' : 'Dark mode'}</span></button>
        <span className="connection-status"><span className="status-dot"/>Local demo API</span>
      </div></header>
      <div className="content">
        {page === 'workbench' ? <ResearchWorkspaceV3 onSave={saveCurrent} saveDisabled={!result || notebook.some(e => e.result === result)} notebookCount={notebook.length} onNotebook={() => setPage('notebook')} question={question} setQuestion={setQuestion} result={result} setResult={setResult} loading={loading} error={error} setError={setError} onRun={handleRun} history={researchHistory} onClearHistory={() => setResearchHistory([])}/> : page === 'library' ? <div className="v4-page"><div className="v4-pagehead"><div><span className="v4-eyebrow">SOURCE DISCOVERY</span><h1>Guideline Library</h1><p>Browse the registered clinical evidence and inspect document provenance. Development fixtures are not clinical guidelines.</p></div></div><KnowledgeLibrary/></div> : page === 'notebook' ? <ResearchNotebook entries={notebook} setEntries={updateNotebook} openResearch={(item) => { setResult(item); setQuestion(item.response.query); setError(null); setPage('workbench') }}/>: <div className="v4-page"><div className="v4-pagehead"><div><span className="v4-eyebrow">ADVANCED / EXAMINER</span><h1>Technical Inspection</h1><p>Read-only system diagnostics. Execution-specific ontology and LangGraph evidence is available from each research result.</p></div></div><SystemOperations/></div>}
        {storageError && <div role="alert" className="v4-notice">Browser storage is unavailable or full. This Notebook change may not survive refresh.</div>}<footer className="footer"><ShieldCheck size={15}/>For research and demonstration only. Not validated for independent clinical use.</footer>
      </div>
    </main>
  </div>
}
export default App
