import type { DemoQueryResult } from './api'

export type NotebookEntry = { id: string; savedAt: string; result: DemoQueryResult; starred: boolean; notes: string }
export function readNotebook(): NotebookEntry[] {
  try {
    const raw = window.localStorage.getItem('medonto-research-notebook-v1')
    const value: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(value) ? value.filter((v): v is NotebookEntry => { const e = v as NotebookEntry | null; return !!e && typeof e.id === 'string' && typeof e.notes === 'string' && typeof e.result?.response?.query === 'string' }).slice(0, 60) : []
  } catch { return [] }
}
export function writeNotebook(items: NotebookEntry[]) {
  try { window.localStorage.setItem('medonto-research-notebook-v1', JSON.stringify(items.slice(0, 60))); return true }
  catch { return false }
}
