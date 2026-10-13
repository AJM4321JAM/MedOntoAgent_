import { test, expect } from '@playwright/test'

test('technical capability guide shows actual inspection destinations', async ({ page }) => {
  await page.goto('http://127.0.0.1:5173/')
  await page.getByRole('button', { name: /Technical Inspection/i }).first().click()
  await expect(page.getByRole('heading', { name: 'Capability guide' })).toBeVisible()
  await expect(page.getByText('Ontology assistance', { exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'System operations', exact: true })).toBeVisible()
})

test('notebook selection export is available for a saved synthetic finding', async ({ page }) => {
  await page.route('**/api/demo/query', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ response: { query: 'What do guidelines say about ACE inhibitors during pregnancy?', answer: 'SYNTHETIC TEST RESULT. NOT CLINICAL EVIDENCE.', status: 'VERIFIED', execution_mode: 'MEDONTO', citations: [{ chunk_id: 'synthetic-chunk', document_id: 'synthetic-doc', title: 'Synthetic Guideline', section_title: 'Synthetic section', page_number: 1, source_url: null }], ontology_constraints: [], retrieval: { result_count: 1, used_ontology_filter: true, retrieval_relaxed: false } }, execution_trace: [] }) }))
  await page.goto('http://127.0.0.1:5173/')
  await page.getByRole('button', { name: /Medication safety/i }).click()
  await page.getByRole('button', { name: /Analyze evidence/i }).click()
  await page.getByRole('button', { name: /Save to Notebook/i }).click()
  await page.getByRole('button', { name: /Research Notebook/i }).first().click()
  await expect(page.getByRole('button', { name: 'Export selected' })).toBeDisabled()
  await page.getByRole('checkbox', { name: /Select for export/i }).check()
  await expect(page.getByRole('button', { name: 'Export selected' })).toBeEnabled()
})
