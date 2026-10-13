import { test, expect } from '@playwright/test'

test('guided research and evidence explorer use a synthetic response', async ({ page }) => {
  let calls = 0
  await page.route('**/api/demo/query', async route => {
    calls += 1
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      response: { query: 'Synthetic test question', answer: 'SYNTHETIC ANSWER ONLY', status: 'VERIFIED', execution_mode: 'MEDONTO',
        citations: [{ chunk_id: 'chunk-1', document_id: 'doc-1', title: 'Synthetic Guideline', section_title: 'Testing', page_number: 2, source_url: null }],
        ontology_constraints: [], retrieval: { result_count: 1, used_ontology_filter: true, retrieval_relaxed: false } },
      execution_trace: [{ sequence: 1, node: 'initialize', elapsed_since_previous_update_ms: 5 }],
    }) })
  })
  await page.goto('http://127.0.0.1:5173/')
  await expect(page.getByRole('heading', { name: 'Research Workspace' })).toBeVisible()
  await page.getByRole('button', { name: /Medication safety/i }).click()
  await expect(page.getByLabel('QUESTION')).toHaveValue(/ACE inhibitors/)
  await page.getByRole('button', { name: /Analyze evidence/i }).click()
  await expect(page.getByText('SYNTHETIC ANSWER ONLY')).toBeVisible()
  await page.getByRole('button', { name: /Synthetic Guideline/i }).click()
  await expect(page.getByText('chunk-1')).toBeVisible()
  // V4 removed low-value citation filtering; source details remain inspectable.
  await expect(page.getByRole('textbox', { name: 'Filter evidence' })).toHaveCount(0)
  expect(calls).toBe(1)
})
