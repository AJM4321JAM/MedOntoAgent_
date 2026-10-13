import { test, expect } from '@playwright/test'

test('research workspace restores a session result without another provider request', async ({ page }) => {
  let requests = 0
  await page.route('**/api/demo/query', async (route) => {
    requests += 1
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      response: {
        query: 'What do the available guidelines say about ACE inhibitors during pregnancy?',
        answer: 'SYNTHETIC RESEARCH OUTPUT: NOT MEDICAL ADVICE', status: 'VERIFIED', execution_mode: 'MEDONTO',
        citations: [{ chunk_id: 'demo-chunk', document_id: 'demo-doc', title: 'Synthetic Guideline', section_title: 'Research test', page_number: 3, source_url: null }],
        ontology_constraints: [], retrieval: { result_count: 1, used_ontology_filter: true, retrieval_relaxed: false },
      }, execution_trace: [{ sequence: 1, node: 'initialize', elapsed_since_previous_update_ms: 6 }],
    }) })
  })
  await page.goto('http://127.0.0.1:5173/')
  await expect(page.getByRole('heading', { name: 'Research Workspace' })).toBeVisible()
  await page.getByRole('button', { name: /Medication safety/i }).click()
  await page.getByRole('button', { name: /Analyze evidence/i }).click()
  await expect(page.getByText('SYNTHETIC RESEARCH OUTPUT: NOT MEDICAL ADVICE')).toBeVisible()
  await page.getByRole('button', { name: /Synthetic Guideline/i }).click()
  await expect(page.getByText('demo-chunk')).toBeVisible()
  await page.getByRole('button', { name: /How was this result produced/i }).click()
  await expect(page.getByText('Initialize', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: /Labetalol guidance/i }).click()
  await expect(page.getByText('Ready to investigate')).toBeVisible()
  await page.locator('.v3-history').first().click()
  await expect(page.getByText('SYNTHETIC RESEARCH OUTPUT: NOT MEDICAL ADVICE')).toBeVisible()
  expect(requests).toBe(1)
})
