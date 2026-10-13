import { test, expect } from '@playwright/test'

test('researcher can inspect, copy and refine an evidence result without auto-running', async ({ page }) => {
  let calls = 0
  await page.route('**/api/demo/query', async route => {
    calls++
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
      response: {
        query: 'What do the available guidelines say about ACE inhibitors during pregnancy?',
        answer: 'SYNTHETIC RESEARCH OUTPUT ONLY', status: 'VERIFIED', execution_mode: 'MEDONTO',
        citations: [{ chunk_id: 'synthetic-chunk', document_id: 'synthetic-doc', title: 'Synthetic Source', section_title: 'Test section', page_number: 7, source_url: null }],
        ontology_constraints: [{subject_name:'SyntheticConcept', predicate_name:'TEST_RELATION', object_name:'SyntheticObject'}],
        retrieval: { result_count: 3, used_ontology_filter: true, retrieval_relaxed: false },
      }, execution_trace: [{ sequence: 1, node: 'initialize', elapsed_since_previous_update_ms: 2 }],
    }) })
  })
  await page.goto('http://127.0.0.1:5173/')
  await page.getByRole('button', { name: /Medication safety/i }).click()
  await page.getByRole('button', { name: /Analyze evidence/i }).click()
  await expect(page.getByText('One document was cited. This is not a cross-guideline comparison.', { exact: false })).toBeVisible()
  await page.getByRole('button', { name: /Synthetic Source/i }).click()
  await expect(page.getByRole('button', { name: /Copy citation/i })).toBeVisible()
  await page.getByRole('button', { name: /How was this result produced/i }).click()
  await expect(page.getByRole('button', { name: 'Inspect structured constraints' })).toBeVisible()
  await page.getByRole('button', { name: 'Inspect structured constraints' }).click()
  await expect(page.getByText(/SyntheticConcept/)).toBeVisible()
  await page.getByRole('button', { name: 'Edit and re-run' }).click()
  await expect(page.getByText('Ready to investigate')).toBeVisible()
  await expect(page.getByRole('textbox', { name: 'QUESTION' })).toContainText('ACE inhibitors')
  expect(calls).toBe(1)
})
