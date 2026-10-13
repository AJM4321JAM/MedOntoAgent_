import { test, expect } from '@playwright/test'

test('workbench displays a synthetic backend response', async ({ page }) => {
  let requestCount = 0

  await page.route('**/api/demo/query', async (route) => {
    requestCount += 1

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        response: {
          query: 'Synthetic test question',
          answer: 'SYNTHETIC TEST RESPONSE. Not clinical guidance.',
          status: 'VERIFIED',
          execution_mode: 'MEDONTO',
          citations: [
            {
              chunk_id: 'test-chunk-1',
              document_id: 'test-document',
              title: 'Synthetic Test Guideline',
              section_title: 'Test section',
              page_number: 2,
              source_url: null,
            },
          ],
          ontology_constraints: [
            {
              relation: 'synthetic_test_relation',
              description: 'Test-only ontology constraint',
            },
          ],
          retrieval: {
            result_count: 1,
            used_ontology_filter: true,
            retrieval_relaxed: false,
          },
        },
        execution_trace: [
          {
            sequence: 1,
            node: 'initialize',
            elapsed_since_previous_update_ms: 2.5,
          },
          {
            sequence: 2,
            node: 'verified',
            elapsed_since_previous_update_ms: 4.2,
          },
        ],
      }),
    })
  })

  await page.goto('http://127.0.0.1:5173/')

  await page.getByLabel('CLINICAL QUESTION').fill('Synthetic test question')
  await page.getByRole('button', { name: /Run Agent/i }).click()

  await expect(page.getByText('SYNTHETIC TEST RESPONSE. Not clinical guidance.')).toBeVisible()
  await expect(page.getByText('Mode: MEDONTO', { exact: true })).toBeVisible()
  await expect(page.getByText('Synthetic Test Guideline', { exact: true }).first()).toBeVisible()

  await page.getByRole('button', { name: /Decision Audit/i }).click()

  await expect(page.getByText('Initialize', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('Verification completed', { exact: true }).first()).toBeVisible()

  expect(requestCount).toBe(1)
})

test('workbench handles backend errors without fabricating results', async ({ page }) => {
  await page.route('**/api/demo/query', async (route) => {
    await route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({
        detail: 'Synthetic test service failure',
      }),
    })
  })

  await page.goto('http://127.0.0.1:5173/')

  await page.getByLabel('CLINICAL QUESTION').fill('Synthetic test question')
  await page.getByRole('button', { name: /Run Agent/i }).click()

  await expect(page.getByText('Request unsuccessful')).toBeVisible()
  await expect(page.getByText('The clinical service is temporarily unavailable.')).toBeVisible()
  await expect(page.getByText('SYNTHETIC TEST RESPONSE.')).toHaveCount(0)
})
