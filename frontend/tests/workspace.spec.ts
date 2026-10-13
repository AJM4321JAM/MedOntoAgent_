import { expect, test } from '@playwright/test'

test('knowledge library shows governed synthetic documents', async ({ page }) => {
  await page.route('**/api/demo/library', async (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ registry_state: 'available', total_documents: 1, documents: [{ document_id: 'DOC-TEST12345', title: 'Synthetic Governance Reference', publisher: 'Synthetic Publisher', clinical_domain: 'Test domain', document_type: 'GUIDELINE', source_authority: 'GUIDELINE', version: '1', approval_status: 'APPROVED', index_status: 'INDEXED', publication_date: null, effective_date: null, registered_at: '2026-01-01T00:00:00Z', indexed_at: null, source_url: null, supersedes_document_id: null, superseded_by_document_id: null }] }) }))
  await page.goto('http://127.0.0.1:5173/')
  await page.getByRole('button', { name: /Knowledge Library/i }).click()
  await expect(page.getByRole('heading', { name: 'Knowledge Library' })).toBeVisible()
  await expect(page.getByText('Synthetic Governance Reference')).toBeVisible()
  await page.getByLabel('Search documents').fill('no match')
  await expect(page.getByText('No documents match the current filters.')).toBeVisible()
})

test('system operations never invents external health checks', async ({ page }) => {
  await page.route('**/api/demo/operations', async (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ service: 'MedOntoAgent', environment: 'development', status: 'running', app_version: 'test', demo_llm_timeout_seconds: 20, demo_provider_attempt_limit: 8, telemetry_configured: false, historical_metrics_available: false, components: [{ label: 'Gemini', state: 'not_checked', explanation: 'No provider request was performed.' }] }) }))
  await page.goto('http://127.0.0.1:5173/')
  await page.getByRole('button', { name: /System Operations/i }).click()
  await expect(page.getByRole('heading', { name: 'System Operations' })).toBeVisible()
  await expect(page.getByText('No provider request was performed.')).toBeVisible()
  await expect(page.getByText('Not exposed by this API')).toBeVisible()
})
