import { test, expect } from '@playwright/test'

test('guideline shelf separates sources and prepares research question', async ({ page }) => {
  await page.route('**/api/demo/library', route => route.fulfill({ status:200, contentType:'application/json', body:JSON.stringify({ registry_state:'available', total_documents:2, documents:[
    { document_id:'DOC-NICE',title:'Hypertension in pregnancy NICE guideline',publisher:'NICE',clinical_domain:'Hypertension in pregnancy',document_type:'GUIDELINE',source_authority:'GUIDELINE',version:'2019',approval_status:'APPROVED',index_status:'INDEXED',publication_date:'2019-06-25',effective_date:null,registered_at:'2026-01-01',indexed_at:null,source_url:'https://www.nice.org.uk',supersedes_document_id:null,superseded_by_document_id:null },
    { document_id:'DOC-TEST',title:'Synthetic Governance Reference',publisher:'Synthetic Publisher',clinical_domain:'Test',document_type:'GUIDELINE',source_authority:'GUIDELINE',version:'1',approval_status:'APPROVED',index_status:'INDEXED',publication_date:null,effective_date:null,registered_at:'2026-01-01',indexed_at:null,source_url:null,supersedes_document_id:null,superseded_by_document_id:null }
  ]}) }))
  await page.goto('http://127.0.0.1:5173/')
  await page.getByRole('button',{name:'Guideline Library'}).click()
  await expect(page.getByRole('heading',{name:'Hypertension in pregnancy NICE guideline'})).toBeVisible()
  await expect(page.getByText('Synthetic Governance Reference')).toHaveCount(0)
  await page.getByRole('button',{name:/Inspect guideline/}).click()
  await expect(page.getByRole('dialog',{name:'Guideline details'})).toBeVisible()
  await page.getByRole('button',{name:'Close guideline details'}).click()
  await page.getByRole('button',{name:/Ask about this guideline/}).click()
  await expect(page.getByRole('textbox',{name:'QUESTION'})).toContainText('Hypertension in pregnancy NICE guideline')
})
