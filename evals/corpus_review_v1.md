# Clinical Corpus v1 Review

## Scope

Focused evaluation domain:

Hypertension management with pregnancy-related treatment constraints,
contraindications, source applicability, and treatment conditionality.

Corpus version:

`clinical-corpus-v1`

## Frozen sources

### NICE NG133

Document ID:

`DOC-e943ba2bdc31`

Title:

Hypertension in pregnancy: diagnosis and management

Status:

APPROVED

Frozen chunk count:

160

Manually confirmed evidence:

- `DOC-e943ba2bdc31-CHUNK-0022`
  - ACE inhibitors and ARBs are associated with increased risk of congenital abnormalities when taken during pregnancy.
  - Women planning pregnancy should discuss alternative antihypertensive treatment.
  - The cited MHRA safety advice states that ACE inhibitors and angiotensin II receptor antagonists are not for use in pregnancy.

- `DOC-e943ba2bdc31-CHUNK-0023`
  - Women taking ACE inhibitors or ARBs who become pregnant should stop that antihypertensive treatment, preferably within 2 working days of notification of pregnancy, and be offered alternatives.

- `DOC-e943ba2bdc31-CHUNK-0030`
  - Labetalol should be considered for chronic hypertension in pregnant women.
  - Nifedipine should be considered when labetalol is not suitable.
  - Methyldopa should be considered when both labetalol and nifedipine are not suitable.
  - Choice should also consider pre-existing treatment, side-effect profiles, risks including fetal effects, and the woman's preference.

- `DOC-e943ba2bdc31-CHUNK-0112`
  - Severe hypertension in critical care during pregnancy or after birth should be treated immediately with one of:
    - labetalol
    - oral nifedipine
    - intravenous hydralazine

Review conclusion:

NG133 is the principal pregnancy-specific NICE source in the frozen corpus and provides the directly applicable evidence for the ontology assertions concerning ACE inhibitors, ARBs, pregnancy, and the ordered chronic-hypertension treatment options.

## NICE NG136

Document ID:

`DOC-21a51be8081f`

Title:

Hypertension in adults: diagnosis and management

Status:

APPROVED

Frozen chunk count:

98

Manually confirmed evidence:

- `DOC-21a51be8081f-CHUNK-0054`
  - General adult hypertension guidance explicitly redirects women considering pregnancy, pregnant women, and breastfeeding women to the NICE hypertension-in-pregnancy guideline.

- `DOC-21a51be8081f-CHUNK-0057`
  - ACE inhibitors or ARBs are options for some adults beginning step 1 antihypertensive treatment.
  - The same recommendation explicitly carries MHRA safety advice that ACE inhibitors and angiotensin II receptor antagonists are not for use in pregnancy.

Review conclusion:

NG136 provides general adult hypertension evidence, but its own text directs pregnancy-specific management to the pregnancy guideline and preserves the pregnancy restriction on ACE inhibitor and ARB therapy.

## WHO Severe Hypertension in Pregnancy

Document ID:

`DOC-dad82839ba53`

Title:

WHO recommendation on drug treatment for severe hypertension in pregnancy

Status:

APPROVED

Frozen chunk count:

720

Manually confirmed evidence:

- `DOC-dad82839ba53-CHUNK-0006`
  - Choice and route of antihypertensive treatment for severe hypertension during pregnancy should consider clinician experience, cost, and local availability.
  - Hydralazine, alpha methyldopa, beta blockers including labetalol, and nifedipine are described as reasonable choices.
  - ACE inhibitors, angiotensin receptor blockers, and sodium nitroprusside should be avoided because of safety concerns.
  - The recommendation is conditional and based on very low-certainty evidence.

Review conclusion:

This source supports treatment of severe hypertension during pregnancy while preserving uncertainty and context dependence rather than identifying one universally preferred antihypertensive drug.

## WHO Non-Severe Hypertension in Pregnancy

Document ID:

`DOC-d21df49f3ed9`

Title:

WHO recommendations on drug treatment for non-severe hypertension in pregnancy

Status:

APPROVED

Frozen chunk count:

557

Manually confirmed evidence:

- `DOC-d21df49f3ed9-CHUNK-0018`
  - Oral alpha-agonist methyldopa and beta-blockers are described as effective treatment options for non-severe hypertension during pregnancy.
  - Antihypertensive treatment probably reduces development of severe hypertension compared with placebo or no treatment.
  - The source states that evidence is insufficient to conclude that one antihypertensive drug option is superior to the others.
  - The recommendation is context specific.

Review conclusion:

This source supports pharmacological treatment of non-severe hypertension during pregnancy while explicitly preserving uncertainty about superiority between drug options.

## Contamination audit

Clinical Qdrant collection:

`medontoagent_clinical_documents`

Total frozen points:

1535

Frozen document distribution:

- `DOC-e943ba2bdc31`: 160
- `DOC-21a51be8081f`: 98
- `DOC-dad82839ba53`: 720
- `DOC-d21df49f3ed9`: 557

No additional document IDs were found in the clinical collection.

Synthetic integration documents are stored separately from the clinical collection.

## Ontology

Ontology file:

`ontology/medical_ontology.ttl`

Frozen ontology assertion counts:

- Total OntologyAssertion resources: 6
- ACTIVE assertions: 5

ACTIVE clinical assertions use assertion-level provenance.

The synthetic ontology assertion remains diagnostically available but is not eligible to control retrieval or generation.

## Corpus configuration

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimensions: 384
- Maximum chunk words: 600
- Chunk overlap words: 80
- Retrieval top-k: 5
- Qdrant collection: `medontoagent_clinical_documents`
- LLM routing configuration: `openrouter/free`
- Maximum agent retries: 1

## Review result

The reviewed corpus is accepted as:

`clinical-corpus-v1`

The corpus consists of exactly four approved, non-superseded authoritative documents with fixed checksums, fixed indexed chunks, recorded ontology state, and recorded runtime configuration.

It is suitable as the fixed evidence base for construction of the held-back final golden evaluation dataset.
