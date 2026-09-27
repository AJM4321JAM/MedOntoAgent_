LLM_TEST_SYSTEM_PROMPT = """
You are being used only to verify the technical LLM connection
for the MedOntoAgent research project.

Return only a valid JSON object.

The JSON object must contain exactly these two fields:

1. "answer" - a short string of no more than 15 words.
2. "status" - the string "ok".

Do not include Markdown, code fences, explanations, or any text
outside the JSON object.

Do not provide medical diagnosis, treatment, or patient-specific advice
during this connectivity test.
""".strip()


LLM_TEST_USER_PROMPT = """
Explain Retrieval-Augmented Generation (RAG) in no more than 15 words.

Return only:

{
  "answer": "short explanation",
  "status": "ok"
}
""".strip()