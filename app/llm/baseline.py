import json
import time

from pydantic import ValidationError

from app.config.settings import Settings, get_settings
from app.llm.client import (
    LLMServiceError,
    _call_with_one_retry,
    create_llm_client,
)
from app.llm.prompts import (
    BASELINE_RAG_SYSTEM_PROMPT,
)
from app.models.responses import (
    BaselineLLMOutput,
    BaselineQueryResponse,
    CitationReference,
)
from app.models.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.retrieval.retriever import (
    retrieve_baseline,
)


class BaselineRAGError(RuntimeError):
    """
    Raised when the baseline RAG pipeline cannot safely
    produce a response.
    """


def format_evidence(
    chunks: list[RetrievedChunk],
) -> str:
    """
    Convert retrieved chunks into a deterministic evidence
    block for the LLM.
    """
    evidence_blocks: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        block = "\n".join(
            [
                f"EVIDENCE_{index}",
                f"SOURCE_ID: {chunk.chunk_id}",
                f"DOCUMENT_ID: {chunk.document_id}",
                f"TITLE: {chunk.title}",
                (
                    "SECTION: "
                    f"{chunk.section_title or 'Not available'}"
                ),
                (
                    "PAGE: "
                    f"{chunk.page_number or 'Not available'}"
                ),
                f"PUBLISHER: {chunk.publisher}",
                f"VERSION: {chunk.version}",
                (
                    "SOURCE_AUTHORITY: "
                    f"{chunk.source_authority}"
                ),
                "TEXT:",
                chunk.text,
            ]
        )

        evidence_blocks.append(
            block
        )

    return "\n\n---\n\n".join(
        evidence_blocks
    )


def _extract_json_object(
    content: str,
) -> dict:
    """
    Parse a JSON object returned by the development LLM.

    Free routed models may wrap JSON in Markdown code fences.
    """
    cleaned = content.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

        if cleaned.lower().startswith(
            "json"
        ):
            cleaned = (
                cleaned[4:]
                .strip()
            )

    try:
        data = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:
        raise BaselineRAGError(
            (
                "Baseline LLM returned invalid JSON. "
                f"Raw response: {content!r}"
            )
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise BaselineRAGError(
            (
                "Baseline LLM response was JSON but "
                "was not a JSON object."
            )
        )

    return data


def _build_user_prompt(
    *,
    query: str,
    evidence: str,
) -> str:
    """
    Build the baseline RAG user prompt.
    """
    return f"""
QUESTION:

{query}

APPROVED RETRIEVED EVIDENCE:

{evidence}

Answer the QUESTION using only the APPROVED RETRIEVED EVIDENCE.

Return only the required JSON object.

Do not output a safety classification, moderation label,
analysis label, or commentary.

Your response must begin with {{ and end with }}.
""".strip()


def _validate_citations(
    *,
    output: BaselineLLMOutput,
    chunks: list[RetrievedChunk],
) -> None:
    """
    Ensure every model citation refers to an actually
    retrieved chunk.

    This validates citation identity only. It does not yet
    prove that the chunk entails every generated claim.
    """
    retrieved_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    cited_ids = [
        citation.chunk_id
        for citation in output.citations
    ]

    invalid_ids = [
        chunk_id
        for chunk_id in cited_ids
        if chunk_id not in retrieved_ids
    ]

    if invalid_ids:
        raise BaselineRAGError(
            (
                "Baseline LLM cited chunk IDs that were "
                "not retrieved: "
                f"{invalid_ids}"
            )
        )

    if (
        output.status == "ANSWERED"
        and not cited_ids
    ):
        raise BaselineRAGError(
            (
                "ANSWERED baseline response requires "
                "at least one retrieved citation."
            )
        )

    if (
        output.status == "ABSTAINED"
        and cited_ids
    ):
        raise BaselineRAGError(
            (
                "ABSTAINED baseline response must not "
                "contain citations."
            )
        )


def _request_baseline_completion(
    *,
    client,
    settings: Settings,
    user_prompt: str,
):
    """
    Make one baseline generation request.

    Transient HTTP failures receive at most one retry through
    the shared explicit retry helper.
    """
    return _call_with_one_retry(
        lambda: (
            client
            .chat
            .completions
            .create(
                model=settings.gemini_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            BASELINE_RAG_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=min(
                    settings.llm_max_output_tokens,
                    1000,
                ),
                response_format={
                    "type": "json_object",
                },
            )
        ),
        operation_name=(
            "baseline RAG generation"
        ),
    )


def generate_baseline_answer(
    *,
    query: str,
    retrieval: RetrievalResult,
    settings: Settings | None = None,
) -> BaselineLLMOutput:
    """
    Generate one baseline answer from retrieved evidence.

    For the free model router, at most one second generation
    attempt is allowed if the first successful HTTP response
    contains unusable output.
    """
    settings = (
        settings
        or get_settings()
    )

    normalized_query = (
        query.strip()
    )

    if not normalized_query:
        raise ValueError(
            "Baseline query cannot be empty."
        )

    if not retrieval.retrieved_chunks:
        return BaselineLLMOutput(
            answer=(
                "The approved retrieved evidence is "
                "insufficient to answer the question."
            ),
            citations=[],
            status="ABSTAINED",
        )

    evidence = format_evidence(
        retrieval.retrieved_chunks
    )

    user_prompt = _build_user_prompt(
        query=normalized_query,
        evidence=evidence,
    )

    client = create_llm_client(
        settings
    )

    total_generation_attempts = 2
    last_error: Exception | None = None

    for attempt in range(
        1,
        total_generation_attempts + 1,
    ):
        try:
            completion = _request_baseline_completion(
                client=client,
                settings=settings,
                user_prompt=user_prompt,
            )

            if not completion.choices:
                raise BaselineRAGError(
                    "Baseline LLM returned no choices."
                )

            content = (
                completion
                .choices[0]
                .message
                .content
            )

            if not content:
                raise BaselineRAGError(
                    (
                        "Baseline LLM returned an "
                        "empty response."
                    )
                )

            raw_data = _extract_json_object(
                content
            )

            try:
                output = (
                    BaselineLLMOutput
                    .model_validate(
                        raw_data
                    )
                )

            except ValidationError as exc:
                raise BaselineRAGError(
                    (
                        "Baseline LLM JSON did not match "
                        "the expected response schema. "
                        f"Raw response: {content!r}"
                    )
                ) from exc

            _validate_citations(
                output=output,
                chunks=(
                    retrieval
                    .retrieved_chunks
                ),
            )

            return output

        except LLMServiceError:
            raise

        except BaselineRAGError as exc:
            last_error = exc

            if (
                attempt
                == total_generation_attempts
            ):
                break

            time.sleep(1)

    raise BaselineRAGError(
        (
            "Baseline generation did not produce a valid "
            "grounded structured response after one bounded "
            "retry. "
            f"Last error: {last_error}"
        )
    ) from last_error


def run_baseline_rag(
    query: str,
    *,
    settings: Settings | None = None,
) -> BaselineQueryResponse:
    """
    Run the complete conventional baseline RAG pipeline.
    """
    settings = (
        settings
        or get_settings()
    )

    normalized_query = (
        query.strip()
    )

    if not normalized_query:
        raise ValueError(
            "Baseline query cannot be empty."
        )

    retrieval = retrieve_baseline(
        normalized_query,
        settings=settings,
    )

    output = generate_baseline_answer(
        query=normalized_query,
        retrieval=retrieval,
        settings=settings,
    )

    retrieved_chunk_ids = [
        chunk.chunk_id
        for chunk
        in retrieval.retrieved_chunks
    ]

    retrieved_document_ids = list(
        dict.fromkeys(
            chunk.document_id
            for chunk
            in retrieval.retrieved_chunks
        )
    )

    retrieved_contexts = [
        chunk.text
        for chunk
        in retrieval.retrieved_chunks
    ]

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk
        in retrieval.retrieved_chunks
    }

    citations = [
        CitationReference(
            chunk_id=(
                citation.chunk_id
            )
        )
        for citation
        in output.citations
    ]

    citation_document_ids = list(
        dict.fromkeys(
            chunks_by_id[citation.chunk_id]
            .document_id
            for citation
            in output.citations
            if citation.chunk_id in chunks_by_id
        )
    )

    return BaselineQueryResponse(
        query=normalized_query,
        answer=output.answer,
        citations=citations,
        citation_document_ids=(
            citation_document_ids
        ),
        retrieved_chunk_ids=(
            retrieved_chunk_ids
        ),
        retrieved_document_ids=(
            retrieved_document_ids
        ),
        retrieved_contexts=(
            retrieved_contexts
        ),
        retrieved_chunks=(
            retrieval.retrieved_chunks
        ),
        status=output.status,
    )
