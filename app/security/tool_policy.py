from enum import StrEnum


class ContentTrustLevel(
    StrEnum
):
    SYSTEM = "SYSTEM"
    APPLICATION = "APPLICATION"
    USER = "USER"
    RETRIEVED = "RETRIEVED"


class ToolPolicyError(
    RuntimeError
):
    """
    Raised when untrusted content attempts to authorize
    an operation reserved for trusted control logic.
    """


TOOL_AUTHORITY_LEVELS = {
    ContentTrustLevel.SYSTEM,
    ContentTrustLevel.APPLICATION,
}


def may_authorize_tool_action(
    trust_level: ContentTrustLevel,
) -> bool:
    """
    Only trusted system/application control context may
    authorize tool behavior.
    """
    return (
        trust_level
        in TOOL_AUTHORITY_LEVELS
    )


def require_tool_authority(
    trust_level: ContentTrustLevel,
) -> None:
    """
    Fail closed when a caller attempts to authorize
    tool behavior from untrusted content.
    """
    if not may_authorize_tool_action(
        trust_level
    ):
        raise ToolPolicyError(
            "Untrusted content cannot authorize tool actions."
        )


def contains_instruction_like_text(
    text: str,
) -> bool:
    """
    Heuristic only.

    This function does NOT determine whether content is trusted.
    Retrieved and user content remain untrusted regardless of
    whether this detector finds instruction-like text.
    """
    normalized = text.casefold()

    markers = (
        "ignore previous instructions",
        "ignore all previous instructions",
        "reveal the system prompt",
        "reveal your system prompt",
        "reveal api key",
        "reveal all api keys",
        "invoke tools",
        "use every available tool",
        "contact an external service",
        "change these rules",
    )

    return any(
        marker in normalized
        for marker in markers
    )
