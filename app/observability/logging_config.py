import logging


SENSITIVE_MARKERS = (
    "authorization:",
    "bearer ",
    "x-medonto-admin-key",
    "openrouter_api_key",
    "gemini_api_key",
    "qdrant_api_key",
    "admin_api_key",
)


class SensitiveDataFilter(
    logging.Filter
):
    """
    Defensive filter for accidental logging of obvious
    credential-related markers.

    Application code should still avoid logging sensitive
    values in the first place.
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        message = (
            record.getMessage()
            .casefold()
        )

        return not any(
            marker in message
            for marker
            in SENSITIVE_MARKERS
        )


def configure_logging(
    *,
    level: str = "INFO",
) -> None:
    """
    Configure application logging without logging request
    bodies, retrieved clinical text, credentials, or provider
    error payloads.
    """
    root_logger = logging.getLogger()

    root_logger.setLevel(
        level
    )

    if not root_logger.handlers:
        handler = logging.StreamHandler()

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s "
                "%(name)s %(message)s"
            )
        )

        handler.addFilter(
            SensitiveDataFilter()
        )

        root_logger.addHandler(
            handler
        )
    else:
        for handler in root_logger.handlers:
            handler.addFilter(
                SensitiveDataFilter()
            )
