import logging

from app.observability.logging_config import (
    SensitiveDataFilter,
)


def test_normal_log_message_is_allowed() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Request completed successfully",
        args=(),
        exc_info=None,
    )

    filter_ = SensitiveDataFilter()

    assert filter_.filter(
        record
    ) is True


def test_sensitive_marker_is_blocked() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Authorization: Bearer secret-value",
        args=(),
        exc_info=None,
    )

    filter_ = SensitiveDataFilter()

    assert filter_.filter(
        record
    ) is False
