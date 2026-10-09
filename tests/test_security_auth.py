from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.config.settings import (
    Settings,
)
from app.security.auth import (
    require_admin,
)
from app.security.secrets import (
    SecurityConfigurationError,
    require_admin_secret,
)


def test_missing_admin_secret_fails_closed() -> None:
    settings = Settings(
        _env_file=None,
        admin_api_key=None,
    )

    with pytest.raises(
        SecurityConfigurationError,
    ):
        require_admin_secret(
            settings
        )


def test_short_admin_secret_is_rejected() -> None:
    settings = Settings(
        _env_file=None,
        admin_api_key="too-short",
    )

    with pytest.raises(
        SecurityConfigurationError,
    ):
        require_admin_secret(
            settings
        )


def test_valid_admin_secret_is_returned() -> None:
    settings = Settings(
        _env_file=None,
        admin_api_key=(
            "this-is-a-long-test-admin-key"
        ),
    )

    result = require_admin_secret(
        settings
    )

    assert (
        result
        == "this-is-a-long-test-admin-key"
    )


@patch(
    "app.security.auth."
    "require_admin_secret"
)
def test_correct_admin_key_passes(
    mock_secret,
) -> None:
    mock_secret.return_value = (
        "this-is-a-long-test-admin-key"
    )

    result = require_admin(
        "this-is-a-long-test-admin-key"
    )

    assert result is None


@patch(
    "app.security.auth."
    "require_admin_secret"
)
def test_wrong_admin_key_fails(
    mock_secret,
) -> None:
    mock_secret.return_value = (
        "this-is-a-long-test-admin-key"
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        require_admin(
            "this-is-the-wrong-admin-key"
        )

    assert (
        exc_info.value.status_code
        == 401
    )


@patch(
    "app.security.auth."
    "require_admin_secret"
)
def test_missing_admin_header_fails(
    mock_secret,
) -> None:
    mock_secret.return_value = (
        "this-is-a-long-test-admin-key"
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        require_admin(
            None
        )

    assert (
        exc_info.value.status_code
        == 401
    )


@patch(
    "app.security.auth."
    "require_admin_secret"
)
def test_missing_admin_configuration_returns_503(
    mock_secret,
) -> None:
    mock_secret.side_effect = (
        SecurityConfigurationError(
            "missing"
        )
    )

    with pytest.raises(
        HTTPException,
    ) as exc_info:
        require_admin(
            "some-key"
        )

    assert (
        exc_info.value.status_code
        == 503
    )
