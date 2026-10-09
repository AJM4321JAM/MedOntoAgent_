from app.config.settings import (
    Settings,
    get_settings,
)


class SecurityConfigurationError(
    RuntimeError
):
    """
    Raised when required security configuration
    is missing or unsafe.
    """


def require_admin_secret(
    settings: Settings | None = None,
) -> str:
    """
    Return the configured administrative API secret.

    The value must never be logged or returned
    through a public API response.
    """
    settings = (
        settings
        or get_settings()
    )

    if settings.admin_api_key is None:
        raise SecurityConfigurationError(
            "Administrative authentication is not configured."
        )

    value = (
        settings
        .admin_api_key
        .get_secret_value()
        .strip()
    )

    if len(value) < 24:
        raise SecurityConfigurationError(
            "Administrative authentication is not safely configured."
        )

    return value
