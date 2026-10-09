import hmac

from fastapi import (
    Header,
    HTTPException,
    status,
)

from app.security.secrets import (
    SecurityConfigurationError,
    require_admin_secret,
)


ADMIN_HEADER_NAME = (
    "X-MedOnto-Admin-Key"
)


def require_admin(
    x_medonto_admin_key: str | None = Header(
        default=None,
        alias=ADMIN_HEADER_NAME,
    ),
) -> None:
    """
    FastAPI dependency for future administrative operations.

    Authentication fails closed when:
    - no administrative secret is configured,
    - no key is supplied,
    - the supplied key is incorrect.
    """
    try:
        expected = (
            require_admin_secret()
        )

    except SecurityConfigurationError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Administrative authentication "
                "is unavailable."
            ),
        ) from exc

    if x_medonto_admin_key is None:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Administrative authentication "
                "is required."
            ),
        )

    if not hmac.compare_digest(
        x_medonto_admin_key,
        expected,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Administrative authentication "
                "failed."
            ),
        )
