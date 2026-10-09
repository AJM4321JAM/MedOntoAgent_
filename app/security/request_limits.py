from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)


class RequestSizeLimitMiddleware:
    """
    Reject HTTP request bodies larger than the configured limit.

    The middleware reads the incoming body before passing the
    request downstream, so the limit is enforced even when the
    endpoint itself does not consume the request body.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        body_parts: list[bytes] = []
        total_bytes = 0

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            if message["type"] != "http.request":
                continue

            body = message.get(
                "body",
                b"",
            )

            total_bytes += len(
                body
            )

            if total_bytes > self.max_bytes:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (
                                b"content-type",
                                b"application/json",
                            ),
                        ],
                    }
                )

                await send(
                    {
                        "type": "http.response.body",
                        "body": (
                            b'{"detail":"Request body too large."}'
                        ),
                    }
                )

                return

            body_parts.append(
                body
            )

            if not message.get(
                "more_body",
                False,
            ):
                break

        complete_body = b"".join(
            body_parts
        )

        body_sent = False

        async def replay_receive() -> Message:
            nonlocal body_sent

            if not body_sent:
                body_sent = True

                return {
                    "type": "http.request",
                    "body": complete_body,
                    "more_body": False,
                }

            return {
                "type": "http.disconnect",
            }

        await self.app(
            scope,
            replay_receive,
            send,
        )
