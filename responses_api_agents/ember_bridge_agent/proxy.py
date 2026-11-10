# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An episode-scoped loopback endpoint forwarding to Gym's configured model server."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from aiohttp import ClientTimeout, web

from nemo_gym.client import request


@dataclass
class ResponseObserver:
    """Parse one request's stream independently of other concurrent model calls."""

    last_response: dict[str, object] | None = None
    _event_buffer: bytes = field(default=b"", repr=False)

    def observe(self, chunk: bytes, *, streaming: bool) -> None:
        """Retain an actual response envelope without synthesizing assistant messages."""
        if not streaming:
            try:
                value = json.loads(chunk)
            except (ValueError, UnicodeError):
                return
            if isinstance(value, dict) and value.get("object") == "response":
                self.last_response = value
            return
        self._event_buffer += chunk
        while b"\n" in self._event_buffer:
            line, self._event_buffer = self._event_buffer.split(b"\n", 1)
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except (ValueError, UnicodeError):
                continue
            if (
                isinstance(event, dict)
                and event.get("type") in {"response.completed", "response.incomplete", "response.failed"}
                and isinstance(event.get("response"), dict)
            ):
                self.last_response = event["response"]
        if len(self._event_buffer) > 8 * 1024 * 1024:
            # Observation must not interrupt a response already accepted upstream.
            self._event_buffer = b""


@dataclass
class ModelExchange:
    """One accepted proxy request; payloads are retained only when capture is enabled."""

    request: dict[str, object] | None
    started_at: float = field(default_factory=time.time)
    response: dict[str, object] | None = None
    status: int | None = None


@dataclass
class ModelProxy:
    """Keep actor authorization local; provider credentials stay in Gym's model server."""

    upstream: str
    cookies: dict[str, str]
    timeout_seconds: float
    capability: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    last_response: dict[str, object] | None = None
    base_url: str = ""
    capture: bool = False
    exchanges: list[ModelExchange] = field(default_factory=list, repr=False)

    async def forward(self, incoming: web.Request) -> web.StreamResponse:
        """Forward one Responses request once, preserving its body, status and SSE."""
        authorization = incoming.headers.get("Authorization", "")
        expected = f"Bearer {self.capability}"
        if not self.capability or not secrets.compare_digest(authorization.encode(), expected.encode()):
            raise web.HTTPForbidden(text="A live episode capability is required")
        payload = await incoming.read()
        exchange = None
        if self.capture:
            try:
                value = json.loads(payload)
            except (ValueError, UnicodeError):
                value = None
            exchange = ModelExchange(value if isinstance(value, dict) else None)
            self.exchanges.append(exchange)
        upstream = await request(
            "POST",
            self.upstream.rstrip("/") + "/responses",
            data=payload,
            headers={"Content-Type": "application/json"},
            cookies=self.cookies,
            timeout=ClientTimeout(total=self.timeout_seconds),
            _max_num_tries=1,
            _resend_after_disconnect=False,
        )
        try:
            observer = ResponseObserver()
            if exchange is not None:
                exchange.status = upstream.status
            content_type = upstream.headers.get("Content-Type", "application/json")
            outgoing = web.StreamResponse(status=upstream.status, headers={"Content-Type": content_type})
            await outgoing.prepare(incoming)
            streaming = "text/event-stream" in content_type
            if streaming:
                async for chunk in upstream.content.iter_any():
                    observer.observe(chunk, streaming=True)
                    if observer.last_response is not None:
                        self.last_response = observer.last_response
                    await outgoing.write(chunk)
                observer.observe(b"\n", streaming=True)
                if observer.last_response is not None:
                    self.last_response = observer.last_response
            else:
                body = await upstream.read()
                observer.observe(body, streaming=False)
                if observer.last_response is not None:
                    self.last_response = observer.last_response
                await outgoing.write(body)
            await outgoing.write_eof()
            return outgoing
        finally:
            if exchange is not None:
                exchange.response = observer.last_response
            upstream.release()


@asynccontextmanager
async def model_proxy(
    upstream: str, cookies: dict[str, str], timeout_seconds: float, *, capture: bool = False
) -> AsyncIterator[ModelProxy]:
    """Retire the capability and listener when the owning episode ends."""
    proxy = ModelProxy(upstream, cookies, timeout_seconds, capture=capture)
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_post("/v1/responses", proxy.forward)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=10)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port, *_ = runner.addresses[0]
        proxy.base_url = f"http://{host}:{port}/v1"
        yield proxy
    finally:
        proxy.capability = ""
        await runner.cleanup()
