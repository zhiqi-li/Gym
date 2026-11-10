# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from unittest.mock import patch


try:
    from aiohttp import ClientSession, web

    from responses_api_agents.ember_bridge_agent.proxy import ResponseObserver, model_proxy
except ModuleNotFoundError as error:
    if error.name.startswith(("nemo_gym", "responses_api_agents")):
        raise
    raise unittest.SkipTest(f"Install Gym requirements for transport tests: {error.name}") from error


class ProxyTest(unittest.IsolatedAsyncioTestCase):
    def test_split_sse_retains_the_real_response(self) -> None:
        proxy = ResponseObserver()
        response = {"id": "real-response", "object": "response", "output": []}
        event = b"data: " + json.dumps({"type": "response.completed", "response": response}).encode() + b"\r\n\r\n"
        for byte in event:
            proxy.observe(bytes([byte]), streaming=True)
        self.assertEqual(proxy.last_response, response)
        proxy.observe(b"data: [DONE]\n\n", streaming=True)
        self.assertEqual(proxy.last_response, response)

    async def test_authorization_cookie_stream_and_single_forward(self) -> None:
        received = []
        payload = b'{"input":[],"stream":true}'
        envelope = {"id": "actual", "object": "response", "output": []}
        stream = ("data: " + json.dumps({"type": "response.completed", "response": envelope}) + "\n\n").encode()

        async def provider(request: web.Request) -> web.Response:
            received.append((await request.read(), dict(request.cookies), request.headers.get("Authorization")))
            return web.Response(body=stream, content_type="text/event-stream")

        application = web.Application()
        application.router.add_post("/v1/responses", provider)
        runner = web.AppRunner(application)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port, *_ = runner.addresses[0]
        options = []
        try:
            async with ClientSession() as session:

                async def forward(method: str, url: str, **kwargs: object):
                    options.append((kwargs.pop("_max_num_tries"), kwargs.pop("_resend_after_disconnect")))
                    return await session.request(method, url, **kwargs)

                with patch("responses_api_agents.ember_bridge_agent.proxy.request", forward):
                    async with model_proxy(
                        f"http://{host}:{port}/v1", {"session": "rollout"}, 5, capture=True
                    ) as proxy:
                        async with session.post(proxy.base_url + "/responses", data=payload) as response:
                            self.assertEqual(response.status, 403)
                        self.assertEqual(received, [])
                        self.assertEqual(proxy.exchanges, [])
                        async with session.post(
                            proxy.base_url + "/responses",
                            data=payload,
                            headers={"Authorization": f"Bearer {proxy.capability}"},
                        ) as response:
                            self.assertEqual(await response.read(), stream)
                        self.assertEqual(proxy.last_response, envelope)
                        self.assertEqual(len(proxy.exchanges), 1)
                        self.assertEqual(proxy.exchanges[0].request, json.loads(payload))
                        self.assertEqual(proxy.exchanges[0].response, envelope)
                    self.assertEqual(proxy.capability, "")
            self.assertEqual(received, [(payload, {"session": "rollout"}, None)])
            self.assertEqual(options, [(1, False)])
        finally:
            await runner.cleanup()
