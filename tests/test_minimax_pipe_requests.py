"""Exercise the real pipe and HTTP serializer against an offline SSE server."""

import copy
import json
import os
import sys
from unittest.mock import AsyncMock

from aiohttp import web
import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "functions"))

import minimax_pipe
from minimax_pipe import Pipe


@pytest_asyncio.fixture
async def transport(monkeypatch):
    requests = []

    async def complete(request):
        requests.append({
            "path": request.path,
            "authorization": request.headers.get("Authorization"),
            "payload": json.loads(await request.read()),
        })
        return web.Response(
            text='data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n',
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    monkeypatch.setattr(minimax_pipe, "MINIMAX_API_BASE", f"http://127.0.0.1:{port}/v1")
    try:
        yield requests
    finally:
        await runner.cleanup()


async def send(pipe, body, requests):
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    emitter = AsyncMock()
    assert await pipe.pipe(body, __event_emitter__=emitter) is None
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["authorization"] == "Bearer offline-test-key"
    assert any(call.args[0] == {
        "type": "chat:message:delta",
        "data": {"role": "assistant", "content": "Hello"},
    } for call in emitter.call_args_list)
    return requests[0]["payload"]


@pytest.mark.asyncio
async def test_default_picker_to_serialized_request(transport):
    pipe = Pipe()
    assert [p["id"] for p in pipe.pipes()] == [
        "minimax-MiniMax-M3.1-Flash", "minimax-MiniMax-M3",
    ]
    payload = await send(pipe, {
        "model": "minimax_pipe." + pipe.pipes()[0]["id"],
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0,
        "max_tokens": 4096,
        "top_p": 0.9,
        "service_tier": "priority",
        "reasoning_split": False,
    }, transport)
    assert payload == {
        "model": "MiniMax-M3.1-Flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.01,
        "stream": True,
        "max_tokens": 4096,
        "top_p": 0.9,
        "thinking": {"type": "adaptive"},
        "reasoning_effort": "max",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("thinking", ["disabled", {"type": "disabled"}])
async def test_flash_effort_and_adaptive_only(transport, effort, thinking):
    pipe = Pipe()
    pipe.valves.THINKING_MODE = "disabled"
    payload = await send(pipe, {
        "model": "MiniMax-M3.1-Flash",
        "messages": [], "thinking": thinking, "reasoning_effort": effort,
    }, transport)
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["reasoning_effort"] == effort


@pytest.mark.asyncio
@pytest.mark.parametrize("body_effort,valve_effort,expected", [
    ("none", "high", "high"), (None, "max", "max"),
    ("invalid", "none", "max"), ([], "medium", "medium"),
])
async def test_invalid_effort_falls_back(transport, body_effort, valve_effort, expected):
    pipe = Pipe()
    pipe.valves.REASONING_EFFORT = valve_effort
    payload = await send(pipe, {
        "model": "MiniMax-M3.1-Flash", "messages": [],
        "reasoning_effort": body_effort,
    }, transport)
    assert payload["reasoning_effort"] == expected


@pytest.mark.asyncio
async def test_flash_uses_saved_effort(transport):
    pipe = Pipe()
    pipe.valves.REASONING_EFFORT = "low"
    payload = await send(pipe, {"model": "MiniMax-M3.1-Flash", "messages": []}, transport)
    assert payload["reasoning_effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["adaptive", "disabled", {"type": "disabled"}])
async def test_m3_thinking_object_without_flash_effort(transport, thinking):
    payload = await send(Pipe(), {
        "model": "minimax-MiniMax-M3", "messages": [],
        "thinking": thinking, "reasoning_effort": "high",
    }, transport)
    assert payload["thinking"] == {
        "type": thinking if isinstance(thinking, str) else thinking["type"],
    }
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    "MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
    "MiniMax-M3-custom", "account-custom-model",
])
async def test_saved_models_are_never_rewritten(transport, model):
    pipe = Pipe()
    pipe.valves.ENABLED_MODELS = [model]
    saved = list(pipe.valves.ENABLED_MODELS)
    assert [entry["id"] for entry in pipe.pipes()] == ["minimax-" + model]
    payload = await send(pipe, {
        "model": "minimax_pipe.minimax-" + model, "messages": [],
    }, transport)
    assert payload["model"] == model
    assert pipe.valves.ENABLED_MODELS == saved
    assert "reasoning_effort" not in payload
    if model != "MiniMax-M3":
        assert "thinking" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3-custom"])
async def test_saved_chat_id_outside_current_picker_is_preserved(transport, model):
    payload = await send(Pipe(), {
        "model": "minimax_pipe.minimax-" + model, "messages": [],
    }, transport)
    assert payload["model"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
async def test_multimodal_request_preserved(transport, model):
    parts = [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": {"url": "https://example.com/i.png", "detail": "high"}},
        {"type": "video_url", "video_url": {"url": "mm_file://video", "fps": 2}},
    ]
    messages = [
        {"role": "user", "content": parts},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "More", "files": [
            {"type": "image", "url": "data:image/png;base64,AAAA"},
            {"type": "video", "url": "data:video/mp4;base64,AAAA"},
        ]},
    ]
    original = copy.deepcopy(messages)
    payload = await send(Pipe(), {"model": model, "messages": messages}, transport)
    assert payload["messages"][0]["content"] == parts
    assert payload["messages"][1] == messages[1]
    assert payload["messages"][2]["content"] == [
        {"type": "text", "text": "More"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
    ]
    assert messages == original
