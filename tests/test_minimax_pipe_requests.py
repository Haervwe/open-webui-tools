"""Exercise the real pipe and HTTP serializer against an offline SSE server."""

import copy
import inspect
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


DEFAULT_SSE = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'

FLASH_SSE = "".join(
    f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n"
    for d in (
        {"role": "assistant", "reasoning_content": "Let me "},
        {"reasoning_content": "think."},
        {"content": "Hel"},
        {"content": "lo"},
    )
) + "data: [DONE]\n\n"

M3_THINK_SSE = "".join(
    f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n"
    for c in ("<think>hidden", " plan</think>", "Hello")
) + "data: [DONE]\n\n"


class Transport(list):
    """Captured upstream requests plus the SSE body the server replies with."""

    sse = DEFAULT_SSE
    status = 200


@pytest_asyncio.fixture
async def transport(monkeypatch):
    requests = Transport()

    async def complete(request):
        requests.append({
            "path": request.path,
            "authorization": request.headers.get("Authorization"),
            "payload": json.loads(await request.read()),
        })
        if requests.status != 200:
            return web.Response(status=requests.status, text="bad request")
        return web.Response(
            text=requests.sse,
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


async def collect(result):
    """Drain a streaming pipe result into a list of yielded items."""
    assert inspect.isasyncgen(result)
    return [item async for item in result]


def assert_no_message_events(emitter):
    """Content must be pipe output, never chat:message* events (0.12.5 wipes them)."""
    for call in emitter.call_args_list:
        assert not call.args[0]["type"].startswith("chat:message")
        assert call.args[0]["type"] != "chat:completion"


async def send(pipe, body, requests):
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    emitter = AsyncMock()
    items = await collect(
        await pipe.pipe({**body, "stream": True}, __event_emitter__=emitter)
    )
    assert "".join(i for i in items if isinstance(i, str)) == "Hello"
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["authorization"] == "Bearer offline-test-key"
    assert_no_message_events(emitter)
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
async def test_model_outside_enabled_models_is_rejected(transport, model):
    """ENABLED_MODELS stays an allowlist: non-enabled, non-catalog IDs never go upstream."""
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    result = await pipe.pipe({
        "model": "minimax_pipe.minimax-" + model, "messages": [], "stream": True,
    })
    assert result == f"Unknown MiniMax model: minimax_pipe.minimax-{model}"
    assert transport == []


@pytest.mark.asyncio
async def test_flash_reasoning_content_yielded_as_reasoning_chunks(transport):
    transport.sse = FLASH_SSE
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    pipe.valves.STRIP_THINKING = False
    emitter = AsyncMock()
    items = await collect(await pipe.pipe({
        "model": "minimax_pipe.minimax-MiniMax-M3.1-Flash",
        "messages": [{"role": "user", "content": "Hi"}], "stream": True,
    }, __event_emitter__=emitter))
    assert items == [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Let me "}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "think."}}]},
        "Hel",
        "lo",
    ]
    assert_no_message_events(emitter)
    assert emitter.call_args_list[-1].args[0]["data"] == {
        "type": "info", "description": "Done", "done": True,
    }


@pytest.mark.asyncio
async def test_flash_reasoning_content_dropped_when_stripping(transport):
    transport.sse = FLASH_SSE
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    assert pipe.valves.STRIP_THINKING is True
    items = await collect(await pipe.pipe({
        "model": "MiniMax-M3.1-Flash", "messages": [], "stream": True,
    }))
    assert items == ["Hel", "lo"]


@pytest.mark.asyncio
@pytest.mark.parametrize("strip,expected", [
    (True, "Hello"), (False, "<think>hidden plan</think>Hello"),
])
async def test_think_tags_in_stream(transport, strip, expected):
    transport.sse = M3_THINK_SSE
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    pipe.valves.STRIP_THINKING = strip
    items = await collect(await pipe.pipe({
        "model": "minimax-MiniMax-M3", "messages": [], "stream": True,
    }))
    assert "".join(items) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("sse", [DEFAULT_SSE, FLASH_SSE, M3_THINK_SSE])
@pytest.mark.parametrize("body_extra", [{}, {"stream": False}])
async def test_non_stream_returns_content_string(transport, sse, body_extra):
    transport.sse = sse
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    pipe.valves.STRIP_THINKING = False
    emitter = AsyncMock()
    result = await pipe.pipe({
        "model": "minimax-MiniMax-M3.1-Flash", "messages": [], **body_extra,
    }, __event_emitter__=emitter)
    # Reasoning never leaks into the plain-string (task/non-stream) answer.
    expected = "<think>hidden plan</think>Hello" if sse is M3_THINK_SSE else "Hello"
    assert result == expected
    assert transport[0]["payload"]["stream"] is True
    assert_no_message_events(emitter)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_upstream_error_is_returned_as_output(transport, stream):
    transport.status = 400
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "offline-test-key"
    emitter = AsyncMock()
    result = await pipe.pipe({
        "model": "MiniMax-M3", "messages": [], "stream": stream,
    }, __event_emitter__=emitter)
    if stream:
        result = "".join(await collect(result))
    assert result == "MiniMax API error: HTTP 400: bad request"
    assert "offline-test-key" not in result
    last = emitter.call_args_list[-1].args[0]
    assert last["type"] == "status" and last["data"]["type"] == "error"


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
