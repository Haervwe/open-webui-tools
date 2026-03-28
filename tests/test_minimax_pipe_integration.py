"""
Integration tests for the MiniMax LLM Pipe.

These tests call the real MiniMax API and require MINIMAX_API_KEY to be set.
Skip automatically when the key is not available.
"""

import asyncio
import json
import os
import sys
import unittest

import pytest

# Mocks for open_webui
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("open_webui", MagicMock())
sys.modules.setdefault("open_webui.models", MagicMock())
sys.modules.setdefault("open_webui.models.users", MagicMock())
sys.modules.setdefault("open_webui.routers", MagicMock())
sys.modules.setdefault("open_webui.routers.files", MagicMock())
sys.modules.setdefault("fastapi", MagicMock())
sys.modules.setdefault("starlette", MagicMock())
sys.modules.setdefault("starlette.datastructures", MagicMock())

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "functions")
)

from minimax_pipe import Pipe

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set — skipping integration tests",
)


@pytest.fixture
def pipe():
    p = Pipe()
    p.valves.MINIMAX_API_KEY = MINIMAX_API_KEY
    return p


@pytest.mark.asyncio
async def test_streaming_completion(pipe):
    """Test a real streaming chat completion with MiniMax M2.7-highspeed."""
    emitter = AsyncMock()
    result = await pipe.pipe(
        body={
            "model": "minimax-MiniMax-M2.7-highspeed",
            "messages": [
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            "temperature": 0.5,
        },
        __user__={"id": "integration-test"},
        __event_emitter__=emitter,
    )

    # result is None because content is streamed via events
    assert result is None

    # Check that deltas were emitted
    delta_calls = [
        c
        for c in emitter.call_args_list
        if c[0][0].get("type") == "chat:message:delta"
    ]
    assert len(delta_calls) > 0, "Expected at least one streamed delta"

    # Check completion event was emitted
    completion_calls = [
        c
        for c in emitter.call_args_list
        if c[0][0].get("type") == "chat:completion"
    ]
    assert len(completion_calls) == 1


@pytest.mark.asyncio
async def test_think_tag_stripping(pipe):
    """Test that think tags are stripped from streaming output."""
    emitter = AsyncMock()
    pipe.valves.STRIP_THINKING = True
    result = await pipe.pipe(
        body={
            "model": "minimax-MiniMax-M2.7-highspeed",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: Hello World",
                }
            ],
            "temperature": 0.3,
        },
        __user__={"id": "integration-test"},
        __event_emitter__=emitter,
    )
    assert result is None

    # Collect all streamed content
    content = ""
    for call in emitter.call_args_list:
        event = call[0][0]
        if event.get("type") == "chat:message:delta":
            content += event["data"].get("content", "")

    # Content should not contain <think> tags
    assert "<think>" not in content


@pytest.mark.asyncio
async def test_temperature_zero_handled(pipe):
    """Test that temperature=0 doesn't cause API error."""
    emitter = AsyncMock()
    result = await pipe.pipe(
        body={
            "model": "minimax-MiniMax-M2.7-highspeed",
            "messages": [
                {"role": "user", "content": "Say 'ok' and nothing else."}
            ],
            "temperature": 0.0,
        },
        __user__={"id": "integration-test"},
        __event_emitter__=emitter,
    )
    assert result is None

    # No error events should have been emitted
    error_calls = [
        c
        for c in emitter.call_args_list
        if c[0][0].get("type") == "status"
        and c[0][0].get("data", {}).get("type") == "error"
    ]
    assert len(error_calls) == 0, f"Got errors: {error_calls}"


if __name__ == "__main__":
    unittest.main()
