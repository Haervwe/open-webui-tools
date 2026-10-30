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


async def _collect(result):
    """Drain the pipe output: content text plus any error chunks."""
    content, errors = "", []
    if isinstance(result, str):
        return result, errors
    async for chunk in result:
        if isinstance(chunk, str):
            content += chunk
        elif isinstance(chunk, dict) and "error" in chunk:
            errors.append(chunk)
    return content, errors


@pytest.mark.asyncio
async def test_streaming_completion(pipe):
    """Test a real streaming chat completion with MiniMax M3."""
    emitter = AsyncMock()
    result = await pipe.pipe(
        body={
            "model": "minimax-MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            "temperature": 0.5,
            "stream": True,
        },
        __user__={"id": "integration-test"},
        __event_emitter__=emitter,
    )

    # Content is the pipe output (Open WebUI >= 0.12 persists only that)
    content, errors = await _collect(result)
    assert not errors, f"Got errors: {errors}"
    assert content.strip(), "Expected streamed content"


@pytest.mark.asyncio
async def test_think_tag_stripping(pipe):
    """Test that think tags are stripped from streaming output."""
    emitter = AsyncMock()
    pipe.valves.STRIP_THINKING = True
    result = await pipe.pipe(
        body={
            "model": "minimax-MiniMax-M3",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: Hello World",
                }
            ],
            "temperature": 0.3,
            "stream": True,
        },
        __user__={"id": "integration-test"},
        __event_emitter__=emitter,
    )
    content, _ = await _collect(result)

    # Content should not contain <think> tags
    assert "<think>" not in content


@pytest.mark.asyncio
async def test_temperature_zero_handled(pipe):
    """Test that temperature=0 doesn't cause API error."""
    emitter = AsyncMock()
    result = await pipe.pipe(
        body={
            "model": "minimax-MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Say 'ok' and nothing else."}
            ],
            "temperature": 0.0,
        },
        __user__={"id": "integration-test"},
        __event_emitter__=emitter,
    )
    # Non-stream requests return the final content string
    assert isinstance(result, str) and result.strip()

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
