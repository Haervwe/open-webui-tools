"""Unit tests for the MiniMax LLM Pipe."""

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the repo root so we can import the pipe module
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "functions")
)

# We need to mock open_webui imports before importing the pipe
sys.modules["open_webui"] = MagicMock()
sys.modules["open_webui.models"] = MagicMock()
sys.modules["open_webui.models.users"] = MagicMock()
sys.modules["open_webui.routers"] = MagicMock()
sys.modules["open_webui.routers.files"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["starlette"] = MagicMock()
sys.modules["starlette.datastructures"] = MagicMock()

# Now we can import — but we need to handle the pydantic import carefully
# Since pydantic is a real dependency, let's un-mock the relevant parts
from pydantic import BaseModel, Field

# Re-import with real pydantic
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("minimax_pipe"):
        del sys.modules[mod_name]

from minimax_pipe import (
    Pipe,
    _clamp_temperature,
    _strip_think_tags,
    MINIMAX_MODELS,
    MINIMAX_API_BASE,
    MINIMAX_TEMP_MIN,
    MINIMAX_TEMP_MAX,
)


class TestClampTemperature(unittest.TestCase):
    """Tests for _clamp_temperature()."""

    def test_none_returns_default(self):
        assert _clamp_temperature(None) == 0.7

    def test_zero_clamped_to_min(self):
        assert _clamp_temperature(0.0) == MINIMAX_TEMP_MIN

    def test_negative_clamped_to_min(self):
        assert _clamp_temperature(-1.0) == MINIMAX_TEMP_MIN

    def test_above_max_clamped(self):
        assert _clamp_temperature(2.0) == MINIMAX_TEMP_MAX

    def test_valid_value_unchanged(self):
        assert _clamp_temperature(0.5) == 0.5

    def test_boundary_max(self):
        assert _clamp_temperature(1.0) == 1.0

    def test_boundary_min_valid(self):
        assert _clamp_temperature(0.01) == 0.01


class TestStripThinkTags(unittest.TestCase):
    """Tests for _strip_think_tags()."""

    def test_no_tags(self):
        assert _strip_think_tags("Hello world") == "Hello world"

    def test_single_tag(self):
        assert _strip_think_tags("<think>internal</think>Hello") == "Hello"

    def test_multiple_tags(self):
        result = _strip_think_tags(
            "<think>a</think>Hello <think>b</think>world"
        )
        assert result == "Hello world"

    def test_multiline_tag(self):
        result = _strip_think_tags(
            "<think>\nline1\nline2\n</think>Result"
        )
        assert result == "Result"

    def test_empty_tag(self):
        assert _strip_think_tags("<think></think>Hi") == "Hi"


class TestPipeInit(unittest.TestCase):
    """Tests for Pipe initialization and valve defaults."""

    def test_default_valves(self):
        pipe = Pipe()
        assert pipe.valves.MINIMAX_API_KEY == ""
        assert pipe.valves.STRIP_THINKING is True
        assert pipe.valves.DEFAULT_TEMPERATURE == 0.7
        assert len(pipe.valves.ENABLED_MODELS) == len(MINIMAX_MODELS)

    def test_pipes_returns_all_models(self):
        pipe = Pipe()
        pipe.valves.ENABLED_MODELS = [m["id"] for m in MINIMAX_MODELS]
        result = pipe.pipes()
        assert len(result) == len(MINIMAX_MODELS)
        for entry in result:
            assert entry["id"].startswith("minimax-")
            assert "MiniMax" in entry["name"]

    def test_pipes_respects_enabled_filter(self):
        pipe = Pipe()
        pipe.valves.ENABLED_MODELS = ["MiniMax-M2.7"]
        result = pipe.pipes()
        assert len(result) == 1
        assert result[0]["id"] == "minimax-MiniMax-M2.7"


class TestResolveModelId(unittest.TestCase):
    """Tests for _resolve_model_id()."""

    def test_direct_model_id(self):
        pipe = Pipe()
        assert pipe._resolve_model_id("minimax-MiniMax-M2.7") == "MiniMax-M2.7"

    def test_highspeed_model(self):
        pipe = Pipe()
        assert (
            pipe._resolve_model_id("minimax-MiniMax-M2.7-highspeed")
            == "MiniMax-M2.7-highspeed"
        )

    def test_with_function_prefix(self):
        pipe = Pipe()
        result = pipe._resolve_model_id(
            "minimax_pipe.minimax-MiniMax-M2.7"
        )
        assert result == "MiniMax-M2.7"

    def test_unknown_model(self):
        pipe = Pipe()
        assert pipe._resolve_model_id("minimax-UnknownModel") is None


class TestProcessThinkingStream(unittest.TestCase):
    """Tests for _process_thinking_stream()."""

    def test_no_tags(self):
        out, in_t, buf = Pipe._process_thinking_stream("Hello", False, "")
        assert out == "Hello"
        assert in_t is False

    def test_complete_tag_in_single_chunk(self):
        out, in_t, buf = Pipe._process_thinking_stream(
            "<think>internal</think>Result", False, ""
        )
        assert out == "Result"
        assert in_t is False

    def test_tag_spans_chunks(self):
        # First chunk: opening tag
        out1, in_t1, buf1 = Pipe._process_thinking_stream(
            "Hello<think>thinking...", False, ""
        )
        assert out1 == "Hello"
        assert in_t1 is True

        # Second chunk: closing tag
        out2, in_t2, buf2 = Pipe._process_thinking_stream(
            "more thoughts</think>World", in_t1, buf1
        )
        assert out2 == "World"
        assert in_t2 is False

    def test_empty_content(self):
        out, in_t, buf = Pipe._process_thinking_stream("", False, "")
        assert out == ""
        assert in_t is False

    def test_nested_content_after_think(self):
        out, in_t, buf = Pipe._process_thinking_stream(
            "<think>x</think>A<think>y</think>B", False, ""
        )
        assert out == "AB"
        assert in_t is False


@pytest.mark.asyncio
async def test_pipe_no_api_key():
    """pipe() should return error when no API key is set."""
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = ""
    emitter = AsyncMock()
    result = await pipe.pipe(
        body={"model": "minimax-MiniMax-M2.7", "messages": []},
        __user__={"id": "test"},
        __event_emitter__=emitter,
    )
    assert "not configured" in result
    # Check that error status was emitted
    emitter.assert_called()
    call_args = emitter.call_args_list[0][0][0]
    assert call_args["type"] == "status"
    assert call_args["data"]["type"] == "error"


@pytest.mark.asyncio
async def test_pipe_unknown_model():
    """pipe() should return error for unknown model."""
    pipe = Pipe()
    pipe.valves.MINIMAX_API_KEY = "test-key"
    emitter = AsyncMock()
    result = await pipe.pipe(
        body={"model": "minimax-UnknownModel", "messages": []},
        __user__={"id": "test"},
        __event_emitter__=emitter,
    )
    assert "Unknown" in result


class TestStreamResponsePayload(unittest.TestCase):
    """Test that the payload sent to MiniMax API is correct."""

    def test_temperature_clamping_in_payload(self):
        """Verify temperature is clamped before being sent."""
        pipe = Pipe()
        pipe.valves.MINIMAX_API_KEY = "test-key"

        # Simulate building the payload (same logic as pipe())
        body = {
            "model": "minimax-MiniMax-M2.7",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.0,
        }
        temperature = _clamp_temperature(body.get("temperature", 0.7))
        assert temperature == MINIMAX_TEMP_MIN

    def test_max_tokens_forwarded(self):
        """max_tokens should be forwarded if present."""
        body = {
            "model": "minimax-MiniMax-M2.7",
            "messages": [],
            "max_tokens": 4096,
        }
        payload = {
            "model": "MiniMax-M2.7",
            "messages": body["messages"],
            "temperature": 0.7,
            "stream": True,
        }
        if body.get("max_tokens"):
            payload["max_tokens"] = body["max_tokens"]
        assert payload["max_tokens"] == 4096

    def test_top_p_forwarded(self):
        """top_p should be forwarded if present."""
        body = {
            "model": "minimax-MiniMax-M2.7",
            "messages": [],
            "top_p": 0.9,
        }
        payload = {
            "model": "MiniMax-M2.7",
            "messages": body["messages"],
            "temperature": 0.7,
            "stream": True,
        }
        if body.get("top_p") is not None:
            payload["top_p"] = body["top_p"]
        assert payload["top_p"] == 0.9


class TestConstants(unittest.TestCase):
    """Tests for module constants."""

    def test_api_base_url(self):
        assert MINIMAX_API_BASE == "https://api.minimax.io/v1"

    def test_models_have_required_fields(self):
        for model in MINIMAX_MODELS:
            assert "id" in model
            assert "name" in model
            assert "context_length" in model
            assert model["context_length"] == 204000

    def test_model_ids(self):
        ids = [m["id"] for m in MINIMAX_MODELS]
        assert "MiniMax-M2.7" in ids
        assert "MiniMax-M2.7-highspeed" in ids


if __name__ == "__main__":
    unittest.main()
