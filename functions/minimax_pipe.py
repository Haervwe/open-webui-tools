"""
title: MiniMax LLM Pipe
author: octopus
author_url: https://github.com/octo-patch
version: 1.0.0
required_open_webui_version: 0.6.0
description: MiniMax LLM Pipe for Open WebUI — routes chat completions to MiniMax's
OpenAI-compatible API (api.minimax.io/v1). Supports MiniMax-M2.7 and
MiniMax-M2.7-highspeed models with streaming, temperature clamping, and
automatic think-tag handling.
"""

import aiohttp
import json
import logging
import re
import time
import traceback
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)

from fastapi import Request
from pydantic import BaseModel, Field

logger = logging.getLogger("minimax_pipe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINIMAX_API_BASE = "https://api.minimax.io/v1"

MINIMAX_MODELS = [
    {
        "id": "MiniMax-M2.7",
        "name": "MiniMax M2.7",
        "context_length": 204000,
    },
    {
        "id": "MiniMax-M2.7-highspeed",
        "name": "MiniMax M2.7 Highspeed",
        "context_length": 204000,
    },
]

# Temperature must be in (0.0, 1.0] for MiniMax
MINIMAX_TEMP_MIN = 0.01
MINIMAX_TEMP_MAX = 1.0

# Regex to strip <think>…</think> blocks from streamed content
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clamp_temperature(value: Optional[float]) -> float:
    """Clamp temperature to MiniMax's accepted range (0.01, 1.0]."""
    if value is None:
        return 0.7
    if value <= 0.0:
        return MINIMAX_TEMP_MIN
    if value > MINIMAX_TEMP_MAX:
        return MINIMAX_TEMP_MAX
    return value


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from assistant output."""
    return _THINK_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Pipe class
# ---------------------------------------------------------------------------


class Pipe:
    """Open WebUI Function Pipe that proxies chat completions to MiniMax."""

    class Valves(BaseModel):
        MINIMAX_API_KEY: str = Field(
            default="",
            description="MiniMax API key (get one at https://platform.minimaxi.com)",
            json_schema_extra={"input": {"type": "password"}},
        )
        ENABLED_MODELS: List[str] = Field(
            default_factory=lambda: [m["id"] for m in MINIMAX_MODELS],
            description="Which MiniMax models to expose (model IDs)",
        )
        STRIP_THINKING: bool = Field(
            default=True,
            description="Strip <think>…</think> blocks from MiniMax responses",
        )
        DEFAULT_TEMPERATURE: float = Field(
            default=0.7,
            description="Default temperature when none is specified (0.01–1.0)",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # pipes() — register models with Open WebUI
    # ------------------------------------------------------------------

    def pipes(self) -> List[Dict[str, str]]:
        """Return the list of model entries this pipe exposes."""
        enabled = set(self.valves.ENABLED_MODELS)
        return [
            {"id": f"minimax-{m['id']}", "name": f"MiniMax {m['name']}"}
            for m in MINIMAX_MODELS
            if m["id"] in enabled
        ]

    # ------------------------------------------------------------------
    # pipe() — handle a chat completion request
    # ------------------------------------------------------------------

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any] = {},
        __event_emitter__: Optional[
            Callable[[Dict[str, Any]], Awaitable[None]]
        ] = None,
        __request__: Optional[Request] = None,
    ) -> Optional[str]:
        """Route a chat completion request to the MiniMax API."""

        # --- Validate API key ---
        if not self.valves.MINIMAX_API_KEY:
            error_msg = (
                "MiniMax API key not configured. "
                "Please set MINIMAX_API_KEY in the pipe's Valves settings."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "type": "error",
                            "description": error_msg,
                            "done": True,
                        },
                    }
                )
            return error_msg

        # --- Resolve model ID ---
        pipe_id = body.get("model", "")
        model_id = self._resolve_model_id(pipe_id)
        if not model_id:
            error_msg = f"Unknown MiniMax model: {pipe_id}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "type": "error",
                            "description": error_msg,
                            "done": True,
                        },
                    }
                )
            return error_msg

        # --- Build payload ---
        messages = body.get("messages", [])
        temperature = _clamp_temperature(
            body.get("temperature", self.valves.DEFAULT_TEMPERATURE)
        )

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        # Forward optional parameters
        if body.get("max_tokens"):
            payload["max_tokens"] = body["max_tokens"]
        if body.get("top_p") is not None:
            payload["top_p"] = body["top_p"]

        headers = {
            "Authorization": f"Bearer {self.valves.MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        }

        # --- Emit status ---
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "type": "info",
                        "description": f"Generating response with {model_id}...",
                        "done": False,
                    },
                }
            )

        # --- Stream from MiniMax ---
        try:
            result = await self._stream_response(
                payload, headers, __event_emitter__
            )
        except Exception as exc:
            error_msg = f"MiniMax API error: {exc}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "type": "error",
                            "description": error_msg,
                            "done": True,
                        },
                    }
                )
            return error_msg

        # --- Done ---
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "type": "info",
                        "description": "Done",
                        "done": True,
                    },
                }
            )
            await __event_emitter__(
                {"type": "chat:completion", "data": {"done": True}}
            )

        return None  # content already streamed via events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_id(self, pipe_id: str) -> Optional[str]:
        """Extract the MiniMax model ID from the pipe model string."""
        # The pipe ID is formatted as "minimax-<model_id>"
        # Open WebUI may also prepend the function name, e.g.
        # "minimax_pipe.minimax-MiniMax-M2.7"
        # Check longer IDs first to avoid "M2.7" matching before "M2.7-highspeed"
        sorted_models = sorted(
            MINIMAX_MODELS, key=lambda m: len(m["id"]), reverse=True
        )
        for model in sorted_models:
            if model["id"] in pipe_id:
                return model["id"]
        return None

    async def _stream_response(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
    ) -> str:
        """Stream a chat completion from MiniMax and emit deltas."""
        url = f"{MINIMAX_API_BASE}/chat/completions"
        full_content = ""
        thinking_buffer = ""
        in_thinking = False

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"HTTP {resp.status}: {error_text[:500]}"
                    )

                buffer = b""
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = event.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue

                        # Handle think-tag stripping in streaming mode
                        if self.valves.STRIP_THINKING:
                            content, in_thinking, thinking_buffer = (
                                self._process_thinking_stream(
                                    content, in_thinking, thinking_buffer
                                )
                            )

                        if content and emitter:
                            full_content += content
                            await emitter(
                                {
                                    "type": "chat:message:delta",
                                    "data": {
                                        "role": "assistant",
                                        "content": content,
                                    },
                                }
                            )

        return full_content

    @staticmethod
    def _process_thinking_stream(
        text: str,
        in_thinking: bool,
        buffer: str,
    ) -> tuple:
        """
        Process streaming text to strip <think>…</think> blocks.

        Returns (output_text, in_thinking, buffer).
        """
        output = ""
        i = 0

        while i < len(text):
            if in_thinking:
                # Look for </think>
                end_pos = text.find("</think>", i)
                if end_pos != -1:
                    in_thinking = False
                    buffer = ""
                    i = end_pos + len("</think>")
                else:
                    # Still inside <think>, consume everything
                    buffer += text[i:]
                    i = len(text)
            else:
                # Look for <think>
                start_pos = text.find("<think>", i)
                if start_pos != -1:
                    # Emit text before <think>
                    output += text[i:start_pos]
                    in_thinking = True
                    buffer = ""
                    i = start_pos + len("<think>")
                else:
                    # Check for partial "<think" at end
                    partial = ""
                    for plen in range(min(6, len(text) - i), 0, -1):
                        candidate = text[len(text) - plen :]
                        if "<think>".startswith(candidate):
                            partial = candidate
                            break
                    if partial:
                        output += text[i : len(text) - len(partial)]
                        buffer = partial
                    else:
                        output += text[i:]
                        # Flush any prior partial buffer
                        if buffer:
                            output = buffer + output
                            buffer = ""
                    i = len(text)

        return output, in_thinking, buffer
