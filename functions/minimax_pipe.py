"""
title: MiniMax LLM Pipe
author: octopus
author_url: https://github.com/octo-patch
version: 1.3.1
required_open_webui_version: 0.6.0
description: MiniMax LLM Pipe for Open WebUI — routes chat completions to MiniMax's
OpenAI-compatible API (api.minimax.io/v1). Supports MiniMax-M3.1-Flash (default)
and MiniMax-M3 with image and video input, model-aware thinking controls,
streaming (content and reasoning returned as pipe output), temperature
clamping, and automatic think-tag handling.
"""

import aiohttp
import json
import logging
import re
import traceback
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
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
        "id": "MiniMax-M3.1-Flash",
        "name": "MiniMax M3.1 Flash",
        "context_length": 1000000,
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive"],
    },
    {
        "id": "MiniMax-M3",
        "name": "MiniMax M3",
        "context_length": 1000000,
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
    },
]

FLASH_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Thinking modes accepted by the chat-completions API. Each model declares
# which of these it supports in MINIMAX_MODELS[*]["thinking"].
THINKING_MODE_ADAPTIVE = "adaptive"
THINKING_MODE_DISABLED = "disabled"
# Kept for backward compatibility: no built-in model declares it any more;
# legacy/custom IDs simply have no "thinking" list and omit the field.
THINKING_MODE_ALWAYS_ON = "always_on"

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
            description="MiniMax API key (get one at https://platform.minimax.io)",
            json_schema_extra={"input": {"type": "password"}},
        )
        ENABLED_MODELS: List[str] = Field(
            default_factory=lambda: [m["id"] for m in MINIMAX_MODELS],
            description=(
                "Which MiniMax models to expose (model IDs). Only these "
                "(plus the built-in catalog) are ever sent upstream."
            ),
        )
        STRIP_THINKING: bool = Field(
            default=True,
            description=(
                "Hide MiniMax reasoning: strips <think>…</think> blocks and "
                "drops M3.1-Flash reasoning_content. When off, reasoning is "
                "shown in Open WebUI's collapsible reasoning block."
            ),
        )
        DEFAULT_TEMPERATURE: float = Field(
            default=0.7,
            description="Default temperature when none is specified (0.01–1.0)",
        )
        THINKING_MODE: str = Field(
            default=THINKING_MODE_ADAPTIVE,
            description=(
                "Thinking mode for models that support it: "
                "'adaptive' (let the model decide) or 'disabled'. "
                "Flash always uses adaptive; ignored by legacy models."
            ),
        )
        REASONING_EFFORT: str = Field(
            default="max",
            description="Flash thinking depth: low, medium, high, xhigh, or max.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # pipes() — register models with Open WebUI
    # ------------------------------------------------------------------

    def pipes(self) -> List[Dict[str, str]]:
        """Return the list of model entries this pipe exposes."""
        enabled = set(self.valves.ENABLED_MODELS)
        entries = [
            {"id": f"minimax-{m['id']}", "name": f"MiniMax {m['name']}"}
            for m in MINIMAX_MODELS
            if m["id"] in enabled
        ]
        # Keep explicitly configured legacy/custom choices without adding them
        # to the ordinary catalog or changing saved valve settings.
        known = {m["id"] for m in MINIMAX_MODELS}
        for model_id in dict.fromkeys(self.valves.ENABLED_MODELS):
            if model_id not in known:
                entries.append({
                    "id": f"minimax-{model_id}",
                    "name": f"MiniMax {model_id.replace('MiniMax-', 'MiniMax ', 1)}",
                })
        return entries

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
    ) -> Union[str, AsyncGenerator[Union[str, Dict[str, Any]], None]]:
        """
        Route a chat completion request to the MiniMax API.

        The response is returned as the pipe's output so Open WebUI persists
        it: for streaming requests an async generator yielding content text
        and OpenAI-style ``reasoning_content`` delta chunks, otherwise the
        final content string.
        """

        # --- Validate API key ---
        if not self.valves.MINIMAX_API_KEY:
            error_msg = (
                "MiniMax API key not configured. "
                "Please set MINIMAX_API_KEY in the pipe's Valves settings."
            )
            await self._emit_status(__event_emitter__, "error", error_msg, True)
            return error_msg

        # --- Resolve model ID ---
        pipe_id = body.get("model", "")
        model_id = self._resolve_model_id(pipe_id)
        if not model_id:
            error_msg = f"Unknown MiniMax model: {pipe_id}"
            await self._emit_status(__event_emitter__, "error", error_msg, True)
            return error_msg

        # --- Build payload ---
        messages = self._normalize_messages(
            body.get("messages", []), model_id
        )
        temperature = _clamp_temperature(
            body.get("temperature", self.valves.DEFAULT_TEMPERATURE)
        )

        # Upstream is always streamed; the pipe aggregates for non-stream
        # callers (e.g. Open WebUI title/tag generation tasks).
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

        # Flash always thinks; M3 also supports disabling thinking.
        thinking_config = self._resolve_thinking(model_id, body)
        if thinking_config is not None:
            payload["thinking"] = {"type": thinking_config}
        if model_id == "MiniMax-M3.1-Flash":
            effort = body.get("reasoning_effort", self.valves.REASONING_EFFORT)
            if effort not in FLASH_REASONING_EFFORTS:
                effort = self.valves.REASONING_EFFORT
            payload["reasoning_effort"] = (
                effort if effort in FLASH_REASONING_EFFORTS else "max"
            )

        headers = {
            "Authorization": f"Bearer {self.valves.MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        }

        # --- Emit status ---
        await self._emit_status(
            __event_emitter__,
            "info",
            f"Generating response with {model_id}...",
            False,
        )

        if body.get("stream", False):
            return self._stream_output(payload, headers, __event_emitter__)

        # --- Non-streaming: aggregate content into a single string ---
        content = ""
        try:
            async for kind, text in self._stream_response(payload, headers):
                if kind == "content":
                    content += text
        except Exception as exc:
            return await self._report_error(exc, __event_emitter__)

        await self._emit_status(__event_emitter__, "info", "Done", True)
        return content

    async def _stream_output(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Yield the MiniMax stream as Open WebUI pipe output.

        Content is yielded as plain text; reasoning is yielded as an
        OpenAI-style chunk with ``delta.reasoning_content`` which Open WebUI's
        middleware renders as a reasoning block.
        """
        try:
            async for kind, text in self._stream_response(payload, headers):
                if kind == "reasoning":
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": text},
                            }
                        ]
                    }
                else:
                    yield text
        except Exception as exc:
            yield await self._report_error(exc, emitter)
            return

        await self._emit_status(emitter, "info", "Done", True)

    @staticmethod
    async def _emit_status(
        emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
        status_type: str,
        description: str,
        done: bool,
    ) -> None:
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "type": status_type,
                        "description": description,
                        "done": done,
                    },
                }
            )

    async def _report_error(
        self,
        exc: Exception,
        emitter: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
    ) -> str:
        error_msg = f"MiniMax API error: {exc}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        await self._emit_status(emitter, "error", error_msg, True)
        return error_msg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _message_content_to_str(content: Any) -> str:
        """Flatten a message's content into a single string."""
        if isinstance(content, str):
            return content
        parts: List[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(
                        item.get("text"), str
                    ):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
                elif isinstance(item, str):
                    parts.append(item)
        return "".join(parts)

    def _model_supports_modality(
        self, model_id: str, modality: str
    ) -> bool:
        """True when the resolved model lists the given input modality."""
        for model in MINIMAX_MODELS:
            if model["id"] == model_id:
                return modality in model.get("input_modalities", ["text"])
        return False

    def _normalize_messages(
        self, messages: List[Dict[str, Any]], model_id: str
    ) -> List[Dict[str, Any]]:
        """
        Normalize Open WebUI messages for the MiniMax API.

        For multimodal models (MiniMax-M3), inline attachments carried on the
        Open WebUI message are converted to OpenAI-style multi-part content:
        text plus image_url / video_url parts. For text-only models the content
        is flattened to a string so the API never receives image or video
        parts it cannot handle. The supplied ``model_id`` selects which path
        applies.
        """
        normalized: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            files = message.get("files")

            # Already multi-part content from the caller
            if isinstance(content, list):
                normalized.append({"role": role, "content": content})
                continue

            text_content = (
                content if isinstance(content, str) else self._message_content_to_str(content)
            )

            if files and self._model_supports_modality(
                model_id, "image"
            ):
                parts: List[Dict[str, Any]] = []
                if text_content:
                    parts.append({"type": "text", "text": text_content})
                for file in files:
                    url = self._attachment_url(file)
                    if not url:
                        continue
                    kind = file.get("type") or self._attachment_kind(url)
                    if kind == "video" and self._model_supports_modality(
                        model_id, "video"
                    ):
                        parts.append(
                            {"type": "video_url", "video_url": {"url": url}}
                        )
                    else:
                        parts.append(
                            {"type": "image_url", "image_url": {"url": url}}
                        )
                normalized.append({"role": role, "content": parts or text_content})
            else:
                normalized.append({"role": role, "content": text_content})
        return normalized

    @staticmethod
    def _attachment_url(file: Any) -> Optional[str]:
        """Extract a URL or data URL from an Open WebUI file attachment."""
        if not isinstance(file, dict):
            return None
        # Direct URL fields used by Open WebUI uploads
        for key in ("url", "uri"):
            value = file.get(key)
            if isinstance(value, str) and value:
                return value
        # Nested { image_url: { url } } / { video_url: { url } }
        for key in ("image_url", "video_url"):
            nested = file.get(key)
            if isinstance(nested, dict):
                url = nested.get("url")
                if isinstance(url, str) and url:
                    return url
        return None

    @staticmethod
    def _attachment_kind(url: str) -> str:
        """Classify an attachment as 'image' or 'video' from its data URL."""
        if url.startswith("data:"):
            head = url.split(",", 1)[0]
            if head.startswith("data:video"):
                return "video"
            return "image"
        lower = url.lower().split("?", 1)[0]
        if lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
            return "video"
        return "image"

    def _resolve_thinking(
        self, model_id: str, body: Dict[str, Any]
    ) -> Optional[str]:
        """
        Resolve the thinking mode to forward to the API.

        Only built-in models that list adaptive/disabled thinking send the
        field; legacy or custom IDs (e.g. a saved MiniMax-M2.7) declare no
        thinking modes and return None so the field is omitted entirely.
        """
        supported: List[str] = []
        for model in MINIMAX_MODELS:
            if model["id"] == model_id:
                supported = model.get("thinking", [])
                break

        if THINKING_MODE_ADAPTIVE not in supported and (
            THINKING_MODE_DISABLED not in supported
        ):
            return None

        requested = body.get("thinking", self.valves.THINKING_MODE)
        if isinstance(requested, dict):
            requested = requested.get("type")
        if isinstance(requested, str) and requested in supported:
            return requested
        # Fall back to the valve default when it is supported, otherwise pick
        # the first supported adaptive/disabled mode.
        if self.valves.THINKING_MODE in supported:
            return self.valves.THINKING_MODE
        for mode in (THINKING_MODE_ADAPTIVE, THINKING_MODE_DISABLED):
            if mode in supported:
                return mode
        return None

    def _resolve_model_id(self, pipe_id: str) -> Optional[str]:
        """Extract the MiniMax model ID from the pipe model string."""
        if not isinstance(pipe_id, str):
            return None
        # Strip only the pipe/function wrapper, never a model suffix. A saved
        # legacy or custom MiniMax ID must not become a different active model.
        match = re.fullmatch(r"(?:[^.]+\.)?minimax-(.+)", pipe_id)
        model_id = match.group(1) if match else pipe_id
        # Only IDs this pipe exposes (enabled valve entries, which may be
        # legacy/custom) or the built-in catalog are sent upstream, so the
        # ENABLED_MODELS valve remains an effective allowlist.
        if model_id in self.valves.ENABLED_MODELS:
            return model_id
        if any(m["id"] == model_id for m in MINIMAX_MODELS):
            return model_id
        return None

    async def _stream_response(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> AsyncGenerator[tuple, None]:
        """
        Stream a chat completion from MiniMax.

        Yields ``("reasoning", text)`` for ``delta.reasoning_content``
        (M3.1-Flash) and ``("content", text)`` for answer text, applying the
        STRIP_THINKING valve to both.
        """
        url = f"{MINIMAX_API_BASE}/chat/completions"
        strip = self.valves.STRIP_THINKING
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
                done = False
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            done = True
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = event.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta") or {}

                        reasoning = delta.get("reasoning_content")
                        if reasoning and isinstance(reasoning, str) and not strip:
                            yield ("reasoning", reasoning)

                        content = delta.get("content", "")
                        if not content:
                            continue

                        # Handle think-tag stripping in streaming mode
                        if strip:
                            content, in_thinking, thinking_buffer = (
                                self._process_thinking_stream(
                                    content, in_thinking, thinking_buffer
                                )
                            )

                        if content:
                            yield ("content", content)
                    if done:
                        break

        # Flush a trailing partial "<think" prefix that never completed.
        if strip and not in_thinking and thinking_buffer:
            yield ("content", thinking_buffer)

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
