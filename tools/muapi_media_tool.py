"""
title: MuAPI Media Generator
description: Generate images and videos through MuAPI's asynchronous API.
author: Anil-matcha
author_url: https://github.com/Anil-matcha
version: 1.0.0
license: MIT
required_open_webui_version: 0.9.1
requirements:aiohttp
"""

import asyncio
import html
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, urlparse

import aiohttp
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

DEFAULT_API_BASE_URL = "https://api.muapi.ai/api/v1"
DEFAULT_IMAGE_MODEL = "nano-banana"
DEFAULT_VIDEO_MODEL = "seedance-2.5-text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "seedance-2.5-image-to-video"

COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success", "done"})
FAILED_STATUSES = frozenset(
    {"failed", "failure", "error", "cancelled", "canceled", "timeout"}
)
OUTPUT_CONTAINER_KEYS = frozenset(
    {
        "data",
        "image",
        "image_url",
        "images",
        "output",
        "outputs",
        "result",
        "results",
        "url",
        "urls",
        "video",
        "video_url",
        "videos",
    }
)

EventEmitter = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


class MuAPIError(RuntimeError):
    """Raised when MuAPI rejects or cannot complete a generation request."""


class Tools:
    class UserValves(BaseModel):
        MUAPI_API_KEY: Optional[str] = Field(
            default=None,
            description="Override the MuAPI API key for this user.",
            json_schema_extra={"input": {"type": "password"}},
        )
        IMAGE_MODEL: Optional[str] = Field(
            default=None,
            description="Override the text-to-image endpoint slug.",
        )
        VIDEO_MODEL: Optional[str] = Field(
            default=None,
            description="Override the text-to-video endpoint slug.",
        )
        IMAGE_TO_VIDEO_MODEL: Optional[str] = Field(
            default=None,
            description="Override the image-to-video endpoint slug.",
        )

    class Valves(BaseModel):
        MUAPI_API_KEY: str = Field(
            default="",
            description="MuAPI API key.",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_BASE_URL: str = Field(
            default=DEFAULT_API_BASE_URL,
            description="MuAPI API base URL.",
        )
        IMAGE_MODEL: str = Field(
            default=DEFAULT_IMAGE_MODEL,
            description="Default text-to-image endpoint slug.",
        )
        VIDEO_MODEL: str = Field(
            default=DEFAULT_VIDEO_MODEL,
            description="Default text-to-video endpoint slug.",
        )
        IMAGE_TO_VIDEO_MODEL: str = Field(
            default=DEFAULT_IMAGE_TO_VIDEO_MODEL,
            description="Default image-to-video endpoint slug.",
        )
        POLL_INTERVAL_SECONDS: float = Field(default=3.0, ge=0.1)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=1.0)
        RETURN_HTML_EMBED: bool = Field(
            default=True,
            description="Return an inline HTML video player after video generation.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def _resolve_config(
        self, __user__: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve per-user overrides while keeping shared settings in Valves."""
        user_valves: Optional[Tools.UserValves] = None
        if __user__ and isinstance(__user__, dict) and "valves" in __user__:
            value = __user__["valves"]
            if isinstance(value, Tools.UserValves):
                user_valves = value
            elif isinstance(value, dict):
                try:
                    user_valves = Tools.UserValves(**value)
                except Exception:
                    pass
            elif hasattr(value, "__dict__"):
                try:
                    user_valves = Tools.UserValves(**value.__dict__)
                except Exception:
                    pass

        def pick(user_value: Optional[str], default_value: str) -> str:
            if isinstance(user_value, str) and user_value.strip():
                return user_value.strip()
            return default_value.strip()

        user_key = user_valves.MUAPI_API_KEY if user_valves else None
        api_key = (
            user_key.strip()
            if isinstance(user_key, str) and user_key.strip()
            else self.valves.MUAPI_API_KEY.strip()
        )

        return {
            "api_key": api_key,
            "base_url": self.valves.API_BASE_URL.rstrip("/"),
            "image_model": pick(
                user_valves.IMAGE_MODEL if user_valves else None,
                self.valves.IMAGE_MODEL,
            ),
            "video_model": pick(
                user_valves.VIDEO_MODEL if user_valves else None,
                self.valves.VIDEO_MODEL,
            ),
            "image_to_video_model": pick(
                user_valves.IMAGE_TO_VIDEO_MODEL if user_valves else None,
                self.valves.IMAGE_TO_VIDEO_MODEL,
            ),
            "poll_interval": self.valves.POLL_INTERVAL_SECONDS,
            "timeout": self.valves.GENERATION_TIMEOUT_SECONDS,
            "return_html_embed": self.valves.RETURN_HTML_EMBED,
        }

    @staticmethod
    async def _emit_status(
        emitter: EventEmitter,
        description: str,
        *,
        done: bool,
    ) -> None:
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    @staticmethod
    async def _json_response(response: aiohttp.ClientResponse) -> Dict[str, Any]:
        try:
            payload = await response.json()
        except (aiohttp.ContentTypeError, ValueError) as exc:
            body = (await response.text()).strip()
            raise MuAPIError(
                f"MuAPI returned a non-JSON response ({response.status}): {body[:300]}"
            ) from exc

        if not isinstance(payload, dict):
            raise MuAPIError("MuAPI returned an invalid JSON payload.")

        if response.status >= 400:
            detail = payload.get("message") or payload.get("detail") or payload.get("error")
            raise MuAPIError(
                f"MuAPI request failed ({response.status}): {detail or payload}"
            )
        return payload

    @staticmethod
    def _data(payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, dict):
            merged = dict(payload)
            merged.update(data)
            return merged
        return payload

    @staticmethod
    def _model_url(base_url: str, model: str) -> str:
        model_slug = quote(model.strip(), safe="")
        if not model_slug:
            raise MuAPIError("A MuAPI model endpoint must be configured.")
        return f"{base_url.rstrip('/')}/{model_slug}"

    @staticmethod
    def _prediction_url(base_url: str, request_id: str) -> str:
        return f"{base_url.rstrip('/')}/predictions/{quote(request_id, safe='')}/result"

    @staticmethod
    def _extract_request_id(payload: Dict[str, Any]) -> Optional[str]:
        data = Tools._data(payload)
        for key in ("id", "request_id", "prediction_id", "task_id"):
            value = data.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
        return None

    @classmethod
    def _extract_output_urls(cls, payload: Dict[str, Any]) -> List[str]:
        """Extract output URLs from the documented nested result shape and variants."""
        data = cls._data(payload)
        found: List[str] = []

        def collect(value: Any, key: str = "") -> None:
            if isinstance(value, str):
                candidate = value.strip()
                if (
                    candidate.startswith("http://")
                    or candidate.startswith("https://")
                ):
                    if candidate not in found:
                        found.append(candidate)
                return
            if isinstance(value, list):
                for item in value:
                    collect(item, key)
                return
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    if child_key.lower() in OUTPUT_CONTAINER_KEYS:
                        collect(child_value, child_key.lower())

        for key in OUTPUT_CONTAINER_KEYS:
            if key in data:
                collect(data[key], key)
        return found

    @staticmethod
    def _validate_prompt(prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise MuAPIError("Prompt must not be empty.")
        return prompt.strip()

    @staticmethod
    def _validate_image_url(image_url: str) -> str:
        if not isinstance(image_url, str) or not image_url.strip():
            raise MuAPIError("An image_url is required for image-to-video generation.")
        value = image_url.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise MuAPIError(
                "image_url must be an absolute HTTP or HTTPS URL that MuAPI can fetch."
            )
        return value

    @staticmethod
    def _validate_video_options(
        duration: Optional[int],
        aspect_ratio: Optional[str],
        resolution: Optional[str],
    ) -> None:
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 4
            or duration > 30
        ):
            raise MuAPIError("duration must be an integer from 4 through 30 seconds.")
        if aspect_ratio is not None and (
            not isinstance(aspect_ratio, str) or not aspect_ratio.strip()
        ):
            raise MuAPIError("aspect_ratio must be a non-empty string when provided.")
        if resolution is not None and resolution not in {"1k", "2k", "4k"}:
            raise MuAPIError("resolution must be one of: 1k, 2k, or 4k.")

    @staticmethod
    def _extract_image_from_messages(
        messages: Optional[List[Dict[str, Any]]],
    ) -> Optional[str]:
        if not messages:
            return None

        markdown_pattern = re.compile(
            r"!\[[^\]]*\]\((https?://[^\s\)]+)\)"
        )
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        image_value = item.get("image_url")
                        if isinstance(image_value, dict):
                            image_value = image_value.get("url")
                        if isinstance(image_value, str) and image_value.strip():
                            return image_value.strip()
                    if item.get("type") == "image":
                        source = item.get("source") or item.get("image_url")
                        if isinstance(source, dict):
                            source = source.get("url") or source.get("data")
                        if isinstance(source, str) and source.strip():
                            return source.strip()
            elif isinstance(content, str):
                match = markdown_pattern.search(content)
                if match:
                    return match.group(1)

            for collection_key in ("files", "images"):
                files = message.get(collection_key) or []
                if not isinstance(files, list):
                    continue
                for file_info in files:
                    if not isinstance(file_info, dict):
                        continue
                    value = file_info.get("url") or file_info.get("path")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return None

    async def _submit_and_wait(
        self,
        model: str,
        payload: Dict[str, Any],
        config: Dict[str, Any],
        emitter: EventEmitter,
    ) -> List[str]:
        api_key = config["api_key"]
        if not api_key:
            raise MuAPIError(
                "MuAPI API key is not configured. Set MUAPI_API_KEY in tool Valves or User Valves."
            )

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=config["timeout"] + 30)
        submit_url = self._model_url(config["base_url"], model)

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(submit_url, json=payload) as response:
                submit_payload = await self._json_response(response)

            request_id = self._extract_request_id(submit_payload)
            if not request_id:
                raise MuAPIError("MuAPI did not return a request ID.")

            deadline = time.monotonic() + config["timeout"]
            last_status: Optional[str] = None
            while time.monotonic() < deadline:
                result_url = self._prediction_url(config["base_url"], request_id)
                async with session.get(result_url) as response:
                    result_payload = await self._json_response(response)

                result_data = self._data(result_payload)
                status = str(result_data.get("status") or "").strip().lower()
                if status in COMPLETED_STATUSES:
                    outputs = self._extract_output_urls(result_payload)
                    if not outputs:
                        raise MuAPIError(
                            "MuAPI completed without an HTTP or HTTPS output URL."
                        )
                    return outputs

                if status in FAILED_STATUSES:
                    detail = (
                        result_data.get("error")
                        or result_data.get("message")
                        or status
                    )
                    raise MuAPIError(f"MuAPI generation failed: {detail}")

                # Some successful responses omit a status but include the output.
                if not status:
                    outputs = self._extract_output_urls(result_payload)
                    if outputs:
                        return outputs

                if status != last_status:
                    last_status = status
                    await self._emit_status(
                        emitter,
                        f"MuAPI generation status: {status or 'processing'}",
                        done=False,
                    )

                remaining = deadline - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(config["poll_interval"], remaining))

        raise MuAPIError("MuAPI generation timed out.")

    @staticmethod
    def _format_image_outputs(outputs: List[str]) -> str:
        images = "\n".join(f"![Generated image]({url})" for url in outputs)
        links = "\n".join(f"- {url}" for url in outputs)
        return f"{images}\n\nDownload links:\n{links}"

    def _format_video_outputs(
        self, outputs: List[str], config: Dict[str, Any], message: str
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        links = "\n".join(f"- [Download generated video]({url})" for url in outputs)
        context = f"{message}\n\n{links}"
        if not config["return_html_embed"]:
            return f"{message}\n\n{links}"

        players = "".join(
            '<video controls src="{}" width="960" style="max-width:100%"></video>'.format(
                html.escape(url, quote=True)
            )
            for url in outputs
        )
        return (
            HTMLResponse(
                content=players,
                headers={"content-disposition": "inline"},
            ),
            context,
        )

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate an image with the configured MuAPI image endpoint."""
        config = self._resolve_config(__user__)
        try:
            request_payload: Dict[str, Any] = {
                "prompt": self._validate_prompt(prompt),
            }
            if aspect_ratio is not None:
                if not isinstance(aspect_ratio, str) or not aspect_ratio.strip():
                    raise MuAPIError("aspect_ratio must be a non-empty string when provided.")
                request_payload["aspect_ratio"] = aspect_ratio.strip()
            if resolution is not None:
                if resolution not in {"1k", "2k", "4k"}:
                    raise MuAPIError("resolution must be one of: 1k, 2k, or 4k.")
                request_payload["resolution"] = resolution

            await self._emit_status(
                __event_emitter__, "Generating image with MuAPI", done=False
            )
            outputs = await self._submit_and_wait(
                config["image_model"],
                request_payload,
                config,
                __event_emitter__,
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(__event_emitter__, f"MuAPI error: {exc}", done=True)
            return f"MuAPI image generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "MuAPI image generated", done=True
        )
        return self._format_image_outputs(outputs)

    async def generate_video(
        self,
        prompt: str,
        duration: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        high_bitrate: Optional[bool] = None,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Generate a video from a text prompt with MuAPI."""
        config = self._resolve_config(__user__)
        try:
            prompt_value = self._validate_prompt(prompt)
            self._validate_video_options(duration, aspect_ratio, None)
            request_payload: Dict[str, Any] = {"prompt": prompt_value}
            if duration is not None:
                request_payload["duration"] = duration
            if aspect_ratio is not None:
                request_payload["aspect_ratio"] = aspect_ratio.strip()
            if seed is not None:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise MuAPIError("seed must be an integer when provided.")
                request_payload["seed"] = seed
            if high_bitrate is not None:
                if not isinstance(high_bitrate, bool):
                    raise MuAPIError("high_bitrate must be true or false when provided.")
                request_payload["high_bitrate"] = high_bitrate

            await self._emit_status(
                __event_emitter__, "Generating video with MuAPI", done=False
            )
            outputs = await self._submit_and_wait(
                config["video_model"],
                request_payload,
                config,
                __event_emitter__,
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(__event_emitter__, f"MuAPI error: {exc}", done=True)
            return f"MuAPI video generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "MuAPI video generated", done=True
        )
        return self._format_video_outputs(
            outputs, config, "Video generated successfully with MuAPI."
        )

    async def generate_video_from_image(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        high_bitrate: Optional[bool] = None,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Animate an HTTP(S) image URL with MuAPI's image-to-video endpoint."""
        config = self._resolve_config(__user__)
        try:
            prompt_value = self._validate_prompt(prompt)
            target_image = image_url or self._extract_image_from_messages(__messages__)
            target_image = self._validate_image_url(target_image or "")
            self._validate_video_options(duration, aspect_ratio, None)

            request_payload: Dict[str, Any] = {
                "prompt": prompt_value,
                "image_url": target_image,
            }
            if duration is not None:
                request_payload["duration"] = duration
            if aspect_ratio is not None:
                request_payload["aspect_ratio"] = aspect_ratio.strip()
            if seed is not None:
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise MuAPIError("seed must be an integer when provided.")
                request_payload["seed"] = seed
            if high_bitrate is not None:
                if not isinstance(high_bitrate, bool):
                    raise MuAPIError("high_bitrate must be true or false when provided.")
                request_payload["high_bitrate"] = high_bitrate

            await self._emit_status(
                __event_emitter__, "Generating video from image with MuAPI", done=False
            )
            outputs = await self._submit_and_wait(
                config["image_to_video_model"],
                request_payload,
                config,
                __event_emitter__,
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(__event_emitter__, f"MuAPI error: {exc}", done=True)
            return f"MuAPI image-to-video generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "MuAPI image-to-video generation complete", done=True
        )
        return self._format_video_outputs(
            outputs, config, "Video generated from image successfully with MuAPI."
        )
