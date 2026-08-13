"""
title: Atlas Cloud Media Generator
description: Generate images and videos through the Atlas Cloud Media API.
author: binyangzhu000-sudo
author_url: https://github.com/binyangzhu000-sudo
version: 1.0.0
license: MIT
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

IMAGE_ENDPOINT = "/model/generateImage"
VIDEO_ENDPOINT = "/model/generateVideo"
DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-pro/text-to-image"
DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success"})
FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled", "timeout"})

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


class AtlasCloudError(RuntimeError):
    """Raised when Atlas Cloud rejects or fails a generation request."""


class Tools:
    class Valves(BaseModel):
        ATLASCLOUD_API_KEY: str = Field(
            default="",
            description="Atlas Cloud API key.",
            json_schema_extra={"input": {"type": "password"}},
        )
        API_BASE_URL: str = Field(
            default="https://api.atlascloud.ai/api/v1",
            description="Atlas Cloud Media API base URL.",
        )
        IMAGE_MODEL: str = Field(
            default=DEFAULT_IMAGE_MODEL,
            description="Default Atlas Cloud text-to-image model ID.",
        )
        VIDEO_MODEL: str = Field(
            default=DEFAULT_VIDEO_MODEL,
            description="Default Atlas Cloud text-to-video model ID.",
        )
        POLL_INTERVAL_SECONDS: float = Field(default=3.0, ge=0.1)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=1.0)

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def _emit_status(
        self,
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
    async def _json_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            payload = await response.json()
        except (aiohttp.ContentTypeError, ValueError) as exc:
            body = (await response.text()).strip()
            raise AtlasCloudError(
                f"Atlas Cloud returned a non-JSON response ({response.status}): {body[:300]}"
            ) from exc

        if not isinstance(payload, dict):
            raise AtlasCloudError("Atlas Cloud returned an invalid JSON payload.")

        if response.status >= 400:
            detail = payload.get("message") or payload.get("error") or payload
            raise AtlasCloudError(
                f"Atlas Cloud request failed ({response.status}): {detail}"
            )
        return payload

    @staticmethod
    def _data(payload: dict[str, Any]) -> dict[str, Any]:
        code = payload.get("code")
        if code not in (None, 0, 200):
            detail = payload.get("message") or payload.get("error") or f"code {code}"
            raise AtlasCloudError(f"Atlas Cloud request failed: {detail}")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise AtlasCloudError("Atlas Cloud returned an invalid response payload.")
        return data

    async def _submit_and_wait(
        self,
        endpoint: str,
        payload: dict[str, Any],
        emitter: EventEmitter,
    ) -> list[str]:
        api_key = self.valves.ATLASCLOUD_API_KEY.strip()
        if not api_key:
            raise AtlasCloudError(
                "Atlas Cloud API key is not configured in the tool Valves."
            )

        base_url = self.valves.API_BASE_URL.rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(
            total=self.valves.GENERATION_TIMEOUT_SECONDS + 30
        )

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(f"{base_url}{endpoint}", json=payload) as response:
                submit_data = self._data(await self._json_response(response))

            prediction_id = submit_data.get("id") or submit_data.get("request_id")
            if not prediction_id:
                raise AtlasCloudError("Atlas Cloud did not return a prediction ID.")

            deadline = time.monotonic() + self.valves.GENERATION_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                async with session.get(
                    f"{base_url}/model/prediction/{prediction_id}"
                ) as response:
                    prediction = self._data(await self._json_response(response))

                status = str(prediction.get("status", "")).lower()
                if status in COMPLETED_STATUSES:
                    outputs = (
                        prediction.get("outputs") or prediction.get("output") or []
                    )
                    if isinstance(outputs, str):
                        outputs = [outputs]
                    if not isinstance(outputs, list) or not all(
                        isinstance(item, str) for item in outputs
                    ):
                        raise AtlasCloudError(
                            "Atlas Cloud returned invalid output URLs."
                        )
                    if not outputs:
                        raise AtlasCloudError(
                            "Atlas Cloud completed without output URLs."
                        )
                    return outputs

                if status in FAILED_STATUSES:
                    detail = (
                        prediction.get("error") or prediction.get("message") or status
                    )
                    raise AtlasCloudError(f"Atlas Cloud generation failed: {detail}")

                await self._emit_status(
                    emitter,
                    f"Atlas Cloud generation status: {status or 'processing'}",
                    done=False,
                )
                await asyncio.sleep(self.valves.POLL_INTERVAL_SECONDS)

        raise AtlasCloudError("Atlas Cloud generation timed out.")

    async def generate_image(
        self,
        prompt: str,
        size: str = "2048*2048",
        output_format: str = "jpeg",
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Generate an image from a text prompt with Atlas Cloud."""
        await self._emit_status(
            __event_emitter__, "Generating image with Atlas Cloud", done=False
        )
        try:
            outputs = await self._submit_and_wait(
                IMAGE_ENDPOINT,
                {
                    "model": self.valves.IMAGE_MODEL,
                    "prompt": prompt,
                    "size": size,
                    "output_format": output_format,
                },
                __event_emitter__,
            )
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {exc}", done=True
            )
            return f"Atlas Cloud image generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "Atlas Cloud image generated", done=True
        )
        images = "\n".join(f"![Generated image]({url})" for url in outputs)
        links = "\n".join(f"- {url}" for url in outputs)
        return f"{images}\n\nDownload links:\n{links}"

    async def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
    ) -> str:
        """Generate a video from a text prompt with Atlas Cloud."""
        await self._emit_status(
            __event_emitter__, "Generating video with Atlas Cloud", done=False
        )
        try:
            outputs = await self._submit_and_wait(
                VIDEO_ENDPOINT,
                {
                    "model": self.valves.VIDEO_MODEL,
                    "prompt": prompt,
                    "duration": duration,
                    "resolution": resolution,
                    "ratio": ratio,
                    "generate_audio": generate_audio,
                },
                __event_emitter__,
            )
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {exc}", done=True
            )
            return f"Atlas Cloud video generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "Atlas Cloud video generated", done=True
        )
        links = "\n".join(f"- [Download generated video]({url})" for url in outputs)
        return f"Generated video:\n{links}"
