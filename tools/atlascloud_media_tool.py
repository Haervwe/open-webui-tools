"""
title: Atlas Cloud Media Generator
description: Generate images, edit images, and create videos (text-to-video, image-to-video, audio-to-video) through the Atlas Cloud Media API.
author: binyangzhu000-sudo & Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
version: 1.1.0
license: MIT
required_open_webui_version: 0.9.1
"""

import asyncio
import base64
import io
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

IMAGE_ENDPOINT = "/model/generateImage"
VIDEO_ENDPOINT = "/model/generateVideo"
UPLOAD_ENDPOINT = "/model/uploadMedia"

DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-pro/text-to-image"
DEFAULT_IMAGE_EDIT_MODEL = "bytedance/seedream-v5.0-pro/image-to-image"
DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "bytedance/seedance-2.5/image-to-video"
DEFAULT_AUDIO_TO_VIDEO_MODEL = "bytedance/seedance-2.5/reference-to-video"

COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success"})
FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled", "timeout"})

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


class AtlasCloudError(RuntimeError):
    """Raised when Atlas Cloud rejects or fails a generation request."""


class Tools:
    class UserValves(BaseModel):
        ATLASCLOUD_API_KEY: Optional[str] = Field(
            default=None,
            description="Override Atlas Cloud API key for personal account.",
            json_schema_extra={"input": {"type": "password"}},
        )
        IMAGE_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred text-to-image model ID.",
        )
        IMAGE_EDIT_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred image-editing / image-to-image model ID.",
        )
        VIDEO_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred text-to-video model ID.",
        )
        IMAGE_TO_VIDEO_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred image-to-video model ID.",
        )
        AUDIO_TO_VIDEO_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred audio-to-video / reference-to-video model ID.",
        )

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
            description="Default text-to-image model ID.",
        )
        IMAGE_EDIT_MODEL: str = Field(
            default=DEFAULT_IMAGE_EDIT_MODEL,
            description="Default image-editing model ID.",
        )
        VIDEO_MODEL: str = Field(
            default=DEFAULT_VIDEO_MODEL,
            description="Default text-to-video model ID.",
        )
        IMAGE_TO_VIDEO_MODEL: str = Field(
            default=DEFAULT_IMAGE_TO_VIDEO_MODEL,
            description="Default image-to-video model ID.",
        )
        AUDIO_TO_VIDEO_MODEL: str = Field(
            default=DEFAULT_AUDIO_TO_VIDEO_MODEL,
            description="Default audio-to-video / reference-to-video model ID.",
        )
        POLL_INTERVAL_SECONDS: float = Field(default=3.0, ge=0.1)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=1.0)
        RETURN_HTML_EMBED: bool = Field(
            default=True,
            description="Return an inline HTML video/audio player upon completion.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.UserValves = Tools.UserValves

    def _resolve_config(
        self, __user__: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolves user-specific valve overrides or falls back to admin valves."""
        user_valves: Optional[Tools.UserValves] = None
        if __user__ and isinstance(__user__, dict) and "valves" in __user__:
            v = __user__["valves"]
            if isinstance(v, Tools.UserValves):
                user_valves = v
            elif isinstance(v, dict):
                user_valves = Tools.UserValves(**v)

        def pick(user_val: Optional[str], default_val: str) -> str:
            if user_val and isinstance(user_val, str) and user_val.strip():
                return user_val.strip()
            return default_val

        api_key = (
            user_valves.ATLASCLOUD_API_KEY.strip()
            if (user_valves and user_valves.ATLASCLOUD_API_KEY and user_valves.ATLASCLOUD_API_KEY.strip())
            else self.valves.ATLASCLOUD_API_KEY.strip()
        )

        return {
            "api_key": api_key,
            "base_url": self.valves.API_BASE_URL.rstrip("/"),
            "image_model": pick(
                user_valves.IMAGE_MODEL if user_valves else None,
                self.valves.IMAGE_MODEL,
            ),
            "image_edit_model": pick(
                user_valves.IMAGE_EDIT_MODEL if user_valves else None,
                self.valves.IMAGE_EDIT_MODEL,
            ),
            "video_model": pick(
                user_valves.VIDEO_MODEL if user_valves else None,
                self.valves.VIDEO_MODEL,
            ),
            "image_to_video_model": pick(
                user_valves.IMAGE_TO_VIDEO_MODEL if user_valves else None,
                self.valves.IMAGE_TO_VIDEO_MODEL,
            ),
            "audio_to_video_model": pick(
                user_valves.AUDIO_TO_VIDEO_MODEL if user_valves else None,
                self.valves.AUDIO_TO_VIDEO_MODEL,
            ),
            "poll_interval": self.valves.POLL_INTERVAL_SECONDS,
            "timeout": self.valves.GENERATION_TIMEOUT_SECONDS,
            "return_html_embed": self.valves.RETURN_HTML_EMBED,
        }

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

    async def _upload_media(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        file_bytes: bytes,
        filename: str = "media_input.png",
        content_type: str = "image/png",
    ) -> str:
        """Uploads raw file bytes to Atlas Cloud uploadMedia endpoint and returns the hosted URL."""
        data = aiohttp.FormData()
        data.add_field(
            "file",
            file_bytes,
            filename=filename,
            content_type=content_type,
        )

        async with session.post(f"{base_url}{UPLOAD_ENDPOINT}", data=data) as resp:
            res_data = self._data(await self._json_response(resp))
            url = res_data.get("url") or res_data.get("file_url")
            if not url or not isinstance(url, str):
                raise AtlasCloudError("Atlas Cloud uploadMedia did not return a valid URL.")
            return url

    async def _ensure_atlas_media_url(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        media_input: str,
    ) -> str:
        """Converts base64 data URIs, local files, or relative URLs into an Atlas Cloud hosted URL."""
        if not media_input or not isinstance(media_input, str):
            raise AtlasCloudError("No valid media input provided.")

        s_input = media_input.strip()

        # If it's a data URI (e.g. data:image/png;base64,...)
        if s_input.startswith("data:"):
            header, data_str = s_input.split(",", 1) if "," in s_input else ("", s_input)
            mime_type = "image/png"
            if "image/jpeg" in header:
                mime_type = "image/jpeg"
            elif "image/webp" in header:
                mime_type = "image/webp"
            elif "audio/mp3" in header or "audio/mpeg" in header:
                mime_type = "audio/mpeg"
            elif "audio/wav" in header:
                mime_type = "audio/wav"

            ext = mime_type.split("/")[-1]
            raw_bytes = base64.b64decode(data_str)
            return await self._upload_media(
                session, base_url, raw_bytes, filename=f"input_file.{ext}", content_type=mime_type
            )

        # If it's a local file path
        if os.path.exists(s_input) and os.path.isfile(s_input):
            ext = os.path.splitext(s_input)[1].lstrip(".") or "png"
            mime_type = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp") else f"audio/{ext}"
            with open(s_input, "rb") as f:
                raw_bytes = f.read()
            return await self._upload_media(
                session, base_url, raw_bytes, filename=os.path.basename(s_input), content_type=mime_type
            )

        # If it's a public web URL
        if s_input.startswith("http://") or s_input.startswith("https://"):
            # If it's an external URL (not localhost OWUI backend), return directly
            if not ("localhost" in s_input or "127.0.0.1" in s_input or "/api/v1/files/" in s_input):
                return s_input
            # If it points to OWUI backend files, attempt to download raw bytes and upload
            try:
                async with session.get(s_input) as resp:
                    if resp.status == 200:
                        raw_bytes = await resp.read()
                        c_type = resp.headers.get("Content-Type", "image/png")
                        return await self._upload_media(
                            session, base_url, raw_bytes, filename="owui_file.png", content_type=c_type
                        )
            except Exception:
                pass
            return s_input

        # Fallback: assume raw base64 string
        try:
            raw_bytes = base64.b64decode(s_input)
            return await self._upload_media(session, base_url, raw_bytes)
        except Exception as exc:
            raise AtlasCloudError(f"Failed to process media input: {exc}") from exc

    def _extract_media_from_messages(
        self, messages: Optional[List[Dict[str, Any]]], media_type: str = "image"
    ) -> List[str]:
        """Extracts image/audio URLs or base64 strings attached in chat messages."""
        if not messages:
            return []

        media_items: List[str] = []
        for message in reversed(messages):
            content = message.get("content")

            # Check structured content blocks (OWUI standard)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        b_type = item.get("type")
                        if media_type == "image" and b_type == "image_url":
                            url_obj = item.get("image_url", {})
                            url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                            if url:
                                media_items.append(url)
                        elif media_type == "audio" and b_type in ("audio", "input_audio"):
                            aud_obj = item.get("input_audio") or item.get("audio") or {}
                            url = aud_obj.get("url") or aud_obj.get("data") if isinstance(aud_obj, dict) else None
                            if url:
                                media_items.append(url)

            # Check text markdown syntax e.g. ![name](url) or http links
            if isinstance(content, str):
                if media_type == "image":
                    matches = re.findall(r'!\[.*?\]\((https?://[^\s\)]+|data:image/[^\s\)]+)\)', content)
                    media_items.extend(matches)

            # Check attached files list in OWUI message dict
            files = message.get("files") or message.get("images") or []
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, dict):
                        url = f.get("url") or f.get("path")
                        if url:
                            media_items.append(url)

            if media_items:
                break

        return media_items

    async def _submit_and_wait(
        self,
        endpoint: str,
        payload: dict[str, Any],
        config: Dict[str, Any],
        emitter: EventEmitter,
    ) -> list[str]:
        api_key = config["api_key"]
        if not api_key:
            raise AtlasCloudError(
                "Atlas Cloud API key is not configured. Please set ATLASCLOUD_API_KEY in tool Valves or User Valves."
            )

        base_url = config["base_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=config["timeout"] + 30)

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.post(f"{base_url}{endpoint}", json=payload) as response:
                submit_data = self._data(await self._json_response(response))

            prediction_id = submit_data.get("id") or submit_data.get("request_id")
            if not prediction_id:
                raise AtlasCloudError("Atlas Cloud did not return a prediction ID.")

            deadline = time.monotonic() + config["timeout"]
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
                await asyncio.sleep(config["poll_interval"])

        raise AtlasCloudError("Atlas Cloud generation timed out.")

    async def generate_image(
        self,
        prompt: str,
        size: str = "2048*2048",
        output_format: str = "jpeg",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate an image from a text prompt with Atlas Cloud."""
        config = self._resolve_config(__user__)
        await self._emit_status(
            __event_emitter__, "Generating image with Atlas Cloud", done=False
        )
        try:
            outputs = await self._submit_and_wait(
                IMAGE_ENDPOINT,
                {
                    "model": config["image_model"],
                    "prompt": prompt,
                    "size": size,
                    "output_format": output_format,
                },
                config,
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

    async def edit_image(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        size: str = "2048*2048",
        output_format: str = "jpeg",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Edit or transform an image using Atlas Cloud image-to-image model. Image is automatically retrieved from attached chat message or explicit image_url."""
        config = self._resolve_config(__user__)
        await self._emit_status(
            __event_emitter__, "Processing image reference for editing...", done=False
        )

        target_image = image_url
        if not target_image:
            extracted = self._extract_media_from_messages(__messages__, "image")
            if extracted:
                target_image = extracted[0]

        if not target_image:
            return "Error: No image reference provided or found in chat messages. Please attach an image or provide an image_url."

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                hosted_url = await self._ensure_atlas_media_url(
                    session, config["base_url"], target_image
                )

            await self._emit_status(
                __event_emitter__, "Submitting image edit request to Atlas Cloud...", done=False
            )

            outputs = await self._submit_and_wait(
                IMAGE_ENDPOINT,
                {
                    "model": config["image_edit_model"],
                    "prompt": prompt,
                    "image_url": hosted_url,
                    "size": size,
                    "output_format": output_format,
                },
                config,
                __event_emitter__,
            )
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud image edit error: {exc}", done=True
            )
            return f"Atlas Cloud image editing failed: {exc}"

        await self._emit_status(
            __event_emitter__, "Atlas Cloud image edit complete!", done=True
        )
        images = "\n".join(f"![Edited image]({url})" for url in outputs)
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
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Generate a video from a text prompt with Atlas Cloud."""
        config = self._resolve_config(__user__)
        await self._emit_status(
            __event_emitter__, "Generating video with Atlas Cloud", done=False
        )
        try:
            outputs = await self._submit_and_wait(
                VIDEO_ENDPOINT,
                {
                    "model": config["video_model"],
                    "prompt": prompt,
                    "duration": duration,
                    "resolution": resolution,
                    "ratio": ratio,
                    "generate_audio": generate_audio,
                },
                config,
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
        url = outputs[0]
        context = f"🎬 Video generated successfully. Link: {url}"
        if config["return_html_embed"]:
            html_player = (
                f'<video controls src="{url}" width="960"'
                f' style="max-width:100%"></video>'
            )
            return (
                HTMLResponse(
                    content=html_player,
                    headers={"content-disposition": "inline"},
                ),
                context,
            )
        links = "\n".join(
            f"- [Download generated video]({u})" for u in outputs
        )
        return f"Generated video:\n{links}"

    async def generate_video_from_image(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Generate a video using a reference image (image-to-video) with Atlas Cloud. Image is automatically retrieved from attached chat message or explicit image_url."""
        config = self._resolve_config(__user__)
        await self._emit_status(
            __event_emitter__, "Processing image reference for video...", done=False
        )

        target_image = image_url
        if not target_image:
            extracted = self._extract_media_from_messages(__messages__, "image")
            if extracted:
                target_image = extracted[0]

        if not target_image:
            return "Error: No reference image provided or found in chat messages. Please attach an image or pass image_url."

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                hosted_url = await self._ensure_atlas_media_url(
                    session, config["base_url"], target_image
                )

            await self._emit_status(
                __event_emitter__, "Submitting image-to-video request to Atlas Cloud...", done=False
            )

            outputs = await self._submit_and_wait(
                VIDEO_ENDPOINT,
                {
                    "model": config["image_to_video_model"],
                    "prompt": prompt,
                    "image_url": hosted_url,
                    "duration": duration,
                    "resolution": resolution,
                    "ratio": ratio,
                    "generate_audio": generate_audio,
                },
                config,
                __event_emitter__,
            )
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {exc}", done=True
            )
            return f"Atlas Cloud image-to-video generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "Atlas Cloud video generated from image!", done=True
        )
        url = outputs[0]
        context = f"🎬 Video generated from image successfully. Link: {url}"
        if config["return_html_embed"]:
            html_player = (
                f'<video controls src="{url}" width="960"'
                f' style="max-width:100%"></video>'
            )
            return (
                HTMLResponse(
                    content=html_player,
                    headers={"content-disposition": "inline"},
                ),
                context,
            )
        links = "\n".join(
            f"- [Download generated video]({u})" for u in outputs
        )
        return f"Generated video:\n{links}"

    async def generate_video_from_audio(
        self,
        prompt: str,
        audio_url: Optional[str] = None,
        image_url: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Generate a video using a reference audio clip (audio-to-video / music-to-video) with Atlas Cloud."""
        config = self._resolve_config(__user__)
        await self._emit_status(
            __event_emitter__, "Processing audio reference for video...", done=False
        )

        target_audio = audio_url
        if not target_audio:
            extracted_audio = self._extract_media_from_messages(__messages__, "audio")
            if extracted_audio:
                target_audio = extracted_audio[0]

        target_image = image_url
        if not target_image:
            extracted_img = self._extract_media_from_messages(__messages__, "image")
            if extracted_img:
                target_image = extracted_img[0]

        if not target_audio and not target_image:
            return "Error: No audio or image reference provided or found in chat messages."

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                payload: Dict[str, Any] = {
                    "model": config["audio_to_video_model"],
                    "prompt": prompt,
                    "duration": duration,
                    "resolution": resolution,
                    "ratio": ratio,
                }
                if target_audio:
                    payload["audio_url"] = await self._ensure_atlas_media_url(
                        session, config["base_url"], target_audio
                    )
                if target_image:
                    payload["image_url"] = await self._ensure_atlas_media_url(
                        session, config["base_url"], target_image
                    )

            await self._emit_status(
                __event_emitter__, "Submitting audio-to-video request to Atlas Cloud...", done=False
            )

            outputs = await self._submit_and_wait(
                VIDEO_ENDPOINT,
                payload,
                config,
                __event_emitter__,
            )
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {exc}", done=True
            )
            return f"Atlas Cloud audio-to-video generation failed: {exc}"

        await self._emit_status(
            __event_emitter__, "Atlas Cloud video generated from audio!", done=True
        )
        url = outputs[0]
        context = f"🎬 Video generated from audio successfully. Link: {url}"
        if config["return_html_embed"]:
            html_player = (
                f'<video controls src="{url}" width="960"'
                f' style="max-width:100%"></video>'
            )
            return (
                HTMLResponse(
                    content=html_player,
                    headers={"content-disposition": "inline"},
                ),
                context,
            )
        links = "\n".join(
            f"- [Download generated video]({u})" for u in outputs
        )
        return f"Generated video:\n{links}"
