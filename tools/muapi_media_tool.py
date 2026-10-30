"""
title: MuAPI Media Generator
description: Generate images and videos through MuAPI's asynchronous API. Outputs are stored in Open WebUI file storage and shown in an inline viewer.
author: Anil-matcha, Haervwe
author_url: https://github.com/Anil-matcha
version: 1.1.0
license: MIT
required_open_webui_version: 0.9.1
requirements:aiohttp
"""

import asyncio
import base64
import binascii
import html
import io
import ipaddress
import logging
import mimetypes
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, urlparse

import aiohttp
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
from open_webui.storage.provider import Storage
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://api.muapi.ai/api/v1"
DEFAULT_IMAGE_MODEL = "nano-banana"
DEFAULT_VIDEO_MODEL = "seedance-2.5-text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "seedance-2.5-image-to-video"

COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success", "done"})
FAILED_STATUSES = frozenset(
    {"failed", "failure", "error", "cancelled", "canceled", "timeout"}
)
# Documented result locations (``outputs`` / ``output``) are checked first; the
# rest is a fallback for response variants. Input-echo keys such as
# ``image_url`` are deliberately excluded so the source image of an
# image-to-video job is never mistaken for an output.
DOCUMENTED_OUTPUT_KEYS = ("outputs", "output")
FALLBACK_OUTPUT_KEYS = (
    "video",
    "videos",
    "video_url",
    "image",
    "images",
    "result",
    "results",
    "url",
    "urls",
    "data",
)
NESTED_OUTPUT_KEYS = frozenset(DOCUMENTED_OUTPUT_KEYS + FALLBACK_OUTPUT_KEYS)

MAX_ERROR_DETAIL_CHARS = 200
# MuAPI's upload endpoint accepts images up to 10 MB.
MAX_INPUT_IMAGE_BYTES = 10 * 1024 * 1024
OWUI_FILE_PATH_RE = re.compile(r"^/api/v1/files/([A-Za-z0-9-]+)(?:/content)?/?$")
NON_SAVED_CHAT_ID_PREFIXES = ("temporary:", "local:", "channel:")

EventEmitter = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


class MuAPIError(RuntimeError):
    """Raised when MuAPI rejects or cannot complete a generation request."""


def _short(value: Any, limit: int = MAX_ERROR_DETAIL_CHARS) -> str:
    text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
            description="Return an inline HTML viewer after image or video generation.",
        )
        MAX_DOWNLOAD_MB: int = Field(
            default=200,
            ge=1,
            description="Maximum size of a single MuAPI output downloaded into Open WebUI storage.",
        )
        ALLOW_PROVIDER_URL_FALLBACK: bool = Field(
            default=False,
            description=(
                "If saving an output to Open WebUI storage fails, show the MuAPI URL "
                "instead of returning an error. Provider URLs may expire."
            ),
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
            "max_download_bytes": self.valves.MAX_DOWNLOAD_MB * 1024 * 1024,
            "allow_provider_url_fallback": self.valves.ALLOW_PROVIDER_URL_FALLBACK,
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
            log.warning(
                "MuAPI returned a non-JSON response (%s): %s",
                response.status,
                body[:1000],
            )
            raise MuAPIError(
                f"MuAPI returned a non-JSON response (HTTP {response.status})."
            ) from exc

        if not isinstance(payload, dict):
            raise MuAPIError("MuAPI returned an invalid JSON payload.")

        if response.status >= 400:
            detail = payload.get("message") or payload.get("detail") or payload.get("error")
            log.warning("MuAPI request failed (%s): %s", response.status, payload)
            suffix = f": {_short(detail)}" if detail else "."
            raise MuAPIError(f"MuAPI request failed (HTTP {response.status}){suffix}")
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
        """Extract output URLs, preferring the documented ``outputs``/``output`` shape."""
        data = cls._data(payload)

        def collect(value: Any, found: List[str]) -> None:
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
                    collect(item, found)
                return
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    if str(child_key).lower() in NESTED_OUTPUT_KEYS:
                        collect(child_value, found)

        documented: List[str] = []
        for key in DOCUMENTED_OUTPUT_KEYS:
            if key in data:
                collect(data[key], documented)
        if documented:
            return documented

        fallback: List[str] = []
        for key in FALLBACK_OUTPUT_KEYS:
            if key in data:
                collect(data[key], fallback)
        return fallback

    @staticmethod
    def _validate_prompt(prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise MuAPIError("Prompt must not be empty.")
        return prompt.strip()

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
        """Find the latest attached image: http(s) URL, data: URI or /api/v1/files path."""
        if not messages:
            return None

        markdown_pattern = re.compile(
            r"!\[[^\]]*\]\(((?:https?://|data:image/|/api/v1/files/)[^\s\)]+)\)"
        )
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in ("image_url", "input_image"):
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
                    if file_info.get("type") not in (None, "image", "file"):
                        continue
                    value = file_info.get("url") or file_info.get("path")
                    if (
                        not value
                        and file_info.get("type") == "image"
                        and file_info.get("id")
                    ):
                        value = f"/api/v1/files/{file_info['id']}/content"
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return None

    # ------------------------------------------------------------------
    # Image-to-video input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_data_uri(value: str) -> Tuple[bytes, str]:
        header, _, encoded = value.partition(",")
        match = re.match(r"^data:(image/[A-Za-z0-9.+-]+)(;[^,]*)?$", header)
        if not match or ";base64" not in (match.group(2) or ""):
            raise MuAPIError("Attached image must be a base64 data:image/* URI.")
        try:
            raw = base64.b64decode(encoded, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise MuAPIError("Attached image data could not be decoded.") from exc
        if not raw:
            raise MuAPIError("Attached image is empty.")
        return raw, match.group(1).lower()

    @staticmethod
    async def _read_owui_file(file_id: str, user_obj: Any) -> Tuple[bytes, str]:
        """Read an Open WebUI file the user may access, server-side via Files + Storage."""
        if user_obj is None:
            raise MuAPIError("Cannot read the attached Open WebUI file without a user context.")
        file_item = await Files.get_file_by_id(file_id)
        if not file_item:
            raise MuAPIError("The attached Open WebUI file was not found.")

        allowed = file_item.user_id == user_obj.id or user_obj.role == "admin"
        if not allowed:
            try:
                from open_webui.utils.access_control.files import has_access_to_file

                allowed = await has_access_to_file(file_id, "read", user_obj)
            except Exception:
                allowed = False
        if not allowed:
            raise MuAPIError("You do not have access to the attached Open WebUI file.")

        meta = file_item.meta or {}
        content_type = (meta.get("content_type") or "").lower() or (
            mimetypes.guess_type(file_item.filename or "")[0] or ""
        )
        if not content_type.startswith("image/"):
            raise MuAPIError("The attached Open WebUI file is not an image.")

        def _read() -> bytes:
            local_path = Storage.get_file(file_item.path)
            with open(local_path, "rb") as handle:
                return handle.read()

        raw = await asyncio.to_thread(_read)
        if not raw:
            raise MuAPIError("The attached Open WebUI file is empty.")
        return raw, content_type

    @staticmethod
    def _owui_file_id(value: str) -> Optional[str]:
        parsed = urlparse(value)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            return None
        match = OWUI_FILE_PATH_RE.match(parsed.path or "")
        return match.group(1) if match else None

    async def _upload_to_muapi(
        self,
        raw: bytes,
        content_type: str,
        config: Dict[str, Any],
    ) -> str:
        """Upload image bytes with MuAPI's ``/upload_file`` endpoint and return the hosted URL."""
        if len(raw) > MAX_INPUT_IMAGE_BYTES:
            raise MuAPIError("Attached image is larger than MuAPI's 10 MB upload limit.")
        api_key = config["api_key"]
        if not api_key:
            raise MuAPIError(
                "MuAPI API key is not configured. Set MUAPI_API_KEY in tool Valves or User Valves."
            )
        extension = (mimetypes.guess_extension(content_type) or ".png").lstrip(".")
        if extension == "jpe":
            extension = "jpg"
        form = aiohttp.FormData()
        form.add_field(
            "file",
            raw,
            filename=f"owui_input_{uuid.uuid4().hex[:8]}.{extension}",
            content_type=content_type,
        )
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(
            timeout=timeout, headers={"x-api-key": api_key}
        ) as session:
            async with session.post(
                f"{config['base_url']}/upload_file", data=form
            ) as response:
                payload = await self._json_response(response)

        data = self._data(payload)
        for key in ("url", "file_url", "image_url"):
            value = data.get(key)
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                return value.strip()
        raise MuAPIError("MuAPI upload did not return a hosted file URL.")

    async def _resolve_image_input(
        self,
        image: str,
        config: Dict[str, Any],
        user_obj: Any,
    ) -> str:
        """Turn an attached/provided image into a URL MuAPI can fetch."""
        if not isinstance(image, str) or not image.strip():
            raise MuAPIError(
                "An image is required for image-to-video generation. Attach an image or pass image_url."
            )
        value = image.strip()

        if value.startswith("data:"):
            raw, content_type = self._decode_data_uri(value)
            return await self._upload_to_muapi(raw, content_type, config)

        file_id = self._owui_file_id(value)
        if file_id:
            # Relative /api/v1/files paths are always Open WebUI files. Absolute
            # URLs with that path are treated as such only when the id exists
            # locally; otherwise they are ordinary public URLs.
            is_relative = not urlparse(value).netloc
            if is_relative or await Files.get_file_by_id(file_id):
                raw, content_type = await self._read_owui_file(file_id, user_obj)
                return await self._upload_to_muapi(raw, content_type, config)

        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            # Public URL: MuAPI fetches it directly.
            return value
        raise MuAPIError(
            "image_url must be an attached image, an Open WebUI file, a data:image URI, "
            "or an absolute HTTP(S) URL."
        )

    # ------------------------------------------------------------------
    # MuAPI job lifecycle
    # ------------------------------------------------------------------

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
                    log.warning("MuAPI generation %s failed: %s", request_id, detail)
                    raise MuAPIError(f"MuAPI generation failed: {_short(detail)}")

                # Some successful responses omit a status but include the output.
                if not status:
                    outputs = self._extract_output_urls(result_payload)
                    if outputs:
                        return outputs

                if status != last_status:
                    last_status = status
                    await self._emit_status(
                        emitter,
                        f"MuAPI generation status: {_short(status or 'processing', 40)}",
                        done=False,
                    )

                remaining = deadline - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(config["poll_interval"], remaining))

        raise MuAPIError("MuAPI generation timed out.")

    # ------------------------------------------------------------------
    # Output storage in Open WebUI
    # ------------------------------------------------------------------

    @staticmethod
    def _is_local_host(host: str) -> bool:
        host = (host or "").strip("[]").lower()
        if not host:
            return True
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_unspecified
            or address.is_multicast
        )

    def _check_download_url(self, url: str, base_url: str) -> None:
        """Only download provider outputs from public HTTPS hosts (unless MuAPI itself is local)."""
        parsed = urlparse(url)
        base = urlparse(base_url)
        base_is_local = self._is_local_host(base.hostname or "")
        allowed_schemes = {"https", "http"} if base.scheme == "http" else {"https"}
        if parsed.scheme not in allowed_schemes or not parsed.hostname:
            raise MuAPIError("MuAPI returned an output URL with an unsupported scheme.")
        if self._is_local_host(parsed.hostname) and not base_is_local:
            raise MuAPIError("MuAPI returned an output URL on a private or local host.")

    async def _download_output(
        self, url: str, config: Dict[str, Any]
    ) -> Tuple[bytes, str]:
        """Download an output without the MuAPI key, enforcing a size cap."""
        self._check_download_url(url, config["base_url"])
        max_bytes = config["max_download_bytes"]
        timeout = aiohttp.ClientTimeout(total=max(120.0, config["timeout"]))
        # A fresh session with no default headers: the API key must never reach a CDN.
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise MuAPIError(
                        f"Downloading the MuAPI output failed (HTTP {response.status})."
                    )
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise MuAPIError("MuAPI output exceeds the MAX_DOWNLOAD_MB limit.")
                content_type = (
                    response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                )
                chunks: List[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(1024 * 256):
                    total += len(chunk)
                    if total > max_bytes:
                        raise MuAPIError("MuAPI output exceeds the MAX_DOWNLOAD_MB limit.")
                    chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            raise MuAPIError("MuAPI output was empty.")
        return data, content_type

    @staticmethod
    def _media_type(url: str, content_type: str, kind: str) -> Optional[str]:
        """Return the output MIME type, or None when it does not match ``kind``."""
        guessed = mimetypes.guess_type(urlparse(url).path)[0] or ""
        for candidate in (content_type, guessed):
            if candidate.startswith(f"{kind}/"):
                return candidate
        if content_type.startswith(
            ("image/", "video/", "audio/", "text/", "application/json", "application/xml")
        ):
            return None
        if guessed and not guessed.startswith(f"{kind}/"):
            return None
        # Generic content type (e.g. application/octet-stream) and no extension.
        return "image/png" if kind == "image" else "video/mp4"

    @staticmethod
    def _chat_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        metadata = metadata if isinstance(metadata, dict) else {}
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        if (
            isinstance(chat_id, str)
            and chat_id
            and not chat_id.startswith(NON_SAVED_CHAT_ID_PREFIXES)
            and isinstance(message_id, str)
            and message_id
        ):
            return {"chat_id": chat_id, "message_id": message_id}
        return {}

    async def _save_outputs(
        self,
        outputs: List[str],
        kind: str,
        config: Dict[str, Any],
        request: Any,
        user_obj: Any,
        metadata: Optional[Dict[str, Any]],
        emitter: EventEmitter,
    ) -> List[str]:
        """Store every MuAPI output in Open WebUI and return /api/v1/files URLs."""
        if request is None or user_obj is None:
            raise MuAPIError(
                "Cannot store the generated media: Open WebUI request or user context is missing."
            )

        await self._emit_status(
            emitter, f"Saving generated {kind} to Open WebUI", done=False
        )
        chat_meta = self._chat_metadata(metadata)
        stored: List[str] = []
        stored_ids: List[str] = []
        for index, url in enumerate(outputs):
            try:
                data, content_type = await self._download_output(url, config)
                mime = self._media_type(url, content_type, kind)
                if not mime:
                    log.info("Skipping MuAPI output %s: not a %s", index, kind)
                    continue
                extension = (mimetypes.guess_extension(mime) or "").lstrip(".") or (
                    "png" if kind == "image" else "mp4"
                )
                if extension == "jpe":
                    extension = "jpg"
                upload = UploadFile(
                    file=io.BytesIO(data),
                    filename=f"muapi_{kind}_{uuid.uuid4().hex[:8]}.{extension}",
                    headers={"content-type": mime},
                )
                file_item = await upload_file_handler(
                    request,
                    file=upload,
                    metadata=dict(chat_meta),
                    process=False,
                    user=user_obj,
                )
                file_id = getattr(file_item, "id", None)
                if not file_id:
                    raise MuAPIError("Open WebUI did not return a file ID.")
            except MuAPIError:
                raise
            except Exception as exc:
                detail = getattr(exc, "detail", None) or exc
                log.warning("Storing MuAPI output %s failed: %s", index, detail)
                raise MuAPIError(
                    f"Saving the generated {kind} to Open WebUI failed: "
                    f"{_short(detail) or type(exc).__name__}"
                ) from exc
            stored.append(f"/api/v1/files/{file_id}/content")
            stored_ids.append(file_id)

        if not stored:
            raise MuAPIError(f"MuAPI did not return a {kind} output.")

        if chat_meta and stored_ids:
            # Link the files to the chat message so shared chats can read them.
            try:
                await Chats.insert_chat_files(
                    chat_id=chat_meta["chat_id"],
                    message_id=chat_meta["message_id"],
                    file_ids=stored_ids,
                    user_id=user_obj.id,
                )
            except Exception as exc:
                log.debug("Linking MuAPI files to chat failed: %s", exc)
        return stored

    async def _store_or_fallback(
        self,
        outputs: List[str],
        kind: str,
        config: Dict[str, Any],
        request: Any,
        user_obj: Any,
        metadata: Optional[Dict[str, Any]],
        emitter: EventEmitter,
    ) -> List[str]:
        try:
            return await self._save_outputs(
                outputs, kind, config, request, user_obj, metadata, emitter
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if not config["allow_provider_url_fallback"]:
                raise MuAPIError(str(exc) or f"Saving the generated {kind} failed.") from exc
            log.warning("Falling back to MuAPI output URLs: %s", exc)
            await self._emit_status(
                emitter,
                f"Could not save the {kind} to Open WebUI; showing the MuAPI link instead",
                done=False,
            )
            return list(outputs)

    # ------------------------------------------------------------------
    # Output rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_viewer_html(kind: str, urls: List[str], prompt: str) -> str:
        """Inline viewer matching the repo's image/video embeds, with token-auth blob loading."""
        if kind == "image":
            media = "".join(
                '<div class="media-wrapper"><img data-muapi-src="{}" alt="Generated image" /></div>'.format(
                    html.escape(url, quote=True)
                )
                for url in urls
            )
        else:
            media = "".join(
                '<div class="media-wrapper"><video controls playsinline preload="metadata" '
                'data-muapi-src="{}"></video></div>'.format(html.escape(url, quote=True))
                for url in urls
            )
        safe_prompt = html.escape(prompt or "", quote=True)
        return f"""<!DOCTYPE html>
<html style="margin:0; padding:0; overflow:hidden;">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .container {{ margin: 0; padding: 0; border: none; }}
        .media-wrapper {{
            margin: 0 0 8px 0;
            padding: 0;
            border-radius: 12px;
            overflow: hidden;
            line-height: 0;
        }}
        .media-wrapper img, .media-wrapper video {{
            max-width: 100%;
            width: 100%;
            height: auto;
            display: block;
            border: none;
            background: #000;
        }}
        .media-wrapper img {{ background: transparent; }}
        .prompt-bubble {{
            background: rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0 0 0;
            font-size: 13px;
            line-height: 1.5;
            color: #333;
            word-wrap: break-word;
        }}
        .prompt-label {{
            font-weight: 600;
            color: #666;
            margin-bottom: 4px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .prompt-text {{ color: #444; }}
    </style>
</head>
<body>
    <div class="container">
        {media}
        <div class="prompt-bubble">
            <div class="prompt-label">Prompt</div>
            <div class="prompt-text">{safe_prompt}</div>
        </div>
    </div>
    <script>
    (function() {{
      function authFetch(url) {{
        var opts = {{ credentials: 'include' }};
        try {{
          var token = localStorage.getItem('token');
          if (token) {{ opts.headers = {{ 'Authorization': 'Bearer ' + token }}; }}
        }} catch (e) {{}}
        return fetch(url, opts);
      }}
      function fetchBlobUrl(url, callback) {{
        // Only attach the Open WebUI token to same-origin relative /api/ paths;
        // provider URLs (fallback mode) must never receive it.
        if (!url || url.indexOf('/api/') !== 0) {{ callback(url); return; }}
        authFetch(url).then(function(resp) {{
          if (!resp.ok) throw new Error('HTTP ' + resp.status);
          return resp.blob();
        }}).then(function(blob) {{
          callback(URL.createObjectURL(blob));
        }}).catch(function() {{ callback(url); }});
      }}
      var nodes = document.querySelectorAll('[data-muapi-src]');
      for (var i = 0; i < nodes.length; i++) {{
        (function(node) {{
          fetchBlobUrl(node.getAttribute('data-muapi-src'), function(src) {{ node.src = src; }});
        }})(nodes[i]);
      }}
    }})();
    </script>
</body>
</html>"""

    def _format_outputs(
        self,
        kind: str,
        urls: List[str],
        config: Dict[str, Any],
        prompt: str,
        message: str,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        label = "Image" if kind == "image" else "Video"
        links = "\n".join(f"- {label} {i + 1}: {url}" for i, url in enumerate(urls))
        if not config["return_html_embed"]:
            return f"{message}\n\n{links}"

        context = (
            f"{message} The {kind} is displayed above in the chat; do not repeat it "
            f"as markdown.\n\n{links}"
        )
        return (
            HTMLResponse(
                content=self._build_viewer_html(kind, urls, prompt),
                headers={"Content-Disposition": "inline"},
            ),
            context,
        )

    @staticmethod
    async def _resolve_user(__user__: Optional[Dict[str, Any]]) -> Any:
        if not __user__ or not isinstance(__user__, dict) or not __user__.get("id"):
            return None
        try:
            return await Users.get_user_by_id(__user__["id"])
        except Exception as exc:
            log.debug("Could not resolve Open WebUI user: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate an image from a text prompt with MuAPI. The image is shown to the user automatically.

        :param prompt: Detailed description of the image to create.
        :param aspect_ratio: Optional aspect ratio such as "1:1", "16:9", "9:16", "4:3" or "3:4".
        :param resolution: Optional output resolution: "1k", "2k" or "4k".
        """
        config = self._resolve_config(__user__)
        try:
            prompt_value = self._validate_prompt(prompt)
            request_payload: Dict[str, Any] = {"prompt": prompt_value}
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
            urls = await self._store_or_fallback(
                outputs,
                "image",
                config,
                __request__,
                await self._resolve_user(__user__),
                __metadata__,
                __event_emitter__,
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            detail = _short(exc) if str(exc) else type(exc).__name__
            await self._emit_status(__event_emitter__, f"MuAPI error: {detail}", done=True)
            return f"MuAPI image generation failed: {detail}"

        await self._emit_status(
            __event_emitter__, "MuAPI image generated", done=True
        )
        return self._format_outputs(
            "image",
            urls,
            config,
            prompt_value,
            "Image generated successfully with MuAPI and displayed above.",
        )

    async def generate_video(
        self,
        prompt: str,
        duration: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        seed: Optional[int] = None,
        high_bitrate: Optional[bool] = None,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate a video from a text prompt with MuAPI. The video is shown to the user automatically.

        :param prompt: Description of the scene, motion and style of the video.
        :param duration: Optional length in seconds, an integer from 4 through 30.
        :param aspect_ratio: Optional aspect ratio such as "16:9", "9:16" or "1:1".
        :param seed: Optional integer seed for reproducible results.
        :param high_bitrate: Optional; true for higher visual fidelity and larger files.
        """
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
            urls = await self._store_or_fallback(
                outputs,
                "video",
                config,
                __request__,
                await self._resolve_user(__user__),
                __metadata__,
                __event_emitter__,
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            detail = _short(exc) if str(exc) else type(exc).__name__
            await self._emit_status(__event_emitter__, f"MuAPI error: {detail}", done=True)
            return f"MuAPI video generation failed: {detail}"

        await self._emit_status(
            __event_emitter__, "MuAPI video generated", done=True
        )
        return self._format_outputs(
            "video",
            urls,
            config,
            prompt_value,
            "Video generated successfully with MuAPI.",
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
        __request__: Optional[Any] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Animate an image into a video with MuAPI. Uses the image attached to the chat when image_url is omitted.

        :param prompt: Description of the motion and style to apply to the image.
        :param image_url: Optional public HTTP(S) image URL. Leave empty to use the image attached in the chat.
        :param duration: Optional length in seconds, an integer from 4 through 30.
        :param aspect_ratio: Optional aspect ratio such as "16:9", "9:16" or "1:1".
        :param seed: Optional integer seed for reproducible results.
        :param high_bitrate: Optional; true for higher visual fidelity and larger files.
        """
        config = self._resolve_config(__user__)
        try:
            prompt_value = self._validate_prompt(prompt)
            self._validate_video_options(duration, aspect_ratio, None)
            if not config["api_key"]:
                raise MuAPIError(
                    "MuAPI API key is not configured. Set MUAPI_API_KEY in tool Valves or User Valves."
                )
            user_obj = await self._resolve_user(__user__)
            source_image = image_url or self._extract_image_from_messages(__messages__)
            if source_image and not source_image.strip().startswith(("http://", "https://")):
                await self._emit_status(
                    __event_emitter__, "Uploading attached image to MuAPI", done=False
                )
            target_image = await self._resolve_image_input(
                source_image or "", config, user_obj
            )

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
            urls = await self._store_or_fallback(
                outputs,
                "video",
                config,
                __request__,
                user_obj,
                __metadata__,
                __event_emitter__,
            )
        except (MuAPIError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            detail = _short(exc) if str(exc) else type(exc).__name__
            await self._emit_status(__event_emitter__, f"MuAPI error: {detail}", done=True)
            return f"MuAPI image-to-video generation failed: {detail}"

        await self._emit_status(
            __event_emitter__, "MuAPI image-to-video generation complete", done=True
        )
        return self._format_outputs(
            "video",
            urls,
            config,
            prompt_value,
            "Video generated from image successfully with MuAPI.",
        )
