"""
title: ComfyUI VibeVoice TTS Generator
description: Tool to generate speech using VibeVoice workflows via ComfyUI API. Supports both single speaker and multi-speaker (up to 4) voice cloning. The voice files must be pre-loaded in ComfyUI.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.2.0
required_open_webui_version: 0.9.0
"""

import asyncio
import html
import io
import json
import logging
import mimetypes
import os
import random
import re
import uuid
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from urllib.parse import urlencode

import aiohttp
from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers

from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import upload_file_handler

# Default workflow for single speaker TTS
DEFAULT_SINGLE_SPEAKER_WORKFLOW = json.dumps(
    {
        "15": {
            "inputs": {"audio": "Voice.mp3", "audioUI": ""},
            "class_type": "LoadAudio",
            "_meta": {"title": "CargarAudio"},
        },
        "16": {
            "inputs": {
                "filename_prefix": "audio/ComfyUI",
                "quality": "V0",
                "audioUI": "",
                "audio": ["45", 0],
            },
            "class_type": "SaveAudioMP3",
            "_meta": {"title": "Save Audio (MP3)"},
        },
        "44": {
            "inputs": {
                "text": "Hello, this is a test of the VibeVoice text-to-speech system.",
                "model": "VibeVoice-1.5B",
                "attention_type": "auto",
                "quantize_llm": "full precision",
                "free_memory_after_generate": True,
                "diffusion_steps": 20,
                "seed": 42,
                "cfg_scale": 1.3,
                "use_sampling": False,
                "temperature": 0.95,
                "top_p": 0.95,
                "max_words_per_chunk": 250,
                "voice_speed_factor": 1,
            },
            "class_type": "VibeVoiceSingleSpeakerNode",
            "_meta": {"title": "VibeVoice Single Speaker"},
        },
        "45": {
            "inputs": {"anything": ["44", 0]},
            "class_type": "easy cleanGpuUsed",
            "_meta": {"title": "Clean VRAM Used"},
        },
    }
)

# Default workflow for multi-speaker TTS
DEFAULT_MULTI_SPEAKER_WORKFLOW = json.dumps(
    {
        "15": {
            "inputs": {"audio": "Voice1.mp3", "audioUI": ""},
            "class_type": "LoadAudio",
            "_meta": {"title": "CargarAudio"},
        },
        "16": {
            "inputs": {
                "filename_prefix": "audio/ComfyUI",
                "quality": "V0",
                "audioUI": "",
                "audio": ["39", 0],
            },
            "class_type": "SaveAudioMP3",
            "_meta": {"title": "Save Audio (MP3)"},
        },
        "17": {
            "inputs": {"audio": "Voice2.mp3", "audioUI": ""},
            "class_type": "LoadAudio",
            "_meta": {"title": "CargarAudio"},
        },
        "36": {
            "inputs": {
                "text": "[1]: Hello, this is the first speaker.\n[2]: Hi there, I'm the second speaker.\n[1]: Nice to meet you!\n[2]: Nice to meet you too!",
                "model": "VibeVoice-1.5B",
                "attention_type": "auto",
                "quantize_llm": "full precision",
                "free_memory_after_generate": True,
                "diffusion_steps": 20,
                "seed": 42,
                "cfg_scale": 1.3,
                "use_sampling": False,
                "temperature": 0.95,
                "top_p": 0.95,
                "voice_speed_factor": 1,
            },
            "class_type": "VibeVoiceMultipleSpeakersNode",
            "_meta": {"title": "VibeVoice Multiple Speakers"},
        },
        "39": {
            "inputs": {"anything": ["36", 0]},
            "class_type": "easy cleanGpuUsed",
            "_meta": {"title": "Clean VRAM Used"},
        },
    }
)


async def wait_for_completion_ws(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_id: str,
    client_id: str,
    max_wait_time: int,
    event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Waits for ComfyUI job completion using WebSocket for real-time updates.
    Returns the job output data upon successful execution.
    """
    start_time = asyncio.get_event_loop().time()
    job_data_output = None

    try:
        async with aiohttp.ClientSession(headers=headers).ws_connect(
            f"{comfyui_ws_url}?clientId={client_id}"
        ) as ws:
            async for msg in ws:
                if asyncio.get_event_loop().time() - start_time > max_wait_time:
                    raise TimeoutError(f"Job timed out after {max_wait_time}s")

                if msg.type == aiohttp.WSMsgType.TEXT:
                    message = json.loads(msg.data)
                    msg_type = message.get("type")

                    if msg_type == "executing":
                        data = message.get("data", {})
                        node = data.get("node")
                        if node is None and data.get("prompt_id") == prompt_id:
                            # Job completed
                            if event_emitter:
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "Generation completed, retrieving audio..."
                                        },
                                    }
                                )

                            # Fetch final job data
                            async with aiohttp.ClientSession(headers=headers) as session:
                                async with session.get(
                                    f"{comfyui_http_url}/history/{prompt_id}"
                                ) as resp:
                                    if resp.status == 200:
                                        history = await resp.json()
                                        job_data_output = history.get(prompt_id, {})
                            break

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

    except asyncio.TimeoutError:
        raise TimeoutError(f"Overall wait/connection timed out after {max_wait_time}s")
    except aiohttp.ClientError:
        raise  # handled by the caller without echoing the ComfyUI host
    except Exception as e:
        raise Exception(f"Error during WebSocket communication: {e}")

    if job_data_output is None:
        raise Exception("No job data received from ComfyUI")

    return job_data_output


def extract_audio_files(job_data: Dict[str, Any]) -> list[Dict[str, str]]:
    """Extract audio file paths from completed job data."""
    audio_files: list[Dict[str, str]] = []
    node_outputs_dict = cast(Dict[str, Any], job_data.get("outputs", {}))
    for _node_id, node_output_content in node_outputs_dict.items():
        if isinstance(node_output_content, dict):
            node_output_dict: Dict[str, Any] = cast(Dict[str, Any], node_output_content)
            for key_holding_files in [
                "audio",
                "files",
                "filenames",
                "output",
                "outputs",
                "audios",
            ]:
                if key_holding_files in node_output_dict:
                    potential_files_raw: Any = node_output_dict.get(key_holding_files)
                    if isinstance(potential_files_raw, list):
                        potential_files_list: list[Union[Dict[str, Any], str]] = cast(
                            list[Union[Dict[str, Any], str]], potential_files_raw
                        )
                        for file_info_item in potential_files_list:
                            filename = None
                            subfolder = ""
                            if isinstance(file_info_item, dict):
                                file_info_dict: Dict[str, Any] = file_info_item
                                fn_val: Any = file_info_dict.get("filename")
                                filename = fn_val if isinstance(fn_val, str) else None
                                subfolder_val: Any = file_info_dict.get("subfolder", "")
                                subfolder = (
                                    str(subfolder_val)
                                    if subfolder_val is not None
                                    else ""
                                )
                            else:
                                # Treat any non-dict entry as a filename string
                                filename = str(file_info_item)

                            if filename is not None and filename.lower().endswith(
                                (".wav", ".mp3", ".flac", ".ogg")
                            ):
                                audio_files.append(
                                    {
                                        "filename": filename,
                                        "subfolder": subfolder.strip("/"),
                                    }
                                )
    return audio_files


# ==========================================================================
# Open WebUI storage helpers (canonical copy, Open WebUI 0.9+ / 0.12.5)
# ==========================================================================

log = logging.getLogger(__name__)

# Mirrors open_webui.utils.chat_id (v0.12.5). Kept local so the tool also
# imports on older servers that do not ship that module.
TEMPORARY_CHAT_ID_PREFIXES = ("temporary:", "local:")
CHANNEL_CHAT_ID_PREFIX = "channel:"
NON_SAVED_CHAT_ID_PREFIXES = (*TEMPORARY_CHAT_ID_PREFIXES, CHANNEL_CHAT_ID_PREFIX)

# Preferred extensions (mimetypes.guess_extension gives odd ones, e.g. .mpga).
MEDIA_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "image/svg+xml": ".svg",
}

TRANSIENT_HTTP_STATUSES = frozenset({429, 502, 503, 504})


class MediaStorageError(Exception):
    """Saving/downloading media failed. ``str(exc)`` is safe to show the user
    and the LLM: it never contains provider URLs, keys or response bodies."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def is_saved_chat_id(chat_id: Any) -> bool:
    """True for a persisted chat (not temporary/local/channel)."""
    return isinstance(chat_id, str) and bool(chat_id) and not chat_id.startswith(
        NON_SAVED_CHAT_ID_PREFIXES
    )


def normalize_content_type(content_type: Optional[str], default: str = "application/octet-stream") -> str:
    """'Audio/MPEG; charset=x' -> 'audio/mpeg'."""
    value = (content_type or "").split(";", 1)[0].strip().lower()
    return value if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", value or "") else default


def extension_for(content_type: str) -> str:
    ct = normalize_content_type(content_type)
    return MEDIA_EXTENSIONS.get(ct) or mimetypes.guess_extension(ct) or ".bin"


def safe_filename(name: Optional[str], content_type: str, prefix: str = "media") -> str:
    """Basename only, conservative charset, extension that matches the type."""
    ext = extension_for(content_type)
    base = os.path.basename(str(name or "")).strip()
    stem, given_ext = os.path.splitext(base)
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")[:80]
    if not stem:
        clean_prefix = re.sub(r"[^\w-]+", "_", str(prefix or "")).strip("_") or "media"
        stem = f"{clean_prefix}_{uuid.uuid4().hex[:8]}"
    given_ext = given_ext.lower()
    if given_ext and re.fullmatch(r"\.[a-z0-9]{1,5}", given_ext) and (
        mimetypes.guess_type(f"x{given_ext}")[0] == normalize_content_type(content_type)
    ):
        ext = given_ext
    return f"{stem}{ext}"


async def resolve_user(user: Any) -> Optional[UserModel]:
    """Accept the ``__user__`` dict (or a UserModel) and return a UserModel."""
    if isinstance(user, UserModel):
        return user
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not user_id:
        return None
    try:
        return await Users.get_user_by_id(user_id)
    except Exception as exc:  # DB hiccup -> treated as "no user"
        log.warning("Could not resolve Open WebUI user (%s)", type(exc).__name__)
        return None


def chat_context(
    metadata: Optional[dict],
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """(chat_id, message_id) from ``__metadata__`` or ``__chat_id__``/``__message_id__``."""
    meta = metadata if isinstance(metadata, dict) else {}
    cid = chat_id or meta.get("chat_id")
    mid = message_id or meta.get("message_id")
    return (cid if isinstance(cid, str) and cid else None, mid if isinstance(mid, str) and mid else None)


def _upload_metadata(chat_id: Optional[str], message_id: Optional[str], extra: Optional[dict]) -> dict:
    meta: Dict[str, Any] = {}
    if isinstance(extra, dict):
        # Never let callers smuggle routing keys; we set them below.
        meta.update({k: v for k, v in extra.items() if k not in ("chat_id", "message_id", "channel_id", "file_hash")})
    if chat_id and chat_id.startswith(CHANNEL_CHAT_ID_PREFIX):
        channel_id = chat_id[len(CHANNEL_CHAT_ID_PREFIX):]
        if channel_id:
            meta["channel_id"] = channel_id  # upload_file_handler adds file to channel
    elif chat_id:
        meta["chat_id"] = chat_id
    if message_id:
        meta["message_id"] = message_id
    return meta


def _file_id(file_item: Any) -> Optional[str]:
    if isinstance(file_item, dict):  # process=True returns {'status': True, **model_dump()}
        value = file_item.get("id")
    else:
        value = getattr(file_item, "id", None)
    return value if isinstance(value, str) and value else None


async def save_media_bytes(
    data: bytes,
    *,
    request: Any,
    user: Any,
    content_type: str,
    filename: Optional[str] = None,
    metadata: Optional[dict] = None,
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    prefix: str = "media",
    allow_fallback: bool = False,
    fallback_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Store ``data`` in Open WebUI and link it to the chat message.

    :param request: the tool's ``__request__`` (required).
    :param user: ``__user__`` dict or a UserModel (required).
    :param content_type: real MIME type of ``data`` (e.g. "audio/mpeg").
    :param metadata: the tool's ``__metadata__`` (chat_id/message_id read from it).
    :param allow_fallback/fallback_url: explicit opt-in only. When both are set
        and saving fails, the provider URL is returned instead of raising.
    :returns: {"id", "url" ("/api/v1/files/{id}/content"), "name",
        "content_type", "size", "linked", "fallback"}
    :raises MediaStorageError: on any failure (unless the opt-in fallback applies).
    """
    try:
        return await _save_media_bytes(
            data,
            request=request,
            user=user,
            content_type=content_type,
            filename=filename,
            metadata=metadata,
            chat_id=chat_id,
            message_id=message_id,
            extra_metadata=extra_metadata,
            prefix=prefix,
        )
    except MediaStorageError as exc:
        if allow_fallback and is_http_url(fallback_url):
            log.warning("Media storage failed, using opt-in provider URL fallback: %s", exc)
            return {
                "id": None,
                "url": fallback_url,
                "name": None,
                "content_type": normalize_content_type(content_type),
                "size": len(data) if isinstance(data, (bytes, bytearray)) else 0,
                "linked": False,
                "fallback": True,
                "error": str(exc),
            }
        raise


async def _save_media_bytes(
    data: bytes,
    *,
    request: Any,
    user: Any,
    content_type: str,
    filename: Optional[str],
    metadata: Optional[dict],
    chat_id: Optional[str],
    message_id: Optional[str],
    extra_metadata: Optional[dict],
    prefix: str,
) -> Dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise MediaStorageError("The generated file is empty.")
    if request is None:
        raise MediaStorageError("Open WebUI request context is unavailable.")
    user_model = await resolve_user(user)
    if user_model is None:
        raise MediaStorageError("The requesting Open WebUI user could not be resolved.")

    ct = normalize_content_type(content_type)
    name = safe_filename(filename, ct, prefix)
    cid, mid = chat_context(metadata, chat_id, message_id)
    upload = UploadFile(
        file=io.BytesIO(bytes(data)),
        filename=name,
        headers=Headers({"content-type": ct}),
    )
    try:
        file_item = await upload_file_handler(
            request,
            file=upload,
            metadata=_upload_metadata(cid, mid, extra_metadata),
            process=False,
            user=user_model,
        )
    except HTTPException as exc:
        if exc.status_code == 413:
            detail = exc.detail if isinstance(exc.detail, str) else ""
            limit = re.search(r"less than ([0-9.]+ ?MB)", detail or "")
            raise MediaStorageError(
                "The generated file ({:.1f} MB) exceeds Open WebUI's upload size limit{}. "
                "Ask an admin to raise Admin Settings > Documents > Max Upload Size.".format(
                    len(data) / (1024 * 1024), f" ({limit.group(1)})" if limit else ""
                ),
                status_code=413,
            ) from None
        raise MediaStorageError(
            f"Open WebUI rejected the file (HTTP {exc.status_code}).", status_code=exc.status_code
        ) from None
    except Exception as exc:  # storage backend / DB error: deterministic here, no retry
        log.exception("upload_file_handler failed")
        raise MediaStorageError(f"Saving the file to Open WebUI failed ({type(exc).__name__}).") from None

    file_id = _file_id(file_item)
    if not file_id:
        raise MediaStorageError("Open WebUI did not return a file ID.")

    linked = False
    if is_saved_chat_id(cid) and mid:
        # Same as upload_image: makes the file part of the chat (shared-chat access).
        try:
            linked = bool(
                await Chats.insert_chat_files(
                    chat_id=cid, message_id=mid, file_ids=[file_id], user_id=user_model.id
                )
            )
        except Exception as exc:  # not fatal: the owner can still read the file
            log.warning("Could not link file to chat (%s)", type(exc).__name__)

    meta = (file_item.get("meta") if isinstance(file_item, dict) else getattr(file_item, "meta", None)) or {}
    return {
        "id": file_id,
        "url": f"/api/v1/files/{file_id}/content",
        "name": meta.get("name") or name,
        "content_type": meta.get("content_type") or ct,
        "size": meta.get("size") or len(data),
        "linked": linked,
        "fallback": False,
    }


def is_http_url(url: Any) -> bool:
    return isinstance(url, str) and re.match(r"^https?://[^\s/]+", url, re.I) is not None


async def fetch_media_bytes(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 120.0,
    max_bytes: Optional[int] = 512 * 1024 * 1024,
    retries: int = 2,
    retry_delay: float = 2.0,
    min_bytes: int = 1,
    ssl: Any = None,
    allow_redirects: bool = True,
) -> Tuple[bytes, str]:
    """GET a provider/backend output. Returns (bytes, normalized content_type).

    Retries ONLY transient network failures (connection errors, timeouts,
    HTTP 429/502/503/504) up to ``retries`` extra attempts. Other HTTP errors
    (404, 401, ...) and oversize bodies fail immediately. Error messages never
    contain the URL (signed URLs / keys) or the response body.

    aiohttp drops Authorization/Cookie on cross-origin redirects but NOT custom
    key headers (e.g. "x-api-key"): pass allow_redirects=False when ``headers``
    carry such a key and the output URL may redirect to another host.
    """
    if not is_http_url(url):
        raise MediaStorageError("The provider returned an invalid output URL.")
    attempts = max(0, int(retries)) + 1
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        transient = False
        try:
            own = session is None
            sess = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
            try:
                async with sess.get(url, headers=headers, ssl=ssl, allow_redirects=allow_redirects) as resp:
                    if resp.status != 200:
                        transient = resp.status in TRANSIENT_HTTP_STATUSES
                        last_error = f"HTTP {resp.status}"
                        if not transient:
                            raise MediaStorageError(
                                f"Downloading the generated file failed ({last_error}).", status_code=resp.status
                            )
                    else:
                        if max_bytes and resp.content_length and resp.content_length > max_bytes:
                            raise MediaStorageError("The generated file is too large to download.")
                        buf = bytearray()
                        async for chunk in resp.content.iter_chunked(1 << 16):
                            buf.extend(chunk)
                            if max_bytes and len(buf) > max_bytes:
                                raise MediaStorageError("The generated file is too large to download.")
                        if len(buf) < max(1, min_bytes):
                            # Output not ready yet (some backends 200 with an empty body).
                            transient = True
                            last_error = "empty response"
                        else:
                            return bytes(buf), normalize_content_type(resp.headers.get("Content-Type"))
            finally:
                if own:
                    await sess.close()
        except MediaStorageError:
            raise
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as exc:
            transient = True
            last_error = type(exc).__name__
        except aiohttp.ClientError as exc:  # e.g. InvalidURL, TooManyRedirects
            raise MediaStorageError(f"Downloading the generated file failed ({type(exc).__name__}).") from None
        if transient and attempt < attempts:
            log.info("Transient download failure (%s), retry %d/%d", last_error, attempt, attempts - 1)
            await asyncio.sleep(retry_delay * attempt)
            continue
        break
    raise MediaStorageError(f"Downloading the generated file failed ({last_error}).")


async def download_to_storage(
    url: str,
    *,
    request: Any,
    user: Any,
    metadata: Optional[dict] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 120.0,
    retries: int = 2,
    prefix: str = "media",
    allow_fallback: bool = False,
    ssl: Any = None,
    allow_redirects: bool = True,
) -> Dict[str, Any]:
    """fetch_media_bytes + save_media_bytes. ``content_type`` overrides the
    response header (use it when the backend sends application/octet-stream).
    With ``allow_fallback=True`` (explicit admin opt-in) a failed download or
    save returns ``url`` itself with ``fallback=True`` instead of raising."""
    try:
        data, served_type = await fetch_media_bytes(
            url, headers=headers, session=session, timeout=timeout, retries=retries, ssl=ssl,
            allow_redirects=allow_redirects,
        )
    except MediaStorageError as exc:
        if allow_fallback:
            log.warning("Download failed, using opt-in provider URL fallback: %s", exc)
            return {"id": None, "url": url, "name": None, "content_type": content_type,
                    "size": 0, "linked": False, "fallback": True, "error": str(exc)}
        raise
    ct = content_type or served_type
    if ct == "application/octet-stream" and filename:
        ct = mimetypes.guess_type(filename)[0] or ct
    return await save_media_bytes(
        data,
        request=request,
        user=user,
        content_type=ct,
        filename=filename,
        metadata=metadata,
        prefix=prefix,
        allow_fallback=allow_fallback,
        fallback_url=url,
    )


# ==========================================================================
# Inline embed helpers (canonical copy): escaping, shared media runtime
# ==========================================================================

def esc(value: Any) -> str:
    """HTML-escape any text for element content or a quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


def script_json(value: Any) -> str:
    """JSON for inclusion inside <script>: '<', '>', '&' and U+2028/9 escaped,
    so '</script>' or '<!--' in data can never terminate the block."""
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )



# --------------------------------------------------------------------------
# Shared runtime (plain string: no f-string brace escaping, data via JSON)
# --------------------------------------------------------------------------

_RUNTIME_JS = r"""
(function () {
  var M = window.OWUIMedia = {};
  // Same-origin relative Open WebUI API path ("/api/..."), never "//host" or "/\host".
  M.isLocalApi = function (u) {
    return typeof u === 'string' && /^\/api\/[A-Za-z0-9]/.test(u) && u.indexOf('\\') === -1;
  };
  M.isHttp = function (u) { return typeof u === 'string' && /^https?:\/\/[^\s\/]+/i.test(u); };
  M.isSafe = function (u) { return M.isLocalApi(u) || M.isHttp(u); };
  // Document origin. NOT location.origin: for about:srcdoc that is always "null".
  M.origin = (function () {
    try { return typeof self.origin === 'string' ? self.origin : 'null'; } catch (e) { return 'null'; }
  })();
  // allow-same-origin on? (srcdoc then shares the chat page origin; parent is readable)
  M.sameOrigin = (function () {
    try {
      return M.origin !== 'null' && window.parent !== window && window.parent.origin === M.origin &&
        !!window.parent.document;
    } catch (e) { return false; }
  })();
  function resolvesHere(u) {
    try { return new URL(u, document.baseURI).origin === M.origin; } catch (e) { return false; }
  }
  function token() {
    if (!M.sameOrigin) return null;
    try { return window.localStorage.getItem('token') || null; } catch (e) { return null; }
  }
  // Bearer fetch -> blob: URL. Only for same-origin relative /api/ paths.
  M.blobUrl = function (u) {
    if (!M.sameOrigin || !M.isLocalApi(u) || !resolvesHere(u)) {
      return Promise.reject(new Error('not authorized for this URL'));
    }
    var t = token(), opts = { credentials: 'same-origin' };
    if (t) opts.headers = { 'Authorization': 'Bearer ' + t };
    return fetch(u, opts).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.blob();
    }).then(function (b) { return URL.createObjectURL(b); });
  };

  var pending = false;
  M.postHeight = function () {
    if (pending) return;
    pending = true;
    (window.requestAnimationFrame || setTimeout)(function () {
      pending = false;
      var root = document.getElementById('owui-root') || document.body;
      var h = Math.ceil(root.getBoundingClientRect().height +
        (parseFloat(getComputedStyle(document.body).paddingTop) || 0) +
        (parseFloat(getComputedStyle(document.body).paddingBottom) || 0));
      try { window.parent.postMessage({ type: 'iframe:height', height: h }, '*'); } catch (e) {}
    });
  };
  window.addEventListener('load', M.postHeight);
  window.addEventListener('resize', M.postHeight);
  if (window.ResizeObserver) {
    try { new ResizeObserver(M.postHeight).observe(document.getElementById('owui-root') || document.body); } catch (e) {}
  }

  M.fallback = function (box, u, what) {
    if (!box) return;
    box.innerHTML = '';
    box.className += ' owui-fallback';
    var p = document.createElement('div');
    p.className = 'owui-fallback-text';
    p.textContent = 'The ' + (what || 'media') + ' could not be loaded inside this embed.';
    box.appendChild(p);
    if (M.isSafe(u)) {
      var a = document.createElement('a');
      a.href = u; a.target = '_blank'; a.rel = 'noopener noreferrer';
      a.className = 'owui-btn';
      a.textContent = 'Open / download ' + (what || 'file');
      box.appendChild(a);
    }
    if (M.isLocalApi(u) && !M.sameOrigin) {
      var h = document.createElement('div');
      h.className = 'owui-fallback-hint';
      h.textContent = 'To view it inline, enable Settings > Interface > "iframe Sandbox Allow Same Origin".';
      box.appendChild(h);
    }
    M.postHeight();
  };

  // Load url into an <img>/<audio>/<video>. onFail() after every strategy failed.
  M.load = function (el, u, onFail) {
    var tried = false;
    function fail() { if (onFail) onFail(); }
    if (!M.isSafe(u)) { fail(); return; }
    el.addEventListener('error', function () {
      if (tried || !M.isLocalApi(u) || !M.sameOrigin) { fail(); return; }
      tried = true;  // direct src failed (e.g. no cookie) -> Bearer blob
      M.blobUrl(u).then(function (b) { el.dataset.blob = b; el.src = b; }).catch(fail);
    });
    el.addEventListener('load', M.postHeight);
    el.addEventListener('loadedmetadata', M.postHeight);
    el.src = u;  // direct first: streams/range requests, uses the cookie when it is sent
  };

  // Download: same-origin -> blob with Bearer; otherwise a top-level tab (cookie is sent).
  M.download = function (u, name, el) {
    if (!M.isSafe(u)) return;
    function save(href) {
      var a = document.createElement('a');
      a.href = href; a.download = name || ''; a.rel = 'noopener';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }
    function openTab() {
      var a = document.createElement('a');
      a.href = u; a.target = '_blank'; a.rel = 'noopener noreferrer';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }
    if (el && el.dataset && el.dataset.blob) { save(el.dataset.blob); return; }
    if (M.isLocalApi(u) && M.sameOrigin) {
      M.blobUrl(u).then(function (b) { save(b); setTimeout(function () { URL.revokeObjectURL(b); }, 30000); })
        .catch(openTab);
      return;
    }
    openTab();
  };
})();
"""

_BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { background: transparent; overflow: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
.owui-btn {
  display: inline-block; padding: 8px 14px; margin-top: 8px; border-radius: 8px;
  font-size: 12px; font-weight: 600; text-decoration: none; cursor: pointer;
  color: #333; background: rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.12);
}
.owui-btn:hover { background: rgba(0,0,0,0.1); }
.owui-fallback { padding: 14px 16px; border-radius: 12px; line-height: 1.5;
  background: rgba(0,0,0,0.05); border: 1px dashed rgba(0,0,0,0.2); color: #444; font-size: 13px; }
.owui-fallback-hint { margin-top: 8px; font-size: 11px; color: #777; }
.prompt-bubble {
  background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); border-radius: 12px;
  padding: 12px 16px; margin: 8px 0 0 0; font-size: 13px; line-height: 1.5; color: #333;
  word-wrap: break-word; white-space: pre-wrap;
}
.prompt-label { font-weight: 600; color: #666; margin-bottom: 4px; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.5px; white-space: normal; }
.prompt-text { color: #444; }
@media (prefers-color-scheme: dark) {
  .owui-btn { color: #e0e0e0; background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); }
  .owui-btn:hover { background: rgba(255,255,255,0.15); }
  .owui-fallback { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.2); color: #ccc; }
  .owui-fallback-hint { color: #999; }
  .prompt-bubble { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.1); color: #ddd; }
  .prompt-label { color: #aaa; }
  .prompt-text { color: #ccc; }
}
"""


def _page(title: str, css: str, body: str, data: Any, script: str) -> str:
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        f"<title>{esc(title)}</title>\n<style>{_BASE_CSS}{css}</style>\n</head>\n<body>\n"
        f"<div id=\"owui-root\">{body}</div>\n"
        f"<script>{_RUNTIME_JS}</script>\n"
        f"<script>(function (DATA) {{{script}}})({script_json(data)});</script>\n"
        "</body>\n</html>"
    )


# --------------------------------------------------------------------------
# Audio player (minimax_tts / acestep look: dark glass card, custom controls)
# --------------------------------------------------------------------------

_AUDIO_CSS = """
body { display: flex; justify-content: center; padding: 8px 0 12px 0; }
#owui-root { width: 100%; max-width: 440px; }
.player { position: relative; overflow: hidden; border-radius: 18px; padding: 22px; color: #fff;
  background: rgba(20,20,25,0.82); border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 8px 24px rgba(0,0,0,0.22); }
.player.accent { background: linear-gradient(135deg, var(--a0), var(--a1)); }
.player.accent::before { content: ''; position: absolute; inset: 0; background: rgba(0,0,0,0.22); pointer-events: none; }
.player > * { position: relative; }
.hdr { text-align: center; margin-bottom: 16px; }
.kicker { font-size: 9px; font-weight: 700; letter-spacing: 2.2px; text-transform: uppercase; opacity: .65; margin-bottom: 5px; }
.ttl { font-size: 19px; font-weight: 700; letter-spacing: -0.2px; line-height: 1.25; word-wrap: break-word; }
.track { width: 100%; height: 5px; border-radius: 99px; background: rgba(255,255,255,0.2); cursor: pointer; margin-bottom: 8px; }
.fill { height: 100%; width: 0%; border-radius: 99px; background: rgba(255,255,255,0.9); transition: width .1s linear; pointer-events: none; }
.times { display: flex; justify-content: space-between; font-size: 10px; font-weight: 600; opacity: .65;
  font-variant-numeric: tabular-nums; margin-bottom: 14px; }
.ctrls { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.vol { display: flex; align-items: center; gap: 7px; }
.ibtn { width: 30px; height: 30px; border-radius: 50%; border: none; cursor: pointer; color: #fff;
  background: rgba(255,255,255,0.18); display: flex; align-items: center; justify-content: center; padding: 0; }
.ibtn svg { width: 13px; height: 13px; fill: currentColor; pointer-events: none; }
.vtrack { width: 52px; height: 4px; border-radius: 99px; background: rgba(255,255,255,0.22); cursor: pointer; }
.vfill { height: 100%; width: 100%; border-radius: 99px; background: rgba(255,255,255,0.85); pointer-events: none; }
.play { width: 56px; height: 56px; border-radius: 50%; border: none; cursor: pointer; background: #fff; color: #222;
  display: flex; align-items: center; justify-content: center; box-shadow: 0 0 22px rgba(255,255,255,0.35), 0 6px 18px rgba(0,0,0,0.28);
  transition: transform .14s; flex-shrink: 0; }
.play:hover { transform: scale(1.05); }
.play:disabled { opacity: .5; cursor: default; transform: none; }
.play svg { width: 24px; height: 24px; fill: currentColor; pointer-events: none; }
.save { border-radius: 999px; padding: 7px 13px; font-size: 10px; font-weight: 700; letter-spacing: .4px;
  color: #fff; text-decoration: none; background: rgba(255,255,255,0.18); display: flex; align-items: center; gap: 5px; }
.save svg { width: 11px; height: 11px; fill: currentColor; }
.vers { display: none; gap: 7px; flex-wrap: wrap; justify-content: center; margin-top: 16px; }
.vers button { border: none; cursor: pointer; padding: 5px 16px; border-radius: 999px; font-size: 11px; font-weight: 700;
  background: rgba(255,255,255,0.18); color: #fff; }
.vers button.on { background: #fff; color: #222; }
.info { margin-top: 16px; border-radius: 12px; background: rgba(0,0,0,0.22); border: 1px solid rgba(255,255,255,0.1); }
.info-row { padding: 10px 14px; border-top: 1px solid rgba(255,255,255,0.08); }
.info-row:first-child { border-top: none; }
.info-label { font-size: 8px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; opacity: .55; margin-bottom: 5px; }
.info-value { font-size: 12px; line-height: 1.55; opacity: .9; word-wrap: break-word; white-space: pre-wrap; }
.info-value.long { max-height: 160px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.25) transparent; }
.player .owui-fallback { margin-top: 12px; color: #ddd; background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.25); }
.player .owui-fallback-hint { color: #aaa; }
.player .owui-btn { color: #fff; background: rgba(255,255,255,0.15); border-color: rgba(255,255,255,0.2); }
"""

_AUDIO_JS = r"""
var M = window.OWUIMedia, $ = function (id) { return document.getElementById(id); };
var audio = $('audio'), playBtn = $('play'), fill = $('fill'), trackEl = $('track');
var tc = $('tc'), td = $('td'), vtrack = $('vtrack'), vfill = $('vfill'), vbtn = $('vbtn');
var PLAY = '<path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/>';
var PAUSE = '<path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>';
var VOL = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>';
var MUTE = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zM19 12c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.8 8.8 0 0 0 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3 3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a9 9 0 0 0 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4 9.91 6.09 12 8.18V4z"/>';
function fmt(s) { if (!s || !isFinite(s)) return '0:00'; var m = Math.floor(s / 60), x = Math.floor(s % 60); return m + ':' + (x < 10 ? '0' : '') + x; }
function setIcon(p) { playBtn.querySelector('svg').innerHTML = p ? PAUSE : PLAY; }
var cur = 0, failed = false;
function select(i) {
  cur = i; failed = false;
  var box = $('fallback'); box.innerHTML = ''; box.className = ''; playBtn.disabled = false;
  var bs = document.querySelectorAll('.vers button');
  for (var k = 0; k < bs.length; k++) bs[k].className = (k === i ? 'on' : '');
  audio.pause(); setIcon(false); fill.style.width = '0%'; tc.textContent = '0:00'; td.textContent = '0:00';
  delete audio.dataset.blob;
  var u = DATA.tracks[i].url;
  var a2 = audio.cloneNode(false); audio.parentNode.replaceChild(a2, audio); audio = a2; bind();
  M.load(audio, u, function () { failed = true; playBtn.disabled = true; M.fallback(box, u, 'audio'); });
}
function bind() {
  audio.addEventListener('timeupdate', function () {
    fill.style.width = (audio.duration ? audio.currentTime / audio.duration * 100 : 0) + '%';
    tc.textContent = fmt(audio.currentTime);
  });
  audio.addEventListener('loadedmetadata', function () { td.textContent = fmt(audio.duration); });
  audio.addEventListener('durationchange', function () { td.textContent = fmt(audio.duration); });
  audio.addEventListener('ended', function () { setIcon(false); });
  audio.addEventListener('pause', function () { setIcon(false); });
  audio.addEventListener('play', function () { setIcon(true); });
}
playBtn.addEventListener('click', function () {
  if (failed) return;
  if (audio.paused) audio.play().catch(function () {}); else audio.pause();
});
trackEl.addEventListener('click', function (e) {
  var r = trackEl.getBoundingClientRect();
  if (audio.duration) audio.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * audio.duration;
});
vtrack.addEventListener('click', function (e) {
  var r = vtrack.getBoundingClientRect(), p = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  audio.volume = p; audio.muted = false; vfill.style.width = (p * 100) + '%';
  vbtn.querySelector('svg').innerHTML = p === 0 ? MUTE : VOL;
});
vbtn.addEventListener('click', function () {
  audio.muted = !audio.muted;
  vbtn.querySelector('svg').innerHTML = audio.muted ? MUTE : VOL;
  vfill.style.width = audio.muted ? '0%' : (audio.volume * 100) + '%';
});
$('save').addEventListener('click', function (e) {
  e.preventDefault(); var t = DATA.tracks[cur]; M.download(t.url, t.name, audio);
});
if (DATA.tracks.length > 1) {
  var vers = $('vers'); vers.style.display = 'flex';
  DATA.tracks.forEach(function (t, i) {
    var b = document.createElement('button'); b.textContent = t.label || ('v' + (i + 1));
    b.addEventListener('click', function () { select(i); }); vers.appendChild(b);
  });
}
if (DATA.tracks.length) select(0); else { playBtn.disabled = true; M.fallback($('fallback'), '', 'audio'); }
"""


def audio_player_embed(
    tracks: Union[str, Sequence[Union[str, dict]]],
    *,
    title: str = "Generated Audio",
    kicker: str = "Audio",
    info: Optional[Sequence[Tuple[str, Any]]] = None,
    long_text: Optional[Tuple[str, str]] = None,
    download_name: str = "audio.mp3",
    accent: Optional[Tuple[str, str]] = None,
) -> str:
    """Custom audio player.

    :param tracks: one URL, a list of URLs, or dicts {"url", "name"?, "label"?}
        (several tracks -> version buttons v1, v2, ... like the AceStep player).
    :param info: [(label, value)] short rows, e.g. [("Voice", "narrator · speech-2.8")].
    :param long_text: (label, text) scrollable block, e.g. ("Lyrics", lyrics) / ("Text", text).
    :param accent: optional two CSS colors for a gradient card, e.g. ("#6d28d9", "#db2777").
        Validated as #hex; anything else is ignored.
    """
    if isinstance(tracks, str):
        tracks = [tracks]
    items = []
    for i, t in enumerate(tracks or []):
        if isinstance(t, dict):
            url, name, label = t.get("url"), t.get("name"), t.get("label")
        else:
            url, name, label = t, None, None
        if not isinstance(url, str) or not url.strip():
            continue
        if not name:
            stem, dot, ext = download_name.rpartition(".")
            name = download_name if len(tracks) == 1 or not dot else f"{stem}-v{i + 1}.{ext}"
        items.append({"url": url, "name": str(name), "label": str(label) if label else None})

    style = ""
    cls = "player"
    if accent and len(accent) == 2 and all(isinstance(c, str) and _is_hex_color(c) for c in accent):
        cls += " accent"
        style = f' style="--a0:{accent[0]};--a1:{accent[1]}"'

    rows = "".join(
        f'<div class="info-row"><div class="info-label">{esc(k)}</div><div class="info-value">{esc(v)}</div></div>'
        for k, v in (info or [])
        if v not in (None, "")
    )
    if long_text and long_text[1]:
        rows += (
            f'<div class="info-row"><div class="info-label">{esc(long_text[0])}</div>'
            f'<div class="info-value long">{esc(long_text[1])}</div></div>'
        )
    info_html = f'<div class="info">{rows}</div>' if rows else ""

    body = f"""<div class="{cls}"{style}>
  <div class="hdr"><div class="kicker">{esc(kicker)}</div><div class="ttl">{esc(title)}</div></div>
  <audio id="audio" preload="metadata"></audio>
  <div class="track" id="track"><div class="fill" id="fill"></div></div>
  <div class="times"><span id="tc">0:00</span><span id="td">0:00</span></div>
  <div class="ctrls">
    <div class="vol">
      <button class="ibtn" id="vbtn" title="Mute"><svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg></button>
      <div class="vtrack" id="vtrack"><div class="vfill" id="vfill"></div></div>
    </div>
    <button class="play" id="play" title="Play"><svg viewBox="0 0 24 24"><path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/></svg></button>
    <a class="save" id="save" href="#"><svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>Save</a>
  </div>
  <div class="vers" id="vers"></div>
  <div id="fallback"></div>
  {info_html}
</div>"""
    return _page(title, _AUDIO_CSS, body, {"tracks": items}, _AUDIO_JS)


def _is_hex_color(value: str) -> bool:
    v = value.strip()
    return v.startswith("#") and len(v) in (4, 7, 9) and all(c in "0123456789abcdefABCDEF" for c in v[1:])


# --------------------------------------------------------------------------
# Return helper
# --------------------------------------------------------------------------


def embed_result(html_content: str, context: Union[str, dict]):
    """Return value for a tool method: (HTMLResponse inline embed, LLM context).

    Open WebUI (utils/middleware.py process_tool_result) renders the HTML as an
    embed and gives ``context`` (str/dict/list) to the model instead of the HTML.
    """
    from fastapi.responses import HTMLResponse

    return (
        HTMLResponse(content=html_content, headers={"Content-Disposition": "inline"}),
        context,
    )


_AUDIO_EXT_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg"}


def comfyui_view_url(http_url: str, filename: str, subfolder: str = "", file_type: str = "output") -> str:
    """ComfyUI /view URL for an output file (server side only: never shown to the LLM)."""
    params = {"filename": filename, "type": file_type}
    if subfolder:
        params["subfolder"] = subfolder
    return f"{http_url.rstrip('/')}/view?{urlencode(params)}"


def audio_content_type(filename: str, default: str = "audio/mpeg") -> str:
    """Audio MIME type from a ComfyUI output filename (its /view may send a generic type)."""
    return _AUDIO_EXT_TYPES.get(os.path.splitext(filename or "")[1].lower(), default)


async def get_loaded_models_async(
    api_url: str = "http://localhost:11434",
) -> list[Dict[str, Any]]:
    """Get all currently loaded models in VRAM"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url.rstrip('/')}/api/ps") as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return cast(list[Dict[str, Any]], data.get("models", []))
    except Exception as e:
        print(f"Error fetching loaded models: {e}")
        return []


async def unload_all_models_async(api_url: str = "http://localhost:11434") -> bool:
    """Unload all currently loaded models from VRAM"""
    try:
        loaded_models = await get_loaded_models_async(api_url)
        if not loaded_models:
            return True

        async with aiohttp.ClientSession() as session:
            for model in loaded_models:
                model_name = model.get("name", model.get("model", ""))
                if model_name:
                    payload = {"model": model_name, "keep_alive": 0}
                    async with session.post(
                        f"{api_url.rstrip('/')}/api/generate", json=payload
                    ) as resp:
                        pass

        for _ in range(5):
            await asyncio.sleep(1)
            remaining = await get_loaded_models_async(api_url)
            if not remaining:
                print("All models successfully unloaded.")
                return True

        print("Warning: Some models might still be loaded after timeout.")
        return False

    except Exception as e:
        print(f"Error unloading models: {e}")
        return False


def generate_audio_player_embed(
    audio_url: str,
    text: str,
    speaker_info: str,
    generation_type: str = "Single Speaker",
    download_name: str = "speech.wav",
) -> str:
    """The repo's shared audio player (custom controls, Save button, voice + text)."""
    return audio_player_embed(
        audio_url,
        title="VibeVoice TTS",
        kicker=generation_type,
        info=[("Voice Configuration", speaker_info)],
        long_text=("Text", text),
        download_name=download_name,
    )


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188",
            description="ComfyUI HTTP API endpoint.",
        )
        comfyui_api_key: str = Field(
            default="",
            description="API key for ComfyUI authentication (Bearer token). Leave empty if not required.",
            json_schema_extra={"input": {"type": "password"}},
        )
        single_speaker_workflow: str = Field(
            default=DEFAULT_SINGLE_SPEAKER_WORKFLOW,
            description="Full VibeVoice Single Speaker workflow JSON (string).",
        )
        multi_speaker_workflow: str = Field(
            default=DEFAULT_MULTI_SPEAKER_WORKFLOW,
            description="Full VibeVoice Multi-Speaker workflow JSON (string).",
        )
        single_text_node: str = Field(
            default="44",
            description="Node ID for text input in single speaker workflow.",
        )
        single_seed_node: str = Field(
            default="44",
            description="Node ID for seed input in single speaker workflow.",
        )
        multi_text_node: str = Field(
            default="36",
            description="Node ID for text input in multi-speaker workflow.",
        )
        multi_seed_node: str = Field(
            default="36",
            description="Node ID for seed input in multi-speaker workflow.",
        )
        max_wait_time: int = Field(
            default=300,
            description="Max wait time for generation (seconds).",
        )
        unload_ollama_models: bool = Field(
            default=False,
            description="Unload all Ollama models before calling ComfyUI.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL.",
        )
        save_local: bool = Field(
            default=True,
            description=(
                "Copy the generated audio to the Open WebUI storage backend (recommended). If saving fails, "
                "the tool reports an error. Turn off only if users' browsers can reach ComfyUI directly: "
                "the player then streams from ComfyUI's /view URL, which exposes the ComfyUI address "
                "and is not protected by Open WebUI's access control."
            ),
        )

        show_player_embed: bool = Field(
            default=True,
            description="Show the embedded audio player. If false, only returns download link.",
        )

    class UserValves(BaseModel):
        seed: int = Field(
            default=-1,
            description="Seed for audio generation. Use -1 for random seed, or any positive number for reproducible results.",
            ge=-1,
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _deliver_audio(
        self,
        filename: str,
        subfolder: str,
        text: str,
        *,
        speaker_info: str,
        generation_type: str,
        comfyui_http_url: str,
        comfyui_headers: Dict[str, str],
        __user__: Optional[Dict[str, Any]],
        __request__: Optional[Request],
        __metadata__: Optional[Dict[str, Any]],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]],
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Store the ComfyUI output in Open WebUI (chat-linked) and build the reply."""
        view_url = comfyui_view_url(comfyui_http_url, filename, subfolder)
        content_type = audio_content_type(filename, "audio/wav")
        download_name = safe_filename(f"vibevoice{extension_for(content_type)}", content_type, "vibevoice")

        if self.valves.save_local:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Saving to Open WebUI...", "done": False},
                    }
                )
            try:
                saved = await download_to_storage(
                    view_url,
                    request=__request__,
                    user=__user__,
                    metadata=__metadata__,
                    content_type=content_type,
                    headers=comfyui_headers,
                    prefix="vibevoice",
                )
            except MediaStorageError as exc:
                # No fallback to the internal ComfyUI URL.
                error_msg = f"Generated, but could not save the audio to Open WebUI: {exc}"
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": error_msg, "done": True}}
                    )
                return error_msg
            audio_url = saved["url"]
            source_note = ""
        else:
            # Explicit admin opt-out of storage: stream from ComfyUI.
            audio_url = view_url
            source_note = " The audio is streamed from ComfyUI and is not saved in Open WebUI."

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generation complete!", "done": True},
                }
            )

        if self.valves.show_player_embed:
            html_content = generate_audio_player_embed(
                audio_url, text, speaker_info, generation_type, download_name=download_name
            )
            what = "Multi-speaker speech audio" if generation_type == "Multi-Speaker" else "Speech audio"
            return embed_result(
                html_content,
                f"{what} generated and shown to the user in an audio player above. "
                "Do not repeat it as a link or markdown." + source_note,
            )
        if self.valves.save_local:
            # audio_url is the stored /api/v1/files/... link, safe to share.
            return f"Audio generated successfully!\n\nDownload: {audio_url}"
        # Never hand the internal ComfyUI /view URL to the model.
        return (
            "Audio was generated, but it was not saved to Open WebUI (save_local is off) "
            "and the player is disabled (show_player_embed is off), so there is no link to share. "
            "Ask an admin to enable show_player_embed or save_local."
        )

    async def generate_single_speaker_tts(
        self,
        text: str,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate natural, expressive speech from text using voice cloning.

        Creates high-quality, conversational audio that maintains the speaker's identity and prosody.
        Perfect for voiceovers, narration, audiobooks, podcasts, and any content requiring natural-sounding speech.

        Args:
            text: The complete text to convert to speech. Can be any length.

        Example:
            text = "Welcome to today's podcast. We'll be exploring the fascinating world of artificial intelligence and its impact on our daily lives."
        """
        try:
            # Get user valves (Open WebUI passes them in __user__["valves"])
            user_valves = (__user__ or {}).get("valves") or self.UserValves()
            if not isinstance(user_valves, self.UserValves):
                user_valves = self.UserValves(**(user_valves if isinstance(user_valves, dict) else {}))
            # Unload Ollama models if requested
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Unloading Ollama models to free VRAM...",
                                "done": False,
                            },
                        }
                    )
                await unload_all_models_async(self.valves.ollama_url)

            # Load workflow
            workflow_str = self.valves.single_speaker_workflow
            if not workflow_str or workflow_str.strip() == "":
                raise ValueError(
                    "Single speaker workflow JSON is not configured in valves"
                )

            workflow = json.loads(workflow_str)

            # Set text input
            if self.valves.single_text_node in workflow:
                workflow[self.valves.single_text_node]["inputs"]["text"] = text
            else:
                raise ValueError(
                    f"Text node {self.valves.single_text_node} not found in workflow"
                )

            # Set seed (random if -1, otherwise use user's value)
            if user_valves.seed == -1:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = user_valves.seed

            if self.valves.single_seed_node in workflow:
                workflow[self.valves.single_seed_node]["inputs"]["seed"] = seed

            # Submit to ComfyUI
            client_id = str(uuid.uuid4())
            comfyui_http_url = self.valves.comfyui_api_url.rstrip("/")
            comfyui_ws_url = comfyui_http_url.replace("http", "ws", 1) + "/ws"

            comfyui_headers = {}
            if self.valves.comfyui_api_key:
                comfyui_headers["Authorization"] = f"Bearer {self.valves.comfyui_api_key}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generating speech...", "done": False},
                    }
                )

            async with aiohttp.ClientSession(headers=comfyui_headers) as session:
                async with session.post(
                    f"{comfyui_http_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log.warning("ComfyUI /prompt returned HTTP %s: %s", resp.status, error_text[:1000])
                        raise Exception(
                            f"ComfyUI API error: HTTP {resp.status}. Check the workflow and the server logs."
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

            # Wait for completion
            job_data = await wait_for_completion_ws(
                comfyui_ws_url,
                comfyui_http_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
                __event_emitter__,
                headers=comfyui_headers,
            )

            # Extract audio files
            audio_files = extract_audio_files(job_data)
            if not audio_files:
                raise Exception("No audio files generated")

            audio_info = audio_files[0]
            return await self._deliver_audio(
                audio_info.get("filename", ""),
                audio_info.get("subfolder", ""),
                text,
                speaker_info="Single voice clone (pre-loaded in ComfyUI)",
                generation_type="Single Speaker",
                comfyui_http_url=comfyui_http_url,
                comfyui_headers=comfyui_headers,
                __user__=__user__,
                __request__=__request__,
                __metadata__=__metadata__,
                __event_emitter__=__event_emitter__,
            )

        except Exception as e:
            if isinstance(e, aiohttp.ClientError):
                # aiohttp messages carry the ComfyUI host:port; keep them in the log only.
                log.warning("ComfyUI request failed: %s", e)
                detail = f"could not reach ComfyUI ({type(e).__name__})"
            else:
                detail = str(e)
            error_msg = f"Error generating single speaker TTS: {detail}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

    async def generate_multi_speaker_tts(
        self,
        text: str,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
                Generate dynamic multi-speaker conversations with distinct cloned voices.

                Creates natural dialogue between up to 4 different speakers, each with their own voice characteristics.
                Perfect for podcasts, interviews, audiobooks with multiple characters, or any conversational content.

                Args:
                    text: Dialogue script where each line starts with [1], [2], [3], or [4] to indicate the speaker, followed by a colon and their dialogue.

                Example:
                    text = '''[1]: Welcome to the AI Ethics Roundtable. I'm your host Sarah.
        [2]: Thanks for having me, Sarah. I'm excited to discuss this important topic.
        [1]: Let's start with your recent paper on algorithmic bias. Can you tell us more?
        [2]: Absolutely. We found that bias can emerge in unexpected ways during training.
        [1]: That's fascinating. How do you think we should address this?'''

                Note: Each number [1] through [4] represents a different speaker with a unique voice.
        """
        try:
            # Get user valves (Open WebUI passes them in __user__["valves"])
            user_valves = (__user__ or {}).get("valves") or self.UserValves()
            if not isinstance(user_valves, self.UserValves):
                user_valves = self.UserValves(**(user_valves if isinstance(user_valves, dict) else {}))

            # Unload Ollama models if requested
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Unloading Ollama models to free VRAM...",
                                "done": False,
                            },
                        }
                    )
                await unload_all_models_async(self.valves.ollama_url)

            # Load workflow
            workflow_str = self.valves.multi_speaker_workflow
            if not workflow_str or workflow_str.strip() == "":
                raise ValueError(
                    "Multi-speaker workflow JSON is not configured in valves"
                )

            workflow = json.loads(workflow_str)

            # Set text input
            if self.valves.multi_text_node in workflow:
                workflow[self.valves.multi_text_node]["inputs"]["text"] = text
            else:
                raise ValueError(
                    f"Text node {self.valves.multi_text_node} not found in workflow"
                )

            # Set seed (random if -1, otherwise use user's value)
            if user_valves.seed == -1:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = user_valves.seed

            if self.valves.multi_seed_node in workflow:
                workflow[self.valves.multi_seed_node]["inputs"]["seed"] = seed

            # Submit to ComfyUI
            client_id = str(uuid.uuid4())
            comfyui_http_url = self.valves.comfyui_api_url.rstrip("/")
            comfyui_ws_url = comfyui_http_url.replace("http", "ws", 1) + "/ws"

            comfyui_headers = {}
            if self.valves.comfyui_api_key:
                comfyui_headers["Authorization"] = f"Bearer {self.valves.comfyui_api_key}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generating speech...", "done": False},
                    }
                )

            async with aiohttp.ClientSession(headers=comfyui_headers) as session:
                async with session.post(
                    f"{comfyui_http_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log.warning("ComfyUI /prompt returned HTTP %s: %s", resp.status, error_text[:1000])
                        raise Exception(
                            f"ComfyUI API error: HTTP {resp.status}. Check the workflow and the server logs."
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

            # Wait for completion
            job_data = await wait_for_completion_ws(
                comfyui_ws_url,
                comfyui_http_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
                __event_emitter__,
                headers=comfyui_headers,
            )

            # Extract audio files
            audio_files = extract_audio_files(job_data)
            if not audio_files:
                raise Exception("No audio files generated")

            audio_info = audio_files[0]
            return await self._deliver_audio(
                audio_info.get("filename", ""),
                audio_info.get("subfolder", ""),
                text,
                speaker_info="Multiple voice clones (pre-loaded in ComfyUI) - [1], [2], [3], [4]",
                generation_type="Multi-Speaker",
                comfyui_http_url=comfyui_http_url,
                comfyui_headers=comfyui_headers,
                __user__=__user__,
                __request__=__request__,
                __metadata__=__metadata__,
                __event_emitter__=__event_emitter__,
            )

        except Exception as e:
            if isinstance(e, aiohttp.ClientError):
                # aiohttp messages carry the ComfyUI host:port; keep them in the log only.
                log.warning("ComfyUI request failed: %s", e)
                detail = f"could not reach ComfyUI ({type(e).__name__})"
            else:
                detail = str(e)
            error_msg = f"Error generating multi-speaker TTS: {detail}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg
