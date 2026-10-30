"""
title: Atlas Cloud Media Generator
description: Generate images, edit images, create videos (text-to-video, image-to-video, audio-to-video) and 3D models (text-to-3D, image-to-3D as GLB) through the Atlas Cloud Media API. Outputs are stored in Open WebUI file storage and shown in an inline viewer.
author: binyangzhu000-sudo & Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
version: 1.3.0
license: MIT
required_open_webui_version: 0.9.0
"""

import asyncio
import base64
import binascii
import html
import io
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import aiohttp
from fastapi import HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers

from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import upload_file_handler
from open_webui.storage.provider import Storage

log = logging.getLogger(__name__)

IMAGE_ENDPOINT = "/model/generateImage"
VIDEO_ENDPOINT = "/model/generateVideo"
UPLOAD_ENDPOINT = "/model/uploadMedia"

DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-pro/text-to-image"
DEFAULT_IMAGE_EDIT_MODEL = "bytedance/seedream-v5.0-pro/image-to-image"
DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "bytedance/seedance-2.5/image-to-video"
DEFAULT_AUDIO_TO_VIDEO_MODEL = "bytedance/seedance-2.5/reference-to-video"
DEFAULT_TEXT_TO_3D_MODEL = "tencent/hunyuan3d-rapid/text-to-3d"
DEFAULT_IMAGE_TO_3D_MODEL = "tencent/hunyuan3d-rapid/image-to-3d"

# Supported 3D model families. "image_field" is the request field that carries
# the reference image for the image-to-3d variant (it differs per provider).
MODEL_3D_FAMILIES: Dict[str, Dict[str, Any]] = {
    "hunyuan3d-rapid": {
        "label": "Hunyuan3D Rapid",
        "text": "tencent/hunyuan3d-rapid/text-to-3d",
        "image": "tencent/hunyuan3d-rapid/image-to-3d",
        "image_field": "image",
        "cost": "economy (about $0.02 per model)",
        "max_prompt": None,
    },
    "hunyuan3d-pro": {
        "label": "Hunyuan3D Pro",
        "text": "tencent/hunyuan3d-pro/text-to-3d",
        "image": "tencent/hunyuan3d-pro/image-to-3d",
        "image_field": "image",
        "cost": "economy (about $0.02 per model)",
        "max_prompt": None,
    },
    "tripo-h3.1": {
        "label": "Tripo H3.1",
        "text": "tripo-h3.1/text-to-3d",
        "image": "tripo-h3.1/image-to-3d",
        "image_field": "image_url",
        "cost": "premium (about $0.10 per model)",
        "max_prompt": 1024,
    },
    "meshy-v7": {
        "label": "Meshy v7",
        "text": "meshy-v7/text-to-3d",
        "image": "meshy-v7/image-to-3d",
        "image_field": "image",
        "cost": "economy (about $0.02 per model)",
        "max_prompt": 800,
    },
}
# model id -> (family, "text" | "image")
SUPPORTED_3D_MODELS: Dict[str, Tuple[str, str]] = {
    spec[mode]: (family, mode)
    for family, spec in MODEL_3D_FAMILIES.items()
    for mode in ("text", "image")
}
GLB_CONTENT_TYPE = "model/gltf-binary"
GLB_MAGIC = b"glTF"
MAX_3D_MODEL_BYTES = 200 * 1024 * 1024
PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
# Google <model-viewer> (Apache-2.0), pinned exact version + SRI (jsDelivr file hash).
MODEL_VIEWER_VERSION = "4.4.0"
MODEL_VIEWER_MODULE_URL = (
    f"https://cdn.jsdelivr.net/npm/@google/model-viewer@{MODEL_VIEWER_VERSION}/dist/model-viewer.min.js"
)
MODEL_VIEWER_INTEGRITY = "sha256-GLLnEQNapHBe4oFTClXcIzj/Be7T1wrhbVQQUoKnXtU="

COMPLETED_STATUSES = frozenset({"completed", "succeeded", "success"})
FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled", "timeout"})
# Poll GETs that hit a gateway hiccup are retried on the next tick instead of
# aborting a (paid) generation; only this many failures in a row give up.
TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_CONSECUTIVE_POLL_ERRORS = 5

MAX_ERROR_DETAIL_CHARS = 200
# Reference media read from Open WebUI / data: URIs before re-uploading to Atlas Cloud.
MAX_INPUT_MEDIA_BYTES = 50 * 1024 * 1024
# Relative (or same-path absolute) Open WebUI file references: /api/v1/files/{id}[/content]
OWUI_FILE_PATH_RE = re.compile(r"^/api/v1/files/([A-Za-z0-9-]+)(?:/content)?/?$")
DATA_URI_RE = re.compile(r"^data:([A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+)((?:;[^,;]*)*),(.*)$", re.S)
# Output types we are willing to store and serve from Open WebUI.
ALLOWED_OUTPUT_TYPES = {
    "image": frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"}),
    "video": frozenset({"video/mp4", "video/webm", "video/quicktime"}),
}
OUTPUT_FORMAT_TYPES = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
INPUT_TYPE_ALIASES = {"audio/mp3": "audio/mpeg", "audio/x-wav": "audio/wav", "image/jpg": "image/jpeg"}

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


class AtlasCloudError(RuntimeError):
    """Raised when Atlas Cloud rejects or fails a generation request."""


class AtlasCloudTransientError(AtlasCloudError):
    """A poll request failed in a way that is worth retrying (5xx/429, non-JSON)."""


def _short(value: Any, limit: int = MAX_ERROR_DETAIL_CHARS) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Open WebUI media storage helpers (canonical, v0.12.5)
# ---------------------------------------------------------------------------

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
    "model/gltf-binary": ".glb",
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



# ---------------------------------------------------------------------------
# Inline embed builders (canonical, v0.12.5)
# ---------------------------------------------------------------------------


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


def _clean_urls(urls: Union[str, Iterable[str], None]) -> List[str]:
    if isinstance(urls, str):
        urls = [urls]
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


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


def _prompt_bubble(label: str, text: Optional[str]) -> str:
    if not text:
        return ""
    return (
        '<div class="prompt-bubble"><div class="prompt-label">'
        f"{esc(label)}</div><div class=\"prompt-text\">{esc(text)}</div></div>"
    )


# --------------------------------------------------------------------------
# Image gallery (comfyui_text_to_image / muapi look)
# --------------------------------------------------------------------------

_GALLERY_CSS = """
.gallery { display: grid; grid-template-columns: 1fr; gap: 8px; }
.gallery.multi { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.media-wrapper { position: relative; border-radius: 12px; overflow: hidden; line-height: 0; }
.media-wrapper img { max-width: 100%; width: 100%; height: auto; display: block; border: none; }
.media-wrapper .dl {
  position: absolute; right: 8px; bottom: 8px; width: 30px; height: 30px; border-radius: 50%;
  border: none; cursor: pointer; background: rgba(0,0,0,0.55); color: #fff; opacity: 0;
  transition: opacity .15s; display: flex; align-items: center; justify-content: center;
}
.media-wrapper:hover .dl, .media-wrapper .dl:focus { opacity: 1; }
.media-wrapper .dl svg { width: 14px; height: 14px; fill: currentColor; }
.media-wrapper.owui-fallback { line-height: 1.5; }
"""

_GALLERY_JS = r"""
var M = window.OWUIMedia, g = document.getElementById('gallery');
if (DATA.items.length > 1) g.className += ' multi';
DATA.items.forEach(function (it, i) {
  var w = document.createElement('div'); w.className = 'media-wrapper';
  var img = document.createElement('img');
  img.alt = it.alt || ('Generated image ' + (i + 1)); img.decoding = 'async';
  w.appendChild(img);
  var b = document.createElement('button'); b.className = 'dl'; b.title = 'Download';
  b.innerHTML = '<svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>';
  b.addEventListener('click', function () { M.download(it.url, it.name, img); });
  w.appendChild(b);
  g.appendChild(w);
  M.load(img, it.url, function () { M.fallback(w, it.url, 'image'); });
});
"""


def image_gallery_embed(
    urls: Union[str, Sequence[str]],
    *,
    prompt: str = "",
    prompt_label: str = "Prompt",
    title: str = "Generated Image",
    download_names: Optional[Sequence[str]] = None,
) -> str:
    """One or more images (2-column grid for >1), hover download button, prompt bubble."""
    clean = _clean_urls(urls)
    items = []
    for i, u in enumerate(clean):
        name = (download_names[i] if download_names and i < len(download_names) else None) or f"image-{i + 1}.png"
        items.append({"url": u, "name": name, "alt": f"{title} {i + 1}" if len(clean) > 1 else title})
    body = '<div id="gallery" class="gallery"></div>' + _prompt_bubble(prompt_label, prompt)
    return _page(title, _GALLERY_CSS, body, {"items": items}, _GALLERY_JS)


# --------------------------------------------------------------------------
# Video player (text_to_video_comfyui / muapi look)
# --------------------------------------------------------------------------

_VIDEO_CSS = """
.media-wrapper { margin: 0 0 8px 0; border-radius: 12px; overflow: hidden; line-height: 0; background: #000; }
.media-wrapper video { width: 100%; max-width: 100%; height: auto; display: block; border: none; background: #000; }
.media-wrapper.owui-fallback { background: transparent; line-height: 1.5; }
.video-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: -2px 0 8px 0; }
.video-actions .owui-btn { margin-top: 0; }
"""

_VIDEO_JS = r"""
var M = window.OWUIMedia, list = document.getElementById('videos');
DATA.items.forEach(function (it, i) {
  var w = document.createElement('div'); w.className = 'media-wrapper';
  var v = document.createElement('video');
  v.controls = true; v.playsInline = true; v.preload = 'metadata';
  if (DATA.loop) v.loop = true;
  if (DATA.muted) v.muted = true;
  w.appendChild(v); list.appendChild(w);
  var acts = document.createElement('div'); acts.className = 'video-actions';
  var b = document.createElement('a'); b.href = '#'; b.className = 'owui-btn';
  b.textContent = DATA.items.length > 1 ? ('Download video ' + (i + 1)) : 'Download video';
  b.addEventListener('click', function (e) { e.preventDefault(); M.download(it.url, it.name, v); });
  acts.appendChild(b); list.appendChild(acts);
  M.load(v, it.url, function () { acts.style.display = 'none'; M.fallback(w, it.url, 'video'); });
});
"""


def video_player_embed(
    urls: Union[str, Sequence[str]],
    *,
    prompt: str = "",
    prompt_label: str = "Prompt",
    title: str = "Generated Video",
    download_names: Optional[Sequence[str]] = None,
    loop: bool = False,
    muted: bool = False,
) -> str:
    """One or more <video controls> players with a download button and prompt bubble."""
    clean = _clean_urls(urls)
    items = []
    for i, u in enumerate(clean):
        name = (download_names[i] if download_names and i < len(download_names) else None) or f"video-{i + 1}.mp4"
        items.append({"url": u, "name": name})
    body = '<div id="videos"></div>' + _prompt_bubble(prompt_label, prompt)
    return _page(title, _VIDEO_CSS, body, {"items": items, "loop": bool(loop), "muted": bool(muted)}, _VIDEO_JS)


# --------------------------------------------------------------------------
# 3D model viewer (Google <model-viewer>, GLB)
# --------------------------------------------------------------------------

_MODEL_3D_CSS = """
.mv-wrap { position: relative; width: 100%; height: __H__px; border-radius: 12px; overflow: hidden;
  background: radial-gradient(circle at 50% 35%, #ffffff 0%, #e9ebef 70%, #dfe2e7 100%); }
.mv-wrap:fullscreen { height: 100%; border-radius: 0; }
.mv-wrap:-webkit-full-screen { height: 100%; border-radius: 0; }
.mv-wrap model-viewer { position: absolute; inset: 0; width: 100%; height: 100%;
  background: transparent; --poster-color: transparent; --progress-bar-height: 0px; }
.mv-poster { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.mv-progress { position: absolute; left: 0; right: 0; bottom: 0; height: 3px; background: rgba(0,0,0,0.08); }
.mv-bar { height: 100%; width: 0; background: #3b82f6; transition: width .2s; }
.mv-status { position: absolute; left: 12px; bottom: 12px; padding: 4px 10px; border-radius: 999px;
  font-size: 11px; color: #333; background: rgba(255,255,255,0.75); }
.mv-wrap.mv-failed { height: auto; background: transparent; overflow: visible; }
.mv-wrap.mv-failed .mv-poster { position: static; display: block; max-height: __H__px;
  border-radius: 12px; margin-bottom: 8px; }
.mv-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0 0 0; }
.mv-actions .owui-btn { margin-top: 0; }
.mv-meta { font-size: 11px; color: #777; margin-left: auto; }
@media (prefers-color-scheme: dark) {
  .mv-wrap { background: radial-gradient(circle at 50% 35%, #3a3d44 0%, #24262b 70%, #1c1d21 100%); }
  .mv-wrap.mv-failed { background: transparent; }
  .mv-progress { background: rgba(255,255,255,0.1); }
  .mv-status { color: #ddd; background: rgba(0,0,0,0.55); }
  .mv-meta { color: #999; }
}
"""

_MODEL_3D_JS = r"""
var M = window.OWUIMedia, wrap = document.getElementById('mv'), stage = document.getElementById('mv-stage');
var bar = document.getElementById('mv-bar'), status = document.getElementById('mv-status');
var acts = document.getElementById('mv-actions'), holder = { dataset: {} }, mv = null, failed = false;
var poster = null;
if (DATA.poster) {
  poster = document.createElement('img');
  poster.className = 'mv-poster'; poster.alt = DATA.alt; poster.decoding = 'async';
  stage.appendChild(poster);
  M.load(poster, DATA.poster, function () { if (poster && poster.parentNode) poster.parentNode.removeChild(poster); poster = null; });
}
function setStatus(t) { if (status) { status.textContent = t; status.style.display = t ? '' : 'none'; } }
function fail() {
  if (failed) return;
  failed = true;
  if (mv && mv.parentNode) mv.parentNode.removeChild(mv);
  wrap.className += ' mv-failed';
  var bars = wrap.querySelectorAll('.mv-progress, .mv-status');
  for (var i = 0; i < bars.length; i++) bars[i].parentNode.removeChild(bars[i]);
  acts.style.display = 'none';
  var box = document.createElement('div');
  wrap.appendChild(box);
  M.fallback(box, DATA.glb, '3D model');
}
// GLB / poster source: same-origin Bearer blob for /api/ paths, else the URL itself.
function source(u) {
  if (M.isLocalApi(u) && M.sameOrigin) return M.blobUrl(u);
  if (M.isSafe(u)) return Promise.resolve(u);
  return Promise.reject(new Error('unsafe URL'));
}
function loadModule() {
  if (!window.customElements || !DATA.module) return Promise.reject(new Error('no custom elements'));
  if (customElements.get('model-viewer')) return Promise.resolve();
  return new Promise(function (resolve, reject) {
    var s = document.createElement('script');
    s.type = 'module'; s.src = DATA.module; s.crossOrigin = 'anonymous';
    if (DATA.integrity) s.integrity = DATA.integrity;
    s.onerror = function () { reject(new Error('module failed to load')); };
    customElements.whenDefined('model-viewer').then(resolve);
    setTimeout(function () { reject(new Error('module load timed out')); }, 20000);
    document.head.appendChild(s);
  });
}
setStatus('Loading 3D viewer...');
Promise.all([loadModule(), source(DATA.glb)]).then(function (res) {
  var src = res[1];
  if (src.indexOf('blob:') === 0) holder.dataset.blob = src;
  mv = document.createElement('model-viewer');
  mv.setAttribute('camera-controls', '');
  mv.setAttribute('touch-action', 'pan-y');
  mv.setAttribute('shadow-intensity', '1');
  mv.setAttribute('environment-image', 'neutral');
  mv.setAttribute('exposure', '1');
  mv.setAttribute('interaction-prompt', 'auto');
  mv.setAttribute('alt', DATA.alt);
  if (DATA.autoRotate) mv.setAttribute('auto-rotate', '');
  mv.addEventListener('progress', function (e) {
    var p = (e && e.detail && e.detail.totalProgress) || 0;
    bar.style.width = Math.round(p * 100) + '%';
    setStatus(p < 1 ? 'Loading model ' + Math.round(p * 100) + '%' : '');
    if (p >= 1 && bar.parentNode) bar.parentNode.style.display = 'none';
  });
  mv.addEventListener('load', function () {
    setStatus('');
    if (bar.parentNode) bar.parentNode.style.display = 'none';
    if (poster && poster.parentNode) poster.parentNode.removeChild(poster);
    M.postHeight();
  });
  mv.addEventListener('error', fail);
  stage.appendChild(mv);
  mv.src = src;
  setStatus('Loading model...');
}).catch(fail);

var rot = document.getElementById('mv-rotate');
rot.textContent = DATA.autoRotate ? 'Stop rotation' : 'Auto-rotate';
rot.addEventListener('click', function (e) {
  e.preventDefault();
  if (!mv) return;
  mv.autoRotate = !mv.autoRotate;
  rot.textContent = mv.autoRotate ? 'Stop rotation' : 'Auto-rotate';
});
var fs = document.getElementById('mv-fullscreen');
var canFs = (document.fullscreenEnabled || document.webkitFullscreenEnabled) &&
  (wrap.requestFullscreen || wrap.webkitRequestFullscreen);
if (!canFs) { fs.style.display = 'none'; } else {
  fs.addEventListener('click', function (e) {
    e.preventDefault();
    var cur = document.fullscreenElement || document.webkitFullscreenElement;
    try {
      if (cur) { (document.exitFullscreen || document.webkitExitFullscreen).call(document); }
      else { var r = (wrap.requestFullscreen || wrap.webkitRequestFullscreen).call(wrap); if (r && r.catch) r.catch(function () {}); }
    } catch (err) {}
  });
}
document.getElementById('mv-download').addEventListener('click', function (e) {
  e.preventDefault(); M.download(DATA.glb, DATA.name, holder);
});
if (DATA.meta) document.getElementById('mv-meta').textContent = DATA.meta;
"""


def model_3d_viewer_embed(
    glb_url: str,
    poster_url: Optional[str] = None,
    title: str = "Generated 3D Model",
    prompt: str = "",
    *,
    prompt_label: str = "Prompt",
    download_name: str = "model.glb",
    model_label: str = "",
    auto_rotate: bool = True,
    height: int = 440,
) -> str:
    """Interactive GLB viewer (<model-viewer>): orbit controls, auto-rotate toggle,
    shadow + neutral environment lighting, poster, progress bar, fullscreen and
    download. Falls back to the poster image + download link when the viewer
    module or the GLB cannot be loaded."""
    h = max(240, min(int(height or 440), 900))
    data = {
        "glb": glb_url if isinstance(glb_url, str) else "",
        "poster": poster_url if isinstance(poster_url, str) and poster_url.strip() else None,
        "name": download_name or "model.glb",
        "alt": title or "Generated 3D model",
        "meta": model_label or "",
        "autoRotate": bool(auto_rotate),
        "module": MODEL_VIEWER_MODULE_URL,
        "integrity": MODEL_VIEWER_INTEGRITY,
    }
    body = (
        '<div id="mv" class="mv-wrap"><div id="mv-stage"></div>'
        '<div class="mv-progress"><div id="mv-bar" class="mv-bar"></div></div>'
        '<div id="mv-status" class="mv-status"></div></div>'
        '<div id="mv-actions" class="mv-actions">'
        '<a href="#" id="mv-rotate" class="owui-btn">Stop rotation</a>'
        '<a href="#" id="mv-fullscreen" class="owui-btn">Fullscreen</a>'
        '<a href="#" id="mv-download" class="owui-btn">Download GLB</a>'
        '<span id="mv-meta" class="mv-meta"></span></div>'
        + _prompt_bubble(prompt_label, prompt)
    )
    return _page(title, _MODEL_3D_CSS.replace("__H__", str(h)), body, data, _MODEL_3D_JS)


def _url_path(url: Any) -> str:
    """Lower-cased URL path (query string / fragment ignored)."""
    if not isinstance(url, str):
        return ""
    try:
        return (urlparse(url.strip()).path or "").lower()
    except ValueError:
        return ""


def _prediction_files(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = prediction.get("files")
    return [f for f in files if isinstance(f, dict)] if isinstance(files, list) else []


def _prediction_outputs(prediction: Dict[str, Any]) -> List[str]:
    outputs = prediction.get("outputs") or prediction.get("output") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    return [o for o in outputs if isinstance(o, str) and o.strip()] if isinstance(outputs, list) else []


def select_glb_url(prediction: Dict[str, Any]) -> str:
    """The GLB mesh of a completed 3D prediction: a files[] entry named/typed as GLB,
    else the first outputs URL whose path ends in .glb. Raises AtlasCloudError."""
    for entry in _prediction_files(prediction):
        url = entry.get("url")
        if not is_http_url(url):
            continue
        name = str(entry.get("file_name") or entry.get("name") or "").lower()
        ctype = str(entry.get("content_type") or "").lower()
        ftype = str(entry.get("type") or "").lower()
        if name.endswith(".glb") or _url_path(url).endswith(".glb") or "gltf-binary" in ctype or ftype == "glb":
            return url
    for url in _prediction_outputs(prediction):
        if is_http_url(url) and _url_path(url).endswith(".glb"):
            return url
    raise AtlasCloudError("The Atlas Cloud model returned no GLB file.")


def select_preview_url(prediction: Dict[str, Any], exclude: Optional[str] = None) -> Optional[str]:
    """Preview render: data.thumbnail, else an image in files[] / outputs."""
    thumb = prediction.get("thumbnail")
    if isinstance(thumb, dict):
        thumb = thumb.get("url")
    if is_http_url(thumb) and thumb != exclude:
        return thumb
    for entry in _prediction_files(prediction):
        url = entry.get("url")
        if not is_http_url(url) or url == exclude:
            continue
        name = str(entry.get("file_name") or entry.get("name") or "").lower()
        ctype = str(entry.get("content_type") or "").lower()
        if ctype.startswith("image/") or name.endswith(PREVIEW_EXTENSIONS) or _url_path(url).endswith(PREVIEW_EXTENSIONS):
            return url
    for url in reversed(_prediction_outputs(prediction)):  # preview image is listed last
        if is_http_url(url) and url != exclude and _url_path(url).endswith(PREVIEW_EXTENSIONS):
            return url
    return None


def model_slug(text: Optional[str], fallback: str = "atlascloud_3d") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")[:48].strip("-")
    return slug or f"{fallback}_{uuid.uuid4().hex[:8]}"


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


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


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
        TEXT_TO_3D_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred text-to-3D model ID (one of the supported 3D models).",
        )
        IMAGE_TO_3D_MODEL: Optional[str] = Field(
            default=None,
            description="Preferred image-to-3D model ID (one of the supported 3D models).",
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
        TEXT_TO_3D_MODEL: str = Field(
            default=DEFAULT_TEXT_TO_3D_MODEL,
            description=(
                "Default text-to-3D model ID. Supported: "
                + ", ".join(spec["text"] for spec in MODEL_3D_FAMILIES.values())
            ),
        )
        IMAGE_TO_3D_MODEL: str = Field(
            default=DEFAULT_IMAGE_TO_3D_MODEL,
            description=(
                "Default image-to-3D model ID. Supported: "
                + ", ".join(spec["image"] for spec in MODEL_3D_FAMILIES.values())
            ),
        )
        ENABLE_PBR_3D: bool = Field(
            default=True,
            description="Request PBR materials (metallic/roughness/normal maps) for 3D models.",
        )
        FACE_LIMIT_3D: Optional[int] = Field(
            default=None,
            ge=1,
            description=(
                "Optional target face count for 3D meshes (Tripo: face_limit 1000-2000000, "
                "Hunyuan3D Pro: face_count 40000-1500000; clamped). Empty = provider default."
            ),
        )
        MAX_WAIT_3D_SECONDS: float = Field(
            default=900.0,
            ge=30.0,
            description="Maximum seconds to wait for a 3D generation (they take minutes).",
        )
        POLL_INTERVAL_SECONDS: float = Field(default=3.0, ge=0.1)
        GENERATION_TIMEOUT_SECONDS: float = Field(default=600.0, ge=1.0)
        RETURN_HTML_EMBED: bool = Field(
            default=True,
            description="Return an inline HTML image viewer / video player / 3D viewer upon completion.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

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
                try:
                    user_valves = Tools.UserValves(**v)
                except Exception:
                    pass
            elif hasattr(v, "__dict__"):
                try:
                    user_valves = Tools.UserValves(**v.__dict__)
                except Exception:
                    pass

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
            "text_to_3d_model": pick(
                user_valves.TEXT_TO_3D_MODEL if user_valves else None,
                self.valves.TEXT_TO_3D_MODEL,
            ),
            "image_to_3d_model": pick(
                user_valves.IMAGE_TO_3D_MODEL if user_valves else None,
                self.valves.IMAGE_TO_3D_MODEL,
            ),
            "enable_pbr_3d": bool(self.valves.ENABLE_PBR_3D),
            "face_limit_3d": self.valves.FACE_LIMIT_3D,
            "timeout_3d": self.valves.MAX_WAIT_3D_SECONDS,
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
    def _error_text(exc: BaseException) -> str:
        """User/LLM-safe error text: our own messages as-is, network errors by type only
        (aiohttp messages can contain request URLs)."""
        if isinstance(exc, (AtlasCloudError, MediaStorageError)):
            return str(exc)
        if isinstance(exc, asyncio.TimeoutError):
            return "the request to Atlas Cloud timed out"
        return f"network error ({type(exc).__name__})"

    @staticmethod
    async def _json_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            payload = await response.json()
        except (aiohttp.ContentTypeError, ValueError) as exc:
            raise AtlasCloudError(
                f"Atlas Cloud returned a non-JSON response (HTTP {response.status})."
            ) from exc

        if not isinstance(payload, dict):
            raise AtlasCloudError("Atlas Cloud returned an invalid JSON payload.")

        if response.status >= 400:
            detail = payload.get("message") or payload.get("error") or "request rejected"
            raise AtlasCloudError(
                f"Atlas Cloud request failed ({response.status}): {_short(detail)}"
            )
        return payload

    @staticmethod
    def _data(payload: dict[str, Any]) -> dict[str, Any]:
        code = payload.get("code")
        if code not in (None, 0, 200):
            detail = payload.get("message") or payload.get("error") or f"code {code}"
            raise AtlasCloudError(f"Atlas Cloud request failed: {_short(detail)}")
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

    # ------------------------------------------------------------------
    # Reference media (image_url / audio_url) handling
    #
    # Accepted: http(s) URLs (passed to Atlas Cloud as-is, never fetched by
    # Open WebUI), base64 data: URIs, and Open WebUI file references
    # (/api/v1/files/{id}[/content]) read server-side only after an access
    # check. Local filesystem paths and anything else are rejected.
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_data_uri(value: str, kind: str) -> Tuple[bytes, str]:
        match = DATA_URI_RE.match(value.strip())
        if not match or ";base64" not in (match.group(2) or "").lower():
            raise AtlasCloudError(f"The {kind} must be a base64 data:{kind}/* URI.")
        content_type = match.group(1).lower()
        content_type = INPUT_TYPE_ALIASES.get(content_type, content_type)
        if not content_type.startswith(f"{kind}/"):
            raise AtlasCloudError(f"The provided data URI is not an {kind}.")
        encoded = re.sub(r"\s+", "", match.group(3))
        if len(encoded) * 3 // 4 > MAX_INPUT_MEDIA_BYTES:
            raise AtlasCloudError(f"The {kind} is too large to upload.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AtlasCloudError(f"The {kind} data could not be decoded.") from exc
        if not raw:
            raise AtlasCloudError(f"The {kind} data is empty.")
        return raw, content_type

    @staticmethod
    def _owui_file_id(value: str) -> Optional[str]:
        """File id for an Open WebUI file reference, else None."""
        try:
            parsed = urlparse(value.strip())
        except ValueError:
            return None
        if parsed.scheme and parsed.scheme not in ("http", "https"):
            return None
        match = OWUI_FILE_PATH_RE.match(parsed.path or "")
        return match.group(1) if match else None

    @staticmethod
    async def _read_owui_file(
        file_id: str, kind: str, user_obj: Optional[UserModel]
    ) -> Tuple[bytes, str, str]:
        """Read an Open WebUI file the user may access (owner, admin or has_access_to_file)."""
        if user_obj is None:
            raise AtlasCloudError("Cannot read the attached Open WebUI file without a user context.")
        file_item = await Files.get_file_by_id(file_id)
        if not file_item:
            raise AtlasCloudError("The attached Open WebUI file was not found.")

        allowed = file_item.user_id == user_obj.id or user_obj.role == "admin"
        if not allowed:
            try:
                from open_webui.utils.access_control.files import has_access_to_file

                allowed = await has_access_to_file(file_id, "read", user_obj)
            except Exception as exc:
                log.warning("File access check failed (%s)", type(exc).__name__)
                allowed = False
        if not allowed:
            # Same message as "not found" would leak existence; keep it explicit but generic.
            raise AtlasCloudError("You do not have access to the attached Open WebUI file.")

        meta = file_item.meta or {}
        content_type = normalize_content_type(
            meta.get("content_type") or mimetypes.guess_type(file_item.filename or "")[0], ""
        )
        content_type = INPUT_TYPE_ALIASES.get(content_type, content_type)
        if not content_type.startswith(f"{kind}/"):
            raise AtlasCloudError(f"The attached Open WebUI file is not an {kind}.")
        size = meta.get("size")
        if isinstance(size, (int, float)) and size > MAX_INPUT_MEDIA_BYTES:
            raise AtlasCloudError(f"The attached {kind} is too large to upload.")
        if not file_item.path:
            raise AtlasCloudError("The attached Open WebUI file has no stored content.")

        def _read() -> bytes:
            local_path = Storage.get_file(file_item.path)
            if os.path.getsize(local_path) > MAX_INPUT_MEDIA_BYTES:
                raise AtlasCloudError(f"The attached {kind} is too large to upload.")
            with open(local_path, "rb") as handle:
                return handle.read()

        try:
            raw = await asyncio.to_thread(_read)
        except AtlasCloudError:
            raise
        except Exception as exc:
            log.warning("Reading Open WebUI file failed (%s)", type(exc).__name__)
            raise AtlasCloudError("The attached Open WebUI file could not be read.") from None
        if not raw:
            raise AtlasCloudError("The attached Open WebUI file is empty.")
        name = safe_filename(file_item.filename, content_type, f"input_{kind}")
        return raw, content_type, name

    async def _ensure_atlas_media_url(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        media_input: str,
        kind: str = "image",
        user_obj: Optional[UserModel] = None,
    ) -> str:
        """Turn an http(s) URL, data: URI or Open WebUI file reference into an Atlas Cloud media URL."""
        if not media_input or not isinstance(media_input, str) or not media_input.strip():
            raise AtlasCloudError(f"No valid {kind} input provided.")

        s_input = media_input.strip()

        if s_input[:5].lower() == "data:":
            raw, content_type = self._decode_data_uri(s_input, kind)
            return await self._upload_media(
                session,
                base_url,
                raw,
                filename=safe_filename(None, content_type, f"input_{kind}"),
                content_type=content_type,
            )

        file_id = self._owui_file_id(s_input)
        if file_id:
            raw, content_type, name = await self._read_owui_file(file_id, kind, user_obj)
            return await self._upload_media(
                session, base_url, raw, filename=name, content_type=content_type
            )

        if is_http_url(s_input):
            # Public URL: Atlas Cloud fetches it. Open WebUI never requests it.
            return s_input

        raise AtlasCloudError(
            f"Unsupported {kind} reference. Attach the {kind} to the chat, or pass an "
            f"http(s) URL, a base64 data: URI or an Open WebUI /api/v1/files/<id> link."
        )

    @staticmethod
    def _file_entry_content_type(entry: Dict[str, Any]) -> str:
        candidates = [entry.get("content_type"), entry.get("mime_type")]
        for nested in (entry.get("meta"), (entry.get("file") or {}).get("meta") if isinstance(entry.get("file"), dict) else None):
            if isinstance(nested, dict):
                candidates.append(nested.get("content_type"))
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return normalize_content_type(value, "")
        name = entry.get("name") or entry.get("filename") or ""
        guessed = mimetypes.guess_type(str(name))[0] if name else None
        return normalize_content_type(guessed, "")

    def _extract_media_from_messages(
        self, messages: Optional[List[Dict[str, Any]]], media_type: str = "image"
    ) -> List[str]:
        """Extracts image/audio references (URL, data: URI or Open WebUI file link) from chat messages."""
        if not messages:
            return []

        media_items: List[str] = []
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")

            # Check structured content blocks (OWUI standard)
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    b_type = item.get("type")
                    if media_type == "image" and b_type == "image_url":
                        url_obj = item.get("image_url", {})
                        url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                        if isinstance(url, str) and url.strip():
                            media_items.append(url.strip())
                    elif media_type == "audio" and b_type in ("audio", "input_audio"):
                        aud_obj = item.get("input_audio") or item.get("audio") or {}
                        if not isinstance(aud_obj, dict):
                            continue
                        url = aud_obj.get("url") or aud_obj.get("data")
                        if not isinstance(url, str) or not url.strip():
                            continue
                        url = url.strip()
                        fmt = str(aud_obj.get("format") or "").lower()
                        if (
                            fmt
                            and not url.lower().startswith(("data:", "http://", "https://", "/api/"))
                            and re.fullmatch(r"[a-z0-9]{1,8}", fmt)
                        ):
                            # OpenAI input_audio: raw base64 + format
                            mime = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
                            url = f"data:{mime};base64,{url}"
                        media_items.append(url)

            # Check text markdown syntax e.g. ![name](url)
            if isinstance(content, str) and media_type == "image":
                matches = re.findall(
                    r"!\[[^\]]*\]\(((?:https?://|data:image/|/api/v1/files/)[^\s\)]+)\)",
                    content,
                )
                media_items.extend(matches)

            # Check attached files list in OWUI message dict (never server storage paths)
            files = message.get("files") or message.get("images") or []
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, str):
                        if media_type == "image" and f.strip():
                            media_items.append(f.strip())
                        continue
                    if not isinstance(f, dict):
                        continue
                    content_type = self._file_entry_content_type(f)
                    f_type = f.get("type")
                    if content_type:
                        if not content_type.startswith(f"{media_type}/"):
                            continue
                    elif media_type == "audio" and f_type == "image":
                        continue
                    elif media_type == "image" and f_type not in (None, "image"):
                        continue
                    url = f.get("url")
                    if not (isinstance(url, str) and url.strip()) and isinstance(f.get("id"), str):
                        url = f"/api/v1/files/{f['id']}/content"
                    if isinstance(url, str) and url.strip():
                        media_items.append(url.strip())

            if media_items:
                break

        return media_items

    async def _submit_and_poll(
        self,
        endpoint: str,
        payload: dict[str, Any],
        config: Dict[str, Any],
        emitter: EventEmitter,
        *,
        timeout: Optional[float] = None,
        heartbeat: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit a prediction and poll until it completes. Returns the completed
        prediction ``data`` dict (outputs/files/thumbnail are validated by callers).

        :param timeout: per-call wait limit (default: GENERATION_TIMEOUT_SECONDS).
        :param heartbeat: when set, re-emit an unchanged status with the elapsed
            time every ``heartbeat`` seconds (long 3D jobs); still deduplicated.
        """
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
        wait = float(timeout if timeout is not None else config["timeout"])
        client_timeout = aiohttp.ClientTimeout(total=wait + 30)

        async with aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session:
            async with session.post(f"{base_url}{endpoint}", json=payload) as response:
                submit_data = self._data(await self._json_response(response))

            prediction_id = submit_data.get("id") or submit_data.get("request_id")
            if not prediction_id:
                raise AtlasCloudError("Atlas Cloud did not return a prediction ID.")

            started = time.monotonic()
            deadline = started + wait
            last_key: Optional[Tuple[str, int]] = None
            poll_errors = 0
            while time.monotonic() < deadline:
                try:
                    prediction = await self._poll_once(
                        session, f"{base_url}/model/prediction/{prediction_id}"
                    )
                except (
                    AtlasCloudTransientError,
                    aiohttp.ClientConnectionError,
                    aiohttp.ClientPayloadError,
                    asyncio.TimeoutError,
                ) as exc:
                    poll_errors += 1
                    if poll_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                        raise AtlasCloudError(
                            f"Atlas Cloud status checks kept failing for prediction "
                            f"{_short(prediction_id, 80)}: {_short(str(exc) or type(exc).__name__)}"
                        ) from exc
                    last_key = None  # re-announce the status once polling recovers
                    await self._emit_status(
                        emitter,
                        f"Atlas Cloud status check failed, retrying ({poll_errors}/"
                        f"{MAX_CONSECUTIVE_POLL_ERRORS - 1})...",
                        done=False,
                    )
                    await asyncio.sleep(config["poll_interval"])
                    continue
                poll_errors = 0

                status = str(prediction.get("status", "")).lower()
                if status in COMPLETED_STATUSES:
                    return prediction

                if status in FAILED_STATUSES:
                    detail = (
                        prediction.get("error") or prediction.get("message") or status
                    )
                    raise AtlasCloudError(f"Atlas Cloud generation failed: {_short(detail)}")

                elapsed = time.monotonic() - started
                tick = int(elapsed // heartbeat) if heartbeat else 0
                if (status, tick) != last_key:
                    last_key = (status, tick)
                    text = f"Atlas Cloud generation status: {_short(status or 'processing', 40)}"
                    if heartbeat and tick:
                        text += f" ({int(elapsed) // 60}m {int(elapsed) % 60:02d}s elapsed)"
                    await self._emit_status(emitter, text, done=False)
                await asyncio.sleep(config["poll_interval"])

        raise AtlasCloudError(
            f"Atlas Cloud generation timed out (prediction {_short(prediction_id, 80)})."
        )

    async def _poll_once(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """One prediction status GET. Gateway errors (429/5xx) and non-JSON bodies
        raise AtlasCloudTransientError; other 4xx and API error codes stay fatal."""
        async with session.get(url) as response:
            if response.status in TRANSIENT_HTTP_STATUSES:
                raise AtlasCloudTransientError(
                    f"Atlas Cloud status check returned HTTP {response.status}."
                )
            try:
                payload = await self._json_response(response)
            except AtlasCloudError as exc:
                if response.status < 400 and "non-JSON" in str(exc):
                    raise AtlasCloudTransientError(str(exc)) from exc
                raise
        return self._data(payload)

    async def _submit_and_wait(
        self,
        endpoint: str,
        payload: dict[str, Any],
        config: Dict[str, Any],
        emitter: EventEmitter,
        *,
        timeout: Optional[float] = None,
    ) -> list[str]:
        prediction = await self._submit_and_poll(
            endpoint, payload, config, emitter, timeout=timeout
        )
        outputs = prediction.get("outputs") or prediction.get("output") or []
        if isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(outputs, list) or not all(
            isinstance(item, str) for item in outputs
        ):
            raise AtlasCloudError("Atlas Cloud returned invalid output URLs.")
        if not outputs:
            raise AtlasCloudError("Atlas Cloud completed without output URLs.")
        return outputs

    # ------------------------------------------------------------------
    # Output persistence + presentation
    # ------------------------------------------------------------------

    async def _persist_outputs(
        self,
        outputs: List[str],
        *,
        kind: str,
        expected_type: str,
        prefix: str,
        request: Any,
        user: Any,
        metadata: Optional[dict],
    ) -> List[Dict[str, Any]]:
        """Download each Atlas Cloud output (no API key sent) and store it in Open WebUI."""
        user_model = await resolve_user(user)
        if user_model is None:
            raise MediaStorageError("The requesting Open WebUI user could not be resolved.")
        allowed = ALLOWED_OUTPUT_TYPES[kind]
        saved: List[Dict[str, Any]] = []
        for index, url in enumerate(outputs, start=1):
            data, served_type = await fetch_media_bytes(
                url, timeout=300.0 if kind == "video" else 120.0
            )
            content_type = INPUT_TYPE_ALIASES.get(served_type, served_type)
            if content_type not in allowed:
                guessed = normalize_content_type(
                    mimetypes.guess_type(urlparse(url).path or "")[0], ""
                )
                if content_type in ("application/octet-stream", "binary/octet-stream", ""):
                    content_type = guessed if guessed in allowed else expected_type
                else:
                    raise MediaStorageError(
                        f"Atlas Cloud returned an unexpected file type for the {kind}."
                    )
            saved.append(
                await save_media_bytes(
                    data,
                    request=request,
                    user=user_model,
                    content_type=content_type,
                    filename=f"{prefix}_{uuid.uuid4().hex[:8]}_{index}",
                    metadata=metadata,
                    prefix=prefix,
                )
            )
        return saved

    async def _deliver(
        self,
        outputs: List[str],
        *,
        kind: str,
        expected_type: str,
        prefix: str,
        title: str,
        prompt: str,
        done_text: str,
        config: Dict[str, Any],
        request: Any,
        user: Any,
        metadata: Optional[dict],
        emitter: EventEmitter,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
        await self._emit_status(emitter, "Saving to Open WebUI...", done=False)
        try:
            saved = await self._persist_outputs(
                outputs,
                kind=kind,
                expected_type=expected_type,
                prefix=prefix,
                request=request,
                user=user,
                metadata=metadata,
            )
        except MediaStorageError as exc:
            msg = f"Atlas Cloud generated the {kind}, but it could not be saved to Open WebUI: {exc}"
            await self._emit_status(emitter, msg, done=True)
            return msg

        urls = [item["url"] for item in saved]
        names = [
            item.get("name") or f"{prefix}_{i + 1}{extension_for(item.get('content_type') or expected_type)}"
            for i, item in enumerate(saved)
        ]
        await self._emit_status(emitter, done_text, done=True)

        noun = "Image" if kind == "image" else "Video"
        if len(urls) > 1:
            noun += "s"
        if config["return_html_embed"]:
            if kind == "image":
                html_doc = image_gallery_embed(urls, prompt=prompt, title=title, download_names=names)
                hint = (
                    " These file URLs can be passed as image_url to the other Atlas Cloud tools"
                    " for follow-up edits or image-to-video."
                )
            else:
                html_doc = video_player_embed(urls, prompt=prompt, title=title, download_names=names)
                hint = ""
            return embed_result(
                html_doc,
                {
                    "message": (
                        f"{noun} generated and shown to the user in the embedded viewer above. "
                        f"Do not repeat it as markdown or links.{hint}"
                    ),
                    "files": urls,
                },
            )

        if kind == "image":
            lines = "\n".join(f"![{title}]({u})" for u in urls)
        else:
            lines = "\n".join(f"- [Download generated video]({u})" for u in urls)
        return f"{noun} generated and saved to Open WebUI:\n{lines}"

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        size: str = "2048*2048",
        output_format: str = "jpeg",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Any = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
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
            error = self._error_text(exc)
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {error}", done=True
            )
            return f"Atlas Cloud image generation failed: {error}"

        return await self._deliver(
            outputs,
            kind="image",
            expected_type=OUTPUT_FORMAT_TYPES.get(str(output_format).lower(), "image/jpeg"),
            prefix="atlascloud_image",
            title="Generated Image",
            prompt=prompt,
            done_text="Atlas Cloud image generated",
            config=config,
            request=__request__,
            user=__user__,
            metadata=__metadata__,
            emitter=__event_emitter__,
        )

    async def edit_image(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        size: str = "2048*2048",
        output_format: str = "jpeg",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __request__: Any = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
        """Edit or transform an image using Atlas Cloud image-to-image model. Image is automatically retrieved from attached chat message or explicit image_url (an http(s) URL, data: URI or Open WebUI /api/v1/files/<id> link)."""
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
            msg = "Error: No image reference provided or found in chat messages. Please attach an image or provide an image_url."
            await self._emit_status(__event_emitter__, msg, done=True)
            return msg

        try:
            user_obj = await resolve_user(__user__)
            timeout = aiohttp.ClientTimeout(total=60)
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                hosted_url = await self._ensure_atlas_media_url(
                    session, config["base_url"], target_image, "image", user_obj
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
            error = self._error_text(exc)
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud image edit error: {error}", done=True
            )
            return f"Atlas Cloud image editing failed: {error}"

        return await self._deliver(
            outputs,
            kind="image",
            expected_type=OUTPUT_FORMAT_TYPES.get(str(output_format).lower(), "image/jpeg"),
            prefix="atlascloud_edit",
            title="Edited Image",
            prompt=prompt,
            done_text="Atlas Cloud image edit complete!",
            config=config,
            request=__request__,
            user=__user__,
            metadata=__metadata__,
            emitter=__event_emitter__,
        )

    async def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        generate_audio: bool = True,
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Any = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
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
            error = self._error_text(exc)
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {error}", done=True
            )
            return f"Atlas Cloud video generation failed: {error}"

        return await self._deliver(
            outputs,
            kind="video",
            expected_type="video/mp4",
            prefix="atlascloud_video",
            title="Generated Video",
            prompt=prompt,
            done_text="Atlas Cloud video generated",
            config=config,
            request=__request__,
            user=__user__,
            metadata=__metadata__,
            emitter=__event_emitter__,
        )

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
        __request__: Any = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
        """Generate a video using a reference image (image-to-video) with Atlas Cloud. Image is automatically retrieved from attached chat message or explicit image_url (an http(s) URL, data: URI or Open WebUI /api/v1/files/<id> link)."""
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
            msg = "Error: No reference image provided or found in chat messages. Please attach an image or pass image_url."
            await self._emit_status(__event_emitter__, msg, done=True)
            return msg

        try:
            user_obj = await resolve_user(__user__)
            timeout = aiohttp.ClientTimeout(total=60)
            headers = {"Authorization": f"Bearer {config['api_key']}"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                hosted_url = await self._ensure_atlas_media_url(
                    session, config["base_url"], target_image, "image", user_obj
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
            error = self._error_text(exc)
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {error}", done=True
            )
            return f"Atlas Cloud image-to-video generation failed: {error}"

        return await self._deliver(
            outputs,
            kind="video",
            expected_type="video/mp4",
            prefix="atlascloud_i2v",
            title="Generated Video",
            prompt=prompt,
            done_text="Atlas Cloud video generated from image!",
            config=config,
            request=__request__,
            user=__user__,
            metadata=__metadata__,
            emitter=__event_emitter__,
        )

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
        __request__: Any = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
        """Generate a video using a reference audio clip (audio-to-video / music-to-video) with Atlas Cloud. audio_url / image_url may be http(s) URLs, data: URIs or Open WebUI /api/v1/files/<id> links; attached chat media is used when omitted."""
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
            msg = "Error: No audio or image reference provided or found in chat messages."
            await self._emit_status(__event_emitter__, msg, done=True)
            return msg

        try:
            user_obj = await resolve_user(__user__)
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
                        session, config["base_url"], target_audio, "audio", user_obj
                    )
                if target_image:
                    payload["image_url"] = await self._ensure_atlas_media_url(
                        session, config["base_url"], target_image, "image", user_obj
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
            error = self._error_text(exc)
            await self._emit_status(
                __event_emitter__, f"Atlas Cloud error: {error}", done=True
            )
            return f"Atlas Cloud audio-to-video generation failed: {error}"

        return await self._deliver(
            outputs,
            kind="video",
            expected_type="video/mp4",
            prefix="atlascloud_a2v",
            title="Generated Video",
            prompt=prompt,
            done_text="Atlas Cloud video generated from audio!",
            config=config,
            request=__request__,
            user=__user__,
            metadata=__metadata__,
            emitter=__event_emitter__,
        )

    # ------------------------------------------------------------------
    # 3D models
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_3d_model(model_id: str, mode: str) -> Dict[str, Any]:
        """Validate a configured 3D model id for ``mode`` ("text" | "image")."""
        entry = SUPPORTED_3D_MODELS.get(str(model_id or "").strip())
        valve = "TEXT_TO_3D_MODEL" if mode == "text" else "IMAGE_TO_3D_MODEL"
        if entry is None or entry[1] != mode:
            supported = ", ".join(spec[mode] for spec in MODEL_3D_FAMILIES.values())
            raise AtlasCloudError(
                f"Unsupported {mode}-to-3D model '{_short(model_id, 80)}' in {valve}. Supported: {supported}."
            )
        family = entry[0]
        return {"id": model_id.strip(), "family": family, "mode": mode, **MODEL_3D_FAMILIES[family]}

    @staticmethod
    def _build_3d_payload(
        spec: Dict[str, Any],
        *,
        prompt: str,
        image: Optional[str],
        enable_pbr: bool,
        face_limit: Optional[int],
    ) -> Dict[str, Any]:
        """Request body for one model family. Only documented fields are sent."""
        family = spec["family"]
        payload: Dict[str, Any] = {"model": spec["id"]}
        if spec["mode"] == "image":
            payload[spec["image_field"]] = image
        else:
            payload["prompt"] = prompt
        if family == "hunyuan3d-rapid":
            payload["enable_pbr"] = bool(enable_pbr)
            payload["format"] = "GLB"
        elif family == "hunyuan3d-pro":
            payload["enable_pbr"] = bool(enable_pbr)
            if face_limit:
                payload["face_count"] = max(40000, min(int(face_limit), 1500000))
        elif family == "tripo-h3.1":
            payload["texture"] = True
            payload["pbr"] = bool(enable_pbr)
            if face_limit:
                payload["face_limit"] = max(1000, min(int(face_limit), 2000000))
        elif family == "meshy-v7":
            if spec["mode"] == "text":
                payload["mode"] = "full"  # mesh + texture stages ("preview" is untextured)
            else:
                payload["should_texture"] = True  # enable_pbr requires it
            payload["enable_pbr"] = bool(enable_pbr)
            payload["target_formats"] = ["glb"]
        return payload

    async def _persist_3d(
        self,
        prediction: Dict[str, Any],
        *,
        slug: str,
        request: Any,
        user: Any,
        metadata: Optional[dict],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Download the GLB (+ preview render) right away (provider URLs expire)
        and store them in Open WebUI. Raises AtlasCloudError / MediaStorageError."""
        glb_url = select_glb_url(prediction)
        preview_url = select_preview_url(prediction, exclude=glb_url)
        user_model = await resolve_user(user)
        if user_model is None:
            raise MediaStorageError("The requesting Open WebUI user could not be resolved.")

        data, _served = await fetch_media_bytes(glb_url, timeout=300.0, max_bytes=MAX_3D_MODEL_BYTES)
        if not data.startswith(GLB_MAGIC):
            raise AtlasCloudError("The Atlas Cloud model returned a file that is not a valid GLB.")
        glb = await save_media_bytes(
            data,
            request=request,
            user=user_model,
            content_type=GLB_CONTENT_TYPE,
            filename=f"{slug}.glb",
            metadata=metadata,
            prefix="atlascloud_3d",
        )

        preview: Optional[Dict[str, Any]] = None
        if preview_url:
            try:
                img, served = await fetch_media_bytes(preview_url, timeout=120.0)
                ctype = INPUT_TYPE_ALIASES.get(served, served)
                if ctype not in ALLOWED_OUTPUT_TYPES["image"]:
                    guessed = normalize_content_type(mimetypes.guess_type(_url_path(preview_url))[0], "")
                    ctype = guessed if guessed in ALLOWED_OUTPUT_TYPES["image"] else ""
                if ctype:
                    preview = await save_media_bytes(
                        img,
                        request=request,
                        user=user_model,
                        content_type=ctype,
                        filename=f"{slug}_preview",
                        metadata=metadata,
                        prefix="atlascloud_3d",
                    )
            except MediaStorageError as exc:  # the preview is optional
                log.warning("3D preview image not stored: %s", exc)
        return glb, preview

    async def generate_3d_model(
        self,
        prompt: str = "",
        image_url: str = "",
        __event_emitter__: EventEmitter = None,
        __user__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __files__: Optional[List[Dict[str, Any]]] = None,
        __request__: Any = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
        """Generate a textured 3D model (GLB) with Atlas Cloud and show it in an interactive 3D viewer. Uses image-to-3D when an image is given in image_url or attached to the user's latest message, otherwise text-to-3D from the prompt. Takes a few minutes.

        :param prompt: Description of a single object to model, e.g. "a weathered bronze knight helmet". Required when no image is given; with an image it only names the result.
        :param image_url: Optional reference image to turn into 3D: an http(s) URL, a base64 data: URI or an Open WebUI /api/v1/files/<id>/content link (e.g. an image generated earlier in this chat). Leave empty to use an image attached to the latest message, or for text-to-3D.
        """
        config = self._resolve_config(__user__)
        prompt = (prompt or "").strip()

        target_image = (image_url or "").strip() if isinstance(image_url, str) else ""
        if not target_image:
            # Only the latest user message: an old image elsewhere in the chat must
            # not silently turn a text-to-3D request into image-to-3D.
            last_user = next(
                (m for m in reversed(__messages__ or []) if isinstance(m, dict) and m.get("role") == "user"),
                None,
            )
            extracted = self._extract_media_from_messages([last_user] if last_user else [], "image")
            if not extracted and isinstance(__files__, list):
                extracted = self._extract_media_from_messages([{"files": __files__}], "image")
            if extracted:
                target_image = extracted[0]

        mode = "image" if target_image else "text"
        try:
            spec = self._resolve_3d_model(
                config["image_to_3d_model"] if mode == "image" else config["text_to_3d_model"], mode
            )
            if mode == "text" and not prompt:
                raise AtlasCloudError("Provide a prompt describing the object, or attach an image.")
            if mode == "text" and spec["max_prompt"] and len(prompt) > spec["max_prompt"]:
                raise AtlasCloudError(
                    f"The prompt is too long for {spec['label']} (max {spec['max_prompt']} characters)."
                )
            if not config["api_key"]:
                raise AtlasCloudError(
                    "Atlas Cloud API key is not configured. Please set ATLASCLOUD_API_KEY in tool Valves or User Valves."
                )

            hosted_image: Optional[str] = None
            if mode == "image":
                await self._emit_status(
                    __event_emitter__, "Processing reference image for 3D...", done=False
                )
                user_obj = await resolve_user(__user__)
                headers = {"Authorization": f"Bearer {config['api_key']}"}
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=60), headers=headers
                ) as session:
                    hosted_image = await self._ensure_atlas_media_url(
                        session, config["base_url"], target_image, "image", user_obj
                    )

            await self._emit_status(
                __event_emitter__,
                f"Generating 3D model with {spec['label']} ({mode}-to-3D), this takes a few minutes...",
                done=False,
            )
            prediction = await self._submit_and_poll(
                IMAGE_ENDPOINT,
                self._build_3d_payload(
                    spec,
                    prompt=prompt,
                    image=hosted_image,
                    enable_pbr=config["enable_pbr_3d"],
                    face_limit=config["face_limit_3d"],
                ),
                config,
                __event_emitter__,
                timeout=config["timeout_3d"],
                heartbeat=30.0,
            )
            select_glb_url(prediction)  # fail fast before any download
        except (AtlasCloudError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            error = self._error_text(exc)
            await self._emit_status(__event_emitter__, f"Atlas Cloud error: {error}", done=True)
            return f"Atlas Cloud 3D model generation failed: {error}"

        await self._emit_status(__event_emitter__, "Saving 3D model to Open WebUI...", done=False)
        slug = model_slug(prompt)
        try:
            glb, preview = await self._persist_3d(
                prediction, slug=slug, request=__request__, user=__user__, metadata=__metadata__
            )
        except (AtlasCloudError, MediaStorageError) as exc:
            msg = f"Atlas Cloud generated the 3D model, but it could not be saved to Open WebUI: {exc}"
            await self._emit_status(__event_emitter__, msg, done=True)
            return msg

        await self._emit_status(__event_emitter__, "Atlas Cloud 3D model generated", done=True)
        glb_link = glb["url"]
        used = f"{spec['label']} ({spec['id']}), cost tier: {spec['cost']}"
        if not config["return_html_embed"]:
            return (
                f"3D model generated and saved to Open WebUI: [Download {slug}.glb]({glb_link})\n"
                f"Model: {used}."
            )

        title = (prompt[:80] if prompt else "Generated 3D Model") or "Generated 3D Model"
        viewer = model_3d_viewer_embed(
            glb_link,
            preview["url"] if preview else None,
            title,
            prompt,
            download_name=glb.get("name") or f"{slug}.glb",
            model_label=f"{spec['label']} - {mode}-to-3D",
        )
        message = (
            "The 3D model is shown to the user in the interactive 3D viewer above; do not repeat it "
            f"as markdown. GLB download link: {glb_link}. Model used: {used}."
        )
        return embed_result(viewer, message)
