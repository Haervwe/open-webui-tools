"""
title: ComfyUI Text-to-Video Tool
description: Generate video from text prompt via ComfyUI workflow JSON. Uses ComfyUI HTTP+WebSocket API with robust fallback polling, supports unloading Ollama models before run, randomizes seed, saves the final video to Open WebUI storage (linked to the chat) and shows it in an inline video player.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.0
required_open_webui_version: 0.11.0
license: MIT
"""

import aiohttp
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
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from urllib.parse import urlencode
from pydantic import BaseModel, Field
from fastapi import HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.datastructures import Headers
from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import upload_file_handler  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log = logger

# --- Open WebUI media storage helpers (shared across open-webui-tools media plugins) ---
# Saves bytes with upload_file_handler(process=False) and links the file to the
# chat message (Chats.insert_chat_files) so shared chats can play it. Failures
# raise MediaStorageError with a user-safe message; there is no silent fallback
# to the internal ComfyUI URL.

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



# --- Inline video player embed (escaped text, iframe:height, same-origin-only auth) ---

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



# --- Constants ---

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")

DEFAULT_WORKFLOW: Dict[str, Any] = {
    "71": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Cargar CLIP"},
    },
    "72": {
        "inputs": {"text": "", "clip": ["71", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "73": {
        "inputs": {"vae_name": "wan/wan_2.1_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Cargar VAE"},
    },
    "74": {
        "inputs": {"width": 848, "height": 480, "length": 41, "batch_size": 1},
        "class_type": "EmptyHunyuanLatentVideo",
        "_meta": {"title": "EmptyHunyuanLatentVideo"},
    },
    "75": {
        "inputs": {
            "unet_name": "wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Cargar Modelo de Difusión"},
    },
    "76": {
        "inputs": {
            "unet_name": "wan/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Cargar Modelo de Difusión"},
    },
    "78": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
            "model": ["86", 0],
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["81", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Avanzado)"},
    },
    "80": {
        "inputs": {
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "codec": "auto",
            "video-preview": "",
            "video": ["88", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Guardar video"},
    },
    "81": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 742951153577776,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "model": ["82", 0],
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["74", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Avanzado)"},
    },
    "82": {
        "inputs": {"shift": 5.000000000000001, "model": ["83", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "MuestreoDeModeloSD3"},
    },
    "83": {
        "inputs": {
            "lora_name": "wan2_2/14b/lightning/high_noise_model.safetensors",
            "strength_model": 1.0000000000000002,
            "model": ["75", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "CargadorLoRAModeloSolo"},
    },
    "85": {
        "inputs": {
            "lora_name": "wan2_2/14b/lightning/low_noise_model.safetensors",
            "strength_model": 1.0000000000000002,
            "model": ["76", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "CargadorLoRAModeloSolo"},
    },
    "86": {
        "inputs": {"shift": 5.000000000000001, "model": ["85", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "MuestreoDeModeloSD3"},
    },
    "88": {
        "inputs": {"fps": 16, "images": ["114", 0]},
        "class_type": "CreateVideo",
        "_meta": {"title": "Crear video"},
    },
    "89": {
        "inputs": {
            "text": "A golden retriever playing an electric guitar at a concert.",
            "clip": ["71", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "114": {
        "inputs": {
            "tile_size": 128,
            "overlap": 64,
            "temporal_size": 20,
            "temporal_overlap": 8,
            "samples": ["78", 0],
            "vae": ["73", 0],
        },
        "class_type": "VAEDecodeTiled",
        "_meta": {"title": "VAE Decodificar (Mosaico)"},
    },
    "115": {
        "inputs": {"anything": ["88", 0]},
        "class_type": "easy cleanGpuUsed",
        "_meta": {"title": "Clean VRAM Used"},
    },
}

# --- Helper Functions (Module-level) ---


async def get_loaded_models_async(
    api_url: str = "http://localhost:11434",
) -> list[Dict[str, Any]]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url.rstrip('/')}/api/ps") as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return cast(list[Dict[str, Any]], data.get("models", []))
    except Exception as e:
        logger.debug("Error fetching loaded models: %s", e)
        return []


async def unload_all_models_async(api_url: str = "http://localhost:11434") -> bool:
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
                logger.info("All models successfully unloaded.")
                return True

        logger.warning("Some models might still be loaded after timeout.")
        return False

    except Exception as e:
        logger.error("Error unloading models: %s", e)
        return False


async def connect_submit_and_wait(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_payload: Dict[str, Any],
    client_id: str,
    max_wait_time: int,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    start_time = asyncio.get_event_loop().time()
    prompt_id = None

    async with aiohttp.ClientSession(headers=headers) as session:
        ws_url = f"{comfyui_ws_url}?clientId={client_id}"
        try:
            async with session.ws_connect(ws_url) as ws:
                async with session.post(
                    f"{comfyui_http_url}/prompt", json=prompt_payload
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise Exception(
                            f"Failed to queue prompt: {resp.status} - {err_text}"
                        )
                    resp_json = await resp.json()
                    prompt_id = resp_json.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id received from ComfyUI")

                last_poll_time = 0
                poll_interval = 3.0

                while True:
                    execute_poll = False
                    if asyncio.get_event_loop().time() - start_time > max_wait_time:
                        raise TimeoutError(
                            f"Generation timed out after {max_wait_time}s"
                        )

                    if asyncio.get_event_loop().time() - last_poll_time > poll_interval:
                        execute_poll = True
                        last_poll_time = asyncio.get_event_loop().time()

                    if execute_poll and prompt_id:
                        try:
                            async with session.get(
                                f"{comfyui_http_url}/history/{prompt_id}"
                            ) as history_resp:
                                if history_resp.status == 200:
                                    history = await history_resp.json()
                                    if prompt_id in history:
                                        return history[prompt_id]
                        except Exception:
                            pass

                    try:
                        msg = await ws.receive(timeout=1.0)

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            message = json.loads(msg.data)
                            msg_type = message.get("type")
                            data = message.get("data", {})

                            if (
                                msg_type == "execution_cached" or msg_type == "executed"
                            ) and data.get("prompt_id") == prompt_id:
                                async with session.get(
                                    f"{comfyui_http_url}/history/{prompt_id}"
                                ) as final_resp:
                                    if final_resp.status == 200:
                                        history = await final_resp.json()
                                        if prompt_id in history:
                                            return history[prompt_id]

                            elif (
                                msg_type == "execution_error"
                                and data.get("prompt_id") == prompt_id
                            ):
                                error_details = data.get(
                                    "exception_message", "Unknown error"
                                )
                                node_id = data.get("node_id", "N/A")
                                raise Exception(
                                    f"ComfyUI job failed on node {node_id}. Error: {error_details}"
                                )

                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning(
                                "WebSocket connection lost. Switching to pure polling."
                            )
                            break

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning("WS Error: %s", e)
                        break

        except Exception as e:
            if not prompt_id:
                raise e
            logger.warning("WebSocket failed (%s). Fallback to pure polling.", e)

        if prompt_id:
            while asyncio.get_event_loop().time() - start_time <= max_wait_time:
                await asyncio.sleep(2)
                try:
                    async with session.get(
                        f"{comfyui_http_url}/history/{prompt_id}"
                    ) as h_resp:
                        if h_resp.status == 200:
                            history = await h_resp.json()
                            if prompt_id in history:
                                return history[prompt_id]
                except Exception:
                    pass
            raise TimeoutError(f"Generation timed out (polling) after {max_wait_time}s")

        raise Exception("Failed to start generation flow.")


def extract_video_files(job_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Robustly extract video filenames from ComfyUI job outputs
    by recursively searching the data.
    """
    candidates: List[Tuple[str, str]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, str) and any(
            obj.lower().endswith(ext) for ext in VIDEO_EXTS
        ):
            candidates.append((obj, ""))
        elif isinstance(obj, dict):
            filename_val = None
            for k in ("filename", "file", "path", "name"):
                v = obj.get(k)
                if isinstance(v, str) and any(
                    v.lower().endswith(ext) for ext in VIDEO_EXTS
                ):
                    filename_val = v
                    break

            if filename_val:
                subfolder_val = ""
                for sf_key in ("subfolder", "subdir", "folder", "directory"):
                    sf = obj.get(sf_key)
                    if isinstance(sf, str):
                        subfolder_val = sf.strip("/ ")
                        break
                candidates.append((filename_val, subfolder_val))

            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(job_data)

    seen = set()
    video_files: List[Dict[str, str]] = []
    for fname, sub in candidates:
        filename = os.path.basename(str(fname))
        subfolder = str(sub).strip("/ ")
        key = f"{subfolder}/{filename}" if subfolder else filename
        if key not in seen:
            seen.add(key)
            video_files.append({"filename": filename, "subfolder": subfolder})

    return video_files


def video_content_type(filename: str) -> str:
    """MIME type from the output's extension (ComfyUI may send octet-stream)."""
    guessed = mimetypes.guess_type(filename)[0] or ""
    return guessed if guessed.startswith("video/") else "video/mp4"


async def download_and_upload_to_owui(
    comfyui_http_url: str,
    filename: str,
    subfolder: str,
    request: Any,
    user: Any,
    headers: Optional[Dict[str, str]] = None,
    metadata: Optional[dict] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Download the video from ComfyUI and save it to Open WebUI storage.

    Returns the save_media_bytes() result ({"id", "url", "name", ...}); ``url``
    is the relative /api/v1/files/{id}/content path. Raises MediaStorageError
    (user-safe message, e.g. the 413 "Max Upload Size" case) and never falls back
    to the internal ComfyUI URL.
    """
    params: Dict[str, str] = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder
    query = urlencode(params)
    data: Optional[bytes] = None
    served = ""
    # Core ComfyUI /view serves the original output bytes. VideoHelperSuite's
    # /api/viewvideo re-encodes to a realtime VP9 WebM preview when ffmpeg is
    # present, so it is only a fallback for setups where /view returns 404.
    for path in ("/view", "/api/viewvideo"):
        try:
            data, served = await fetch_media_bytes(
                f"{comfyui_http_url}{path}?{query}", headers=headers, timeout=timeout
            )
            break
        except MediaStorageError as exc:
            if exc.status_code == 404 and path != "/api/viewvideo":
                continue
            raise MediaStorageError(f"Could not fetch the video from ComfyUI: {exc}", exc.status_code) from None
    if data is None:
        raise MediaStorageError("Could not fetch the video from ComfyUI.")
    # Trust a real video/* type from the server (e.g. video/webm from viewvideo)
    # so the stored type and the download extension match the bytes.
    served_type = (served or "").split(";", 1)[0].strip().lower()
    content_type = served_type if served_type.startswith("video/") else video_content_type(filename)
    return await save_media_bytes(
        data,
        request=request,
        user=user,
        content_type=content_type,
        filename=filename,
        metadata=metadata,
        prefix="comfyui_video",
    )


# --- Main Tool Class ---


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188", description="ComfyUI HTTP API endpoint."
        )
        comfyui_api_key: str = Field(
            default="",
            description="API key for ComfyUI authentication (Bearer token). Leave empty if not required.",
            json_schema_extra={"input": {"type": "password"}},
        )
        workflow_json: str = Field(
            default=json.dumps(DEFAULT_WORKFLOW),
            description="ComfyUI Workflow JSON string. If empty, a default is used.",
        )
        prompt_node_id: str = Field(
            default="89", description="Node ID for the text prompt input."
        )
        prompt_field_name: str = Field(
            default="text",
            description="Name of the input field where the prompt text goes.",
        )
        seed_node_id: str = Field(
            default="81",
            description="Node ID containing the seed parameter to randomize.",
        )
        seed_field_name: str = Field(
            default="noise_seed",
            description="Name of the seed input field on the seed node.",
        )
        max_wait_time: int = Field(
            default=600, description="Max wait time in seconds for job completion."
        )
        unload_ollama_models: bool = Field(
            default=False, description="Unload Ollama models before calling ComfyUI."
        )
        ollama_api_url: str = Field(
            default="http://localhost:11434",
            description="Ollama API URL for unloading models.",
        )
        return_html_embed: bool = Field(
            default=True, description="Return an HTML video embed upon completion."
        )

    def __init__(self):
        self.valves = self.Valves()

    async def generate_video(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Generate a video from a text prompt using the provided ComfyUI workflow."""
        try:
            # --- Unload Ollama models if requested ---
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "⏳ Unloading Ollama models...",
                                "done": False,
                            },
                        }
                    )
                unloaded = await unload_all_models_async(self.valves.ollama_api_url)
                await asyncio.sleep(2)
                if not unloaded:
                    logger.warning("Ollama models may not have fully unloaded.")

            # --- Parse workflow ---
            try:
                workflow = json.loads(self.valves.workflow_json)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid Workflow JSON in valves: {e}")

            # --- Inject prompt ---
            prompt_node_id = self.valves.prompt_node_id
            prompt_field = self.valves.prompt_field_name
            if prompt_node_id not in workflow:
                raise ValueError(
                    f"Prompt node ID '{prompt_node_id}' not found in the workflow."
                )
            workflow[prompt_node_id].setdefault("inputs", {})[prompt_field] = prompt

            # --- Inject random seed ---
            seed_node_id = self.valves.seed_node_id
            seed_field = self.valves.seed_field_name
            if seed_node_id in workflow:
                workflow[seed_node_id].setdefault("inputs", {})[seed_field] = (
                    random.randint(1, 2**31 - 1)
                )

            # --- Prepare connection ---
            http_api_url = self.valves.comfyui_api_url.rstrip("/")
            ws_api_url = (
                http_api_url.replace("http://", "ws://").replace("https://", "wss://")
                + "/ws"
            )
            client_id = str(uuid.uuid4())
            payload: Dict[str, Any] = {
                "prompt": workflow,
                "client_id": client_id,
            }

            comfyui_headers = {}
            if self.valves.comfyui_api_key:
                comfyui_headers["Authorization"] = f"Bearer {self.valves.comfyui_api_key}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "🎬 Submitting to ComfyUI...",
                            "done": False,
                        },
                    }
                )

            # --- Submit and wait (robust WS + polling fallback) ---
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⏳ Waiting for video generation...",
                            "done": False,
                        },
                    }
                )

            result_data = await connect_submit_and_wait(
                ws_api_url,
                http_api_url,
                payload,
                client_id,
                self.valves.max_wait_time,
                headers=comfyui_headers,
            )

            # --- Extract video files ---
            video_files = extract_video_files(result_data)
            if not video_files:
                logger.warning(
                    "No video files extracted from job data: %s",
                    json.dumps(result_data, indent=2),
                )
                return "ComfyUI job completed, but no video output was found in the results."

            video_info = video_files[0]

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "💾 Saving video to Open WebUI...",
                            "done": False,
                        },
                    }
                )

            saved = await download_and_upload_to_owui(
                http_api_url,
                video_info["filename"],
                video_info["subfolder"],
                __request__,
                __user__,
                headers=comfyui_headers,
                metadata=__metadata__,
                timeout=float(max(60, self.valves.max_wait_time)),
            )
            final_url = saved["url"]

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "✅ Video Generated!",
                            "done": True,
                        },
                    }
                )
            if self.valves.return_html_embed:
                html_player = video_player_embed(
                    final_url,
                    prompt=prompt,
                    title="Generated Video",
                    download_names=[saved.get("name") or video_info["filename"]],
                )
                return embed_result(
                    html_player,
                    "🎬 Video generated and shown to the user in a video player above. "
                    "Do not repeat it as a link or markdown.",
                )
            return f"🎬 Video generated successfully. Link: {final_url}"

        except MediaStorageError as e:
            logger.error("Video storage failed: %s", e)
            message = f"The video was generated, but could not be saved to Open WebUI: {e}"
        except aiohttp.ClientError as e:
            logger.error("ComfyUI request failed: %s", e, exc_info=True)
            message = f"An error occurred: could not reach ComfyUI ({type(e).__name__})."
        except Exception as e:
            logger.error("Error during video generation: %s", e, exc_info=True)
            message = f"An error occurred: {e}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"❌ {message}",
                        "done": True,
                    },
                }
            )
        return message
