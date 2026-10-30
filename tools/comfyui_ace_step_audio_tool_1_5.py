"""
title: ComfyUI ACE Step 1.5 Audio Generator
description: Tool to generate songs using the ACE Step 1.5 workflow via the ComfyUI API.
Supports advanced parameters like key, tempo, language, and batch size.
Requires [ComfyUI-Unload-Model](https://github.com/SeanScripts/ComfyUI-Unload-Model) for model unloading functionality (can be customized via unload_node ID).
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools/
version: 0.7.0
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
                        log.warning("ComfyUI /prompt returned HTTP %s: %s", resp.status, err_text[:1000])
                        raise Exception(
                            f"Failed to queue prompt: ComfyUI returned HTTP {resp.status}"
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
                            print(
                                "[Warning] WebSocket connection lost. Switching to pure polling."
                            )
                            break

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"[Warning] WS Error: {e}")
                        break

        except Exception as e:
            if not prompt_id:
                raise e
            print(f"[Warning] WebSocket failed ({e}). Fallback to pure polling.")

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
                except:
                    pass
            raise TimeoutError(f"Generation timed out (polling) after {max_wait_time}s")

        raise Exception("Failed to start generation flow.")


def extract_audio_files(job_data: Dict[str, Any]) -> list[Dict[str, str]]:
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


# Player behaviour. Plain string (no f-string escaping); data arrives as JSON via
# script_json(). Media loading/downloading goes through window.OWUIMedia (above):
# direct src first, Bearer blob only for same-origin /api/ paths, fallback panel
# with an "Open / download" link otherwise, and iframe:height messages.
_PLAYER_JS = r"""
var M = window.OWUIMedia, P = DATA.pid, tracks = DATA.tracks, c0 = DATA.c0;
function $(id) { return document.getElementById(id + '_' + P); }
var aud = $('aud'), pb = $('pb'), pi = $('pi'), pt = $('pt'), pf = $('pf'), pthumb = $('pthumb');
var tc = $('tc'), td = $('td'), dl = $('dl'), vbtn = $('vbtn'), vtrack = $('vtrack'), vfill = $('vfill');
var ww = $('ww'), fb = $('fb');
var vBtns = document.querySelectorAll('.vbtn_' + P);
var curIdx = 0, failed = false, styleOpen = false, lastVol = 1.0;

var styleHdr = $('styleHdr'), styleBody = $('styleBody'), styleChev = $('styleChev');
styleHdr.addEventListener('click', function () {
  styleOpen = !styleOpen;
  styleBody.style.display = styleOpen ? 'block' : 'none';
  styleChev.style.transform = styleOpen ? 'rotate(180deg)' : 'rotate(0deg)';
  M.postHeight();
});

var PLAY  = '<path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/>';
var PAUSE = '<rect x="6" y="4" width="4" height="16" rx="2"/><rect x="14" y="4" width="4" height="16" rx="2"/>';
var REDO  = '<path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>';
var VON   = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>';
var VOFF  = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';

function fmt(s) {
  if (!s || !isFinite(s)) return '0:00';
  var m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}
function setWave(on) {
  ww.style.opacity = on ? '1' : '0.45';
  var bars = ww.querySelectorAll('.wavebar_' + P);
  for (var i = 0; i < bars.length; i++) bars[i].style.animationPlayState = on ? 'running' : 'paused';
}
function setProgress(pct) { pf.style.width = pct + '%'; pthumb.style.left = pct + '%'; }

function bindAudio() {
  aud.addEventListener('play',  function () { pi.innerHTML = PAUSE; pi.style.marginLeft = '0';   setWave(true);  });
  aud.addEventListener('pause', function () { pi.innerHTML = PLAY;  pi.style.marginLeft = '3px'; setWave(false); });
  aud.addEventListener('ended', function () { pi.innerHTML = REDO;  pi.style.marginLeft = '0';   setWave(false); });
  aud.addEventListener('loadedmetadata', function () { td.textContent = fmt(aud.duration); });
  aud.addEventListener('timeupdate', function () {
    var pct = aud.duration ? (aud.currentTime / aud.duration * 100) : 0;
    setProgress(pct); tc.textContent = fmt(aud.currentTime); td.textContent = fmt(aud.duration);
  });
}

function loadTrack(idx) {
  var t = tracks[idx];
  failed = false; pb.disabled = false;
  fb.innerHTML = ''; fb.className = '';
  // Fresh <audio> per track so listeners of the previous load never fire again.
  var vol = aud.volume, muted = aud.muted;
  aud.pause();
  if (aud.dataset.blob) { try { URL.revokeObjectURL(aud.dataset.blob); } catch (e) {} }
  delete aud.dataset.blob;
  var a2 = aud.cloneNode(false);
  a2.removeAttribute('src');
  aud.parentNode.replaceChild(a2, aud); aud = a2;
  aud.volume = vol; aud.muted = muted;
  bindAudio();
  pi.innerHTML = PLAY; pi.style.marginLeft = '3px'; setWave(false);
  setProgress(0); tc.textContent = '0:00'; td.textContent = '0:00';
  for (var k = 0; k < vBtns.length; k++) {
    var on = parseInt(vBtns[k].getAttribute('data-idx'), 10) === idx;
    vBtns[k].style.background = on ? 'rgba(255,255,255,0.88)' : 'rgba(255,255,255,0.18)';
    vBtns[k].style.color = on ? c0 : '#fff';
  }
  M.load(aud, t.url, function () { failed = true; pb.disabled = true; M.fallback(fb, t.url, 'audio'); });
}

for (var b = 0; b < vBtns.length; b++) {
  (function (btn) {
    btn.addEventListener('click', function () {
      var idx = parseInt(btn.getAttribute('data-idx'), 10);
      if (idx !== curIdx && tracks[idx]) {
        var wasPlaying = !aud.paused;
        curIdx = idx; loadTrack(idx);
        if (wasPlaying) aud.play().catch(function () {});
      }
    });
  })(vBtns[b]);
}

pb.addEventListener('click', function () {
  if (failed) return;
  if (aud.ended) { aud.currentTime = 0; aud.play().catch(function () {}); }
  else if (aud.paused) { aud.play().catch(function () {}); }
  else { aud.pause(); }
});

pt.addEventListener('click', function (e) {
  var r = pt.getBoundingClientRect();
  if (aud.duration) aud.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * aud.duration;
});

dl.addEventListener('click', function (e) {
  e.preventDefault();
  var t = tracks[curIdx];
  if (t) M.download(t.url, t.name, aud);
});

vbtn.addEventListener('click', function () {
  if (aud.muted) {
    aud.muted = false; aud.volume = lastVol;
    vfill.style.width = (lastVol * 100) + '%';
    vbtn.querySelector('svg').innerHTML = VON;
  } else {
    lastVol = aud.volume; aud.muted = true;
    vfill.style.width = '0%';
    vbtn.querySelector('svg').innerHTML = VOFF;
  }
});
vtrack.addEventListener('click', function (e) {
  var r = vtrack.getBoundingClientRect();
  var v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  aud.volume = v; aud.muted = false; lastVol = v;
  vfill.style.width = (v * 100) + '%';
  vbtn.querySelector('svg').innerHTML = v > 0 ? VON : VOFF;
});

if (tracks.length) loadTrack(0);
else { pb.disabled = true; M.fallback(fb, '', 'audio'); }
"""


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


async def get_llamacpp_models_async(
    api_url: str = "http://localhost:8082",
) -> list[Dict[str, Any]]:
    """Fetch all models currently known to the llama-server router via GET /v1/models."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url.rstrip('/')}/v1/models") as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return cast(list[Dict[str, Any]], data.get("data", []))
    except Exception as e:
        print(f"Error fetching llama.cpp models: {e}")
        return []


async def unload_all_llamacpp_models_async(
    api_url: str = "http://localhost:8082",
) -> bool:
    """
    Unload all models loaded in a llama-server router instance.

    Uses GET /v1/models to list models, then POST /models/unload for each one
    that is currently loaded (in_cache == True or no in_cache field present).
    """
    try:
        models = await get_llamacpp_models_async(api_url)
        if not models:
            print("No llama.cpp models found or server unreachable.")
            return True

        base_url = api_url.rstrip("/")
        unloaded_any = False

        async with aiohttp.ClientSession() as session:
            for model in models:
                model_id = model.get("id", "")
                # in_cache is present in router mode; if absent we still attempt unload
                in_cache = model.get("in_cache", True)
                if not model_id or not in_cache:
                    continue

                print(f"Unloading llama.cpp model: {model_id}")
                try:
                    async with session.post(
                        f"{base_url}/models/unload",
                        json={"model": model_id},
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status in (200, 204):
                            unloaded_any = True
                        else:
                            body = await resp.text()
                            print(
                                f"Warning: /models/unload returned {resp.status} for '{model_id}': {body}"
                            )
                except Exception as e:
                    print(f"Warning: failed to unload '{model_id}': {e}")

        if not unloaded_any:
            print("No llama.cpp models were unloaded (none in cache?).")

        return True

    except Exception as e:
        print(f"Error unloading llama.cpp models: {e}")
        return False


# ---------------------------------------------------------------------------
# PALETTE — generate 5 vivid rainbow colors from a random seed.
# Hues are spaced evenly around the wheel (72° apart) from a random start.
# ---------------------------------------------------------------------------


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (h 0-360, s 0-1, l 0-1) → #RRGGBB"""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    r, g, b = int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255)
    return f"#{r:02X}{g:02X}{b:02X}"


def _random_rainbow_palette(seed_val: int) -> list:
    """5 vivid colors evenly spread around the hue wheel from a random start."""
    rng = random.Random(seed_val)
    start_hue = rng.randint(0, 359)
    colors = []
    for i in range(5):
        hue = (start_hue + i * 72) % 360
        sat = rng.uniform(0.88, 1.0)
        light = rng.uniform(0.50, 0.60)
        colors.append(_hsl_to_hex(hue, sat, light))
    return colors


def generate_audio_player_embed(
    tracks: list[Dict[str, str]],
    song_title: str,
    tags: str,
    lyrics: Optional[str] = None,
    palette_seed: Optional[int] = None,
    colorful: bool = True,
) -> str:
    """
    Generate an audio player embed.
    When *colorful* is True, uses a vivid rainbow-gradient style.
    When False, uses a minimalistic grey card style.
    Tags are collapsed by default; lyrics are the main focus.
    Pure vanilla HTML/CSS/JS — no external dependencies.
    """
    if palette_seed is None:
        palette_seed = random.randint(0, 9999999)

    if colorful:
        c0, c1, c2, c3, c4 = _random_rainbow_palette(palette_seed)
    else:
        # Minimalistic grey palette
        c0 = c1 = c2 = c3 = c4 = "#3a3a3e"

    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    safe_title = _esc(song_title)
    safe_tags = _esc(tags)
    safe_lyrics = _esc(lyrics or "Instrumental")

    pid = uuid.uuid4().hex[:8]
    multi = len(tracks) > 1

    # Version selector buttons
    version_btns = "".join(
        f'<button class="vbtn_{pid}" data-idx="{i}" '
        f'style="border:none;cursor:pointer;padding:5px 16px;border-radius:999px;'
        f"font-size:11px;font-weight:700;letter-spacing:0.4px;transition:all .18s;"
        f'background:rgba(255,255,255,0.18);color:#fff;">'
        f"v{i + 1}</button>"
        for i in range(len(tracks))
    )

    # 24 waveform bars — heights & speeds vary naturally
    bars_html = "".join(
        f'<div style="width:3px;min-width:3px;border-radius:3px;'
        f"background:rgba(255,255,255,0.92);"
        f"animation:bb_{pid} {0.55 + (i % 8) * 0.07:.2f}s ease-in-out infinite alternate;"
        f"animation-delay:{i * 0.04:.2f}s;"
        f'animation-play-state:paused;height:4px;" class="wavebar_{pid}"></div>'
        for i in range(24)
    )

    # Build background layers based on colorful flag
    if colorful:
        bg_layer = f"""  <!-- ── Animated rainbow gradient background ── -->
  <div style="position:absolute;inset:0;z-index:0;
    background:linear-gradient(125deg,{c0},{c1},{c2},{c3},{c4},{c0});
    background-size:500% 500%;
    animation:mesh_{pid} 10s ease infinite;"></div>

  <!-- Soft dark overlay so text is always legible -->
  <div style="position:absolute;inset:0;z-index:1;background:rgba(0,0,0,0.22);"></div>

  <!-- Floating glow orbs for depth -->
  <div style="position:absolute;z-index:1;width:220px;height:220px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,255,255,0.16) 0%,transparent 68%);
    top:-70px;right:-50px;
    animation:orb_{pid} 7s ease-in-out infinite;pointer-events:none;"></div>
  <div style="position:absolute;z-index:1;width:160px;height:160px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,255,255,0.10) 0%,transparent 68%);
    bottom:-40px;left:-30px;
    animation:orb_{pid} 11s ease-in-out infinite reverse;pointer-events:none;"></div>"""
    else:
        bg_layer = """  <!-- ── Transparent backdrop ── -->
  <div style="position:absolute;inset:0;z-index:0;background:rgba(0,0,0,0.12);
    backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);"></div>"""

    html = f"""
<div style="display:flex;justify-content:center;width:100%;
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">

<div style="position:relative;overflow:hidden;border-radius:22px;
  max-width:420px;width:100%;
  box-shadow:{"0 24px 64px rgba(0,0,0,0.45),0 0 0 1px rgba(255,255,255,0.1)" if colorful else "0 4px 24px rgba(0,0,0,0.18),0 0 0 1px rgba(255,255,255,0.06)"};
  color:#fff;box-sizing:border-box;margin-bottom:18px;">

{bg_layer}

  <!-- ── CONTENT ── -->
  <div style="position:relative;z-index:2;padding:24px 22px 22px;">

    <!-- Title -->
    <div style="text-align:center;margin-bottom:18px;">
      <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
        opacity:0.65;margin-bottom:5px;">♪ ACE Step 1.5</div>
      <div style="font-size:19px;font-weight:800;letter-spacing:-0.2px;line-height:1.2;
        text-shadow:0 2px 14px rgba(0,0,0,0.35);">{safe_title}</div>
    </div>

    <!-- Waveform bars -->
    <div id="ww_{pid}" style="display:flex;align-items:center;justify-content:center;
      gap:3px;height:28px;margin-bottom:18px;opacity:0.45;transition:opacity .3s;">
      {bars_html}
    </div>

    <!-- Progress bar -->
    <div id="pt_{pid}" style="width:100%;height:5px;border-radius:99px;cursor:pointer;
      background:rgba(255,255,255,0.22);margin-bottom:8px;position:relative;">
      <div id="pf_{pid}" style="height:100%;border-radius:99px;width:0%;
        background:rgba(255,255,255,0.92);
        box-shadow:0 0 10px rgba(255,255,255,0.55);
        transition:width .1s linear;pointer-events:none;"></div>
      <div id="pthumb_{pid}" style="position:absolute;top:50%;width:13px;height:13px;
        border-radius:50%;background:#fff;transform:translate(-50%,-50%);left:0%;
        box-shadow:0 0 8px rgba(255,255,255,0.7);transition:left .1s linear;
        opacity:0;pointer-events:none;"></div>
    </div>

    <!-- Time row -->
    <div style="display:flex;justify-content:space-between;font-size:10px;font-weight:600;
      opacity:0.65;margin-bottom:18px;font-variant-numeric:tabular-nums;letter-spacing:0.3px;">
      <span id="tc_{pid}">0:00</span>
      <span id="td_{pid}">0:00</span>
    </div>

    <!-- Controls -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:{"16px" if multi else "20px"};">

      <!-- Volume pill -->
      <div style="display:flex;align-items:center;gap:7px;">
        <button id="vbtn_{pid}" style="background:rgba(255,255,255,0.2);border:none;
          color:#fff;width:30px;height:30px;border-radius:50%;cursor:pointer;
          display:flex;align-items:center;justify-content:center;transition:background .18s;">
          <svg viewBox="0 0 24 24" style="width:13px;height:13px;fill:#fff;pointer-events:none;">
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
          </svg>
        </button>
        <div id="vtrack_{pid}" style="width:48px;height:4px;border-radius:99px;cursor:pointer;
          background:rgba(255,255,255,0.22);">
          <div id="vfill_{pid}" style="height:100%;width:100%;border-radius:99px;
            background:rgba(255,255,255,0.85);pointer-events:none;"></div>
        </div>
      </div>

      <!-- Play button -->
      <button id="pb_{pid}" style="width:60px;height:60px;border-radius:50%;border:none;
        cursor:pointer;background:#fff;
        box-shadow:0 0 28px rgba(255,255,255,0.55),0 6px 20px rgba(0,0,0,0.28);
        display:flex;align-items:center;justify-content:center;
        transition:transform .14s,box-shadow .14s;outline:none;flex-shrink:0;">
        <svg id="pi_{pid}" viewBox="0 0 24 24"
          style="width:26px;height:26px;pointer-events:none;fill:{c0 if colorful else "#888"};margin-left:3px;">
          <path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/>
        </svg>
      </button>

      <!-- Download -->
      <a id="dl_{pid}" href="#" download style="background:rgba(255,255,255,0.2);
        border-radius:999px;padding:7px 13px;font-size:10px;font-weight:700;
        letter-spacing:0.4px;color:#fff;text-decoration:none;
        display:flex;align-items:center;gap:5px;transition:background .18s;">
        <svg viewBox="0 0 24 24" style="width:11px;height:11px;fill:#fff;flex-shrink:0;">
          <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
        </svg>
        Save
      </a>
    </div>

    <!-- Version selector (only when batch > 1) -->
    <div id="ver_{pid}" style="display:{"flex" if multi else "none"};
      gap:7px;flex-wrap:wrap;justify-content:center;margin-bottom:18px;">
      <div style="width:100%;font-size:8px;font-weight:800;letter-spacing:1.5px;
        text-transform:uppercase;opacity:0.55;text-align:center;margin-bottom:3px;">Versions</div>
      {version_btns}
    </div>

    <div id="fb_{pid}"></div>

    <!-- ── Info panel ── -->
    <div style="border-radius:14px;overflow:hidden;
      background:rgba(0,0,0,0.22);border:1px solid rgba(255,255,255,0.1);">

      <!-- Style row — collapsed by default, click to expand -->
      <div id="styleHdr_{pid}" style="display:flex;align-items:center;justify-content:space-between;
        padding:10px 14px;cursor:pointer;user-select:none;">
        <div style="display:flex;align-items:center;gap:8px;overflow:hidden;">
          <span style="font-size:8px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;
            opacity:0.55;flex-shrink:0;">Style</span>
          <span id="stylePrev_{pid}" style="font-size:11px;opacity:0.75;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;">
            {safe_tags[:60] + ("…" if len(safe_tags) > 60 else "")}
          </span>
        </div>
        <svg id="styleChev_{pid}" viewBox="0 0 24 24"
          style="width:14px;height:14px;fill:rgba(255,255,255,0.6);flex-shrink:0;
          transition:transform .22s;transform:rotate(0deg);">
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </div>

      <!-- Style body — hidden by default -->
      <div id="styleBody_{pid}" style="display:none;padding:0 14px 12px;
        font-size:11px;line-height:1.55;opacity:0.85;
        border-top:1px solid rgba(255,255,255,0.08);">
        {safe_tags}
      </div>

      <!-- Divider -->
      <div style="height:1px;background:rgba(255,255,255,0.08);margin:0;"></div>

      <!-- Lyrics — always visible, scrollable -->
      <div style="padding:12px 14px 14px;">
        <div style="font-size:8px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;
          opacity:0.55;margin-bottom:8px;">Lyrics</div>
        <div id="lyr_{pid}" style="max-height:160px;overflow-y:auto;font-size:13px;
          line-height:1.75;white-space:pre-wrap;word-wrap:break-word;opacity:0.92;
          scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.25) transparent;">
          {safe_lyrics}
        </div>
      </div>

    </div><!-- /info panel -->

  </div><!-- /content -->
</div><!-- /card -->
</div>

<style>
  {"@keyframes mesh_" + pid + " { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }" if colorful else ""}
  {"@keyframes orb_" + pid + " { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(-18px) scale(1.07); } }" if colorful else ""}
  @keyframes bb_{pid} {{
    from {{ height:3px;  opacity:0.45; }}
    to   {{ height:24px; opacity:1; }}
  }}
  #pb_{pid}:hover  {{ transform:scale(1.07) !important; box-shadow:0 0 38px rgba(255,255,255,{"0.7" if colorful else "0.3"}),0 8px 24px rgba(0,0,0,0.3) !important; }}
  #pb_{pid}:active {{ transform:scale(0.95) !important; }}
  #vbtn_{pid}:hover, #dl_{pid}:hover {{ background:rgba(255,255,255,0.34) !important; }}
  .vbtn_{pid}:hover {{ background:rgba(255,255,255,0.34) !important; }}
  #styleHdr_{pid}:hover {{ background:rgba(255,255,255,0.05); }}
  #pt_{pid}:hover #pthumb_{pid} {{ opacity:1 !important; }}
  #lyr_{pid}::-webkit-scrollbar {{ width:3px; }}
  #lyr_{pid}::-webkit-scrollbar-thumb {{ background:rgba(255,255,255,0.25);border-radius:3px; }}
</style>

<audio id="aud_{pid}" preload="auto" style="display:none;"></audio>
"""
    fallback_css = f"""
#fb_{pid}.owui-fallback {{ margin:0 0 16px 0; color:#f2f2f2; background:rgba(0,0,0,0.28);
  border-color:rgba(255,255,255,0.3); text-align:center; }}
#fb_{pid} .owui-btn {{ color:#fff; background:rgba(255,255,255,0.18); border-color:rgba(255,255,255,0.28); }}
#fb_{pid} .owui-fallback-hint {{ color:rgba(255,255,255,0.75); }}
"""
    data = {"pid": pid, "tracks": tracks, "c0": c0}
    return (
        "<style>" + _BASE_CSS + fallback_css + "</style>\n"
        + '<div id="owui-root">' + html + "</div>\n"
        + "<script>" + _RUNTIME_JS + "</script>\n"
        + "<script>(function (DATA) {" + _PLAYER_JS + "})(" + script_json(data) + ");</script>\n"
    )


DEFAULT_WORKFLOW = {
    "3": {
        "inputs": {
            "seed": 0,
            "steps": 8,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["78", 0],
            "positive": ["94", 0],
            "negative": ["47", 0],
            "latent_image": ["98", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "18": {
        "inputs": {"samples": ["3", 0], "vae": ["97", 2]},
        "class_type": "VAEDecodeAudio",
        "_meta": {"title": "VAEDecodeAudio"},
    },
    "47": {
        "inputs": {"conditioning": ["94", 0]},
        "class_type": "ConditioningZeroOut",
        "_meta": {"title": "Acondicionamiento Cero"},
    },
    "78": {
        "inputs": {"shift": 3, "model": ["97", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "94": {
        "inputs": {
            "tags": "",
            "lyrics": "",
            "seed": 0,
            "bpm": 120,
            "duration": 180,
            "timesignature": "4",
            "language": "en",
            "keyscale": "E minor",
            "generate_audio_codes": True,
            "cfg_scale": 2,
            "temperature": 0.85,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0,
            "clip": ["97", 1],
        },
        "class_type": "TextEncodeAceStepAudio1.5",
        "_meta": {"title": "TextEncodeAceStepAudio1.5"},
    },
    "97": {
        "inputs": {"ckpt_name": "ace_step_1.5_turbo_aio.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Cargar Punto de Control"},
    },
    "98": {
        "inputs": {"seconds": 180, "batch_size": 1},
        "class_type": "EmptyAceStep1.5LatentAudio",
        "_meta": {"title": "Empty Ace Step 1.5 Latent Audio"},
    },
    "104": {
        "inputs": {
            "filename_prefix": "audio/ace_step_1_5",
            "quality": "V0",
            "audioUI": "",
            "audio": ["105", 0],
        },
        "class_type": "SaveAudioMP3",
        "_meta": {"title": "Guardar Audio (MP3)"},
    },
    "105": {
        "inputs": {"value": ["18", 0]},
        "class_type": "UnloadAllModels",
        "_meta": {"title": "UnloadAllModels"},
    },
}


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
        unload_ollama_models: bool = Field(
            default=False,
            description="Unload Llamacpp and ollama models before calling ComfyUI.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL.",
        )
        unload_llamacpp_models: bool = Field(
            default=False,
            description="Unload all llama.cpp (llama-server router) models before calling ComfyUI.",
        )
        llamacpp_url: str = Field(
            default="http://localhost:8082",
            description="llama-server API URL (router mode). Used for model unloading via POST /models/unload.",
        )
        save_to_storage: bool = Field(
            default=True,
            description=(
                "Save the generated audio to Open WebUI's file storage (recommended). If saving fails, "
                "the tool reports an error. Turn off only if users' browsers can reach ComfyUI directly: "
                "the player then streams from ComfyUI's /view URL, which exposes the ComfyUI address "
                "and is not protected by Open WebUI's access control."
            ),
        )
        show_player_embed: bool = Field(
            default=True,
            description="Show the embedded audio player.",
        )
        max_batch_size: int = Field(
            default=4,
            description="Maximum batch size users can set.",
        )
        max_duration: int = Field(
            default=180,
            description="Maximum allowed duration in seconds.",
        )
        max_number_of_steps: int = Field(
            default=50,
            description="Maximum allowed sampling steps.",
        )
        max_wait_time: int = Field(
            default=600, description="Max wait time for generation (seconds)."
        )
        unload_comfyui_models: bool = Field(
            default=False,
            description="Unload models after generation using ComfyUI-Unload-Model node.",
        )
        workflow_json: str = Field(
            default=json.dumps(DEFAULT_WORKFLOW),
            description="ComfyUI Workflow JSON.",
        )
        model_name: str = Field(
            default="ace_step_1.5_turbo_aio.safetensors",
            description="Checkpoint name for ACE Step 1.5.",
        )
        checkpoint_node: str = Field(
            default="97", description="Node ID for CheckpointLoaderSimple"
        )
        text_encoder_node: str = Field(
            default="94", description="Node ID for TextEncodeAceStepAudio1.5"
        )
        empty_latent_node: str = Field(
            default="98", description="Node ID for EmptyAceStep1.5LatentAudio"
        )
        sampler_node: str = Field(default="3", description="Node ID for KSampler")
        save_node: str = Field(default="104", description="Node ID for SaveAudioMP3")
        vae_decode_node: str = Field(
            default="18", description="Node ID for VAEDecodeAudio"
        )
        unload_node: str = Field(
            default="105", description="Node ID for UnloadAllModels"
        )
        clip_name_1: str = Field(
            default="",
            description="First CLIP model for DualCLIPLoader (non-AIO). Leave empty for AIO.",
        )
        clip_name_2: str = Field(
            default="",
            description="Second CLIP model for DualCLIPLoader (non-AIO). Leave empty for AIO.",
        )
        vae_name: str = Field(
            default="",
            description="VAE model for VAELoader (non-AIO). Leave empty for AIO.",
        )
        dual_clip_loader_node: str = Field(
            default="111", description="Node ID for DualCLIPLoader (non-AIO)"
        )
        vae_loader_node: str = Field(
            default="110", description="Node ID for VAELoader (non-AIO)"
        )

    class UserValves(BaseModel):
        generate_audio_codes: bool = Field(
            default=True,
            description="Enable generate audio codes for better quality.",
        )
        colorful_player: bool = Field(
            default=True,
            description="Use colorful rainbow gradient player. When off, a minimalistic grey style is used.",
        )
        batch_size: int = Field(
            default=1,
            description="Number of tracks to generate per request.",
        )
        steps: int = Field(default=8, description="Sampling steps.")
        seed: int = Field(default=-1, description="Random seed (-1 for random).")

    def __init__(self):
        self.valves = self.Valves()

    async def generate_song(
        self,
        tags: str,
        lyrics: str,
        song_title: str,
        seed: Optional[int] = None,
        bpm: int = 120,
        duration: int = 180,
        key: str = "E minor",
        language: str = "en",
        time_signature: int = 4,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Union[str, Dict[str, Any]]]]:
        """
        Generate music using ACE Step 1.5 with extended parameters.

        **Prompting Guide for Agents:**
        - **Tags (Style/Genre)**: Be descriptive! Include genre, instruments, mood, tempo, and vocal style.
          Examples: "rock, hard rock, powerful voice, electric guitar, 120 bpm", "lo-fi, chill, study beats, jazz piano", "synthwave, darkwave, retrofuturism."
        - **Lyrics**: Use structure tags `[verse]`, `[chorus]`, `[bridge]` to guide the song arrangement.
          For instrumental, use `[inst]` or describe instruments as tags.
        - **Languages**: Supports 50+ languages. Best performance in EN, ZH, JA. For Japanese, use Katakana.

        :param tags: Comma-separated tags describing style, genre, instruments, mood.
        :param lyrics: Full lyrics with structure tags [verse], [chorus], etc.
        :param song_title: Display title for the player.
        :param seed: Random seed. If None, generated automatically.
        :param bpm: Beats per minute (e.g., 90, 120).
        :param duration: Length in seconds. Capped by max_duration valve.
        :param key: Musical key (e.g. "C major", "F# minor").
        :param language: Language code (e.g. "en", "zh", "ja").
        :param time_signature: Time signature (e.g., 4 for 4/4, 3 for 3/4).
        """
        user_valves = (__user__ or {}).get("valves") or self.UserValves()
        batch_size = min(user_valves.batch_size, self.valves.max_batch_size)

        if duration > self.valves.max_duration:
            duration = self.valves.max_duration

        steps = user_valves.steps
        if steps > self.valves.max_number_of_steps:
            steps = self.valves.max_number_of_steps

        # ── Unload Ollama models ──────────────────────────────────────────────
        if self.valves.unload_ollama_models:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Unloading Ollama models...",
                            "done": False,
                        },
                    }
                )
            unloaded = await unload_all_models_async(self.valves.ollama_url)
            await asyncio.sleep(2)
            if not unloaded:
                print("Warning: Ollama models may not have fully unloaded.")

        # ── Unload llama.cpp models ───────────────────────────────────────────
        if self.valves.unload_llamacpp_models:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Unloading llama.cpp models...",
                            "done": False,
                        },
                    }
                )
            unloaded_cpp = await unload_all_llamacpp_models_async(
                self.valves.llamacpp_url
            )
            await asyncio.sleep(1)
            if not unloaded_cpp:
                print("Warning: llama.cpp models may not have fully unloaded.")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Preparing ACE Step 1.5 workflow...",
                        "done": False,
                    },
                }
            )

        try:
            workflow = json.loads(self.valves.workflow_json)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid Workflow JSON in valves: {e}")

        text_node_id = self.valves.text_encoder_node
        latent_node_id = self.valves.empty_latent_node
        sampler_node_id = self.valves.sampler_node
        checkpoint_node_id = self.valves.checkpoint_node
        save_node_id = self.valves.save_node
        vae_node_id = self.valves.vae_decode_node
        unload_node_id = self.valves.unload_node

        target_seed = seed if seed is not None else user_valves.seed
        if target_seed == -1 or target_seed is None:
            gen_seed = random.randint(1, 1500000000000)
        else:
            gen_seed = target_seed

        if text_node_id in workflow:
            inputs = workflow[text_node_id]["inputs"]
            inputs["tags"] = tags
            inputs["lyrics"] = lyrics
            inputs["bpm"] = bpm
            inputs["duration"] = duration
            inputs["language"] = language
            inputs["keyscale"] = key
            inputs["timesignature"] = str(time_signature)
            inputs["seed"] = gen_seed
            inputs["generate_audio_codes"] = user_valves.generate_audio_codes

        if checkpoint_node_id in workflow:
            ckpoint_node = workflow[checkpoint_node_id]["inputs"].get("ckpt_name", "")
            unet_name = workflow[checkpoint_node_id]["inputs"].get("unet_name", "")
            if ckpoint_node:
                workflow[checkpoint_node_id]["inputs"]["ckpt_name"] = (
                    self.valves.model_name
                )
            if unet_name:
                workflow[checkpoint_node_id]["inputs"]["unet_name"] = (
                    self.valves.model_name
                )

        dual_clip_node_id = self.valves.dual_clip_loader_node
        vae_loader_node_id = self.valves.vae_loader_node

        if self.valves.clip_name_1 and dual_clip_node_id in workflow:
            node_inputs = workflow[dual_clip_node_id].get("inputs", {})
            if "clip_name1" in node_inputs:
                node_inputs["clip_name1"] = self.valves.clip_name_1
            if "clip_name2" in node_inputs:
                node_inputs["clip_name2"] = (
                    self.valves.clip_name_2 or self.valves.clip_name_1
                )

        if self.valves.vae_name and vae_loader_node_id in workflow:
            node_inputs = workflow[vae_loader_node_id].get("inputs", {})
            if "vae_name" in node_inputs:
                node_inputs["vae_name"] = self.valves.vae_name

        if latent_node_id in workflow:
            inputs = workflow[latent_node_id]["inputs"]
            inputs["batch_size"] = batch_size
            inputs["seconds"] = duration

        if sampler_node_id in workflow:
            workflow[sampler_node_id]["inputs"]["seed"] = gen_seed
            workflow[sampler_node_id]["inputs"]["steps"] = steps

        if save_node_id in workflow:
            safe_title = re.sub(r"[^\w\s-]", "", song_title).strip()
            workflow[save_node_id]["inputs"]["filename_prefix"] = f"audio/{safe_title}"

        if not self.valves.unload_comfyui_models:
            if unload_node_id in workflow:
                unload_inputs = workflow[unload_node_id].get("inputs", {})
                source_link = unload_inputs.get("value")
                if source_link:
                    if save_node_id in workflow:
                        save_inputs = workflow[save_node_id].get("inputs", {})
                        current_audio_input = save_inputs.get("audio")
                        if (
                            isinstance(current_audio_input, list)
                            and len(current_audio_input) > 0
                            and str(current_audio_input[0]) == str(unload_node_id)
                        ):
                            workflow[save_node_id]["inputs"]["audio"] = source_link
                del workflow[unload_node_id]

        client_id = str(uuid.uuid4())
        ws_url = (
            self.valves.comfyui_api_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )
            + "/ws"
        )
        http_url = self.valves.comfyui_api_url

        comfyui_headers = {}
        if self.valves.comfyui_api_key:
            comfyui_headers["Authorization"] = f"Bearer {self.valves.comfyui_api_key}"

        # Use the gen_seed to deterministically pick a palette colour
        palette_seed = gen_seed % 1000000

        try:
            prompt_payload = {"prompt": workflow, "client_id": client_id}

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Generating {song_title}...",
                            "done": False,
                        },
                    }
                )

            result_data = await connect_submit_and_wait(
                ws_url, http_url, prompt_payload, client_id, self.valves.max_wait_time,
                headers=comfyui_headers,
            )

            audio_files = extract_audio_files(result_data)
            if not audio_files:
                raise Exception("No audio files generated.")

            track_list = []
            save_errors = []
            user_obj = None
            if self.valves.save_to_storage:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Saving to Open WebUI...", "done": False},
                        }
                    )
                user_obj = await resolve_user(__user__)
                if user_obj is None or __request__ is None:
                    raise MediaStorageError(
                        "the requesting user or request context is unavailable."
                    )

            for idx, finfo in enumerate(audio_files):
                fname = finfo["filename"]
                subfolder = finfo["subfolder"]
                track_title = (
                    song_title
                    if batch_size <= 1 or idx == 0
                    else f"{song_title} {idx + 1}"
                )
                view_url = comfyui_view_url(http_url, fname, subfolder)
                content_type = audio_content_type(fname)
                download_name = safe_filename(f"{track_title}{extension_for(content_type)}", content_type, "ace_step")

                if not self.valves.save_to_storage:
                    # Explicit admin opt-out of storage: stream from ComfyUI.
                    track_list.append({"title": track_title, "url": view_url, "name": download_name})
                    continue
                try:
                    saved = await download_to_storage(
                        view_url,
                        request=__request__,
                        user=user_obj,
                        metadata=__metadata__,
                        filename=download_name,
                        content_type=content_type,
                        headers=comfyui_headers,
                        prefix="ace_step",
                    )
                    track_list.append({"title": track_title, "url": saved["url"], "name": saved["name"]})
                except MediaStorageError as exc:
                    # No fallback to the internal ComfyUI URL: skip the track and say so.
                    log.warning("[ACE Step 1.5] Could not save track %d: %s", idx + 1, exc)
                    save_errors.append(str(exc))
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Warning: track {idx + 1} could not be saved: {exc}",
                                    "done": False,
                                },
                            }
                        )

            if not track_list:
                raise MediaStorageError(save_errors[0] if save_errors else "no audio could be saved.")

            if __event_emitter__:
                done_desc = "Generation complete!"
                if len(track_list) < len(audio_files):
                    done_desc = f"Generation complete! ({len(track_list)}/{len(audio_files)} tracks saved)"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": done_desc, "done": True},
                    }
                )

            stored = self.valves.save_to_storage
            count = f"{len(track_list)} versions" if len(track_list) > 1 else "the song"
            if self.valves.show_player_embed:
                final_html = generate_audio_player_embed(
                    track_list,
                    song_title,
                    tags,
                    lyrics,
                    palette_seed=palette_seed,
                    colorful=user_valves.colorful_player,
                )
                message = (
                    f"Song '{song_title}' generated; {count} shown to the user in an audio player above. "
                    "Do not repeat the audio as a link or markdown."
                )
                if stored:
                    context: Union[str, Dict[str, Any]] = {
                        "message": message,
                        "tracks": [{"title": t["title"], "url": t["url"]} for t in track_list],
                    }
                else:
                    # Never hand the internal ComfyUI URL to the model.
                    context = message + " The audio is streamed from ComfyUI and is not saved in Open WebUI."
                return embed_result(final_html, context)
            elif not stored:
                # Never hand the internal ComfyUI /view URLs to the model.
                return (
                    f"Song '{song_title}' generated ({count}), but it was not saved to Open WebUI "
                    "(save_to_storage is off) and the player is disabled (show_player_embed is off), "
                    "so there is no link to share. Ask an admin to enable show_player_embed or save_to_storage."
                )
            else:
                # Stored tracks: these are /api/v1/files/... links, safe to share.
                track_links = "\n".join(
                    f"- [{t['title']}]({t['url']})" for t in track_list
                )
                return f"Song '{song_title}' generated successfully!\n\nDownload links:\n{track_links}"

        except MediaStorageError as exc:
            error_msg = f"Generated, but could not save it to Open WebUI: {exc}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg
        except Exception as e:
            if isinstance(e, aiohttp.ClientError):
                # aiohttp messages carry the ComfyUI host:port; keep them in the log only.
                log.warning("ComfyUI request failed: %s", e)
                detail = f"could not reach ComfyUI ({type(e).__name__})"
            else:
                detail = str(e)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {detail}", "done": True},
                    }
                )
            return f"Error generating song: {detail}"
