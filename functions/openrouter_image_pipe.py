"""
title: OpenRouter Image Adapter Pipe
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.1.0
required_open_webui_version: 0.9.1
description: OpenRouter Adapter Pipe for Open WebUI Tools.
Features: Settings/valves: API key, allowed models, use websearch. Async streaming inference.
Built-in websearch and image generation support.
Streaming completions only
Citations and image event emission. Images are saved to the owui backend, linked to the chat and URLs are emitted (not base64).
"""

import aiohttp
import base64
import binascii
import html
import io
import json
import logging
import mimetypes
import os
import re
import traceback
import uuid
from typing import (
    List,
    Dict,
    Any,
    Callable,
    Awaitable,
    Iterable,
    Optional,
    AsyncGenerator,
    Sequence,
    Tuple,
    Union,
)

from fastapi import HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import upload_file_handler  # type: ignore

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Open WebUI media storage (stores the file and links it to the chat message)
# --------------------------------------------------------------------------


TEMPORARY_CHAT_ID_PREFIXES = ("temporary:", "local:")


CHANNEL_CHAT_ID_PREFIX = "channel:"


NON_SAVED_CHAT_ID_PREFIXES = (*TEMPORARY_CHAT_ID_PREFIXES, CHANNEL_CHAT_ID_PREFIX)


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


# --------------------------------------------------------------------------
# Inline image embed (escaped text, iframe:height, same-origin-only auth fetch)
# --------------------------------------------------------------------------


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


class Pipe:
    class Valves(BaseModel):
        API_KEY: str = Field(
            default="", description="OpenRouter API key",
            json_schema_extra={"input": {"type": "password"}},
        )
        ALLOWED_MODELS: List[str] = Field(
            default_factory=list, description="Allowed model slugs"
        )
        USE_WEBSEARCH: bool = Field(
            default=False,
            description="Enable web search plugin for all models, you can also add :online to a model ID to enable it per-model basis",
        )
        USE_IMAGE_EMBEDDING: bool = Field(
            default=False,
            description="Enable image embedding as html for images generated by the model, false to emit as message content only in markdown format",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.content_buffer = ""
        self.assistant_content = ""
        self.reasoning_start = None
        self.reasoning_text = ""

    def pipes(self) -> List[Dict[str, str]]:
        return [
            {"id": f"openrouter-{model}-pipe", "name": f"OpenRouter {model} Pipe"}
            for model in self.valves.ALLOWED_MODELS
        ]

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any] = {},
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        __request__: Optional[Request] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
    ) -> Union[str, AsyncGenerator[Any, None], None]:
        # Per-request values stay local: the Pipe instance is shared between
        # concurrent requests, so they must not live on self.
        request = __request__
        user_model = await resolve_user(__user__)
        chat_id, message_id = chat_context(__metadata__, __chat_id__, __message_id__)
        pipe_id = body.get("model")
        model = None
        if pipe_id:
            if "-" in pipe_id and pipe_id.endswith("-pipe"):
                model = pipe_id.split("-", 1)[1][: -len("-pipe")]
            elif pipe_id.endswith("-pipe"):
                model = pipe_id[: -len("-pipe")]
            else:
                model = pipe_id
        messages = body.get("messages", [])
        modalities = body.get("modalities")

        def get_slug(model_id: str) -> str:
            return model_id.split("/")[-1]

        allowed_slugs = [get_slug(m) for m in self.valves.ALLOWED_MODELS]
        model_slug = get_slug(model) if model else None
        if model_slug not in allowed_slugs:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "type": "error",
                            "description": f"Model '{model}' not allowed",
                            "done": True,
                        },
                    }
                )
            return
        if not model:
            return

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "include_reasoning": True,
        }

        async def get_model_capabilities(model_id: str) -> List[str]:
            url = "https://openrouter.ai/api/v1/models"
            headers = {"Authorization": f"Bearer {self.valves.API_KEY}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logging.error(f"Failed to fetch models: {resp.status}")
                        return ["text"]
                    data = await resp.json()
                    for m in data.get("data", []):
                        if m.get("id") == model_id or get_slug(
                            m.get("id", "")
                        ) == get_slug(model_id):
                            # OpenRouter lists these under architecture.output_modalities
                            architecture = m.get("architecture") or {}
                            return (
                                architecture.get("output_modalities")
                                or m.get("modalities")
                                or ["text"]
                            )
            return ["text"]

        modalities = await get_model_capabilities(model)
        payload["modalities"] = modalities

        if getattr(self.valves, "USE_WEBSEARCH", False):
            payload["plugins"] = [{"id": "web"}]

        headers = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json",
        }

        # Text and reasoning are yielded (not emitted as chat:message events) so
        # they become part of the pipe output, which OWUI >= 0.12 persists.
        async def stream_response(
            payload: Dict[str, Any], headers: Dict[str, str]
        ) -> AsyncGenerator[Any, None]:
            self.assistant_content = ""
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logging.error(f"OpenRouter API error: {resp.status}")
                        yield {
                            "error": {
                                "detail": f"OpenRouter API error: {resp.status} {error_text}"
                            }
                        }
                        return
                    buffer = b""
                    async for chunk in resp.content.iter_any():
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            try:
                                line = line.decode().strip()
                                if len(line) > 1000:
                                    logging.info(
                                        f"Raw line from OpenRouter: {line[:1000]}..."
                                    )
                                else:
                                    logging.info(f"Raw line from OpenRouter: {line}")
                                if not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    break

                                try:
                                    event = json.loads(data)
                                except Exception as e:
                                    logging.error(
                                        f"Failed to parse JSON: {e}, data: {data[:1000]}"
                                    )
                                    continue
                                delta = event.get("choices", [{}])[0].get("delta", {})

                                content = delta.get("content")
                                reasoning = delta.get("reasoning")

                                # Handle reasoning_details first if present in delta
                                if "reasoning_details" in delta:
                                    pass  # Not used for emission

                                # Reasoning is yielded as an OpenAI-style delta; OWUI
                                # renders and closes the reasoning block itself.
                                if reasoning:
                                    self.reasoning_text += reasoning
                                    yield {
                                        "choices": [
                                            {"delta": {"reasoning": reasoning}}
                                        ]
                                    }

                                if content:
                                    logging.info(
                                        f"Emitting content: {content[:1000]}..."
                                    )
                                    self.assistant_content += content
                                    yield content

                                if "images" in delta:
                                    urls: List[str] = []
                                    names: List[str] = []
                                    for image in delta["images"]:
                                        data_url = (image.get("image_url") or {}).get("url")
                                        if not (
                                            isinstance(data_url, str)
                                            and data_url.startswith("data:image/")
                                        ):
                                            continue
                                        try:
                                            saved = await self._save_image(
                                                request,
                                                data_url,
                                                user_model,
                                                chat_id,
                                                message_id,
                                            )
                                        except MediaStorageError as e:
                                            msg = f"Generated an image, but could not save it to Open WebUI: {e}"
                                            if __event_emitter__:
                                                await __event_emitter__(
                                                    {
                                                        "type": "status",
                                                        "data": {
                                                            "description": msg,
                                                            "done": True,
                                                        },
                                                    }
                                                )
                                            notice = f"\n\n> {msg}\n\n"
                                            self.assistant_content += notice
                                            yield notice
                                            continue
                                        urls.append(saved["url"])
                                        names.append(saved.get("name") or "image.png")
                                    if urls:
                                        if self.valves.USE_IMAGE_EMBEDDING and __event_emitter__:
                                            await __event_emitter__(
                                                {
                                                    "type": "embeds",
                                                    "data": {
                                                        "embeds": [
                                                            image_gallery_embed(
                                                                urls,
                                                                download_names=names,
                                                            )
                                                        ]
                                                    },
                                                }
                                            )
                                        else:
                                            markdown_links = "\n".join(
                                                [
                                                    f"![generate_image]({url})"
                                                    for url in urls
                                                ]
                                            )
                                            self.assistant_content += f"\n{markdown_links}\n"
                                            yield f"\n{markdown_links}\n"

                                if "annotations" in delta:
                                    for annotation in delta["annotations"]:
                                        citation_data: Dict[str, Any] = {}
                                        url: Optional[str] = None
                                        title: Optional[str] = None
                                        document: List[str] = []
                                        metadata: List[Dict[str, Any]] = []

                                        if annotation.get("type") == "url_citation":
                                            url_info = annotation.get(
                                                "url_citation", {}
                                            )
                                            url = url_info.get("url")
                                            title = url_info.get("title")

                                            if title:
                                                document.append(title)

                                            meta = {"source": url, "title": title}
                                            if "start_index" in url_info:
                                                meta["start_index"] = url_info[
                                                    "start_index"
                                                ]
                                            if "end_index" in url_info:
                                                meta["end_index"] = url_info[
                                                    "end_index"
                                                ]
                                            metadata.append(meta)

                                        if not document and annotation:
                                            document.append(str(annotation))
                                        citation_data = {
                                            "document": document,
                                            "metadata": metadata,
                                            "source": {
                                                "name": title or url or "Citation",
                                                "url": url or "",
                                            },
                                        }
                                        if __event_emitter__:
                                            await __event_emitter__(
                                                {
                                                    "type": "citation",
                                                    "data": citation_data,
                                                }
                                            )
                            except Exception as e:
                                if __event_emitter__:
                                    await __event_emitter__(
                                        {
                                            "type": "status",
                                            "data": {
                                                "type": "error",
                                                "description": f"Streaming error: {e}\n{traceback.format_exc()}",
                                                "done": True,
                                            },
                                        }
                                    )
                                raise

                    # Clear accumulators
                    self.content_buffer = ""
                    self.reasoning_start = None
                    self.reasoning_text = ""

        # OWUI emits chat:completion itself once the pipe output is consumed.
        if body.get("stream", False):
            return stream_response(payload, headers)

        # Non-stream callers (e.g. title/tag tasks) need a plain string.
        result = ""
        async for chunk in stream_response(payload, headers):
            if isinstance(chunk, str):
                result += chunk
            elif isinstance(chunk, dict) and "error" in chunk:
                raise RuntimeError(chunk["error"]["detail"])
        return result

    async def _save_image(
        self,
        request: Optional[Request],
        data_url: str,
        user: Optional[UserModel],
        chat_id: Optional[str],
        message_id: Optional[str],
    ) -> Dict[str, Any]:
        """Decode an OpenRouter data: URL, store it in Open WebUI and link it to
        the chat message. Raises MediaStorageError with a user-safe message."""
        try:
            header, b64data = data_url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
        except (ValueError, binascii.Error):
            raise MediaStorageError("The model returned invalid image data.") from None
        content_type = normalize_content_type(
            header[len("data:"):].split(";", 1)[0], "image/png"
        )
        if not content_type.startswith("image/"):
            content_type = "image/png"
        return await save_media_bytes(
            img_bytes,
            request=request,
            user=user,
            content_type=content_type,
            chat_id=chat_id,
            message_id=message_id,
            prefix="openrouter_image",
        )
