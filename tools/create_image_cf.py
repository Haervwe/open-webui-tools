"""
title: Cloudflare Workers AI Image Generator
author: tan-yong-sheng
author_url: https://github.com/tan-yong-sheng
description: Generate images using Cloudflare Workers AI text-to-image models with preprocessing for different model types
version: 1.2.0
license: MIT
requirements: aiohttp
required_open_webui_version: 0.9.1
"""

import asyncio
import aiohttp
import base64
import html
import io
import json
import logging
import mimetypes
import os
import re
import uuid
from typing import Dict, Any, Iterable, List, Optional, Callable, Sequence, Tuple, Union
from pydantic import BaseModel, Field

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from starlette.datastructures import Headers
from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import upload_file_handler

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


def sniff_image_type(data: bytes, declared: Optional[str] = None) -> str:
    """Real image MIME type from the magic bytes (FLUX returns JPEG, SD models PNG)."""
    head = bytes(data[:12])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    ct = normalize_content_type(declared, "")
    return ct if ct.startswith("image/") else "image/png"


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


def _get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration and preprocessing parameters."""
    model_configs = {
        "@cf/black-forest-labs/flux-1-schnell": {
            "max_prompt_length": 2048,
            "default_steps": 4,
            "max_steps": 8,
            "supports_negative_prompt": False,
            "supports_size": False,
            "supports_guidance": False,
            "supports_seed": True,
            "output_format": "base64",
            "recommended_for": "fast generation, high quality",
        },
        "@cf/stabilityai/stable-diffusion-xl-base-1.0": {
            "max_prompt_length": 1000,
            "default_steps": 15,  # Optimized: quality plateaus around 15-20 steps
            "max_steps": 20,
            "supports_negative_prompt": True,
            "supports_size": True,
            "supports_guidance": True,
            "supports_seed": True,
            "output_format": "binary",
            "recommended_for": "high quality, detailed images",
            "default_guidance": 12.0,  # Optimized: lower than 7.5 default to reduce over-adherence and distortion
            "guidance_range": "10.0-15.0",  # Sweet spot for SDXL quality without distortion
            "recommended_negative": "blurry, low quality, distorted, unrealistic, bad anatomy, poor quality, jpeg artifacts",
            "notes": "SDXL works best with guidance 10-15, detailed prompts, and comprehensive negative prompts",
        },
        "@cf/bytedance/stable-diffusion-xl-lightning": {
            "max_prompt_length": 1000,
            "default_steps": 4,
            "max_steps": 8,
            "supports_negative_prompt": True,
            "supports_size": True,
            "supports_guidance": True,
            "supports_seed": True,
            "output_format": "binary",
            "recommended_for": "fast generation with good quality",
        },
        "@cf/lykon/dreamshaper-8-lcm": {
            "max_prompt_length": 1000,
            "default_steps": 4,  # LCM models work best with 4-8 steps
            "max_steps": 8,  # Not 20! LCM needs fewer steps
            "supports_negative_prompt": True,
            "supports_size": True,
            "supports_guidance": True,
            "supports_seed": True,
            "output_format": "binary",
            "recommended_for": "photorealistic images (LCM - use 4-8 steps, lower guidance)",
            "guidance_range": "1.0-2.0",  # LCM works better with lower guidance
            "notes": "LCM model - requires 4-8 steps and lower guidance (1.0-2.0) for best results",
        },
    }

    # Check for unsupported models and provide helpful error
    unsupported_models = {
        "@cf/runwayml/stable-diffusion-v1-5-img2img": "Image-to-image functionality is not supported in this tool. Please use text-to-image generation instead.",
        "@cf/runwayml/stable-diffusion-v1-5-inpainting": "Image inpainting functionality is not supported in this tool. Please use text-to-image generation instead.",
    }

    if model_name in unsupported_models:
        raise ValueError(unsupported_models[model_name])

    return model_configs.get(
        model_name, model_configs["@cf/black-forest-labs/flux-1-schnell"]
    )


def _preprocess_prompt(prompt: str, model_name: str) -> str:
    """Preprocess prompt based on model requirements."""
    config = _get_model_config(model_name)
    max_length = config.get("max_prompt_length", 2048)

    # Truncate if too long
    if len(prompt) > max_length:
        prompt = prompt[: max_length - 3] + "..."

    # Model-specific prompt enhancements
    if "flux" in model_name.lower():
        # FLUX works well with detailed, artistic descriptions
        if not any(
            word in prompt.lower()
            for word in ["detailed", "high quality", "masterpiece"]
        ):
            prompt = f"detailed, high quality, {prompt}"

    elif "dreamshaper" in model_name.lower():
        # DreamShaper excels at photorealistic content
        if not any(
            word in prompt.lower() for word in ["photorealistic", "realistic", "photo"]
        ):
            prompt = f"photorealistic, {prompt}"

    elif "xl" in model_name.lower():
        # SDXL models benefit from style descriptors and detailed prompts
        if not any(
            word in prompt.lower()
            for word in [
                "cinematic",
                "artistic",
                "professional",
                "detailed",
                "high quality",
                "masterpiece",
                "ultra realistic",
            ]
        ):
            prompt = f"professional, cinematic, highly detailed, {prompt}"

    return prompt


def _build_request_payload(prompt: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """Build the request payload based on model capabilities."""
    config = _get_model_config(model_name)
    payload = {"prompt": _preprocess_prompt(prompt, model_name)}

    # Add parameters based on model support
    if config.get("supports_negative_prompt"):
        negative_prompt = kwargs.get("negative_prompt", "")

        # Auto-enhance negative prompt for SDXL if none provided
        if (
            "stable-diffusion-xl-base-1.0" in model_name.lower()
            and not negative_prompt.strip()
        ):
            negative_prompt = config.get("recommended_negative", "")

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

    if config.get("supports_size"):
        size = kwargs.get("size", "1024x1024")
        if "x" in size:
            width, height = map(int, size.split("x"))
            payload["width"] = max(256, min(2048, width))
            payload["height"] = max(256, min(2048, height))

    if config.get("supports_guidance"):
        guidance_value = kwargs.get("guidance", config.get("default_guidance", 7.5))

        # Special handling for LCM models - use lower guidance
        if "dreamshaper-8-lcm" in model_name.lower():
            guidance_value = min(guidance_value, 2.0)  # Cap at 2.0 for LCM
            if guidance_value > 2.0:
                guidance_value = 1.5  # Use 1.5 as default for LCM
        # Special handling for SDXL - use optimized guidance range
        elif "stable-diffusion-xl-base-1.0" in model_name.lower():
            # Use optimized guidance range for SDXL: 10-15 for better quality
            if guidance_value == 7.5:  # If using default, upgrade to optimized value
                guidance_value = config.get("default_guidance", 12.0)
            guidance_value = max(
                10.0, min(guidance_value, 15.0)
            )  # Clamp to optimal range

        payload["guidance"] = guidance_value

    if config.get("supports_seed") and kwargs.get("seed"):
        payload["seed"] = kwargs["seed"]

    # Steps configuration - use correct parameter name for each model
    steps = kwargs.get("steps", config.get("default_steps", 4))
    if "flux" in model_name.lower():
        payload["steps"] = min(steps, config.get("max_steps", 8))  # FLUX uses "steps"
    else:
        payload["num_steps"] = min(
            steps, config.get("max_steps", 20)
        )  # Others use "num_steps"

    # Handle image input for img2img and inpainting models
    if config.get("supports_image_input") and kwargs.get("image_b64"):
        payload["image_b64"] = kwargs["image_b64"]
        if kwargs.get("strength"):
            payload["strength"] = max(0.1, min(1.0, kwargs["strength"]))

    # Handle mask for inpainting
    if config.get("supports_mask") and kwargs.get("mask"):
        payload["mask"] = kwargs["mask"]

    return payload


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    class Valves(BaseModel):
        cloudflare_api_token: str = Field(
            default="", description="Your Cloudflare API Token",
            json_schema_extra={"input": {"type": "password"}},
        )
        cloudflare_account_id: str = Field("", description="Your Cloudflare Account ID")
        default_model: str = Field(
            "@cf/black-forest-labs/flux-1-schnell",
            description="Default image generation model (always used) - FLUX recommended over SDXL",
        )

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        negative_prompt: str = "",
        steps: int = 4,
        guidance: float = 7.5,
        seed: Optional[int] = None,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable] = None,
        __metadata__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Generate an image using Cloudflare Workers AI text-to-image models.

        :param prompt: Text description of the image to generate
        :param size: Image size in format 'widthxheight' (e.g., '1024x1024')
        :param negative_prompt: Text describing what to avoid in the image
        :param steps: Number of diffusion steps (model-dependent limits)
        :param guidance: How closely to follow the prompt (1.0-20.0)
        :param seed: Random seed for reproducible results
        """
        if (
            not self.valves.cloudflare_api_token
            or not self.valves.cloudflare_account_id
        ):
            return "Error: Cloudflare API Token and Account ID must be configured in the tool settings."

        if not prompt.strip():
            return "Error: Please provide a prompt for image generation."

        # Always use the configured default model (ignore any LLM model selection)
        model_name = self.valves.default_model

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating image with {model_name}...",
                        "done": False,
                    },
                }
            )

        try:
            # Validate the default model is supported
            config = _get_model_config(model_name)

            # Build request payload with model-specific preprocessing
            payload = _build_request_payload(
                prompt=prompt,
                model_name=model_name,
                size=size,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
            )

            # Log a warning if parameters were ignored due to model limitations
            ignored_params = []

            if negative_prompt and not config.get("supports_negative_prompt"):
                ignored_params.append("negative_prompt")
            if size != "1024x1024" and not config.get("supports_size"):
                ignored_params.append("size")
            if guidance != 7.5 and not config.get("supports_guidance"):
                ignored_params.append("guidance")

            if ignored_params and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Note: {', '.join(ignored_params)} not supported by {model_name}",
                            "done": False,
                        },
                    }
                )

            # Make API request
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.valves.cloudflare_account_id}/ai/run/{model_name}"
            headers = {
                "Authorization": f"Bearer {self.valves.cloudflare_api_token}",
                "Content-Type": "application/json",
            }

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Sending request to Cloudflare Workers AI...",
                            "done": False,
                        },
                    }
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        # Cloudflare's own error message only, never the raw body.
                        detail = ""
                        try:
                            err_json = await response.json(content_type=None)
                            errors = err_json.get("errors") if isinstance(err_json, dict) else None
                            if errors and isinstance(errors[0], dict):
                                detail = str(errors[0].get("message") or "")[:200]
                        except Exception:
                            pass
                        error_msg = f"API Error {response.status}" + (f": {detail}" if detail else "")
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Error: {error_msg}",
                                        "done": True,
                                    },
                                }
                            )
                        return f"Error: {error_msg}"

                    # Handle different response formats based on actual Cloudflare API behavior
                    content_type = response.headers.get("content-type", "").lower()
                    image_binary = None

                    if "application/json" in content_type:
                        # JSON response - FLUX models return base64
                        try:
                            result = await response.json()

                            # Extract image data from Cloudflare's nested structure
                            if "result" in result and isinstance(
                                result["result"], dict
                            ):
                                image_data = result["result"].get("image")
                            else:
                                image_data = result.get("image")

                            if not image_data:
                                return f"Error: No image data found in response. Keys: {list(result.keys())}"

                            # Decode base64
                            image_binary = base64.b64decode(image_data)

                        except Exception as e:
                            return f"Error processing JSON response ({type(e).__name__})."

                    else:
                        # Binary response - other Stable Diffusion models
                        image_binary = await response.read()
                        if len(image_binary) < 100:
                            return f"Error: Invalid binary image data received"

            if not image_binary:
                return "Error: Cloudflare returned no image data."

            # FLUX returns JPEG, the Stable Diffusion models return PNG.
            image_type = sniff_image_type(image_binary, content_type)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Saving image to Open WebUI...", "done": False},
                    }
                )
            try:
                saved = await save_media_bytes(
                    image_binary,
                    request=__request__,
                    user=__user__,
                    content_type=image_type,
                    metadata=__metadata__,
                    chat_id=__chat_id__,
                    message_id=__message_id__,
                    prefix="cloudflare_image",
                )
            except MediaStorageError as e:
                error_msg = f"Image generated, but could not save it to Open WebUI: {e}"
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": error_msg, "done": True}}
                    )
                return f"Error: {error_msg}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Image generated", "done": True},
                    }
                )

            return embed_result(
                image_gallery_embed(
                    saved["url"],
                    prompt=prompt,
                    download_names=[saved.get("name") or f"image{extension_for(image_type)}"],
                ),
                "The image was generated and is already displayed to the user in an embed above. "
                "Do not repeat it as markdown or links. "
                f"File URL (only if the user asks for it): {saved['url']}",
            )

        except ValueError as e:
            # Handle unsupported model errors
            error_msg = str(e)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {error_msg}", "done": True},
                    }
                )
            return f"Error: {error_msg}"

        except asyncio.TimeoutError:
            error_msg = "Request timed out. The image generation is taking longer than expected."
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return f"Error: {error_msg}"

        except Exception as e:
            log.exception("Cloudflare image generation failed")
            error_msg = f"Unexpected error ({type(e).__name__})."
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return f"Error: {error_msg}"

    async def list_models(self, __event_emitter__: Optional[Callable] = None) -> str:
        """
        List available Cloudflare Workers AI image generation models with their capabilities.
        """
        models_info = {
            "@cf/black-forest-labs/flux-1-schnell": "FLUX.1 Schnell - Fast, high-quality generation (12B params)",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0": "Stable Diffusion XL - High quality, detailed images",
            "@cf/bytedance/stable-diffusion-xl-lightning": "SDXL Lightning - Fast generation with good quality",
            "@cf/lykon/dreamshaper-8-lcm": "DreamShaper 8 LCM - Photorealistic images",
        }

        result = "Available Cloudflare Workers AI Image Generation Models:\n\n"
        for model_id, description in models_info.items():
            config = _get_model_config(model_id)
            result += f"**{model_id}**\n"
            result += f"  Description: {description}\n"
            result += (
                f"  Recommended for: {config.get('recommended_for', 'general use')}\n"
            )
            result += f"  Max steps: {config.get('max_steps', 'N/A')}\n"
            result += f"  Supports negative prompts: {config.get('supports_negative_prompt', False)}\n"
            result += (
                f"  Supports custom size: {config.get('supports_size', False)}\n\n"
            )

        result += f"Current default model: {self.valves.default_model}\n"
        result += "\nNote: Image-to-image and inpainting models are not supported in this tool."

        return result
