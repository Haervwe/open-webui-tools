"""
title: Veo 3 Video Generation Pipe
authors:
    - Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
description: Generate videos using Google's Veo 3.1 model via Gemini API.
required_open_webui_version: 0.9.1
requirements: google-genai
version: 2.2.0
license: MIT

This pipe generates videos using Google's Veo 3.1 model through the Gemini API.
It supports text-to-video generation, image-to-video, and reference images.
"""

import asyncio
import base64
import html
import io
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from fastapi import HTTPException, UploadFile
from google import genai
from google.genai.types import Image
from open_webui.models.chats import Chats
from open_webui.models.users import User, UserModel, Users
from open_webui.routers.files import upload_file_handler  # type: ignore
from open_webui.utils.chat import generate_chat_completion  # type: ignore
from pydantic import BaseModel, Field
from starlette.datastructures import Headers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log = logger

# --- Open WebUI media storage helpers (shared across open-webui-tools media plugins) ---
# Saves bytes with upload_file_handler(process=False) and links the file to the
# chat message (Chats.insert_chat_files) so shared chats can play it. Failures
# raise MediaStorageError with a user-safe message (e.g. the 413 "Max Upload
# Size" case).

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


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(
            title="Google API Key",
            default="",
            description="Google API key for Gemini API access.",
            json_schema_extra={"input": {"type": "password"}},
        )
        MODEL: str = Field(
            title="Veo Model",
            default="veo-3.1-generate-preview",
            description="The Veo model to use for video generation.",
        )
        ENHANCE_PROMPT: bool = Field(
            title="Enhance Prompt",
            default=False,
            description="Use vision model to enhance prompt",
        )
        VISION_MODEL_ID: str = Field(
            title="Vision Model ID",
            default="",
            description="Vision model to be used as prompt enhancer",
        )
        ENHANCER_SYSTEM_PROMPT: str = Field(
            title="Enhancer System Prompt",
            default="""
            You are a video prompt engineering assistant.
            For each request, you will receive a user-provided prompt for video generation.
            Generate a single, improved video generation prompt for the Veo model using best practices.
            Be specific and descriptive: use exact color names, detailed adjectives, and clear action verbs.
            Focus on cinematic elements, camera movements, lighting, and visual style.
            Include timing and pacing descriptions where appropriate.
            Output only the final enhanced prompt with no additional explanation or commentary.
            """,
            description="System prompt to be used on the prompt enhancement process",
        )
        MAX_WAIT_TIME: int = Field(
            title="Max Wait Time",
            default=1200,
            description="Max wait time for video generation (seconds).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _extract_last_user_text(self, messages: list[Dict[str, Any]]) -> Optional[str]:
        """Extract the last user message text."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    ]
                    return " ".join(text_parts)
        return None

    def setup_inputs(
        self, messages: list[Dict[str, Any]]
    ) -> tuple[Optional[str], list[Dict[str, str]]]:
        """Extract prompt and base64 images from messages."""
        prompt: Optional[str] = None
        base64_images: list[Dict[str, str]] = []

        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                image_urls: list[str] = []

                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            # Use text from the most recent user message
                            if not prompt:
                                prompt = item.get("text", "")
                        elif item.get("type") == "image_url" and item.get(
                            "image_url", {}
                        ).get("url"):
                            image_urls.append(item["image_url"]["url"])
                elif isinstance(content, str):
                    # Use text from the most recent user message
                    if not prompt:
                        prompt = content

                # Extract base64 images (collect up to 3 images)
                for image_url in image_urls:
                    if len(base64_images) >= 3:
                        break
                    try:
                        if "base64," in image_url:
                            parts = image_url.split("base64,", 1)
                            mime_part = parts[0]
                            if mime_part.startswith("data:"):
                                # "data:image/png;" -> "image/png"
                                mime_type = (
                                    mime_part[5:].split(";", 1)[0].strip()
                                    or "image/jpeg"
                                )
                            else:
                                mime_type = "image/jpeg"  # default
                            base64_data = parts[1]
                            base64_images.append(
                                {"base64": base64_data, "mime_type": mime_type}
                            )
                        elif image_url.startswith("data:") and "," in image_url:
                            mime_part, base64_data = image_url.split(",", 1)
                            mime_type = (
                                mime_part[5:].split(";", 1)[0].strip() or "image/jpeg"
                            )
                            base64_images.append(
                                {"base64": base64_data, "mime_type": mime_type}
                            )
                        else:
                            # assume it's base64 without prefix
                            base64_images.append(
                                {"base64": image_url, "mime_type": "image/jpeg"}
                            )
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error while extracting base64 image: {e}"
                        )

                # Stop after processing the most recent user message
                if prompt or base64_images:
                    break

        return prompt, base64_images

    async def _save_video_and_get_public_url(
        self,
        request: Any,
        video_data: bytes,
        content_type: str,
        user: Any,
        metadata: Optional[dict] = None,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save the video to Open WebUI storage, linked to the chat message.

        Returns the save_media_bytes() result; ``url`` is the relative
        /api/v1/files/{id}/content path. Raises MediaStorageError.
        """
        return await save_media_bytes(
            video_data,
            request=request,
            user=user,
            content_type=content_type or "video/mp4",
            filename=f"veo3_video_{int(time.time())}.mp4",
            metadata=metadata,
            chat_id=chat_id,
            message_id=message_id,
            prefix="veo3_video",
        )

    async def enhance_prompt(
        self,
        prompt: str,
        user: User,
        request: Any,
        event_emitter: Callable[..., Any],
    ) -> str:
        """Enhance the prompt using vision model."""
        try:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "level": "info",
                        "description": "Enhancing prompt...",
                        "done": False,
                    },
                }
            )

            payload: Dict[str, Any] = {
                "model": self.valves.VISION_MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": self.valves.ENHANCER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this video generation prompt: {prompt}",
                    },
                ],
                "stream": False,
            }

            resp_data: Dict[str, Any] = await generate_chat_completion(
                request, payload, user
            )
            enhanced_prompt: str = str(resp_data["choices"][0]["message"]["content"])

            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "level": "info",
                        "description": "Prompt enhanced successfully.",
                        "done": False,
                    },
                }
            )

            return enhanced_prompt
        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}", exc_info=True)
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "level": "warning",
                        "description": "Failed to enhance prompt, using the original prompt.",
                        "done": False,
                    },
                }
            )
            return prompt

    @staticmethod
    def _enhanced_prompt_block(enhanced_prompt: Optional[str]) -> str:
        """Collapsible 'Enhanced Prompt' block for the returned message (LLM text escaped)."""
        if not enhanced_prompt:
            return ""
        return (
            "<details>\n<summary>Enhanced Prompt</summary>\n\n"
            f"{html.escape(enhanced_prompt.strip())}\n\n</details>\n\n"
        )

    @staticmethod
    async def _fail(
        emitter: Callable[..., Any], description: str, prefix: str = ""
    ) -> str:
        """Emit an error status and return the same text as the message content."""
        await emitter(
            {
                "type": "status",
                "data": {
                    "status": "complete",
                    "level": "error",
                    "description": description,
                    "done": True,
                },
            }
        )
        return f"{prefix}❌ {description}"

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Callable[..., Any],
        __request__: Any = None,
        __task__: Any = None,
        __event_call__: Optional[Callable[[Dict[str, Any]], Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Main function of the Pipe class.
        Generates videos using Veo 3.1 via Gemini API.
        """
        # Background tasks (title/tags/follow-ups...) must not start a paid video job.
        if __task__:
            return "Veo 3 Video"

        if __event_emitter__ is None:

            async def __event_emitter__(_event: Dict[str, Any]) -> None:
                return None

        # Per-request state stays local: Open WebUI caches the function module, so
        # one Pipe instance serves concurrent requests (Veo polls for minutes).
        emitter: Callable[..., Any] = __event_emitter__
        request = __request__
        user = await Users.get_user_by_id(__user__["id"])

        # Extract prompt and image from messages
        prompt, base64_images = self.setup_inputs(body.get("messages", []))
        prompt = (prompt or "").strip()

        if not prompt:
            return await self._fail(
                emitter,
                "No prompt provided. Please provide a description for the video to generate."
            )

        # Enhance prompt if enabled
        enhanced_block = ""
        if self.valves.ENHANCE_PROMPT:
            enhanced = await self.enhance_prompt(
                prompt,
                user,
                request,
                emitter,
            )
            if enhanced and enhanced != prompt:
                enhanced_block = self._enhanced_prompt_block(enhanced)
                prompt = enhanced

        # Check API key
        if not self.valves.GOOGLE_API_KEY:
            return await self._fail(
                emitter,
                "Google API key not configured. Please set GOOGLE_API_KEY in valves.",
                enhanced_block,
            )

        await emitter(
            {
                "type": "status",
                "data": {
                    "status": "in_progress",
                    "level": "info",
                    "description": "Generating video...",
                    "done": False,
                },
            }
        )

        try:
            client = genai.Client(api_key=self.valves.GOOGLE_API_KEY)

            # Generate video. google-genai's client is synchronous: run every call in a
            # worker thread so the Open WebUI event loop is never blocked.
            if base64_images:
                # Image-to-video generation
                # Note: The Gemini API currently supports one image for video generation.
                try:
                    first_image = base64_images[0]
                    image_bytes = base64.b64decode(first_image["base64"])
                    image_input = Image(
                        image_bytes=image_bytes, mime_type=first_image["mime_type"]
                    )
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    return await self._fail(
                        emitter,
                        f"Invalid image format provided: {e}", enhanced_block
                    )

                operation = await asyncio.to_thread(
                    client.models.generate_videos,
                    model=self.valves.MODEL,
                    prompt=prompt,
                    image=image_input,
                )
            else:
                # Text-to-video generation
                operation = await asyncio.to_thread(
                    client.models.generate_videos,
                    model=self.valves.MODEL,
                    prompt=prompt,
                )

            # Poll for completion
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > self.valves.MAX_WAIT_TIME:
                    return await self._fail(
                        emitter,
                        f"Video generation timed out after {self.valves.MAX_WAIT_TIME} seconds.",
                        enhanced_block,
                    )

                await asyncio.sleep(10)
                operation = await asyncio.to_thread(client.operations.get, operation)

            # Calculate total elapsed time
            total_elapsed = int(time.time() - start_time)

            op_error = getattr(operation, "error", None)
            if op_error:
                detail = (
                    op_error.get("message") if isinstance(op_error, dict) else None
                )
                raise Exception(
                    f"Veo reported an error: {str(detail)[:300]}"
                    if detail
                    else "Veo reported an error."
                )
            if not operation.response or not operation.response.generated_videos:
                raise Exception(
                    "Video generation operation completed without a response."
                )

            # Download video into memory (no temp file in the server's working directory)
            generated_video = operation.response.generated_videos[0]
            downloaded = await asyncio.to_thread(
                client.files.download, file=generated_video.video
            )
            video_data = (
                downloaded
                if isinstance(downloaded, (bytes, bytearray)) and downloaded
                else getattr(generated_video.video, "video_bytes", None)
            )
            if not video_data:
                raise Exception("Veo returned an empty video.")
            video_mime = (
                getattr(generated_video.video, "mime_type", None) or "video/mp4"
            )

            await emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "level": "info",
                        "description": "Saving video to Open WebUI...",
                        "done": False,
                    },
                }
            )

            # Save to Open WebUI storage, linked to this chat message
            saved = await self._save_video_and_get_public_url(
                request,
                bytes(video_data),
                video_mime,
                user,
                metadata=__metadata__,
                chat_id=__chat_id__,
                message_id=__message_id__,
            )
            public_video_url = saved["url"]
            ext = os.path.splitext(saved.get("name") or "")[1] or ".mp4"

            # Shared video player embed (escaped prompt, iframe:height, auth-aware)
            video_embed = video_player_embed(
                public_video_url,
                prompt=prompt,
                title="Veo 3 Video",
                download_names=[f"veo3_video{ext}"],
            )

            await emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "level": "success",
                        "description": f"Video generated successfully in {total_elapsed} seconds!",
                        "done": True,
                    },
                }
            )

            # Emit the video embed message
            await emitter(
                {
                    "type": "embeds",
                    "data": {
                        "embeds": [video_embed],
                    },
                }
            )

            # Embeds are not rendered in shared / read-only views: keep a link in the text.
            return (
                f"{enhanced_block}🎬 Video generated in {total_elapsed} seconds. "
                f"[Download video]({public_video_url})"
            )

        except MediaStorageError as e:
            logger.error(f"Saving the video failed: {e}")
            return await self._fail(
                emitter,
                f"The video was generated, but could not be saved to Open WebUI: {e}",
                enhanced_block,
            )
        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            return await self._fail(
                emitter,
                f"Video generation failed: {str(e)}", enhanced_block
            )
