"""
title: 3D Model Viewer
description: Show a GLB / glTF 3D model that is already in Open WebUI (an attached file or an /api/v1/files/<id> link) in an interactive inline 3D viewer with orbit controls, auto-rotate, fullscreen and download.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.6
license: MIT
required_open_webui_version: 0.9.0
"""

import asyncio
import html
import json
import logging
import mimetypes
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from open_webui.models.files import Files
from open_webui.models.users import UserModel, Users
from open_webui.storage.provider import Storage

log = logging.getLogger(__name__)

GLB_CONTENT_TYPE = "model/gltf-binary"
GLTF_CONTENT_TYPE = "model/gltf+json"
GLTF_CONTENT_TYPES = frozenset({GLB_CONTENT_TYPE, GLTF_CONTENT_TYPE})
GLTF_EXTENSIONS = (".glb", ".gltf")
GLB_MAGIC = b"glTF"
# Google <model-viewer> (Apache-2.0), pinned exact version + SRI (jsDelivr file hash).
# Kept identical to atlascloud_media_tool.py so both viewers render the same.
MODEL_VIEWER_VERSION = "4.4.0"
MODEL_VIEWER_MODULE_URL = (
    f"https://cdn.jsdelivr.net/npm/@google/model-viewer@{MODEL_VIEWER_VERSION}/dist/model-viewer.min.js"
)
MODEL_VIEWER_INTEGRITY = "sha256-GLLnEQNapHBe4oFTClXcIzj/Be7T1wrhbVQQUoKnXtU="

# Bare Open WebUI file id, and /api/v1/files/{id}[/content] (relative or absolute URL).
FILE_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")
OWUI_FILE_PATH_RE = re.compile(r"^/api/v1/files/([A-Za-z0-9-]+)(?:/content)?/?$")
# Same text for "missing" and "denied", like Open WebUI's own 404 (no existence leak).
NOT_FOUND_TEXT = "The 3D model file was not found, or you do not have access to it."

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


class ViewerError(RuntimeError):
    """User-facing error with a safe message."""


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


# ---------------------------------------------------------------------------
# Inline embed builders (copied verbatim from atlascloud_media_tool.py)
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
# 3D model viewer (Google <model-viewer>, GLB)
# --------------------------------------------------------------------------

_MODEL_3D_CSS = """
.mv-wrap { position: relative; width: 100%; max-width: 960px; margin: 0; height: __H__px; border-radius: 12px; overflow: hidden;
  background: radial-gradient(circle at 50% 35%, #ffffff 0%, #e9ebef 70%, #dfe2e7 100%); }
.mv-wrap:fullscreen { height: 100%; max-width: none; border-radius: 0; }
.mv-wrap:-webkit-full-screen { height: 100%; max-width: none; border-radius: 0; }
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
function reframe() {
  if (!mv || !mv.updateFraming) return;
  Promise.resolve(mv.updateFraming()).then(function () { if (mv && mv.jumpCameraToGoal) mv.jumpCameraToGoal(); });
}
var reframeTimer = null;
window.addEventListener('resize', function () {
  clearTimeout(reframeTimer);
  reframeTimer = setTimeout(reframe, 150);
});
// In a chat the mouse wheel should scroll the conversation, not zoom the
// model: zoom only with Ctrl/Cmd held or after clicking into the viewer.
var engaged = false;
wrap.addEventListener('pointerdown', function () { engaged = true; }, true);
wrap.addEventListener('pointerleave', function () { engaged = false; });
wrap.addEventListener('wheel', function (e) {
  if (!engaged && !e.ctrlKey && !e.metaKey) e.stopPropagation();
}, true);
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


# --------------------------------------------------------------------------
# File discovery / validation
# --------------------------------------------------------------------------

# Content types that say nothing about the format; the file name decides then.
GENERIC_CONTENT_TYPES = frozenset(
    {"", "application/octet-stream", "binary/octet-stream", "application/x-binary", "application/json", "text/plain"}
)
# Attachment entry types that never point at a single stored file.
NON_FILE_ENTRY_TYPES = frozenset({"collection", "web_search", "youtube", "url", "web", "note", "folder"})


def _normalize_type(value: Any) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def parse_file_reference(value: Any) -> Optional[str]:
    """File id from a bare id or an /api/v1/files/<id>[/content] link, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if FILE_ID_RE.match(text):
        return text
    try:
        parsed = urlparse(text)
    except ValueError:
        return None
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return None
    match = OWUI_FILE_PATH_RE.match(parsed.path or "")
    return match.group(1) if match else None


def model_kind(name: Any, content_type: Any) -> Optional[str]:
    """'glb' / 'gltf' from a file name and content type, else None."""
    lname = str(name or "").strip().lower()
    ctype = _normalize_type(content_type)
    if ctype == GLB_CONTENT_TYPE or (lname.endswith(".glb") and ctype in GENERIC_CONTENT_TYPES):
        return "glb"
    if ctype == GLTF_CONTENT_TYPE or (lname.endswith(".gltf") and ctype in GENERIC_CONTENT_TYPES):
        return "gltf"
    return None


def _entry_parts(entry: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
    """(file id, name, content type) of a chat attachment entry."""
    nested = entry.get("file") if isinstance(entry.get("file"), dict) else {}
    meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
    nested_meta = nested.get("meta") if isinstance(nested.get("meta"), dict) else {}
    name = (
        entry.get("name") or entry.get("filename") or nested_meta.get("name") or nested.get("filename")
        or meta.get("name") or ""
    )
    ctype = (
        entry.get("content_type") or entry.get("mime_type") or meta.get("content_type")
        or nested_meta.get("content_type") or ""
    )
    file_id = None
    for candidate in (nested.get("id"), entry.get("url"), entry.get("id")):
        file_id = parse_file_reference(candidate) if isinstance(candidate, str) else None
        if file_id:
            break
    return file_id, str(name), _normalize_type(ctype)


def _glb_ids_in(entries: Any) -> List[str]:
    """Ids of GLB/glTF attachments in a files list, most recent (last) first."""
    found: List[str] = []
    if not isinstance(entries, list):
        return found
    for entry in reversed(entries):
        if not isinstance(entry, dict) or str(entry.get("type") or "").lower() in NON_FILE_ENTRY_TYPES:
            continue
        file_id, name, ctype = _entry_parts(entry)
        if file_id and model_kind(name, ctype) and file_id not in found:
            found.append(file_id)
    return found


def find_attached_model(
    files: Any = None, messages: Any = None, metadata: Any = None
) -> Optional[str]:
    """Most recent GLB/glTF attached in the chat: the current request's files,
    then message file entries (newest message first), then __metadata__ files."""
    ids = _glb_ids_in(files)
    if ids:
        return ids[0]
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict):
                ids = _glb_ids_in(message.get("files"))
                if ids:
                    return ids[0]
    if isinstance(metadata, dict):
        ids = _glb_ids_in(metadata.get("files"))
        if ids:
            return ids[0]
    return None


def human_size(size: Any) -> str:
    if not isinstance(size, (int, float)) or isinstance(size, bool) or size < 0:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""


def _header_matches(path: str, kind: str) -> Optional[bool]:
    """True/False when the stored file's first bytes were read, None when skipped.
    Only for local storage paths: remote providers would download the whole file."""
    if not path or "://" in path:
        return None
    try:
        local_path = Storage.get_file(path)
        if not isinstance(local_path, str) or not os.path.isfile(local_path):
            return None
        with open(local_path, "rb") as handle:
            head = handle.read(64)
    except Exception as exc:
        log.debug("Skipping GLB header check (%s)", type(exc).__name__)
        return None
    if kind == "glb":
        return head[:4] == GLB_MAGIC
    return head.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] == b"{"


async def _has_read_access(file_id: str, user_obj: UserModel) -> bool:
    try:
        from open_webui.utils.access_control.files import has_access_to_file

        return bool(await has_access_to_file(file_id, "read", user_obj))
    except Exception as exc:
        log.warning("File access check failed (%s)", type(exc).__name__)
        return False


async def load_model_file(file_id: str, user_obj: Optional[UserModel]) -> Dict[str, Any]:
    """Look the file up and enforce read access exactly like Open WebUI's
    /api/v1/files/{id}/content (owner, admin, or has_access_to_file 'read')."""
    if user_obj is None:
        raise ViewerError("Cannot open the file without a signed-in user.")
    file_item = await Files.get_file_by_id(file_id)
    if not file_item:
        raise ViewerError(NOT_FOUND_TEXT)
    if not (
        file_item.user_id == user_obj.id
        or user_obj.role == "admin"
        or await _has_read_access(file_id, user_obj)
    ):
        raise ViewerError(NOT_FOUND_TEXT)

    meta = file_item.meta if isinstance(file_item.meta, dict) else {}
    name = str(meta.get("name") or file_item.filename or "model.glb")
    ctype = _normalize_type(meta.get("content_type") or mimetypes.guess_type(name)[0])
    kind = model_kind(name, ctype) or model_kind(file_item.filename, ctype)
    if not kind:
        raise ViewerError("That file is not a GLB / glTF 3D model (expected a .glb or .gltf file).")
    if not file_item.path:
        raise ViewerError("The stored 3D model file is unavailable.")
    if await asyncio.to_thread(_header_matches, file_item.path, kind) is False:
        raise ViewerError(f"That file is named like a {kind.upper()} model but its content is not valid {kind.upper()}.")
    return {"id": file_item.id or file_id, "name": name, "kind": kind, "size": meta.get("size")}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class Tools:
    class Valves(BaseModel):
        AUTO_ROTATE: bool = Field(
            default=True, description="Start the 3D viewer with the model slowly rotating."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    @staticmethod
    async def _emit_status(emitter: EventEmitter, description: str, done: bool) -> None:
        if not emitter:
            return
        try:
            await emitter({"type": "status", "data": {"description": description, "done": done}})
        except Exception as exc:
            log.debug("Status emit failed (%s)", type(exc).__name__)

    async def view_3d_model(
        self,
        file_id: str = "",
        __user__: Optional[dict] = None,
        __files__: Optional[List[dict]] = None,
        __messages__: Optional[List[dict]] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ):
        """
        Show a GLB or glTF 3D model that is already stored in Open WebUI in an interactive 3D viewer.
        Use it when the user asks to view, open, preview or inspect a 3D model (.glb / .gltf).
        :param file_id: Open WebUI file id or /api/v1/files/<id> link of the model. Leave empty to use the most recent GLB attached in this chat.
        :return: The interactive 3D viewer, shown to the user.
        """
        await self._emit_status(__event_emitter__, "Opening 3D model...", done=False)
        try:
            reference = (file_id or "").strip() if isinstance(file_id, str) else ""
            if reference:
                target = parse_file_reference(reference)
                if not target:
                    raise ViewerError(
                        "Unsupported file reference. Pass an Open WebUI file id or an /api/v1/files/<id> link."
                    )
            else:
                target = find_attached_model(__files__, __messages__, __metadata__)
                if not target:
                    raise ViewerError(
                        "No GLB / glTF model is attached to this chat. Attach a .glb file or pass its file id."
                    )
            info = await load_model_file(target, await resolve_user(__user__))
        except ViewerError as exc:
            await self._emit_status(__event_emitter__, str(exc), done=True)
            return f"Could not show the 3D model: {exc}"

        url = f"/api/v1/files/{info['id']}/content"
        size = human_size(info["size"])
        label = f"{info['kind'].upper()} - {size}" if size else info["kind"].upper()
        viewer = model_3d_viewer_embed(
            url,
            None,
            info["name"],
            "",
            download_name=info["name"],
            model_label=label,
            auto_rotate=self.valves.AUTO_ROTATE,
        )
        await self._emit_status(__event_emitter__, "3D model ready", done=True)
        message = (
            f"The 3D model '{info['name']}' is shown to the user in the interactive 3D viewer above; "
            f"do not repeat it as markdown. Download link: {url}."
        )
        if info["kind"] == "gltf":
            message += " Note: a .gltf that references external .bin or texture files may not load; GLB is self-contained."
        return embed_result(viewer, message)
