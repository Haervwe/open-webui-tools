"""
title: Doodle Paint
description: Toggleable filter that opens a paint canvas before sending each message, letting you attach a hand-drawn sketch to your prompt.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.2.0
license: MIT
required_open_webui_version: 0.6.5
"""

import open_webui.models.messages
import logging
import uuid
import base64
import io
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional

from fastapi import UploadFile
from open_webui.constants import TASKS
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
from open_webui.utils.misc import get_last_user_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doodle_paint")


class Filter:
    class Valves(BaseModel):
        canvas_width: int = Field(
            default=512,
            description="Drawing canvas width in pixels.",
        )
        canvas_height: int = Field(
            default=512,
            description="Drawing canvas height in pixels.",
        )
        default_brush_size: int = Field(
            default=5,
            description="Default brush/pen size in pixels.",
        )
        background_color: str = Field(
            default="#FFFFFF",
            description="Canvas background colour (hex).",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self._last_doodle_file = None
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2U9ImN1cnJlbnRDb2xvciI+PHBhdGggc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBkPSJNOS41MyAxNi4xMjJhMyAzIDAgMCAwLTUuNzggMS4xMjggMi4yNSAyLjI1IDAgMCAxLTIuNCAxLjYxNCAuMTEuMTEgMCAwIDAgLjA1NS4yMUE0Ljk3NCA0Ljk3NCAwIDAgMCA1LjI1IDIwLjVjLjczIDAgMS40My0uMTYgMi4wNi0uNDM4QTE1LjAxIDE1LjAxIDAgMCAwIDkuNTMgMTYuMTIyWk05LjUzIDE2LjEyMmExNS4wNDIgMTUuMDQyIDAgMCAxIDMuMDktNC42MTFBMTUuMDgyIDE1LjA4MiAwIDAgMSAxOS43NTIgNS4yODVhLjExLjExIDAgMCAxIC4xNjUgMCAuMTEuMTEgMCAwIDEgLjAyNC4wM0ExNC41NSAxNC41NSAwIDAgMSAyMS42IDguNmwuMDQ0LjA2NGExNS4wMTYgMTUuMDE2IDAgMCAxLTIuMDg1IDUuMjE0Yy0uMjQuMzY2LS40OTYuNzItLjc2OCAxLjA2Ii8+PC9zdmc+"""

    def _build_canvas_js(self) -> str:
        """Build the JavaScript code for the fullscreen paint canvas overlay."""
        cw = self.valves.canvas_width
        ch = self.valves.canvas_height
        bs = self.valves.default_brush_size
        bg = self.valves.background_color

        return f"""
return (function() {{
  return new Promise((resolve) => {{

    // --- Create overlay ---
    const overlay = document.createElement('div');
    overlay.id = '__doodle_overlay__';
    overlay.style.cssText = `
      position: fixed; inset: 0; z-index: 999999;
      background: rgba(0,0,0,0.6); backdrop-filter: blur(12px);
      display: flex; align-items: center; justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    `;

    // --- Panel ---
    const panel = document.createElement('div');
    panel.style.cssText = `
      background: rgba(20, 20, 25, 0.25); backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px; padding: 20px; max-width: 600px; width: 95vw;
      max-height: 95vh; overflow-y: auto;
      box-shadow: 0 8px 24px rgba(0,0,0,0.2);
      display: flex; flex-direction: column; gap: 12px;
    `;
    overlay.appendChild(panel);

    // --- Helper: create element ---
    function el(tag, styles, attrs) {{
      const e = document.createElement(tag);
      if (styles) Object.assign(e.style, typeof styles === 'string' ? {{cssText: styles}} : styles);
      if (attrs) Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k,v));
      return e;
    }}

    // --- Header ---
    const header = el('div');
    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;';
    const titleWrap = el('div');
    const titleEl = el('div');
    titleEl.textContent = '\u270e Doodle Paint';
    titleEl.style.cssText = 'font-size:16px;font-weight:600;color:#f0f0f0;letter-spacing:-0.2px;';
    const subtitleEl = el('div');
    subtitleEl.textContent = 'SKETCH TOOL';
    subtitleEl.style.cssText = 'font-size:10px;color:#888;font-weight:400;text-transform:uppercase;letter-spacing:0.8px;margin-top:2px;';
    titleWrap.appendChild(titleEl);
    titleWrap.appendChild(subtitleEl);
    header.appendChild(titleWrap);

    // Undo/Redo
    const undoRedoBox = el('div');
    undoRedoBox.style.cssText = 'display:flex;gap:4px;';

    function iconBtn(svgPath, titleText) {{
      const b = el('button');
      b.title = titleText;
      b.style.cssText = 'background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);color:#aaa;cursor:pointer;padding:6px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;transition:all 0.2s;';
      b.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;">${{svgPath}}</svg>`;
      b.onmouseenter = () => {{ b.style.background='rgba(255,255,255,0.15)'; b.style.color='#ccc'; }};
      b.onmouseleave = () => {{ b.style.background='rgba(255,255,255,0.08)'; b.style.color='#aaa'; }};
      return b;
    }}

    const btnUndo = iconBtn('<polyline points="9 14 4 9 9 4"/><path d="M20 20v-7a4 4 0 0 0-4-4H4"/>', 'Undo');
    const btnRedo = iconBtn('<polyline points="15 14 20 9 15 4"/><path d="M4 20v-7a4 4 0 0 1 4-4h12"/>', 'Redo');
    undoRedoBox.appendChild(btnUndo);
    undoRedoBox.appendChild(btnRedo);
    header.appendChild(undoRedoBox);
    panel.appendChild(header);

    // --- Canvas ---
    const canvasWrap = el('div');
    canvasWrap.style.cssText = 'border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);line-height:0;background:{bg};';
    const canvas = el('canvas', null, {{width:'{cw}',height:'{ch}'}});
    canvas.style.cssText = 'width:100%;height:auto;display:block;cursor:crosshair;touch-action:none;';
    canvasWrap.appendChild(canvas);
    panel.appendChild(canvasWrap);

    const ctx = canvas.getContext('2d');
    const BG = '{bg}';
    let drawing = false, tool = 'pen', brushColor = '#1a1a1a', brushSize = {bs};
    let lastX = 0, lastY = 0;
    let history = [], historyIdx = -1;
    const MAX_HISTORY = 50;

    function initCanvas() {{ ctx.fillStyle = BG; ctx.fillRect(0,0,canvas.width,canvas.height); saveState(); }}
    function saveState() {{
      historyIdx++;
      history = history.slice(0, historyIdx);
      history.push(canvas.toDataURL());
      if (history.length > MAX_HISTORY) {{ history.shift(); historyIdx--; }}
    }}
    function undo() {{ if (historyIdx > 0) {{ historyIdx--; restoreState(); }} }}
    function redo() {{ if (historyIdx < history.length - 1) {{ historyIdx++; restoreState(); }} }}
    function restoreState() {{
      const img = new Image();
      img.onload = () => {{ ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0); }};
      img.src = history[historyIdx];
    }}
    function getPos(e) {{
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
      let cx, cy;
      if (e.touches) {{ cx = e.touches[0].clientX; cy = e.touches[0].clientY; }}
      else {{ cx = e.clientX; cy = e.clientY; }}
      return [(cx - rect.left) * sx, (cy - rect.top) * sy];
    }}
    function startDraw(e) {{
      e.preventDefault(); drawing = true;
      [lastX, lastY] = getPos(e);
      ctx.beginPath(); ctx.arc(lastX, lastY, brushSize/2, 0, Math.PI*2);
      ctx.fillStyle = tool==='eraser' ? BG : brushColor; ctx.fill();
    }}
    function draw(e) {{
      if (!drawing) return; e.preventDefault();
      const [x,y] = getPos(e);
      ctx.beginPath(); ctx.moveTo(lastX,lastY); ctx.lineTo(x,y);
      ctx.strokeStyle = tool==='eraser' ? BG : brushColor;
      ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.stroke();
      lastX = x; lastY = y;
    }}
    function endDraw() {{ if (drawing) {{ drawing = false; saveState(); }} }}

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);
    canvas.addEventListener('touchstart', startDraw, {{passive:false}});
    canvas.addEventListener('touchmove', draw, {{passive:false}});
    canvas.addEventListener('touchend', endDraw);
    canvas.addEventListener('touchcancel', endDraw);

    // --- Toolbar row 1: tools ---
    const toolbar1 = el('div');
    toolbar1.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;align-items:center;';

    function toolGroupEl() {{
      const g = el('div');
      g.style.cssText = 'display:flex;gap:4px;align-items:center;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:4px;';
      return g;
    }}

    const toolGrp = toolGroupEl();
    const btnPen = iconBtn('<line x1="17" y1="10" x2="3" y2="24"/><path d="M21 3l-7 7"/>', 'Pen');
    btnPen.style.background = 'rgba(255,255,255,0.2)'; btnPen.style.color = '#fff'; btnPen.style.borderColor = 'rgba(255,255,255,0.2)';
    const btnEraser = iconBtn('<path d="M20 20H7L3 16l9-9 8 8-4 4z"/><line x1="6" y1="11" x2="13" y2="18"/>', 'Eraser');
    toolGrp.appendChild(btnPen);
    toolGrp.appendChild(btnEraser);
    toolbar1.appendChild(toolGrp);

    function setTool(t) {{
      tool = t;
      if (t === 'pen') {{
        btnPen.style.background = 'rgba(255,255,255,0.2)'; btnPen.style.color = '#fff'; btnPen.style.borderColor = 'rgba(255,255,255,0.2)';
        btnEraser.style.background = 'rgba(255,255,255,0.08)'; btnEraser.style.color = '#aaa'; btnEraser.style.borderColor = 'rgba(255,255,255,0.1)';
      }} else {{
        btnEraser.style.background = 'rgba(255,255,255,0.2)'; btnEraser.style.color = '#fff'; btnEraser.style.borderColor = 'rgba(255,255,255,0.2)';
        btnPen.style.background = 'rgba(255,255,255,0.08)'; btnPen.style.color = '#aaa'; btnPen.style.borderColor = 'rgba(255,255,255,0.1)';
      }}
    }}
    btnPen.onclick = () => setTool('pen');
    btnEraser.onclick = () => setTool('eraser');

    // Brush size slider
    const sliderGrp = el('div');
    sliderGrp.style.cssText = 'display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:4px 10px;';
    const sliderLabel = el('label');
    sliderLabel.textContent = 'Size';
    sliderLabel.style.cssText = 'font-size:11px;color:#888;white-space:nowrap;';
    const slider = el('input', null, {{type:'range',min:'1',max:'80',value:'{bs}'}});
    slider.style.cssText = 'width:80px;height:4px;accent-color:rgba(255,255,255,0.5);';
    const sizeVal = el('span');
    sizeVal.textContent = '{bs}';
    sizeVal.style.cssText = 'font-size:11px;color:#ccc;min-width:20px;text-align:center;font-variant-numeric:tabular-nums;';
    slider.oninput = () => {{ brushSize = parseInt(slider.value); sizeVal.textContent = brushSize; }};
    sliderGrp.appendChild(sliderLabel);
    sliderGrp.appendChild(slider);
    sliderGrp.appendChild(sizeVal);
    toolbar1.appendChild(sliderGrp);

    // Clear button
    const clearGrp = toolGroupEl();
    const btnClear = iconBtn('<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>', 'Clear');
    btnClear.onclick = () => {{ ctx.fillStyle = BG; ctx.fillRect(0,0,canvas.width,canvas.height); saveState(); }};
    clearGrp.appendChild(btnClear);
    toolbar1.appendChild(clearGrp);
    panel.appendChild(toolbar1);

    // --- Toolbar row 2: palette ---
    const COLORS = ['#1a1a1a','#e2e8f0','#ef4444','#f97316','#f59e0b','#22c55e','#14b8a6','#3b82f6','#8b5cf6','#ec4899','#6366f1','#06b6d4','#84cc16','#a855f7','#f43f5e','#fbbf24','#34d399','#60a5fa'];
    const palette = el('div');
    palette.style.cssText = 'display:flex;flex-wrap:wrap;gap:5px;align-items:center;';

    let activeSwatch = null;
    COLORS.forEach((c, i) => {{
      const sw = el('div');
      sw.style.cssText = `width:24px;height:24px;border-radius:50%;cursor:pointer;transition:all 0.2s;flex-shrink:0;background:${{c}};border:2px solid ${{i===0?'rgba(255,255,255,0.5)':'transparent'}};${{i===0?'transform:scale(1.15);box-shadow:0 0 6px rgba(255,255,255,0.15);':''}}`;
      sw.onclick = () => {{
        brushColor = c;
        if (activeSwatch) {{ activeSwatch.style.borderColor = 'transparent'; activeSwatch.style.transform = 'scale(1)'; activeSwatch.style.boxShadow = 'none'; }}
        sw.style.borderColor = 'rgba(255,255,255,0.5)'; sw.style.transform = 'scale(1.15)'; sw.style.boxShadow = '0 0 6px rgba(255,255,255,0.15)';
        activeSwatch = sw;
        setTool('pen');
      }};
      if (i === 0) activeSwatch = sw;
      palette.appendChild(sw);
    }});

    // Custom colour picker
    const cpWrap = el('div');
    cpWrap.style.cssText = 'width:24px;height:24px;border-radius:50%;overflow:hidden;border:2px dashed rgba(255,255,255,0.25);cursor:pointer;flex-shrink:0;background:conic-gradient(red,yellow,lime,aqua,blue,magenta,red);position:relative;transition:all 0.2s;';
    const cpInput = el('input', null, {{type:'color',value:'#ffffff'}});
    cpInput.style.cssText = 'position:absolute;top:-4px;left:-4px;width:32px;height:32px;border:none;padding:0;cursor:pointer;opacity:0;';
    cpInput.oninput = (e) => {{
      brushColor = e.target.value;
      if (activeSwatch) {{ activeSwatch.style.borderColor = 'transparent'; activeSwatch.style.transform = 'scale(1)'; activeSwatch.style.boxShadow = 'none'; }}
      activeSwatch = null;
      setTool('pen');
    }};
    cpWrap.appendChild(cpInput);
    palette.appendChild(cpWrap);
    panel.appendChild(palette);

    // --- Undo / Redo wiring ---
    btnUndo.onclick = undo;
    btnRedo.onclick = redo;

    // Keyboard shortcuts
    const keyHandler = (e) => {{
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {{
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
      }}
      if (e.key === 'Escape') {{ cleanup(); resolve(null); }}
    }};
    document.addEventListener('keydown', keyHandler);

    // --- Action buttons ---
    const actions = el('div');
    actions.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;';

    function actionBtn(label, primary) {{
      const b = el('button');
      b.textContent = label;
      b.style.cssText = primary
        ? 'background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.15);color:#fff;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:500;flex:1;min-width:120px;cursor:pointer;transition:all 0.2s;font-family:inherit;'
        : 'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);color:#ccc;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:400;flex:1;min-width:120px;cursor:pointer;transition:all 0.2s;font-family:inherit;';
      b.onmouseenter = () => {{ b.style.background = primary ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.12)'; b.style.borderColor = 'rgba(255,255,255,0.2)'; b.style.transform = 'scale(1.02)'; }};
      b.onmouseleave = () => {{ b.style.background = primary ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.06)'; b.style.borderColor = primary ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.08)'; b.style.transform = 'none'; }};
      return b;
    }}

    const btnAttach = actionBtn('\u2714 Attach \u0026 Send', true);
    const btnSkip = actionBtn('Skip (no doodle)', false);

    btnAttach.onclick = () => {{
      const dataUrl = canvas.toDataURL('image/png');
      cleanup();
      resolve(dataUrl);
    }};
    btnSkip.onclick = () => {{
      cleanup();
      resolve(null);
    }};

    actions.appendChild(btnAttach);
    actions.appendChild(btnSkip);
    panel.appendChild(actions);

    // --- Cleanup ---
    function cleanup() {{
      document.removeEventListener('keydown', keyHandler);
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }}

    // --- Mount ---
    document.body.appendChild(overlay);
    initCanvas();
  }});
}})()
"""

    def _save_doodle_file(
        self, data_url: str, user_id: str, request: Any
    ) -> Optional[dict]:
        """
        Save a doodle data-URL as a file using OWUI's upload_file_handler.

        Returns a file entry dict suitable for message.files[], or None on
        failure.
        """
        # Parse the data URL  →  "data:image/png;base64,<payload>"
        try:
            header, b64_data = data_url.split(",", 1)
        except ValueError:
            logger.warning("Doodle Paint: malformed data URL")
            return None

        # Determine content type from header
        content_type = "image/png"
        if "image/" in header:
            ct_part = header.split("image/")[1].split(";")[0]
            content_type = f"image/{ct_part}"

        extension = content_type.split("/")[1]  # e.g. "png"
        raw_bytes = base64.b64decode(b64_data)
        filename = f"doodle_{uuid.uuid4().hex[:8]}.{extension}"

        # Upload via OWUI's native handler (same as all other tools)
        user = Users.get_user_by_id(user_id)
        if not user:
            logger.error("Doodle Paint: could not resolve user")
            return None

        file = UploadFile(
            file=io.BytesIO(raw_bytes),
            filename=filename,
            headers={"content-type": content_type},
        )
        file_item = upload_file_handler(
            request=request, file=file, metadata={}, process=False, user=user
        )

        file_id = getattr(file_item, "id", None)
        if not file_id:
            logger.error("Doodle Paint: upload_file_handler returned no id")
            return None

        file_id = str(file_id)
        return {
            "type": "image",
            "url": f"/api/v1/files/{file_id}/content",
            "id": file_id,
            "name": filename,
            "content_type": content_type,
            "size": len(raw_bytes),
        }

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __event_call__: Callable[[Any], Awaitable[Any]] = None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Any] = None,
        __task__=None,
    ) -> dict:
        # Skip non-default tasks (title generation, search queries, etc.)
        if __task__:
            if __task__ != TASKS.DEFAULT:
                return body

        if not __event_call__:
            logger.warning("Doodle Paint: __event_call__ not available, skipping.")
            return body

        # Execute the canvas JS on the client and wait for result
        try:
            js_code = self._build_canvas_js()
            result = await __event_call__(
                {
                    "type": "execute",
                    "data": {
                        "code": js_code,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Doodle Paint: execute event failed: {e}")
            return body

        # If user skipped or cancelled, send message as-is
        if not result:
            logger.info(f"Doodle Paint: skipped/cancelled (result={result!r})")
            return body

        logger.info(
            f"Doodle Paint: got result type={type(result).__name__}, len={len(str(result)[:200])}"
        )

        # Extract the data URL
        data_url = None
        if isinstance(result, str) and result.startswith("data:image/"):
            data_url = result
        elif isinstance(result, dict):
            data_url = result.get("result") or result.get("value") or result.get("data")
            if isinstance(data_url, str) and not data_url.startswith("data:image/"):
                data_url = None

        if not data_url:
            logger.warning("Doodle Paint: no valid image data received.")
            return body

        logger.info(f"Doodle Paint: image attached ({len(data_url)} chars)")

        # --- Save the doodle as a file in OWUI storage ---
        user_id = __user__.get("id", "") if __user__ else ""
        file_entry = None

        if user_id and __request__:
            try:
                file_entry = self._save_doodle_file(data_url, user_id, __request__)
                if file_entry:
                    logger.info(f"Doodle Paint: file saved (id={file_entry['id']})")
            except Exception as e:
                logger.error(f"Doodle Paint: failed to save file: {e}")

        # Store file entry so the outlet can attach it to the response
        self._last_doodle_file = data_url

        # Inject the image into the last user message as multimodal content
        # This ensures the LLM sees the image on the current request
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "user":
            last_msg = messages[-1]
            current_content = last_msg.get("content", "")

            # Build multimodal content array
            content_parts = []

            # Preserve existing content (could already be multimodal)
            if isinstance(current_content, str):
                if current_content.strip():
                    content_parts.append({"type": "text", "text": current_content})
            elif isinstance(current_content, list):
                content_parts.extend(current_content)

            # Append the doodle image
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

            last_msg["content"] = content_parts
            messages[-1] = last_msg
            body["messages"] = messages

        # Emit the doodle as an inline image embed so the user can see it
        if file_entry and __event_emitter__:
            try:
                await __event_emitter__(
                    {
                        "type": "files",
                        "data": {
                            "files": [
                                {
                                    "type": "image",
                                    "url": f"/api/v1/files/{file_entry['id']}/content",
                                    "id": file_entry["id"],
                                    "name": file_entry.get("name", "doodle.png"),
                                    "content_type": file_entry.get(
                                        "content_type", "image/png"
                                    ),
                                    "size": file_entry.get("size", 0),
                                }
                            ]
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Doodle Paint: failed to emit image embed: {e}")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        # Attach the doodle file to the assistant response so it persists
        # in the chat and is accessible for future reference / image edits.
        data_url = self._last_doodle_file
        messages = body.get("messages", [])
        message = get_last_user_message(messages)
        if message:
            last_msg = messages[-1]
            current_content = last_msg.get("content", "")

            # Build multimodal content array
            content_parts = []

            # Preserve existing content (could already be multimodal)
            if isinstance(current_content, str):
                if current_content.strip():
                    content_parts.append({"type": "text", "text": current_content})
            elif isinstance(current_content, list):
                content_parts.extend(current_content)

            # Append the doodle image
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

        last_msg["content"] = content_parts
        messages[-1] = last_msg
        body["messages"] = messages
        return body
