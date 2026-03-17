"""
title: User Input Tool Set
description: Interactive tools for text, choices, or images with ultra-rounded Open WebUI styling. Optimized for Agent Vision (multimodal LLM perception).
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.6.0
license: MIT
required_open_webui_version: 0.8.10
"""

import json
import logging
from typing import Optional, Any, Callable, Awaitable, List
from fastapi import Request
from open_webui.models.chats import Chats
from open_webui.routers.images import get_image_data, upload_image
from open_webui.models.users import UserModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("user_input_tool_set")

# ---------------------------------------------------------------------------
# Design System (Ultra-Rounded High-Contrast Style)
# ---------------------------------------------------------------------------

def _get_owui_panel_styles() -> str:
    return """
        background-color: #171717;
        border: 1px solid #333;
        border-radius: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        color: #ececec;
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    """

def _get_owui_button_styles(primary: bool = False) -> str:
    if primary:
        # High-contrast Off-white background, black text
        return """
            background-color: #e5e5e5;
            color: #000000;
            border: none;
            padding: 12px 28px;
            border-radius: 9999px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        """
    # Dark gray background, white text
    return """
        background-color: #262626;
        color: #ffffff;
        border: 1px solid #404040;
        padding: 12px 28px;
        border-radius: 9999px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    """

# ---------------------------------------------------------------------------
# JS Builder Functions
# ---------------------------------------------------------------------------

def build_ask_user_js(prompt_text: str, placeholder: str = "Type your response...") -> str:
    prompt_json = json.dumps(prompt_text)
    placeholder_json = json.dumps(placeholder)
    panel_css = _get_owui_panel_styles().replace("\n", " ")
    btn_p = _get_owui_button_styles(True).replace("\n", " ")
    btn_s = _get_owui_button_styles(False).replace("\n", " ")
    
    return f"""
return (function() {{
  return new Promise((resolve) => {{
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;z-index:999999;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(2px);';
    
    const panel = document.createElement('div');
    panel.style.cssText = '{panel_css} width:100%;max-width:540px;padding:32px;display:flex;flex-direction:column;gap:24px;';
    
    const titleEl = document.createElement('div');
    titleEl.textContent = {prompt_json};
    titleEl.style.cssText = 'font-size:20px;font-weight:600;color:#fff;line-height:1.3;';
    panel.appendChild(titleEl);

    const input = document.createElement('textarea');
    input.placeholder = {placeholder_json};
    input.style.cssText = 'background:#111;border:1px solid #333;border-radius:12px;padding:16px;color:#fff;font-size:15px;min-height:100px;resize:none;outline:none;font-family:inherit;width:100%;box-sizing:border-box;';
    input.onfocus = () => input.style.borderColor = '#555';
    input.onblur = () => input.style.borderColor = '#333';
    panel.appendChild(input);

    const footer = document.createElement('div');
    footer.style.cssText = 'display:flex;gap:16px;justify-content:stretch;';
    
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = '{btn_s} flex:1;';
    cancelBtn.onmouseenter = () => cancelBtn.style.backgroundColor = '#333';
    cancelBtn.onmouseleave = () => cancelBtn.style.backgroundColor = '#262626';
    cancelBtn.onclick = () => {{ cleanup(); resolve(null); }};
    
    const submitBtn = document.createElement('button');
    submitBtn.textContent = 'Confirm';
    submitBtn.style.cssText = '{btn_p} flex:1;';
    submitBtn.onmouseenter = () => submitBtn.style.opacity = '0.9';
    submitBtn.onmouseleave = () => submitBtn.style.opacity = '1';
    submitBtn.onclick = () => {{
      const val = input.value.trim();
      if(val) {{ cleanup(); resolve(val); }}
    }};
    
    footer.appendChild(cancelBtn);
    footer.appendChild(submitBtn);
    panel.appendChild(footer);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    input.focus();

    function cleanup() {{
        if(overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }}
    
    overlay.onclick = (e) => {{ if(e.target === overlay) {{ cleanup(); resolve(null); }} }};
    input.onkeydown = (e) => {{
        if((e.ctrlKey || e.metaKey) && e.key === 'Enter') submitBtn.click();
        if(e.key === 'Escape') cancelBtn.click();
    }};
  }});
}})()
    """

def build_give_options_js(prompt_text: str, choices: List[str], context: str = "") -> str:
    prompt_json = json.dumps(prompt_text)
    context_json = json.dumps(context)
    choices_json = json.dumps(choices)
    panel_css = _get_owui_panel_styles().replace("\n", " ")
    btn_s = _get_owui_button_styles(False).replace("\n", " ")
    btn_p = _get_owui_button_styles(True).replace("\n", " ")
    
    return f"""
return (function() {{
  return new Promise((resolve) => {{
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;z-index:999999;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(2px);';
    
    const panel = document.createElement('div');
    panel.style.cssText = '{panel_css} width:100%;max-width:500px;padding:32px;display:flex;flex-direction:column;gap:24px;';
    
    const header = document.createElement('div');
    const titleEl = document.createElement('div');
    titleEl.textContent = {prompt_json};
    titleEl.style.cssText = 'font-size:20px;font-weight:600;color:#fff;margin-bottom:8px;';
    header.appendChild(titleEl);
    
    const ctx = {context_json};
    if(ctx) {{
        const ctxEl = document.createElement('div');
        ctxEl.textContent = ctx;
        ctxEl.style.cssText = 'font-size:15px;color:#a3a3a3;line-height:1.4;';
        header.appendChild(ctxEl);
    }}
    panel.appendChild(header);

    const btnGrid = document.createElement('div');
    btnGrid.style.cssText = 'display:flex;flex-direction:column;gap:12px;';
    
    const CHOICES = {choices_json};
    CHOICES.forEach((c) => {{
      const b = document.createElement('button');
      b.textContent = c;
      b.style.cssText = '{btn_p} text-align:center; padding:14px;';
      b.onmouseenter = () => {{ b.style.opacity = '0.9'; }};
      b.onmouseleave = () => {{ b.style.opacity = '1'; }};
      b.onclick = () => {{ cleanup(); resolve(c); }};
      btnGrid.appendChild(b);
    }});
    panel.appendChild(btnGrid);

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = '{btn_s} width:100%;';
    cancelBtn.onclick = () => {{ cleanup(); resolve(null); }};
    panel.appendChild(cancelBtn);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    function cleanup() {{
        if(overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }}
    overlay.onclick = (e) => {{ if(e.target === overlay) {{ cleanup(); resolve(null); }} }};
  }});
}})()
    """

def build_get_image_js(prompt_text: str) -> str:
    prompt_json = json.dumps(prompt_text)
    panel_css = _get_owui_panel_styles().replace("\n", " ")
    btn_p = _get_owui_button_styles(True).replace("\n", " ")
    btn_s = _get_owui_button_styles(False).replace("\n", " ")
    
    return f"""
return (function() {{
  return new Promise((resolve) => {{
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;z-index:999999;background:rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;padding:20px;';
    
    const panel = document.createElement('div');
    panel.style.cssText = '{panel_css} width:100%;max-width:640px;max-height:90vh;display:flex;flex-direction:column;overflow:hidden;padding:24px;gap:20px;box-sizing:border-box;';
    
    const style = document.createElement('style');
    style.textContent = `
        .tab-btn {{ background:transparent; border:none; color:#737373; padding:10px 20px; cursor:pointer; font-weight:600; border-radius:12px; transition:all 0.2s; font-size:14px; font-family:inherit; }}
        .tab-btn.active {{ background:#262626; color:#fff; }}
        .tab-pane {{ display:none; flex-direction:column; gap:20px; overflow:hidden; flex:1; min-height:0; }}
        .tab-pane.active {{ display:flex; }}
        .drop-zone {{ border:2px dashed #444; border-radius:16px; padding:60px; text-align:center; transition:all 0.2s; color:#777; cursor:pointer; background:#111; flex:1; display:flex; align-items:center; justify-content:center; box-sizing:border-box; }}
        .drop-zone:hover {{ border-color:#666; color:#fff; }}
        #urlIn {{ background:#111; border:1px solid #333; border-radius:12px; padding:16px; color:#fff; width:100%; box-sizing:border-box; font-family:inherit; font-size:15px; outline:none; }}
    `;
    document.head.appendChild(style);

    const titleBar = document.createElement('div');
    titleBar.style.cssText = 'font-size:20px;font-weight:600;color:#fff;';
    titleBar.textContent = {prompt_json};
    panel.appendChild(titleBar);
    
    const tabs = document.createElement('div');
    tabs.style.cssText = 'display:flex;gap:8px;background:#0d0d0d;padding:6px;border-radius:16px;width:fit-content;';
    const tabNames = ['Upload', 'URL', 'Doodle'];
    let currentTab = 0;
    const tabBtns = tabNames.map((name, i) => {{
        const b = document.createElement('button');
        b.className = 'tab-btn' + (i === 0 ? ' active' : '');
        b.textContent = name;
        b.onclick = () => switchTab(i);
        tabs.appendChild(b);
        return b;
    }});
    panel.appendChild(tabs);

    const content = document.createElement('div');
    content.style.cssText = 'flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0;';
    
    // Upload Pane
    const uploadPane = document.createElement('div');
    uploadPane.className = 'tab-pane active';
    uploadPane.innerHTML = '<div class="drop-zone" id="dz">Click or Drag & Drop Image</div>';
    const fileInput = document.createElement('input');
    fileInput.type = 'file'; fileInput.accept = 'image/*';
    uploadPane.appendChild(fileInput);
    fileInput.style.display = 'none';
    const dz = uploadPane.querySelector('#dz');
    dz.onclick = () => fileInput.click();
    fileInput.onchange = (e) => handleFiles(e.target.files);
    content.appendChild(uploadPane);

    // URL Pane
    const urlPane = document.createElement('div');
    urlPane.className = 'tab-pane';
    urlPane.innerHTML = `<input type="text" id="urlIn" placeholder="Paste URL here..." autocomplete="off">`;
    content.appendChild(urlPane);

    // Doodle Pane
    const doodlePane = document.createElement('div');
    doodlePane.className = 'tab-pane';
    doodlePane.style.padding = '0';
    doodlePane.innerHTML = `<div style="display:flex;flex-direction:column;height:100%;gap:12px;min-height:0;">
        <div style="display:flex;gap:12px;align-items:center;">
            <button id="clr" style="{btn_s}">Clear</button>
            <input type="color" id="col" value="#ffffff" style="border:none;background:none;width:36px;height:36px;padding:0;cursor:pointer;">
        </div>
        <div style="flex:1;background:#000;display:flex;align-items:center;justify-content:center;border-radius:16px;overflow:hidden;border:1px solid #333;min-height:0;box-sizing:border-box;">
            <canvas id="canv" width="512" height="512" style="max-width:100%;max-height:100%;cursor:crosshair;touch-action:none;"></canvas>
        </div>
    </div>`;
    const canvas = doodlePane.querySelector('#canv');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000'; ctx.fillRect(0,0,512,512);
    let draw = false, lx=0, ly=0;
    const getP = (e) => {{
        const r = canvas.getBoundingClientRect();
        const scX = 512/r.width, scY = 512/r.height;
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        return [(cx-r.left)*scX, (cy-r.top)*scY];
    }};
    canvas.onmousedown = (e) => {{ draw=true; [lx,ly]=getP(e); }};
    canvas.onmousemove = (e) => {{
        if(!draw) return; e.preventDefault();
        const [x,y] = getP(e);
        ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(x,y);
        ctx.strokeStyle = doodlePane.querySelector('#col').value; ctx.lineWidth = 4; ctx.lineCap='round'; ctx.stroke();
        [lx,ly]=[x,y];
    }};
    window.addEventListener('mouseup', () => draw=false);
    doodlePane.querySelector('#clr').onclick = () => {{ ctx.fillStyle = '#000'; ctx.fillRect(0,0,512,512); }};
    content.appendChild(doodlePane);

    panel.appendChild(content);

    // Footer
    const footer = document.createElement('div');
    footer.style.cssText = 'display:flex;justify-content:flex-end;gap:12px;margin-top:20px;';

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = '{btn_s} border-radius:9999px;';
    cancelBtn.onclick = () => {{ cleanup(); resolve(null); }};

    const confirmBtn = document.createElement('button');
    confirmBtn.textContent = 'Confirm Upload';
    confirmBtn.style.cssText = '{btn_p} border-radius:9999px;';
    confirmBtn.onclick = () => {{
        if(currentTab === 1) {{ 
            const url = urlPane.querySelector('#urlIn').value;
            if(url) finish({{ type:"url", data:url }});
        }} else if(currentTab === 2) {{ 
            finish({{ type:'doodle', data:canvas.toDataURL() }});
        }}
    }};

    const updateButtonLabel = (idx) => {{
        if (idx === 0) confirmBtn.textContent = 'Confirm Upload';
        else if (idx === 1) confirmBtn.textContent = 'Fetch Image';
        else if (idx === 2) confirmBtn.textContent = 'Confirm Sketch';
    }};

    function switchTab(idx) {{
        currentTab = idx;
        tabBtns.forEach((b, i) => i === idx ? b.classList.add('active') : b.classList.remove('active'));
        [uploadPane, urlPane, doodlePane].forEach((p, i) => i === idx ? p.classList.add('active') : p.classList.remove('active'));
        
        if(idx === 0) {{
            confirmBtn.style.display = 'none';
        }} else {{
            confirmBtn.style.display = 'block';
            updateButtonLabel(idx);
        }}
    }}
    
    footer.appendChild(cancelBtn);
    footer.appendChild(confirmBtn);
    panel.appendChild(footer);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    switchTab(0);

    function handleFiles(files) {{
        const f = files[0]; if(!f) return;
        const r = new FileReader();
        r.onload = (e) => finish({{ type:'upload', data: e.target.result, name: f.name, contentType: f.type }});
        r.readAsDataURL(f);
    }}

    function finish(res) {{ cleanup(); resolve(JSON.stringify(res)); }}

    function cleanup() {{
        if(overlay.parentNode) overlay.parentNode.removeChild(overlay);
        if(style.parentNode) style.parentNode.removeChild(style);
    }}
  }});
}})()
    """

# ---------------------------------------------------------------------------
# Tools Class
# ---------------------------------------------------------------------------

class Tools:
    def __init__(self):
        pass

    async def ask_user(
        self,
        prompt_text: str,
        placeholder: str = "Type your response...",
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Ask the user for text input using an ultra-rounded Open WebUI styled modal.
        """
        if not __event_call__: return "Error: Browser interaction not available."
        if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": "Awaiting response...", "done": False}})

        try:
            js = build_ask_user_js(prompt_text, placeholder)
            result = await __event_call__({"type": "execute", "data": {"code": js}})
            selected = result if isinstance(result, str) else (result.get("result") or result.get("value") or result.get("data")) if result else None
            
            if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": "Received" if selected else "Cancelled", "done": True}})
            return f"{selected}" if selected else "User cancelled the prompt."
        except Exception as e:
            logger.error(f"ask_user failed: {e}")
            return f"Error: {e}"

    async def give_options(
        self,
        prompt_text: str,
        choices: List[str],
        context: str = "",
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Show clickable options in an ultra-rounded Open WebUI styled modal.
        """
        if not __event_call__: return "Error: Browser interaction not available."
        if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": "Awaiting selection...", "done": False}})

        try:
            js = build_give_options_js(prompt_text, choices, context)
            result = await __event_call__({"type": "execute", "data": {"code": js}})
            selected = result if isinstance(result, str) else (result.get("result") or result.get("value") or result.get("data")) if result else None
            
            if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": "Selected" if selected else "Cancelled", "done": True}})
            return f"{selected}" if selected else "User cancelled the selection."
        except Exception as e:
            logger.error(f"give_options failed: {e}")
            return f"Error: {e}"

    async def get_image(
        self,
        prompt_text: str = "Please provide an image",
        __user__: dict = None,
        __request__: Request = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __chat_id__: str = None,
        __message_id__: str = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Request an image (Upload, URL, or Doodle/Sketch).
        Handles uploads, URLs, and doodles with correct file type and processing.
        """
        if __request__ is None:
            return json.dumps({"error": "Request context not available"})
        if not __event_call__:
            return json.dumps({"error": "Browser interaction not available"})

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "Awaiting image input...", "done": False}})

        user = UserModel(**__user__) if __user__ else None

        try:
            js = build_get_image_js(prompt_text)
            result = await __event_call__({"type": "execute", "data": {"code": js}})
            raw_data = result if isinstance(result, str) else (result.get("result") or result.get("value") or result.get("data")) if result else None
            if not raw_data:
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": "Cancelled", "done": True}})
                return json.dumps({"status": "cancelled"})

            result_obj = json.loads(raw_data)
            input_type = result_obj.get("type", "upload")
            data = result_obj.get("data", "")
            name = result_obj.get("name", "uploaded-image.png")
            content_type = result_obj.get("contentType", None)

            metadata = {
                "chat_id": __chat_id__,
                "message_id": __message_id__,
            }

            # Handle each input type
            if input_type == "url":
                # Let backend fetch/process the image from URL as built-in tools do
                from open_webui.routers.files import upload_file_handler
                import requests
                try:
                    resp = requests.get(data, timeout=10)
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "image/png")
                    image_data = resp.content
                    # Use filename from URL if possible
                    from urllib.parse import urlparse
                    import os
                    url_path = urlparse(data).path
                    name = os.path.basename(url_path) or "image-from-url.png"
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": f"Failed to fetch image from URL: {e}", "done": True}})
                    return json.dumps({"status": "error", "message": f"Failed to fetch image from URL: {e}"})
                # Upload and process
                file_item, file_url = upload_image(
                    __request__,
                    image_data,
                    content_type,
                    metadata,
                    user,
                )
            elif input_type == "doodle":
                # Doodle is always base64 PNG
                image_data, detected_type = get_image_data(data)
                if image_data is None:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": "Failed to load doodle image data", "done": True}})
                    return json.dumps({"status": "error", "message": "Failed to load doodle image data"})
                content_type = detected_type or "image/png"
                name = "sketch.png"
                # Upload and process
                file_item, file_url = upload_image(
                    __request__,
                    image_data,
                    content_type,
                    metadata,
                    user,
                )
            else:  # upload
                # Uploaded file: decode data URL, preserve filename/content type
                image_data, detected_type = get_image_data(data)
                if image_data is None:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": "Failed to load uploaded image data", "done": True}})
                    return json.dumps({"status": "error", "message": "Failed to load uploaded image data"})
                content_type = content_type or detected_type or "image/png"
                # Upload and process
                file_item, file_url = upload_image(
                    __request__,
                    image_data,
                    content_type,
                    metadata,
                    user,
                )


            # Add file metadata for model context (__files__)
            image_file_entry = {"type": "image", "url": file_url, "name": name, "content_type": content_type}
            image_files = [image_file_entry]
            if __files__ is not None:
                __files__.append(image_file_entry)

            if __chat_id__ and __message_id__:
                db_files = Chats.add_message_files_by_id_and_message_id(
                    __chat_id__,
                    __message_id__,
                    image_files,
                )
                if db_files is not None:
                    image_files = db_files

            if __event_emitter__ and image_files:
                await __event_emitter__({"type": "chat:message:files", "data": {"files": image_files}})
                await __event_emitter__({"type": "status", "data": {"description": "Image received", "done": True}})
                return json.dumps({
                    "status": "success",
                    "message": "The image has been successfully uploaded and is already visible to the user in the chat. You do not need to display or embed the image again - just acknowledge that it has been received.",
                    "images": image_files,
                }, ensure_ascii=False)

            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Image received", "done": True}})

            return json.dumps({
                "status": "success",
                "images": image_files,
            }, ensure_ascii=False)

        except Exception as e:
            logger.exception(f"Error in get_image: {e}")
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Error: {str(e)}", "done": True}})
            return json.dumps({"status": "error", "message": str(e)})
