"""
title: User Input Tool Set
description: Interactive tools for text, choices, or images with ultra-rounded Open WebUI styling. Optimized for Agent Vision (multimodal LLM perception).
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.7.0
license: MIT
required_open_webui_version: 0.9.1
"""
import json
import logging
import base64
import asyncio
import ipaddress
import mimetypes
import socket
import aiohttp
import httpx
import os
from typing import Optional, Any, Callable, Awaitable, List
from fastapi import HTTPException, Request, UploadFile
import io
from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.storage.provider import Storage
from open_webui.routers.images import get_image_data, upload_image
from open_webui.models.users import UserModel
from urllib.parse import urljoin, urlparse

# Open WebUI's own SSRF guards (same ones the web loaders and get_image_data use).
try:
    from open_webui.retrieval.web.utils import validate_url as _owui_validate_url
except ImportError:  # pragma: no cover - very old Open WebUI
    _owui_validate_url = None
try:
    # Connector that re-checks every resolved address at connect time (DNS rebinding).
    from open_webui.retrieval.web.utils import get_ssrf_safe_session as _owui_ssrf_session
except ImportError:  # pragma: no cover - Open WebUI < 0.10
    _owui_ssrf_session = None
try:
    from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL as _AIOHTTP_SSL
except ImportError:  # pragma: no cover
    _AIOHTTP_SSL = True


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
    panel.style.cssText = '{panel_css} width:100%;max-width:640px;max-height:90vh;display:flex;flex-direction:column;overflow:hidden;padding:24px;gap:20px;box-sizing:border-box;position:relative;';
    
    const style = document.createElement('style');
    style.textContent = `
        .tab-btn {{ background:transparent; border:none; color:#737373; padding:10px 20px; cursor:pointer; font-weight:600; border-radius:12px; transition:all 0.2s; font-size:14px; font-family:inherit; }}
        .tab-btn.active {{ background:#262626; color:#fff; }}
        .tab-pane {{ display:none; flex-direction:column; gap:20px; overflow:hidden; flex:1; min-height:0; }}
        .tab-pane.active {{ display:flex; }}
        .drop-zone {{ border:2px dashed #444; border-radius:16px; padding:60px; text-align:center; transition:all 0.2s; color:#777; cursor:pointer; background:#111; flex:1; display:flex; align-items:center; justify-content:center; box-sizing:border-box; }}
        .drop-zone:hover, .drop-zone.dragover {{ border-color:#fff; color:#fff; background:#1a1a1a; }}
        #urlIn {{ background:#111; border:1px solid #333; border-radius:12px; padding:16px; color:#fff; width:100%; box-sizing:border-box; font-family:inherit; font-size:15px; outline:none; }}
        .loading-overlay {{ position:absolute; inset:0; background:rgba(0,0,0,0.7); display:none; align-items:center; justify-content:center; flex-direction:column; gap:16px; z-index:100; border-radius:24px; backdrop-filter:blur(4px); }}
    `;
    document.head.appendChild(style);

    const loading = document.createElement('div');
    loading.className = 'loading-overlay';
    loading.innerHTML = '<div style="width:40px;height:40px;border:3px solid #333;border-top-color:#fff;border-radius:50%;animation:spin 1s linear infinite;"></div><div style="font-weight:600;color:#fff;">Processing Image...</div><style>@keyframes spin {{ to {{ transform:rotate(360deg); }} }}</style>';
    panel.appendChild(loading);

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
    
    // Click to upload
    dz.onclick = () => fileInput.click();
    fileInput.onchange = (e) => handleFiles(e.target.files);

    // Drag and Drop implementation
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {{
        dz.addEventListener(evt, (e) => {{ e.preventDefault(); e.stopPropagation(); }}, false);
    }});
    ['dragenter', 'dragover'].forEach(evt => {{
        dz.addEventListener(evt, () => dz.classList.add('dragover'), false);
    }});
    ['dragleave', 'drop'].forEach(evt => {{
        dz.addEventListener(evt, () => dz.classList.remove('dragover'), false);
    }});
    dz.addEventListener('drop', (e) => {{
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    }}, false);

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
        const cx = (e.touches && e.touches[0]) ? e.touches[0].clientX : e.clientX;
        const cy = (e.touches && e.touches[0]) ? e.touches[0].clientY : e.clientY;
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
            if(url) {{
                loading.style.display = 'flex';
                finish({{ type:"url", data:url }});
            }}
        }} else if(currentTab === 2) {{ 
            loading.style.display = 'flex';
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
        loading.style.display = 'flex';
        const r = new FileReader();
        r.onload = (e) => finish({{ type:'upload', data: e.target.result, name: f.name, contentType: f.type }});
        r.onerror = () => {{
            loading.style.display = 'none';
            alert('Failed to read file.');
        }};
        r.readAsDataURL(f);
    }}

    function finish(res) {{ 
        // Resolve immediately so backend can start processing while UI is closing
        resolve(JSON.stringify(res));
        cleanup(); 
    }}

    function cleanup() {{
        if(overlay.parentNode) overlay.parentNode.removeChild(overlay);
        if(style.parentNode) style.parentNode.removeChild(style);
    }}
  }});
}})()
    """


def build_get_document_js(prompt_text: str) -> str:
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
    panel.style.cssText = '{panel_css} width:100%;max-width:640px;max-height:90vh;display:flex;flex-direction:column;overflow:hidden;padding:24px;gap:20px;box-sizing:border-box;position:relative;';
    
    const style = document.createElement('style');
    style.textContent = `
        .tab-btn {{ background:transparent; border:none; color:#737373; padding:10px 20px; cursor:pointer; font-weight:600; border-radius:12px; transition:all 0.2s; font-size:14px; font-family:inherit; }}
        .tab-btn.active {{ background:#262626; color:#fff; }}
        .tab-pane {{ display:none; flex-direction:column; gap:20px; overflow:hidden; flex:1; min-height:0; }}
        .tab-pane.active {{ display:flex; }}
        .drop-zone {{ border:2px dashed #444; border-radius:16px; padding:60px; text-align:center; transition:all 0.2s; color:#777; cursor:pointer; background:#111; flex:1; display:flex; align-items:center; justify-content:center; box-sizing:border-box; }}
        .drop-zone:hover, .drop-zone.dragover {{ border-color:#fff; color:#fff; background:#1a1a1a; }}
        #urlIn {{ background:#111; border:1px solid #333; border-radius:12px; padding:16px; color:#fff; width:100%; box-sizing:border-box; font-family:inherit; font-size:15px; outline:none; }}
        .loading-overlay {{ position:absolute; inset:0; background:rgba(0,0,0,0.7); display:none; align-items:center; justify-content:center; flex-direction:column; gap:16px; z-index:100; border-radius:24px; backdrop-filter:blur(4px); }}
    `;
    document.head.appendChild(style);

    const loading = document.createElement('div');
    loading.className = 'loading-overlay';
    loading.innerHTML = '<div style="width:40px;height:40px;border:3px solid #333;border-top-color:#fff;border-radius:50%;animation:spin 1s linear infinite;"></div><div style="font-weight:600;color:#fff;">Processing Document...</div><style>@keyframes spin {{ to {{ transform:rotate(360deg); }} }}</style>';
    panel.appendChild(loading);

    const titleBar = document.createElement('div');
    titleBar.style.cssText = 'font-size:20px;font-weight:600;color:#fff;';
    titleBar.textContent = {prompt_json};
    panel.appendChild(titleBar);
    
    const tabs = document.createElement('div');
    tabs.style.cssText = 'display:flex;gap:8px;background:#0d0d0d;padding:6px;border-radius:16px;width:fit-content;';
    const tabNames = ['Upload', 'URL'];
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
    uploadPane.innerHTML = '<div class="drop-zone" id="dz">Click or Drag & Drop Document (PDF, Word, Text, etc.)</div>';
    const fileInput = document.createElement('input');
    fileInput.type = 'file'; fileInput.accept = '.pdf,.doc,.docx,.txt,.csv,.md,.xls,.xlsx,.ppt,.pptx';
    uploadPane.appendChild(fileInput);
    fileInput.style.display = 'none';
    const dz = uploadPane.querySelector('#dz');
    
    dz.onclick = () => fileInput.click();
    fileInput.onchange = (e) => handleFiles(e.target.files);

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {{
        dz.addEventListener(evt, (e) => {{ e.preventDefault(); e.stopPropagation(); }}, false);
    }});
    ['dragenter', 'dragover'].forEach(evt => {{
        dz.addEventListener(evt, () => dz.classList.add('dragover'), false);
    }});
    ['dragleave', 'drop'].forEach(evt => {{
        dz.addEventListener(evt, () => dz.classList.remove('dragover'), false);
    }});
    dz.addEventListener('drop', (e) => {{
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    }}, false);

    content.appendChild(uploadPane);

    // URL Pane
    const urlPane = document.createElement('div');
    urlPane.className = 'tab-pane';
    urlPane.innerHTML = `<input type="text" id="urlIn" placeholder="Paste Document URL here..." autocomplete="off">`;
    content.appendChild(urlPane);

    panel.appendChild(content);

    // Footer
    const footer = document.createElement('div');
    footer.style.cssText = 'display:flex;justify-content:flex-end;gap:12px;margin-top:20px;';

    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = '{btn_s} border-radius:9999px;';
    cancelBtn.onclick = () => {{ cleanup(); resolve(null); }};

    const confirmBtn = document.createElement('button');
    confirmBtn.textContent = 'Fetch Document';
    confirmBtn.style.cssText = '{btn_p} border-radius:9999px;';
    confirmBtn.onclick = () => {{
        if(currentTab === 1) {{ 
            const url = urlPane.querySelector('#urlIn').value;
            if(url) {{
                loading.style.display = 'flex';
                finish({{ type:"url", data:url }});
            }}
        }}
    }};

    function switchTab(idx) {{
        currentTab = idx;
        tabBtns.forEach((b, i) => i === idx ? b.classList.add('active') : b.classList.remove('active'));
        [uploadPane, urlPane].forEach((p, i) => i === idx ? p.classList.add('active') : p.classList.remove('active'));
        
        if(idx === 0) {{
            confirmBtn.style.display = 'none';
        }} else {{
            confirmBtn.style.display = 'block';
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
        loading.style.display = 'flex';
        const r = new FileReader();
        r.onload = (e) => finish({{ type:'upload', data: e.target.result, name: f.name, contentType: f.type }});
        r.onerror = () => {{
            loading.style.display = 'none';
            alert('Failed to read file.');
        }};
        r.readAsDataURL(f);
    }}

    function finish(res) {{ 
        resolve(JSON.stringify(res));
        cleanup(); 
    }}

    function cleanup() {{
        if(overlay.parentNode) overlay.parentNode.removeChild(overlay);
        if(style.parentNode) style.parentNode.removeChild(style);
    }}
  }});
}})()
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_call_error(result: Any) -> Optional[str]:
    """OWUI 0.11+ __event_call__ returns {"error": ...} on disconnect/timeout."""
    if isinstance(result, dict) and result.get("error") and not any(
        result.get(k) for k in ("result", "value", "data")
    ):
        return str(result["error"])
    return None


def _upload_url(ret: Any) -> Any:
    """upload_image returns a {id,url,...} dict since 0.11.2, a URL string before."""
    return ret.get("url") if isinstance(ret, dict) else ret


async def _get_base64_from_file_id(file_id: str) -> Optional[str]:
    """Resolves a file ID to a base64 data URL (works with local, S3, GCS and Azure storage)."""
    try:
        file = await Files.get_file_by_id(file_id)
        if not file or not file.path:
            return None

        # Storage.get_file returns a local path, downloading from cloud storage if needed.
        file_path = await asyncio.to_thread(Storage.get_file, file.path)
        if not file_path or not await asyncio.to_thread(os.path.isfile, file_path):
            return None

        def _read() -> bytes:
            with open(file_path, "rb") as f:
                return f.read()

        data = await asyncio.to_thread(_read)
        encoded = base64.b64encode(data).decode("utf-8")
        mime_type = file.meta.get("content_type", "image/png") if file.meta else "image/png"
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        logger.error(f"User Input Tool Set: Failed to resolve base64 for {file_id}: {e}")
        return None


# Mirrors open_webui.utils.chat_id.NON_SAVED_CHAT_ID_PREFIXES (0.12.x): temporary
# chats ("temporary:", legacy "local:") and channel chats ("channel:") have no chat row.
_NON_SAVED_CHAT_ID_PREFIXES = ("temporary:", "local:", "channel:")

try:
    from open_webui.utils.chat_id import is_saved_chat_id as _owui_is_saved_chat_id
except Exception:  # older Open WebUI without the helper
    _owui_is_saved_chat_id = None


def _is_persistent_chat(chat_id: Optional[str]) -> bool:
    """Only saved chats have a DB row to attach files/messages to."""
    if not chat_id:
        return False
    chat_id = str(chat_id)
    if chat_id.startswith(_NON_SAVED_CHAT_ID_PREFIXES):
        return False
    if _owui_is_saved_chat_id is not None:
        try:
            return bool(_owui_is_saved_chat_id(chat_id))
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# SSRF-safe URL fetching
# ---------------------------------------------------------------------------

MAX_URL_FETCH_BYTES = 50 * 1024 * 1024  # hard cap on bytes read from a user-supplied URL
MAX_URL_REDIRECTS = 5
URL_FETCH_TIMEOUT = 30  # seconds
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class UrlFetchError(Exception):
    """A user-supplied URL was rejected or could not be fetched."""


def _fallback_validate_url(url: str) -> None:
    """Used only if Open WebUI's validate_url is unavailable: http(s) + public addresses only."""
    if any(ch in url for ch in ("\\", "\t", "\n", "\r")):
        raise UrlFetchError("URL contains invalid characters.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UrlFetchError("Only http(s) URLs are allowed.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except (socket.gaierror, UnicodeError):
        raise UrlFetchError("Could not resolve host.") from None
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        mapped = getattr(addr, "ipv4_mapped", None)
        if not addr.is_global or (mapped is not None and not mapped.is_global):
            raise UrlFetchError("URL points to a private or local address.")


async def _validate_fetch_url(url: str) -> None:
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        raise UrlFetchError("Only http(s) URLs are allowed.")
    try:
        if _owui_validate_url is not None:
            # Honors ENABLE_LOCAL_WEB_FETCH and WEB_FETCH_FILTER_LIST like built-in fetches.
            await asyncio.to_thread(_owui_validate_url, url)
        else:
            await asyncio.to_thread(_fallback_validate_url, url)
    except UrlFetchError:
        raise
    except Exception as e:
        raise UrlFetchError(f"URL is not allowed: {e}") from None


async def _safe_fetch_url(url: str, max_bytes: int = MAX_URL_FETCH_BYTES) -> tuple:
    """
    Fetch a user-supplied URL without SSRF exposure.
    Every redirect hop is re-validated, the connection layer re-checks resolved
    addresses (when Open WebUI provides it) and the body is capped at max_bytes.
    Returns (content_bytes, content_type, final_url).
    """
    if _owui_ssrf_session is not None:
        session_cm = _owui_ssrf_session()
    else:
        session_cm = aiohttp.ClientSession()
    timeout = aiohttp.ClientTimeout(total=URL_FETCH_TIMEOUT)

    async with session_cm as session:
        current = url
        for _ in range(MAX_URL_REDIRECTS + 1):
            await _validate_fetch_url(current)
            async with session.get(
                current, allow_redirects=False, timeout=timeout, ssl=_AIOHTTP_SSL
            ) as resp:
                if resp.status in _REDIRECT_STATUSES:
                    location = resp.headers.get("Location")
                    if not location:
                        raise UrlFetchError("Redirect without a Location header.")
                    current = urljoin(current, location)
                    continue
                if resp.status >= 400:
                    raise UrlFetchError(f"HTTP {resp.status} {resp.reason or ''}".strip())
                if resp.content_length is not None and resp.content_length > max_bytes:
                    raise UrlFetchError(
                        f"File is too large (limit {max_bytes // (1024 * 1024)} MB)."
                    )
                buf = bytearray()
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise UrlFetchError(
                            f"File is too large (limit {max_bytes // (1024 * 1024)} MB)."
                        )
                return bytes(buf), resp.headers.get("content-type", ""), current
        raise UrlFetchError("Too many redirects.")


def _looks_like_url(data: Any) -> bool:
    return isinstance(data, str) and data.strip().lower().startswith(("http://", "https://"))


def _name_from_url(url: str, content_type: str, default_stem: str) -> str:
    """File name from the URL path; adds an extension from Content-Type when missing."""
    name = os.path.basename(urlparse(url).path) or default_stem
    if not os.path.splitext(name)[1]:
        mime = (content_type or "").split(";")[0].strip().lower()
        ext = mimetypes.guess_extension(mime) if mime else None
        if ext:
            name = f"{name}{ext}"
    return name


# ---------------------------------------------------------------------------
# Tools Class
# ---------------------------------------------------------------------------

class Tools:
    def __init__(self):
        pass

    # Not named "ask_user": Open WebUI >= 0.12 intercepts any tool call with that
    # name and routes it to its own built-in ask_user tool.
    async def ask_user_input(
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
            call_error = _event_call_error(result)
            if call_error:
                if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": f"Interaction failed: {call_error}", "done": True}})
                return f"Error: {call_error}"
            selected = result if isinstance(result, str) else (result.get("result") or result.get("value") or result.get("data")) if result else None
            
            if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": "Received" if selected else "Cancelled", "done": True}})
            return f"{selected}" if selected else "User cancelled the prompt."
        except Exception as e:
            logger.error(f"ask_user_input failed: {e}")
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
            call_error = _event_call_error(result)
            if call_error:
                if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": f"Interaction failed: {call_error}", "done": True}})
                return f"Error: {call_error}"
            selected = result if isinstance(result, str) else (result.get("result") or result.get("value") or result.get("data")) if result else None
            
            if __event_emitter__: await __event_emitter__({"type": "status", "data": {"description": "Selected" if selected else "Cancelled", "done": True}})
            return f"{selected}" if selected else "User cancelled the selection."
        except Exception as e:
            logger.error(f"give_options failed: {e}")
            return f"Error: {e}"

    async def get_image(
        self,
        prompt_text: str = "Please provide an image",
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __chat_id__: str = None,
        __message_id__: str = None,
        __files__: Optional[list] = None,
        __messages__: Optional[list] = None,
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
            call_error = _event_call_error(result)
            if call_error:
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": f"Interaction failed: {call_error}", "done": True}})
                return json.dumps({"status": "error", "message": call_error})
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
                try:
                    image_data, content_type, final_url = await _safe_fetch_url(data)
                    content_type = (content_type or "").split(";")[0].strip()
                    if not content_type.startswith("image/"):
                        if __event_emitter__:
                            await __event_emitter__({"type": "status", "data": {"description": f"URL does not point to an image (Type: {content_type})", "done": True}})
                        return json.dumps({"status": "error", "message": f"The provided URL does not point to a valid image file. Detected type: {content_type}"})

                    # Use filename from URL if possible
                    name = _name_from_url(final_url, content_type, "image-from-url")
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": f"Failed to fetch image from URL: {e}", "done": True}})
                    return json.dumps({"status": "error", "message": f"Failed to fetch image from URL: {e}"})
                # Upload and process
                file_item, file_url = await upload_image(
                    __request__,
                    image_data,
                    content_type,
                    metadata,
                    user,
                )
            elif input_type == "doodle":
                # Doodle is always base64 PNG (never let it be a server-side URL fetch)
                image_data, detected_type = (None, None) if _looks_like_url(data) else await get_image_data(data)
                if image_data is None:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": "Failed to load doodle image data", "done": True}})
                    return json.dumps({"status": "error", "message": "Failed to load doodle image data"})
                content_type = detected_type or "image/png"
                name = "sketch.png"
                # Upload and process
                file_item, file_url = await upload_image(
                    __request__,
                    image_data,
                    content_type,
                    metadata,
                    user,
                )
            else:  # upload
                # Uploaded file: decode data URL, preserve filename/content type
                image_data, detected_type = (None, None) if _looks_like_url(data) else await get_image_data(data)
                if image_data is None:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": "Failed to load uploaded image data", "done": True}})
                    return json.dumps({"status": "error", "message": "Failed to load uploaded image data"})
                content_type = content_type or detected_type or "image/png"
                # Upload and process
                file_item, file_url = await upload_image(
                    __request__,
                    image_data,
                    content_type,
                    metadata,
                    user,
                )


            # Add file metadata for model context (__files__)
            image_file_entry = {
                "type": "image", 
                "url": _upload_url(file_url), 
                "id": file_item.id if hasattr(file_item, 'id') else (file_item.get('id') if isinstance(file_item, dict) else None),
                "name": name, 
                "content_type": content_type
            }
            image_files = [image_file_entry]
            if __files__ is not None:
                __files__.append(image_file_entry)

            # Vision Secret Sauce: Attach to messages/history and inject Multimodal Content
            if __chat_id__ and __message_id__:
                # 1. Permanent Vision Context (Database Persistence)
                # Attach the file reference to both the Assistant (current) and the User
                # (parent) message in history.messages. Open WebUI resolves image files to
                # vision input on later turns, so no base64 is persisted in the chat JSON.
                if _is_persistent_chat(__chat_id__):
                    try:
                        # Assistant sync
                        await Chats.add_message_files_by_id_and_message_id(__chat_id__, __message_id__, image_files)

                        current_msg = await Chats.get_message_by_id_and_message_id(__chat_id__, __message_id__)
                        parent_id = current_msg.get("parentId") if current_msg else None
                        if parent_id:
                            logger.info(f"User Input Tool Set: Syncing to parent User message {parent_id}")
                            await Chats.add_message_files_by_id_and_message_id(__chat_id__, parent_id, image_files)
                    except Exception as e:
                        logger.warning(f"User Input Tool Set: DB Sync failed: {e}")

                # 2. IMMEDIATE Vision Perception (In-Memory In-place Update for current turn)
                # This ensures the LLM handles vision during the follow-up request in the same turn
                if __messages__:
                    # Find parent user message in the current turn's history
                    parent_msg_in_mem = None
                    for m in reversed(__messages__):
                        if m.get("role") == "user":
                            parent_msg_in_mem = m
                            break
                    
                    if parent_msg_in_mem:
                        logger.info(f"User Input Tool Set: Injecting vision into in-memory history for turn awareness")
                        content = parent_msg_in_mem.get("content", "")
                        if isinstance(content, str):
                            parent_msg_in_mem["content"] = [{"type": "text", "text": content}] if content else []
                        
                        for f in image_files:
                            f_id = f.get("id") or (f["url"].split("/files/")[1].split("/")[0] if "/api/v1/files/" in f.get("url", "") else None)
                            if f_id:
                                base64_url = await _get_base64_from_file_id(f_id)
                                if base64_url:
                                    exists = any(part.get("image_url", {}).get("url") == base64_url for part in parent_msg_in_mem["content"] if part.get("type") == "image_url")
                                    if not exists:
                                        parent_msg_in_mem["content"].append({"type": "image_url", "image_url": {"url": base64_url}})

            if image_files:
                if __event_emitter__:
                    await __event_emitter__({"type": "chat:message:files", "data": {"files": image_files}})
                    await __event_emitter__({"type": "status", "data": {"description": "Image received", "done": True}})

                # Return simple JSON string. The history mutation handles vision projection.
                # Returning JSON prevents 'middleware.py' from stringifying raw multimodal data.
                return json.dumps({
                    "status": "success",
                    "message": "Image received and projected into vision context. You can now perceive and describe this image.",
                    "images": image_files,
                }, ensure_ascii=False)

            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "No image received", "done": True}})

            return "No image was provided."

        except Exception as e:
            logger.exception(f"Error in get_image: {e}")
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Error: {str(e)}", "done": True}})
            return f"Error receiving image: {str(e)}"

    async def ask_for_user_document(
        self,
        prompt_text: str = "Please provide a document",
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __chat_id__: str = None,
        __message_id__: str = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Request a document (Upload or URL).
        Extracts text using Open WebUI's internal retrieval methods.
        """
        if __request__ is None:
            return json.dumps({"error": "Request context not available"})
        if not __event_call__:
            return json.dumps({"error": "Browser interaction not available"})

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "Awaiting document input...", "done": False}})

        user = UserModel(**__user__) if __user__ else None

        try:
            # Import upload_file_handler here to ensure it's available in the venv
            from open_webui.routers.files import upload_file_handler
            
            js = build_get_document_js(prompt_text)
            result = await __event_call__({"type": "execute", "data": {"code": js}})
            call_error = _event_call_error(result)
            if call_error:
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": f"Interaction failed: {call_error}", "done": True}})
                return f"Error: {call_error}"
            raw_data = result if isinstance(result, str) else (result.get("result") or result.get("value") or result.get("data")) if result else None
            
            if not raw_data:
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": "Cancelled", "done": True}})
                return "User cancelled the document request."

            result_obj = json.loads(raw_data)
            input_type = result_obj.get("type", "upload")
            data = result_obj.get("data", "")
            name = result_obj.get("name", "uploaded-document.txt")
            content_type = result_obj.get("contentType", "text/plain")

            metadata = {
                "chat_id": __chat_id__,
                "message_id": __message_id__,
            }

            file_content = None
            if input_type == "url":
                try:
                    file_content, fetched_type, final_url = await _safe_fetch_url(data)
                except Exception as e:
                    if __event_emitter__:
                        await __event_emitter__({"type": "status", "data": {"description": f"Failed to fetch document from URL: {e}", "done": True}})
                    return f"Error: Failed to fetch document from URL: {e}"
                content_type = fetched_type or "application/octet-stream"
                # An extension is needed for the allowed-extensions check and the loader.
                name = _name_from_url(final_url, content_type, "document-from-url")
            else:
                # upload
                if "," in data:
                    header, base64_data = data.split(",", 1)
                    file_content = base64.b64decode(base64_data)
                else:
                    file_content = base64.b64decode(data)

            if not file_content:
                return "Error: Failed to process document data."

            # Wrap in UploadFile for Open WebUI's handler
            upload_file = UploadFile(
                filename=name,
                file=io.BytesIO(file_content),
                headers=httpx.Headers({"content-type": content_type})
            )

            # Call internal handler directly since it's now async
            try:
                file_item_res = await upload_file_handler(
                    __request__,
                    file=upload_file,
                    metadata=json.dumps(metadata),
                    process=True,
                    process_in_background=False,
                    user=user,
                )
            except HTTPException as e:
                # e.g. 400 file type not allowed, 413 file too large
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": f"Upload rejected: {e.detail}", "done": True}})
                return f"Error: Document upload rejected: {e.detail}"

            # upload_file_handler returns a dict with 'status' and file model data if process=True
            if isinstance(file_item_res, dict):
                file_id = file_item_res.get("id")
            else:
                file_id = getattr(file_item_res, "id", None)
            if not file_id:
                return f"Error: Failed to upload and process document. Response: {file_item_res}"

            # Retrieve the processed file to get the extracted content
            processed_file = await Files.get_file_by_id(file_id)
            if not processed_file:
                return "Error: Could not retrieve processed file from database."

            extracted_text = (processed_file.data or {}).get("content", "")
            
            # Attach file to chat for UI visibility
            file_info = {
                "type": "file",
                "id": file_id,
                "name": name,
                "content_type": content_type,
                "url": f"/api/v1/files/{file_id}/content"
            }
            if __files__ is not None:
                __files__.append(file_info)

            if _is_persistent_chat(__chat_id__) and __message_id__:
                try:
                    await Chats.add_message_files_by_id_and_message_id(__chat_id__, __message_id__, [file_info])
                except Exception as e:
                    logger.warning(f"Failed to sync document to chat: {e}")
                # Link the file to the chat (chat_file row) so shared-chat viewers can read it.
                if user is not None and hasattr(Chats, "insert_chat_files"):
                    try:
                        await Chats.insert_chat_files(
                            chat_id=__chat_id__,
                            message_id=__message_id__,
                            file_ids=[file_id],
                            user_id=user.id,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to link document to chat: {e}")

            if __event_emitter__:
                await __event_emitter__({"type": "chat:message:files", "data": {"files": [file_info]}})
                await __event_emitter__({"type": "status", "data": {"description": "Document processed", "done": True}})

            if not extracted_text:
                return f"Document '{name}' uploaded, but no text content could be extracted. (Type: {content_type})"

            return f"Successfully read document '{name}'. Extracted Content:\n\n{extracted_text}"

        except Exception as e:
            logger.exception(f"Error in ask_for_user_document: {e}")
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Error: {str(e)}", "done": True}})
            return f"Error processing document: {str(e)}"
