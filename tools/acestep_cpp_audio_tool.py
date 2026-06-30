"""
title: ACE Step CPP Audio Generator
description: Tool to generate songs using the ACE Step 1.5 backend directly via CPP API.
Supports advanced parameters like key, tempo, and language.
Allows unloading llama models (e.g. from llama-swap) before generation to free VRAM.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.4
required_open_webui_version: 0.9.1
"""

import json
import io
import re
import random
from typing import Optional, Dict, Any, Callable, Awaitable, cast, Union, Tuple
import aiohttp
import asyncio
import uuid
import os
from pydantic import BaseModel, Field
from fastapi import Request, UploadFile
from fastapi.responses import HTMLResponse
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler


async def get_loaded_models_async(
    api_url: str,
) -> list[Dict[str, Any]]:
    """Checks running models on a llama-swap base URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url.rstrip('/')}/running") as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return cast(list[Dict[str, Any]], data.get("running", []))
    except Exception as e:
        print(f"Error fetching loaded models from {api_url}: {e}")
        return []


async def unload_models_async(api_urls_csv: str) -> bool:
    """Triggers /unload on specified llama-swap URLs and waits for completion."""
    if not api_urls_csv:
        return True

    urls = [u.strip().rstrip("/") for u in api_urls_csv.split(",") if u.strip()]
    if not urls:
        return True

    print(f"Triggering model unload for: {urls}")
    async with aiohttp.ClientSession() as session:
        # 1. Trigger unloads
        for url in urls:
            try:
                async with session.get(f"{url}/unload") as resp:
                    print(f"Unload request to {url} returned: {resp.status}")
            except Exception as e:
                print(f"Failed to send unload request to {url}: {e}")

        # 2. Polling for empty status
        for _ in range(30): # max 30 polls
            all_clear = True
            for url in urls:
                running = await get_loaded_models_async(url)
                if running:
                    all_clear = False
                    print(f"Models still running on {url}: {[m.get('model') for m in running]}")
                    break
            
            if all_clear:
                print("All models successfully unloaded.")
                return True
            await asyncio.sleep(2)

    print("Warning: Unload polling timed out.")
    return False


async def download_audio_to_storage(
    request: Request,
    user,
    audio_url: str,
    song_name: str = "",
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: float = 3.0,
) -> Optional[str]:
    """Downloads audio from the backend and saves it to OWUI storage.
    Retries on failure since the audio file may not be immediately available."""
    # Determine extension from URL or default to .mp3
    file_extension = ".mp3"
    if ".wav" in audio_url.lower(): file_extension = ".wav"
    elif ".flac" in audio_url.lower(): file_extension = ".flac"

    if song_name:
        safe_name = re.sub(r"[^\w\s-]", "", song_name).strip().replace(" ", "_")
        local_filename = f"{safe_name}{file_extension}"
    else:
        local_filename = f"acestep_{uuid.uuid4().hex[:8]}{file_extension}"

    mime_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    content_type = mime_map.get(file_extension.lower(), "audio/mpeg")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(audio_url) as response:
                    if response.status == 200:
                        audio_content = await response.read()
                        if not audio_content or len(audio_content) < 100:
                            print(f"[ACEStep] Attempt {attempt}/{max_retries}: Audio content too small ({len(audio_content) if audio_content else 0} bytes) from {audio_url}")
                            last_error = "Audio content too small or empty"
                            if attempt < max_retries:
                                await asyncio.sleep(retry_delay)
                            continue

                        upload_file = UploadFile(
                            file=io.BytesIO(audio_content),
                            filename=local_filename,
                            headers={"content-type": content_type},
                        )

                        file_item = await upload_file_handler(
                            request,
                            file=upload_file,
                            metadata={},
                            process=False,
                            user=user,
                        )

                        if file_item and file_item.id:
                            print(f"[ACEStep] Successfully saved audio to OWUI storage: {file_item.id} ({len(audio_content)} bytes)")
                            return f"/api/v1/files/{file_item.id}/content"

                        print(f"[ACEStep] Attempt {attempt}/{max_retries}: upload_file_handler returned no file ID")
                        last_error = "upload_file_handler returned no file ID"
                    else:
                        body = await response.text()
                        print(f"[ACEStep] Attempt {attempt}/{max_retries}: Failed to download audio: HTTP {response.status} - {body[:200]}")
                        last_error = f"HTTP {response.status}"

        except Exception as e:
            print(f"[ACEStep] Attempt {attempt}/{max_retries}: Error downloading/uploading audio: {str(e)}")
            last_error = str(e)

        if attempt < max_retries:
            print(f"[ACEStep] Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    print(f"[ACEStep] All {max_retries} attempts failed for {audio_url}. Last error: {last_error}")
    return None


# PORTED UI LOGIC (1:1 copy from comfyui_ace_step_audio_tool_1_5.py)

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60: r1, g1, b1 = c, x, 0.0
    elif h < 120: r1, g1, b1 = x, c, 0.0
    elif h < 180: r1, g1, b1 = 0.0, c, x
    elif h < 240: r1, g1, b1 = 0.0, x, c
    elif h < 300: r1, g1, b1 = x, 0.0, c
    else: r1, g1, b1 = c, 0.0, x
    r, g, b = int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255)
    return f"#{r:02X}{g:02X}{b:02X}"

def _random_rainbow_palette(seed_val: int) -> list:
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
    if palette_seed is None:
        palette_seed = random.randint(0, 9999999)

    if colorful:
        c0, c1, c2, c3, c4 = _random_rainbow_palette(palette_seed)
    else:
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
    tracks_json = json.dumps(tracks)
    multi = len(tracks) > 1

    version_btns = "".join(
        f'<button class="vbtn_{pid}" data-idx="{i}" '
        f'style="border:none;cursor:pointer;padding:5px 16px;border-radius:999px;'
        f"font-size:11px;font-weight:700;letter-spacing:0.4px;transition:all .18s;"
        f'background:rgba(255,255,255,0.18);color:#fff;">'
        f"v{i + 1}</button>"
        for i in range(len(tracks))
    )

    bars_html = "".join(
        f'<div style="width:3px;min-width:3px;border-radius:3px;'
        f"background:rgba(255,255,255,0.92);"
        f"animation:bb_{pid} {0.55 + (i % 8) * 0.07:.2f}s ease-in-out infinite alternate;"
        f"animation-delay:{i * 0.04:.2f}s;"
        f'animation-play-state:paused;height:4px;" class="wavebar_{pid}"></div>'
        for i in range(24)
    )

    if colorful:
        bg_layer = f"""
  <div style="position:absolute;inset:0;z-index:0;
    background:linear-gradient(125deg,{c0},{c1},{c2},{c3},{c4},{c0});
    background-size:500% 500%;
    animation:mesh_{pid} 10s ease infinite;"></div>
  <div style="position:absolute;inset:0;z-index:1;background:rgba(0,0,0,0.22);"></div>
  <div style="position:absolute;z-index:1;width:220px;height:220px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,255,255,0.16) 0%,transparent 68%);
    top:-70px;right:-50px;
    animation:orb_{pid} 7s ease-in-out infinite;pointer-events:none;"></div>
  <div style="position:absolute;z-index:1;width:160px;height:160px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,255,255,0.10) 0%,transparent 68%);
    bottom:-40px;left:-30px;
    animation:orb_{pid} 11s ease-in-out infinite reverse;pointer-events:none;"></div>"""
    else:
        bg_layer = """
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

  <div style="position:relative;z-index:2;padding:24px 22px 22px;">
    <div style="text-align:center;margin-bottom:18px;">
      <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
        opacity:0.65;margin-bottom:5px;">♪ ACE Step CPP</div>
      <div style="font-size:19px;font-weight:800;letter-spacing:-0.2px;line-height:1.2;
        text-shadow:0 2px 14px rgba(0,0,0,0.35);">{safe_title}</div>
    </div>

    <div id="ww_{pid}" style="display:flex;align-items:center;justify-content:center;
      gap:3px;height:28px;margin-bottom:18px;opacity:0.45;transition:opacity .3s;">
      {bars_html}
    </div>

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

    <div style="display:flex;justify-content:space-between;font-size:10px;font-weight:600;
      opacity:0.65;margin-bottom:18px;font-variant-numeric:tabular-nums;letter-spacing:0.3px;">
      <span id="tc_{pid}">0:00</span>
      <span id="td_{pid}">0:00</span>
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:{"16px" if multi else "20px"};">
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

    <div id="ver_{pid}" style="display:{"flex" if multi else "none"};
      gap:7px;flex-wrap:wrap;justify-content:center;margin-bottom:18px;">
      <div style="width:100%;font-size:8px;font-weight:800;letter-spacing:1.5px;
        text-transform:uppercase;opacity:0.55;text-align:center;margin-bottom:3px;">Versions</div>
      {version_btns}
    </div>

    <div style="border-radius:14px;overflow:hidden;
      background:rgba(0,0,0,0.22);border:1px solid rgba(255,255,255,0.1);">
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
      <div id="styleBody_{pid}" style="display:none;padding:0 14px 12px;
        font-size:11px;line-height:1.55;opacity:0.85;
        border-top:1px solid rgba(255,255,255,0.08);">
        {safe_tags}
      </div>
      <div style="height:1px;background:rgba(255,255,255,0.08);margin:0;"></div>
      <div style="padding:12px 14px 14px;">
        <div style="font-size:8px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;
          opacity:0.55;margin-bottom:8px;">Lyrics</div>
        <div id="lyr_{pid}" style="max-height:160px;overflow-y:auto;font-size:13px;
          line-height:1.75;white-space:pre-wrap;word-wrap:break-word;opacity:0.92;
          scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.25) transparent;">
          {safe_lyrics}
        </div>
      </div>
    </div>
  </div>
</div>
</div>
<style>
  {"@keyframes mesh_" + pid + " { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }" if colorful else ""}
  {"@keyframes orb_" + pid + " { 0%,100% { transform:translateY(0) scale(1); } 50% { transform:translateY(-18px) scale(1.07); } }" if colorful else ""}
  @keyframes bb_{pid} {{ from {{ height:3px; opacity:0.45; }} to {{ height:24px; opacity:1; }} }}
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
<script>
(function() {{
  var tracks  = {tracks_json};
  var curIdx  = 0;
  var c0      = "{c0}";
  var styleOpen = false;
  var currentBlobUrl = null;
  var aud    = document.getElementById('aud_{pid}');
  var pb     = document.getElementById('pb_{pid}');
  var pi     = document.getElementById('pi_{pid}');
  var pt     = document.getElementById('pt_{pid}');
  var pf     = document.getElementById('pf_{pid}');
  var pthumb = document.getElementById('pthumb_{pid}');
  var tc     = document.getElementById('tc_{pid}');
  var td     = document.getElementById('td_{pid}');
  var dl     = document.getElementById('dl_{pid}');
  var vbtn   = document.getElementById('vbtn_{pid}');
  var vtrack = document.getElementById('vtrack_{pid}');
  var vfill  = document.getElementById('vfill_{pid}');
  var ww     = document.getElementById('ww_{pid}');
  var vBtns  = document.querySelectorAll('.vbtn_{pid}');

  function authFetch(url) {{
    var opts = {{ credentials: 'include' }};
    try {{
      var token = localStorage.getItem('token');
      if (token) {{ opts.headers = {{ 'Authorization': 'Bearer ' + token }}; }}
    }} catch(e) {{}}
    return fetch(url, opts);
  }}

  function fetchBlobUrl(url, callback) {{
    if (url.indexOf('/api/') === -1) {{ callback(url); return; }}
    authFetch(url).then(function(resp) {{
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.blob();
    }}).then(function(blob) {{
      callback(URL.createObjectURL(blob));
    }}).catch(function() {{ callback(url); }});
  }}

  var styleHdr  = document.getElementById('styleHdr_{pid}');
  var styleBody = document.getElementById('styleBody_{pid}');
  var styleChev = document.getElementById('styleChev_{pid}');
  styleHdr.addEventListener('click', function() {{
    styleOpen = !styleOpen;
    styleBody.style.display = styleOpen ? 'block' : 'none';
    styleChev.style.transform = styleOpen ? 'rotate(180deg)' : 'rotate(0deg)';
  }});

  var PLAY  = '<path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/>';
  var PAUSE = '<rect x="6" y="4" width="4" height="16" rx="2"/><rect x="14" y="4" width="4" height="16" rx="2"/>';
  var REDO  = '<path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>';
  var VON   = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>';
  var VOFF  = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';

  function fmt(s) {{
    if (!s || isNaN(s)) return '0:00';
    var m = Math.floor(s/60), sec = Math.floor(s%60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }}

  function setWave(on) {{
    ww.style.opacity = on ? '1' : '0.45';
    ww.querySelectorAll('.wavebar_{pid}').forEach(function(b) {{
      b.style.animationPlayState = on ? 'running' : 'paused';
    }});
  }}

  function setProgress(pct) {{
    pf.style.width     = pct + '%';
    pthumb.style.left  = pct + '%';
  }}

  function loadTrack(idx) {{
    var t = tracks[idx];
    if (currentBlobUrl) {{ URL.revokeObjectURL(currentBlobUrl); currentBlobUrl = null; }}
    pi.innerHTML = PLAY; pi.style.marginLeft = '3px';
    setProgress(0); tc.textContent = '0:00'; td.textContent = '0:00';
    vBtns.forEach(function(b) {{
      var a = parseInt(b.dataset.idx) === idx;
      b.style.background = a ? 'rgba(255,255,255,0.88)' : 'rgba(255,255,255,0.18)';
      b.style.color      = a ? c0 : '#fff';
    }});
    fetchBlobUrl(t.url, function(blobUrl) {{ currentBlobUrl = blobUrl; aud.src = blobUrl; }});
  }}

  if (tracks.length > 0) loadTrack(0);

  vBtns.forEach(function(b) {{
    b.addEventListener('click', function() {{
      var idx = parseInt(b.dataset.idx);
      if (idx !== curIdx) {{
        var wasPlaying = !aud.paused;
        curIdx = idx; loadTrack(idx);
        if (wasPlaying) aud.play();
      }}
    }});
  }});

  pb.addEventListener('click', function() {{
    if (aud.ended) {{ aud.currentTime = 0; aud.play(); }}
    else if (aud.paused) {{ aud.play().catch(function(e) {{ console.error(e); }}); }}
    else {{ aud.pause(); }}
  }});

  aud.addEventListener('play',  function() {{ pi.innerHTML = PAUSE; pi.style.marginLeft = '0'; setWave(true); }});
  aud.addEventListener('pause', function() {{ pi.innerHTML = PLAY;  pi.style.marginLeft = '3px'; setWave(false); }});
  aud.addEventListener('ended', function() {{ pi.innerHTML = REDO;  pi.style.marginLeft = '0'; setWave(false); }});

  aud.addEventListener('timeupdate', function() {{
    var pct = aud.duration ? (aud.currentTime / aud.duration * 100) : 0;
    setProgress(pct); tc.textContent = fmt(aud.currentTime); td.textContent = fmt(aud.duration);
  }});

  pt.addEventListener('click', function(e) {{
    var r = pt.getBoundingClientRect();
    aud.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * aud.duration;
  }});

  dl.addEventListener('click', function(e) {{
    e.preventDefault();
    var t = tracks[curIdx];
    if (t.url.indexOf('/api/') === -1) {{ window.open(t.url, '_blank'); return; }}
    authFetch(t.url).then(function(resp) {{
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.blob();
    }}).then(function(blob) {{
      var dlUrl = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = dlUrl; a.download = t.title + '.mp3';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(dlUrl);
    }}).catch(function(err) {{ window.open(t.url, '_blank'); }});
  }});

  var lastVol = 1.0;
  vbtn.addEventListener('click', function() {{
    if (aud.muted) {{
      aud.muted = false; aud.volume = lastVol;
      vfill.style.width = (lastVol * 100) + '%';
      vbtn.querySelector('svg').innerHTML = VON;
    }} else {{
      lastVol = aud.volume; aud.muted = true;
      vfill.style.width = '0%';
      vbtn.querySelector('svg').innerHTML = VOFF;
    }}
  }});
  vtrack.addEventListener('click', function(e) {{
    var r = vtrack.getBoundingClientRect();
    var v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    aud.volume = v; aud.muted = false; lastVol = v;
    vfill.style.width = (v * 100) + '%';
    vbtn.querySelector('svg').innerHTML = v > 0 ? VON : VOFF;
  }});
}})();
</script>
"""
    return html


class Tools:
    class Valves(BaseModel):
        ACESTEP_CPP_URL: str = Field(
            default="http://localhost:3005",
            description="Base URL for the ACE Step CPP server.",
        )
        ACESTEP_CPP_API_KEY: str = Field(
            default="",
            description="Optional API key for ACE Step CPP.",
        )
        LLAMA_UNLOAD_URLS: str = Field(
            default="",
            description="Comma-separated URLs to trigger model unload (llama-swap).",
        )
        MAX_WAIT_TIME: int = Field(
            default=300,
            description="Maximum wait time for generation in seconds.",
        )
        MAX_BATCH_SIZE: int = Field(
            default=4,
            description="Maximum batch size users can set.",
        )

    class UserValves(BaseModel):
        batch_size: int = Field(
            default=1,
            description="Number of tracks to generate per request.",
        )
        colorful_player: bool = Field(
            default=True,
            description="Use colorful rainbow gradient player.",
        )
        steps: int = Field(default=12, description="Sampling steps (inferenceSteps).")
        seed: int = Field(default=-1, description="Random seed (-1 for random).")
        guidance_scale: float = Field(default=9.0, description="Guidance scale (CFG).")
        lm_cfg_scale: float = Field(default=2.2, description="LM CFG scale.")
        temperature: float = Field(default=0.8, description="LM sampling temperature.")
        top_p: float = Field(default=0.92, description="LM sampling top_p.")
        shift: int = Field(default=3, description="Semantic shift value.")

    def __init__(self):
        self.valves = self.Valves()
        self._token = None

    async def _ensure_token(self):
        """Fetches a token via /api/auth/auto if missing or if using API key."""
        if self.valves.ACESTEP_CPP_API_KEY:
            self._token = self.valves.ACESTEP_CPP_API_KEY
            return

        print("Fetching auto-auth token for ACE Step...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.valves.ACESTEP_CPP_URL.rstrip('/')}/api/auth/auto") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    self._token = data.get("token")
        except Exception as e:
            print(f"Failed to auto-auth: {e}")
            raise

    def _get_headers(self):
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    async def generate_song(
        self,
        prompt: str,
        title: str = "",
        lyrics: str = "",
        tags: str = "",
        language: str = "en",
        key: str = "",
        time_signature: str = "",
        bpm: Optional[int] = None,
        __request__: Request = None,
        __user__: Dict[str, Any] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Dict[str, Any]]]:
        """
        Generate a complete song (music + vocals) using ACE Step 1.5 backend.
        :param prompt: Detailed natural language description of style, mood, and genre (e.g., 'A melancholic lofi track with a soft female vocal').
        :param title: Song title for display and file naming.
        :param lyrics: Optional lyrics to sing. Supports structure tags like [Verse], [Chorus: Anthemic], [Outro] to guide arrangement and performance.
        :param tags: Specific music tags such as genres, instruments, or production styles (e.g., 'lofi, piano, 80s, reverb').
        :param language: Vocal language (en, zh, ja, etc.).
        :param key: Musical key and scale (e.g. C Major).
        :param time_signature: Rhythmic time signature (2, 3, 4, 5, 6).
        :param bpm: Optional beats per minute (e.g., 90, 120).
        """
        user_valves = __user__.get("valves", self.UserValves()) if __user__ else self.UserValves()
        batch_size = min(user_valves.batch_size, self.valves.MAX_BATCH_SIZE)

        song_title = title if title else (prompt[:30].strip() or "ACE Step Song")

        # 1. Unload llama models if configured
        if self.valves.LLAMA_UNLOAD_URLS:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Unloading models...", "done": False},
                    }
                )
            await unload_models_async(self.valves.LLAMA_UNLOAD_URLS)

        # 2. Ensure Auth
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Authenticating...", "done": False},
                }
            )
        await self._ensure_token()

        # 3. Preparation
        base_url = self.valves.ACESTEP_CPP_URL.rstrip("/")
        
        target_seed = user_valves.seed
        if target_seed == -1 or target_seed is None:
            gen_seed = random.randint(1, 1500000000000)
        else:
            gen_seed = target_seed

        full_prompt = f"{tags}, {prompt}" if tags else prompt
        payload = {
            "taskType": "text2music",
            "songDescription": full_prompt,
            "style": full_prompt,
            "lyrics": lyrics,
            "instrumental": not bool(lyrics),
            "vocalLanguage": language,
            "keyScale": key,
            "timeSignature": str(time_signature) if time_signature else "4",
            "title": song_title,
            "randomSeed": False,
            "seed": gen_seed,
            "batchSize": batch_size,
            "audioFormat": "mp3",
            "inferenceSteps": user_valves.steps,
            "guidanceScale": user_valves.guidance_scale,
            "lmCfgScale": user_valves.lm_cfg_scale,
            "lmTemperature": user_valves.temperature,
            "lmTopP": user_valves.top_p,
            "shift": user_valves.shift,
            "useAdg": False,
            "useCotCaption": True,
            "useCotLanguage": True,
            "useCotMetas": True,
            "inferMethod": "ode",
            "lmBackend": "pt",
            "lmModel": "acestep-5Hz-lm-0.6B"
        }

        if bpm is not None:
            payload["bpm"] = bpm

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Generating {song_title}...", "done": False},
                }
            )

        print(f"Starting generation with payload: {payload}")

        async with aiohttp.ClientSession(headers=self._get_headers()) as session:
            # Start Job
            try:
                async with session.post(f"{base_url}/api/generate", json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        error_msg = f"Error starting generation: {resp.status} - {err}"
                        if __event_emitter__:
                            await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                        return error_msg
                    job_data = await resp.json()
                    job_id = job_data.get("jobId")
            except Exception as e:
                error_msg = f"Network error: {str(e)}"
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                return error_msg
            
            if not job_id:
                return "Error: No jobId received from backend."

            # 4. Poll for completion
            start_time = asyncio.get_event_loop().time()
            result_data = None
            while asyncio.get_event_loop().time() - start_time < self.valves.MAX_WAIT_TIME:
                try:
                    async with session.get(f"{base_url}/api/generate/status/{job_id}") as resp:
                        if resp.status == 200:
                            status_data = await resp.json()
                            status = status_data.get("status")
                            if status == "succeeded":
                                result_data = status_data.get("result")
                                break
                            elif status == "failed":
                                error_msg = f"Generation failed: {status_data.get('error', 'Unknown error')}"
                                if __event_emitter__:
                                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                                return error_msg
                        elif resp.status == 429:
                            await asyncio.sleep(10)
                            continue
                except Exception as e:
                    print(f"Polling error: {e}")
                await asyncio.sleep(5)

            if not result_data:
                error_msg = "Generation timed out."
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                return error_msg

            # 5. Handle results (ACE Step returns list of audioUrls)
            audio_urls = result_data.get("audioUrls", [])
            if not audio_urls:
                return "No audio URLs returned from backend."

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Uploading to storage...", "done": False},
                    }
                )

            tracks = []
            user_obj = None
            if __user__ and __user__.get("id"):
                user_obj = await Users.get_user_by_id(__user__["id"])

            for i, rel_url in enumerate(audio_urls):
                # Ensure relative URLs are properly joined (handle leading slash)
                if rel_url.startswith("http://") or rel_url.startswith("https://"):
                    full_audio_url = rel_url
                else:
                    full_audio_url = f"{base_url}{rel_url if rel_url.startswith('/') else '/' + rel_url}"
                track_title = (
                    song_title if batch_size <= 1 else f"{song_title} (Track {i + 1})"
                )

                # Save to OWUI storage (retries internally)
                owui_url = await download_audio_to_storage(
                    __request__,
                    user_obj,
                    full_audio_url,
                    song_name=track_title,
                    headers=self._get_headers()
                )
                if owui_url:
                    tracks.append({"title": track_title, "url": owui_url})
                else:
                    # Do NOT fall back to backend URL - it won't work in the browser
                    print(f"[ACEStep] WARNING: Could not save track {i+1} ({track_title}) to OWUI storage. Backend URL: {full_audio_url}")
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"Warning: Failed to save track {i+1} to storage.", "done": False},
                            }
                        )

            if not tracks:
                error_msg = f"Generation succeeded but all {len(audio_urls)} track(s) failed to save to storage. Check server logs for details."
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                return error_msg

            if __event_emitter__:
                saved_count = len(tracks)
                total_count = len(audio_urls)
                desc = "Generation complete!" if saved_count == total_count else f"Generation complete! ({saved_count}/{total_count} tracks saved)"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": desc, "done": True},
                    }
                )

            # 6. Return Player Embed and LLM Message
            final_html = generate_audio_player_embed(
                tracks=tracks,
                song_title=song_title,
                tags=f"{tags}, {language}, {key}",
                lyrics=lyrics,
                palette_seed=gen_seed % 1000000,
                colorful=user_valves.colorful_player
            )
            
            message = f"Song successfully generated: {song_title}"
            return (
                HTMLResponse(
                    content=final_html,
                    headers={"Content-Disposition": "inline"},
                ),
                {"message": message, "tracks": tracks}
            )
