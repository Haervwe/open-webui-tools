"""
title: ComfyUI ACE Step 1.5 Audio Generator
description: Tool to generate songs using the ACE Step 1.5 workflow via the ComfyUI API.
Supports advanced parameters like key, tempo, language, and batch size.
Requires [ComfyUI-Unload-Model](https://github.com/SeanScripts/ComfyUI-Unload-Model) for model unloading functionality (can be customized via unload_node ID).
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.5.3
"""

import json
import io
import re
import random
from typing import Optional, Dict, Any, Callable, Awaitable, cast, Union
import aiohttp
import asyncio
import uuid
import os
from pydantic import BaseModel, Field
from fastapi import Request, UploadFile
from fastapi.responses import HTMLResponse
from open_webui.models.users import Users
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


async def download_audio_to_storage(
    request: Request,
    user,
    comfyui_http_url: str,
    filename: str,
    subfolder: str = "",
    song_name: str = "",
    headers: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    try:
        file_extension = os.path.splitext(filename)[1] or ".mp3"
        if song_name:
            safe_name = re.sub(r"[^\w\s-]", "", song_name).strip().replace(" ", "_")
            local_filename = f"{safe_name}{file_extension}"
        else:
            local_filename = f"ace_step_{uuid.uuid4().hex[:8]}{file_extension}"

        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        content_type = mime_map.get(file_extension.lower(), "audio/mpeg")

        subfolder_param = f"&subfolder={subfolder}" if subfolder else ""
        comfyui_file_url = (
            f"{comfyui_http_url}/view?filename={filename}&type=output{subfolder_param}"
        )

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(comfyui_file_url) as response:
                if response.status == 200:
                    audio_content = await response.read()

                    upload_file = UploadFile(
                        file=io.BytesIO(audio_content),
                        filename=local_filename,
                        headers={"content-type": content_type},
                    )

                    file_item = upload_file_handler(
                        request,
                        file=upload_file,
                        metadata={},
                        process=False,
                        user=user,
                    )

                    if file_item and file_item.id:
                        return f"/api/v1/files/{file_item.id}/content"

                    print("[DEBUG] upload_file_handler returned no file item")
                    return None
                else:
                    print(
                        f"[DEBUG] Failed to download audio from ComfyUI: HTTP {response.status}"
                    )
                    return None

    except Exception as e:
        print(f"[DEBUG] Error uploading audio to storage: {str(e)}")
        return None


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
    tracks_json = json.dumps(tracks)
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

  // ── Authenticated fetch helper ──
  // Reads JWT from localStorage at call-time (same as OWUI's own frontend).
  // Token is only used in the Authorization header — never in URLs, HTML, or logs.
  // credentials:'include' sends the session cookie as fallback for cookie-auth setups.
  function authFetch(url) {{
    var opts = {{ credentials: 'include' }};
    try {{
      var token = localStorage.getItem('token');
      if (token) {{
        opts.headers = {{ 'Authorization': 'Bearer ' + token }};
      }}
    }} catch(e) {{ /* localStorage blocked — cookie fallback only */ }}
    return fetch(url, opts);
  }}

  // Fetch audio as a blob URL for playback (bypasses <audio> auth limitation).
  // Falls back to direct URL on failure (backward compat with older OWUI).
  function fetchBlobUrl(url, callback) {{
    // External URLs (e.g. direct ComfyUI links) don't need auth
    if (url.indexOf('/api/') === -1) {{
      callback(url);
      return;
    }}
    authFetch(url).then(function(resp) {{
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.blob();
    }}).then(function(blob) {{
      callback(URL.createObjectURL(blob));
    }}).catch(function() {{
      // Fallback: try direct URL (works on older OWUI without strict auth)
      callback(url);
    }});
  }}

  // Style toggle
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
    // Revoke previous blob URL to prevent memory leaks
    if (currentBlobUrl) {{
      URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = null;
    }}
    pi.innerHTML = PLAY;
    pi.style.marginLeft = '3px';
    setProgress(0);
    tc.textContent = '0:00';
    td.textContent = '0:00';
    vBtns.forEach(function(b) {{
      var a = parseInt(b.dataset.idx) === idx;
      b.style.background = a ? 'rgba(255,255,255,0.88)' : 'rgba(255,255,255,0.18)';
      b.style.color      = a ? c0 : '#fff';
    }});
    // Fetch audio with auth and set as blob URL
    fetchBlobUrl(t.url, function(blobUrl) {{
      currentBlobUrl = blobUrl;
      aud.src = blobUrl;
    }});
  }}

  if (tracks.length > 0) loadTrack(0);

  vBtns.forEach(function(b) {{
    b.addEventListener('click', function() {{
      var idx = parseInt(b.dataset.idx);
      if (idx !== curIdx) {{
        var wasPlaying = !aud.paused;
        curIdx = idx;
        loadTrack(idx);
        if (wasPlaying) aud.play();
      }}
    }});
  }});

  pb.addEventListener('click', function() {{
    if (aud.ended) {{
      aud.currentTime = 0; aud.play();
    }} else if (aud.paused) {{
      aud.play().catch(function(e) {{ console.error(e); }});
    }} else {{
      aud.pause();
    }}
  }});

  aud.addEventListener('play',  function() {{ pi.innerHTML = PAUSE; pi.style.marginLeft = '0';  setWave(true);  }});
  aud.addEventListener('pause', function() {{ pi.innerHTML = PLAY;  pi.style.marginLeft = '3px'; setWave(false); }});
  aud.addEventListener('ended', function() {{ pi.innerHTML = REDO;  pi.style.marginLeft = '0';  setWave(false); }});

  aud.addEventListener('timeupdate', function() {{
    var pct = aud.duration ? (aud.currentTime / aud.duration * 100) : 0;
    setProgress(pct);
    tc.textContent = fmt(aud.currentTime);
    td.textContent = fmt(aud.duration);
  }});

  pt.addEventListener('click', function(e) {{
    var r = pt.getBoundingClientRect();
    aud.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * aud.duration;
  }});

  // ── Authenticated download handler ──
  // Uses JS fetch+blob instead of <a href> to include auth headers.
  // The blob URL is revoked immediately after the download to free memory.
  dl.addEventListener('click', function(e) {{
    e.preventDefault();
    var t = tracks[curIdx];
    if (t.url.indexOf('/api/') === -1) {{
      // External URL — normal navigation is fine
      window.open(t.url, '_blank');
      return;
    }}
    authFetch(t.url).then(function(resp) {{
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.blob();
    }}).then(function(blob) {{
      var dlUrl = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = dlUrl;
      a.download = t.title + '.mp3';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(dlUrl);
    }}).catch(function(err) {{
      // Fallback: try direct link
      window.open(t.url, '_blank');
    }});
  }});

  // Volume
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
    var r   = vtrack.getBoundingClientRect();
    var pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    aud.volume = pct; aud.muted = false;
    vfill.style.width = (pct * 100) + '%';
    vbtn.querySelector('svg').innerHTML = pct === 0 ? VOFF : VON;
  }});
  aud.volume = 1.0;

}})();
</script>
"""
    return html


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
            description="Unload all Ollama models before calling ComfyUI.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL.",
        )
        save_to_storage: bool = Field(
            default=True,
            description="Save the generated audio to Open WebUI's file storage.",
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
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str | HTMLResponse:
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
        user_valves = __user__.get("valves", self.UserValves())
        batch_size = min(user_valves.batch_size, self.valves.max_batch_size)

        if duration > self.valves.max_duration:
            duration = self.valves.max_duration

        steps = user_valves.steps
        if steps > self.valves.max_number_of_steps:
            steps = self.valves.max_number_of_steps

        if self.valves.unload_ollama_models:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Unloading models...", "done": False},
                    }
                )
            unloaded = await unload_all_models_async(self.valves.ollama_url)
            await asyncio.sleep(2)
            if not unloaded:
                print("Warning: Ollama models may not have fully unloaded.")

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
            safe_title = re.sub(r"[^\w\s-]", "", song_title).strip().replace(" ", "_")
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

            user_obj = None
            if self.valves.save_to_storage and __request__:
                user_obj = Users.get_user_by_id(__user__["id"])

            for idx, finfo in enumerate(audio_files):
                fname = finfo["filename"]
                subfolder = finfo["subfolder"]
                track_title = (
                    song_title if batch_size <= 1 else f"{song_title} (Track {idx + 1})"
                )

                if user_obj and __request__:
                    storage_url = await download_audio_to_storage(
                        __request__, user_obj, http_url, fname, subfolder, track_title,
                        headers=comfyui_headers,
                    )
                    if storage_url:
                        track_list.append({"title": track_title, "url": storage_url})
                    else:
                        direct_url = f"{http_url}/view?filename={fname}&type=output&subfolder={subfolder}"
                        track_list.append({"title": track_title, "url": direct_url})
                else:
                    direct_url = f"{http_url}/view?filename={fname}&type=output&subfolder={subfolder}"
                    track_list.append({"title": track_title, "url": direct_url})

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generation complete!", "done": True},
                    }
                )

            message = "Song successfully generated, tell the user"
            if self.valves.show_player_embed and track_list:
                final_html = generate_audio_player_embed(
                    track_list,
                    song_title,
                    tags,
                    lyrics,
                    palette_seed=palette_seed,
                    colorful=user_valves.colorful_player,
                )
                await __event_emitter__(
                    {"type": "embeds", "data": {"embeds": [final_html]}}
                )
                message = "The audio player has been successfully embedded above. Inform the user that their song is ready to listen to or download in the above UI component."
            else:
                message += " to use the following download links:"

            return {"message": message, "tracks": track_list}

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )
            return f"Error generating song: {str(e)}"
