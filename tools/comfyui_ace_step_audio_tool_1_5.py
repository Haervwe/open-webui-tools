"""
title: ComfyUI ACE Step 1.5 Audio Generator
description: Tool to generate songs using the ACE Step 1.5 workflow via the ComfyUI API.
Supports advanced parameters like key, tempo, language, and batch size.
Requires [ComfyUI-Unload-Model](https://github.com/SeanScripts/ComfyUI-Unload-Model) for model unloading functionality (can be customized via unload_node ID).
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.1
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
) -> Dict[str, Any]:
    """
    Robustly executes a ComfyUI job:
    1. Connects to WebSocket (preventing race conditions).
    2. Submits the Prompt.
    3. Waits for completion via WebSocket events OR Periodic HTTP Polling (fallback).
    """
    start_time = asyncio.get_event_loop().time()
    prompt_id = None

    async with aiohttp.ClientSession() as session:
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
                                # Fetch final result immediatey
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
    """Extract audio file paths from completed job data."""
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
) -> Optional[str]:
    """Download audio file from ComfyUI and store via OpenWebUI's native file handler."""
    try:
        file_extension = os.path.splitext(filename)[1] or ".mp3"
        # Use sanitized song name for the stored filename
        if song_name:
            safe_name = re.sub(r"[^\w\s-]", "", song_name).strip().replace(" ", "_")
            local_filename = f"{safe_name}{file_extension}"
        else:
            local_filename = f"ace_step_{uuid.uuid4().hex[:8]}{file_extension}"

        # Map common audio extensions to MIME types
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

        async with aiohttp.ClientSession() as session:
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
    """Get all currently loaded models in VRAM (Async)"""
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
    """Unload all currently loaded models from VRAM with verification (Async)"""
    try:
        # 1. Get loaded models
        loaded_models = await get_loaded_models_async(api_url)
        if not loaded_models:
            return True

        # 2. Unload each
        async with aiohttp.ClientSession() as session:
            for model in loaded_models:
                model_name = model.get("name", model.get("model", ""))
                if model_name:
                    payload = {"model": model_name, "keep_alive": 0}
                    async with session.post(
                        f"{api_url.rstrip('/')}/api/generate", json=payload
                    ) as resp:
                        pass  # Fire and forget the unload request properly

        # 3. Wait/Verify cycle (max 5 seconds)
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


def generate_audio_player_embed(
    tracks: list[Dict[str, str]],
    song_title: str,
    tags: str,
    lyrics: Optional[str] = None,
) -> str:
    """Generate a sleek custom audio player embed with version selector."""
    safe_title = (
        song_title.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )
    safe_tags = tags.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    safe_lyrics = (
        (lyrics or "Instrumental")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    player_id = uuid.uuid4().hex[:8]

    # Prepare track data for JS
    tracks_json = json.dumps(tracks)

    html = f"""
    <div style="display: flex; justify-content: center; width: 100%;">
        <div class="player-container" style="background: rgba(20, 20, 25, 0.4); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); max-width: 480px; width: 100%; margin-bottom: 20px; font-family: system-ui, -apple-system, sans-serif; box-sizing: border-box;">
        <style>
             #{player_id}_lyrics::-webkit-scrollbar {{
                width: 5px;
            }}
             #{player_id}_lyrics::-webkit-scrollbar-track {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 3px;
            }}
             #{player_id}_lyrics::-webkit-scrollbar-thumb {{
                background: rgba(255, 255, 255, 0.25);
                border-radius: 3px;
            }}
             #{player_id}_lyrics::-webkit-scrollbar-thumb:hover {{
                background: rgba(255, 255, 255, 0.35);
            }}
        </style>
        <div class="player-header" style="text-align: center; margin-bottom: 20px;">
            <div class="player-title" style="font-size: 18px; font-weight: 600; color: #f0f0f0; margin-bottom: 4px; letter-spacing: -0.2px;">{safe_title}</div>
            <div class="player-subtitle" style="font-size: 10px; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">ACE Step 1.5 Generation</div>
        </div>
        
        <!-- Track Selector -->
        <div id="trackSelectorContainer_{player_id}" style="margin-bottom: 20px; display: {"block" if len(tracks) > 1 else "none"};">
            <label style="display: block; font-size: 10px; color: #666; margin-bottom: 6px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Version Select</label>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                {"".join([f'<button class="track-btn" data-index="{i}" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #ccc; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 11px; transition: all 0.2s;">v{i + 1}</button>' for i in range(len(tracks))])}
            </div>
        </div>

        <audio id="audioPlayer_{player_id}" preload="auto" style="display: none;"></audio>
        
        <div class="custom-player" style="margin: 0 0 18px 0;">
            <div id="loadingText_{player_id}" style="position: absolute; color: #666; font-size: 10px; margin-top: -15px; display: none;">Loading...</div>
            <div class="progress-container" id="progressContainer_{player_id}" style="width: 100%; height: 4px; background: rgba(255, 255, 255, 0.1); border-radius: 2px; cursor: pointer; margin-bottom: 16px; position: relative; overflow: hidden; transition: height 0.2s;">
                <div class="progress-bar" id="progressBar_{player_id}" style="height: 100%; background: #fff; border-radius: 2px; width: 0%; box-shadow: 0 0 10px rgba(255,255,255,0.3); transition: width 0.1s linear;"></div>
            </div>
            
            <div class="controls" style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <button class="play-btn" id="playBtn_{player_id}" type="button" style="width: 50px; height: 50px; min-width: 50px; border-radius: 50%; background: linear-gradient(145deg, #333, #222); border: 1px solid rgba(255,255,255,0.1); color: #fff; font-size: 16px; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center; padding: 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='60%' height='60%' style="fill: #ffffff; pointer-events: none;"><path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/></svg>
                    </button>
                    <div class="time-display" id="timeDisplay_{player_id}" style="font-size: 11px; color: #888; font-variant-numeric: tabular-nums; letter-spacing: 0.5px;">0:00 / 0:00</div>
                    
                    <!-- Volume Control -->
                    <div style="display: flex; align-items: center; gap: 8px; margin-left: 8px;">
                        <button id="volumeBtn_{player_id}" style="background: none; border: none; color: #888; cursor: pointer; padding: 0; display: flex; align-items: center;">
                            <svg viewBox="0 0 24 24" style="width: 14px; height: 14px; fill: currentColor;"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>
                        </button>
                        <div id="volumeContainer_{player_id}" style="width: 40px; height: 4px; background: rgba(255, 255, 255, 0.1); border-radius: 2px; cursor: pointer; position: relative; overflow: hidden;">
                            <div id="volumeBar_{player_id}" style="height: 100%; background: #aaa; border-radius: 2px; width: 100%;"></div>
                        </div>
                    </div>
                </div>
                
                <a id="downloadBtn_{player_id}" href="#" download style="color: #666; text-decoration: none; font-size: 11px; display: flex; align-items: center; gap: 4px; transition: color 0.2s; padding: 6px 10px; border-radius: 4px; background: rgba(255,255,255,0.03);">
                    <svg viewBox="0 0 24 24" style="width: 12px; height: 12px; fill: currentColor;"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
                    Download
                </a>
            </div>
        </div>
        
        <div class="info-section" style="margin-top: 16px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.05);">
            <div class="info-item" style="margin-bottom: 12px;">
                <div class="info-label" style="font-size: 8px; font-weight: 700; color: #555; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.5px;">Style</div>
                <div class="info-content" style="font-size: 12px; color: #aaa; line-height: 1.4;">{safe_tags}</div>
            </div>
            <div class="info-item">
                <div class="info-label" style="font-size: 8px; font-weight: 700; color: #555; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.5px;">Lyrics</div>
                <div id="{player_id}_lyrics" class="lyrics-container" style="max-height: 120px; overflow-y: auto; font-size: 12px; color: #aaa; white-space: pre-wrap; word-wrap: break-word; line-height: 1.6; padding-right: 4px; scrollbar-width: thin; scrollbar-color: rgba(255, 255, 255, 0.25) rgba(255, 255, 255, 0.05);">{safe_lyrics}</div>
            </div>
            </div>
    </div>
    </div>
    
    <script>
        (function() {{
            const tracks = {tracks_json};
            let currentTrackIndex = 0;
            
            const audio = document.getElementById('audioPlayer_{player_id}');
            const playBtn = document.getElementById('playBtn_{player_id}');
            const progressBar = document.getElementById('progressBar_{player_id}');
            const progressContainer = document.getElementById('progressContainer_{player_id}');
            const timeDisplay = document.getElementById('timeDisplay_{player_id}');
            const downloadBtn = document.getElementById('downloadBtn_{player_id}');
            const trackBtns = document.querySelectorAll('#trackSelectorContainer_{player_id} .track-btn');
            const loadingText = document.getElementById('loadingText_{player_id}');
            
            const playIcon = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='60%' height='60%' style="fill: #ffffff; pointer-events: none;"><path d="M8 5.14v13.72a1.14 1.14 0 0 0 1.76.99l10.86-6.86a1.14 1.14 0 0 0 0-1.98L9.76 4.15A1.14 1.14 0 0 0 8 5.14z"/></svg>`;
            const pauseIcon = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='60%' height='60%' style="fill: #ffffff; pointer-events: none;"><rect x="6" y="4" width="4" height="16" rx="1.5" /><rect x="14" y="4" width="4" height="16" rx="1.5" /></svg>`;
            const resetIcon = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='60%' height='60%' style="fill: #ffffff; pointer-events: none;"><path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46A7.93 7.93 0 0 0 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74A7.93 7.93 0 0 0 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/></svg>`;
            const volumeUpIcon = '<svg viewBox="0 0 24 24" style="width: 14px; height: 14px; fill: currentColor;"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>';
            const volumeOffIcon = '<svg viewBox="0 0 24 24" style="width: 14px; height: 14px; fill: currentColor;"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/></svg>';
            
            function loadTrack(index) {{
                const track = tracks[index];
                audio.src = track.url;
                downloadBtn.href = track.url;
                downloadBtn.download = track.title + ".mp3";
                
                // Update active button state
                trackBtns.forEach(btn => {{
                    if (parseInt(btn.dataset.index) === index) {{
                        btn.style.background = '#fff';
                        btn.style.color = '#000';
                        btn.style.borderColor = '#fff';
                    }} else {{
                        btn.style.background = 'rgba(255,255,255,0.05)';
                        btn.style.color = '#ccc';
                        btn.style.borderColor = 'rgba(255,255,255,0.1)';
                    }}
                }});
                
                // Reset UI
                playBtn.innerHTML = playIcon;
                progressBar.style.width = '0%';
                timeDisplay.textContent = '0:00 / 0:00';
            }}

            // Initialize first track
            if (tracks.length > 0) {{
                loadTrack(0);
            }}
            
            // Track selection
            trackBtns.forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const index = parseInt(btn.dataset.index);
                    if (index !== currentTrackIndex) {{
                        const wasPlaying = !audio.paused;
                        currentTrackIndex = index;
                        loadTrack(index);
                        if (wasPlaying) audio.play();
                    }}
                }});
            }});
            
            playBtn.addEventListener('click', () => {{
                if (audio.paused) {{
                    const playPromise = audio.play();
                    if (playPromise !== undefined) {{
                        playPromise.catch(error => {{
                            console.error("Playback failed:", error);
                        }});
                    }}
                    playBtn.innerHTML = pauseIcon;
                }} else {{
                    audio.pause();
                    playBtn.innerHTML = playIcon;
                }}
            }});
            
            audio.addEventListener('waiting', () => {{ loadingText.style.display = 'block'; }});
            audio.addEventListener('playing', () => {{ loadingText.style.display = 'none'; }});
            audio.addEventListener('ended', () => {{ playBtn.innerHTML = resetIcon; }});
            
            audio.addEventListener('timeupdate', () => {{
                const progress = (audio.currentTime / audio.duration) * 100;
                progressBar.style.width = progress + '%';
                
                const current = formatTime(audio.currentTime);
                const duration = formatTime(audio.duration);
                timeDisplay.textContent = current + ' / ' + duration;
            }});
            
            progressContainer.addEventListener('click', (e) => {{
                const rect = progressContainer.getBoundingClientRect();
                const percent = (e.clientX - rect.left) / rect.width;
                audio.currentTime = percent * audio.duration;
            }});
            
            // Volume Control Logic
            const volumeBtn = document.getElementById('volumeBtn_{player_id}');
            const volumeContainer = document.getElementById('volumeContainer_{player_id}');
            const volumeBar = document.getElementById('volumeBar_{player_id}');
            let lastVolume = 1.0;

            volumeBtn.addEventListener('click', () => {{
                if (audio.muted) {{
                    audio.muted = false;
                    audio.volume = lastVolume;
                    volumeBtn.innerHTML = volumeUpIcon;
                    volumeBar.style.width = (lastVolume * 100) + '%';
                }} else {{
                    lastVolume = audio.volume;
                    audio.muted = true;
                    volumeBtn.innerHTML = volumeOffIcon;
                    volumeBar.style.width = '0%';
                }}
            }});

            volumeContainer.addEventListener('click', (e) => {{
                const rect = volumeContainer.getBoundingClientRect();
                const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                audio.volume = percent;
                audio.muted = false;
                
                volumeBar.style.width = (percent * 100) + '%';
                volumeBtn.innerHTML = percent === 0 ? volumeOffIcon : volumeUpIcon;
            }});

            // Init volume
            audio.volume = 1.0;
            
            function formatTime(seconds) {{
                if (isNaN(seconds)) return '0:00';
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return mins + ':' + (secs < 10 ? '0' : '') + secs;
            }}
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
            description="Save the generated audio to Open WebUI's file storage. Files are tracked in the file manager and persist across restarts.",
        )
        show_player_embed: bool = Field(
            default=True,
            description="Show the embedded audio player. If false, only returns download link.",
        )
        batch_size: int = Field(
            default=1,
            description="Number of tracks to generate per request.",
        )
        max_duration: int = Field(
            default=180,
            description="Maximum allowed duration in seconds. Default is 180s.",
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
            description="Unload models after generation using ComfyUI-Unload-Model node. WARNING: Using custom workflows may break this functionality and possibly the whole worflow if node ids are not correclty setted.",
        )
        # workflow configuration
        workflow_json: str = Field(
            default=json.dumps(DEFAULT_WORKFLOW),
            description="ComfyUI Workflow JSON.",
        )
        model_name: str = Field(
            default="ace_step_1.5_turbo_aio.safetensors",
            description="Checkpoint name for ACE Step 1.5.",
        )
        # Node IDs based on Extras/audio_ace_step_1_5_API.json
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
        # Non-AIO workflow support - leave empty to disable
        clip_name_1: str = Field(
            default="",
            description="First CLIP model for DualCLIPLoader (non-AIO workflows). Leave empty for AIO. e.g. qwen_0.6b_ace15.safetensors",
        )
        clip_name_2: str = Field(
            default="",
            description="Second CLIP model for DualCLIPLoader (non-AIO workflows). Leave empty for AIO. e.g. qwen_1.7b_ace15.safetensors",
        )
        vae_name: str = Field(
            default="",
            description="VAE model for VAELoader (non-AIO workflows). Leave empty for AIO. e.g. ace_step/ace_1.5_vae.safetensors",
        )
        dual_clip_loader_node: str = Field(
            default="111", description="Node ID for DualCLIPLoader (non-AIO workflows)"
        )
        vae_loader_node: str = Field(
            default="110", description="Node ID for VAELoader (non-AIO workflows)"
        )

    class UserValves(BaseModel):
        generate_audio_codes: bool = Field(
            default=True,
            description="Enable generate audio codes. If false, disables generate audio codes for faster generation but lower quality.",
        )
        steps: int = Field(
            default=8,
            description="Sampling steps.",
        )
        seed: int = Field(
            default=-1,
            description="Random seed (-1 for random).",
        )

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
        batch_size = self.valves.batch_size
        user_valves = __user__.get("valves", self.UserValves())

        # Cap duration
        if duration > self.valves.max_duration:
            duration = self.valves.max_duration

        # Handle Steps
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

            # Use the new async method and wait extra time for safety
            unloaded = await unload_all_models_async(self.valves.ollama_url)

            # Add an extra safety buffer to ensure VRAM is truly released by the OS/Driver
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

        # Load Workflow from Valve
        try:
            workflow = json.loads(self.valves.workflow_json)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid Workflow JSON in valves: {e}")

        # Handle IDs from Valves
        text_node_id = self.valves.text_encoder_node
        latent_node_id = self.valves.empty_latent_node
        sampler_node_id = self.valves.sampler_node
        checkpoint_node_id = self.valves.checkpoint_node
        save_node_id = self.valves.save_node
        vae_node_id = self.valves.vae_decode_node
        unload_node_id = self.valves.unload_node

        # Parameter Injection
        # Determine Seed:
        # 1. Use function arg 'seed' if provided (not None).
        # 2. Else use user_valves.seed.
        # 3. If the resulting seed is -1 (or None), generate a random one.
        target_seed = seed if seed is not None else user_valves.seed
        if target_seed == -1 or target_seed is None:
            gen_seed = random.randint(1, 1500000000000)
        else:
            gen_seed = target_seed

        # 1. Update Text Encoder Node (94)
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

        # 2. Update Checkpoint Loader (97) - Inject Model Name
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

        # 2b. Non-AIO Workflow Support - DualCLIPLoader and VAELoader
        dual_clip_node_id = self.valves.dual_clip_loader_node
        vae_loader_node_id = self.valves.vae_loader_node

        # DualCLIPLoader - only update if valve is set and node exists
        if self.valves.clip_name_1 and dual_clip_node_id in workflow:
            node_inputs = workflow[dual_clip_node_id].get("inputs", {})
            if "clip_name1" in node_inputs:
                node_inputs["clip_name1"] = self.valves.clip_name_1
            if "clip_name2" in node_inputs:
                node_inputs["clip_name2"] = (
                    self.valves.clip_name_2 or self.valves.clip_name_1
                )

        # VAELoader - only update if valve is set and node exists
        if self.valves.vae_name and vae_loader_node_id in workflow:
            node_inputs = workflow[vae_loader_node_id].get("inputs", {})
            if "vae_name" in node_inputs:
                node_inputs["vae_name"] = self.valves.vae_name

        # 3. Update Empty Latent (98) - Batch Size & Duration Sync
        if latent_node_id in workflow:
            inputs = workflow[latent_node_id]["inputs"]
            inputs["batch_size"] = batch_size
            inputs["seconds"] = duration

        # 3. Update KSampler (3) - Sync Seed & Steps
        if sampler_node_id in workflow:
            workflow[sampler_node_id]["inputs"]["seed"] = gen_seed
            workflow[sampler_node_id]["inputs"]["steps"] = steps

        # 5. Update Save Node filename_prefix with song title
        if save_node_id in workflow:
            safe_title = re.sub(r"[^\w\s-]", "", song_title).strip().replace(" ", "_")
            workflow[save_node_id]["inputs"]["filename_prefix"] = f"audio/{safe_title}"

        # 4. Handle Unload Node (Remove if disabled)
        if not self.valves.unload_comfyui_models:
            if unload_node_id in workflow:
                # Get the input attached to the unload node (pass-through)
                unload_inputs = workflow[unload_node_id].get("inputs", {})
                source_link = unload_inputs.get("value")

                if source_link:
                    # Re-route the Save Node to bypass the unload node
                    if save_node_id in workflow:
                        save_inputs = workflow[save_node_id].get("inputs", {})
                        current_audio_input = save_inputs.get("audio")
                        # Verify connection: SaveAudio -> UnloadNode
                        if (
                            isinstance(current_audio_input, list)
                            and len(current_audio_input) > 0
                        ):
                            if str(current_audio_input[0]) == str(unload_node_id):
                                # Bypass: Connect SaveAudio directly to VAE (source_link)
                                workflow[save_node_id]["inputs"]["audio"] = source_link

                # Remove the node from generation
                del workflow[unload_node_id]

        client_id = str(uuid.uuid4())
        ws_url = (
            self.valves.comfyui_api_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )
            + "/ws"
        )
        http_url = self.valves.comfyui_api_url

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

            # Connect, Submit, and Wait (Atomic Operation)
            result_data = await connect_submit_and_wait(
                ws_url, http_url, prompt_payload, client_id, self.valves.max_wait_time
            )

            # Process Outputs
            audio_files = extract_audio_files(result_data)
            if not audio_files:
                raise Exception("No audio files generated.")

            track_list = []

            # Resolve user object once for all batch uploads
            user_obj = None
            if self.valves.save_to_storage and __request__:
                user_obj = Users.get_user_by_id(__user__["id"])

            # Loop through all generated files (batch support)
            for idx, finfo in enumerate(audio_files):
                fname = finfo["filename"]
                subfolder = finfo["subfolder"]

                # Title differentiation for batches
                track_title = song_title
                if batch_size > 1:
                    track_title = f"{song_title} (Track {idx + 1})"

                if user_obj and __request__:
                    storage_url = await download_audio_to_storage(
                        __request__, user_obj, http_url, fname, subfolder, track_title
                    )

                    if storage_url:
                        track_list.append({"title": track_title, "url": storage_url})
                    else:
                        # Fallback to direct link if storage upload failed
                        direct_url = f"{http_url}/view?filename={fname}&type=output&subfolder={subfolder}"
                        track_list.append({"title": track_title, "url": direct_url})
                else:
                    # Direct link fallback
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
                    track_list, song_title, tags, lyrics
                )
                await __event_emitter__(
                    {
                        "type": "embeds",
                        "data": {
                            "embeds": [final_html],
                        },
                    }
                )
                message = "The audio player has been successfully embedded above. Inform the user that their song is ready to listen to or download in the above UI component."
            else:
                message += "to use the following download links:"

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
