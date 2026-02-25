"""
title: ComfyUI VibeVoice TTS Generator
description: Tool to generate speech using VibeVoice workflows via ComfyUI API. Supports both single speaker and multi-speaker (up to 4) voice cloning. The voice files must be pre-loaded in ComfyUI.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.0
"""

import os
import json
import uuid
import asyncio
import aiohttp
import requests
from typing import Optional, Dict, Any, Callable, Awaitable, cast, Union
from pydantic import BaseModel, Field
from open_webui.models.users import Users
from fastapi.responses import HTMLResponse
from fastapi import Request, UploadFile
from open_webui.routers.files import upload_file_handler
import io
import re

# Default workflow for single speaker TTS
DEFAULT_SINGLE_SPEAKER_WORKFLOW = json.dumps(
    {
        "15": {
            "inputs": {"audio": "Voice.mp3", "audioUI": ""},
            "class_type": "LoadAudio",
            "_meta": {"title": "CargarAudio"},
        },
        "16": {
            "inputs": {
                "filename_prefix": "audio/ComfyUI",
                "quality": "V0",
                "audioUI": "",
                "audio": ["45", 0],
            },
            "class_type": "SaveAudioMP3",
            "_meta": {"title": "Save Audio (MP3)"},
        },
        "44": {
            "inputs": {
                "text": "Hello, this is a test of the VibeVoice text-to-speech system.",
                "model": "VibeVoice-1.5B",
                "attention_type": "auto",
                "quantize_llm": "full precision",
                "free_memory_after_generate": True,
                "diffusion_steps": 20,
                "seed": 42,
                "cfg_scale": 1.3,
                "use_sampling": False,
                "temperature": 0.95,
                "top_p": 0.95,
                "max_words_per_chunk": 250,
                "voice_speed_factor": 1,
            },
            "class_type": "VibeVoiceSingleSpeakerNode",
            "_meta": {"title": "VibeVoice Single Speaker"},
        },
        "45": {
            "inputs": {"anything": ["44", 0]},
            "class_type": "easy cleanGpuUsed",
            "_meta": {"title": "Clean VRAM Used"},
        },
    }
)

# Default workflow for multi-speaker TTS
DEFAULT_MULTI_SPEAKER_WORKFLOW = json.dumps(
    {
        "15": {
            "inputs": {"audio": "Voice1.mp3", "audioUI": ""},
            "class_type": "LoadAudio",
            "_meta": {"title": "CargarAudio"},
        },
        "16": {
            "inputs": {
                "filename_prefix": "audio/ComfyUI",
                "quality": "V0",
                "audioUI": "",
                "audio": ["39", 0],
            },
            "class_type": "SaveAudioMP3",
            "_meta": {"title": "Save Audio (MP3)"},
        },
        "17": {
            "inputs": {"audio": "Voice2.mp3", "audioUI": ""},
            "class_type": "LoadAudio",
            "_meta": {"title": "CargarAudio"},
        },
        "36": {
            "inputs": {
                "text": "[1]: Hello, this is the first speaker.\n[2]: Hi there, I'm the second speaker.\n[1]: Nice to meet you!\n[2]: Nice to meet you too!",
                "model": "VibeVoice-1.5B",
                "attention_type": "auto",
                "quantize_llm": "full precision",
                "free_memory_after_generate": True,
                "diffusion_steps": 20,
                "seed": 42,
                "cfg_scale": 1.3,
                "use_sampling": False,
                "temperature": 0.95,
                "top_p": 0.95,
                "voice_speed_factor": 1,
            },
            "class_type": "VibeVoiceMultipleSpeakersNode",
            "_meta": {"title": "VibeVoice Multiple Speakers"},
        },
        "39": {
            "inputs": {"anything": ["36", 0]},
            "class_type": "easy cleanGpuUsed",
            "_meta": {"title": "Clean VRAM Used"},
        },
    }
)


async def wait_for_completion_ws(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_id: str,
    client_id: str,
    max_wait_time: int,
    event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """
    Waits for ComfyUI job completion using WebSocket for real-time updates.
    Returns the job output data upon successful execution.
    """
    start_time = asyncio.get_event_loop().time()
    job_data_output = None

    try:
        async with aiohttp.ClientSession().ws_connect(
            f"{comfyui_ws_url}?clientId={client_id}"
        ) as ws:
            async for msg in ws:
                if asyncio.get_event_loop().time() - start_time > max_wait_time:
                    raise TimeoutError(f"Job timed out after {max_wait_time}s")

                if msg.type == aiohttp.WSMsgType.TEXT:
                    message = json.loads(msg.data)
                    msg_type = message.get("type")

                    if msg_type == "executing":
                        data = message.get("data", {})
                        node = data.get("node")
                        if node is None and data.get("prompt_id") == prompt_id:
                            # Job completed
                            if event_emitter:
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "Generation completed, retrieving audio..."
                                        },
                                    }
                                )

                            # Fetch final job data
                            async with aiohttp.ClientSession() as session:
                                async with session.get(
                                    f"{comfyui_http_url}/history/{prompt_id}"
                                ) as resp:
                                    if resp.status == 200:
                                        history = await resp.json()
                                        job_data_output = history.get(prompt_id, {})
                            break

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

    except asyncio.TimeoutError:
        raise TimeoutError(f"Overall wait/connection timed out after {max_wait_time}s")
    except Exception as e:
        raise Exception(f"Error during WebSocket communication: {e}")

    if job_data_output is None:
        raise Exception("No job data received from ComfyUI")

    return job_data_output


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
                "audios",
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
                                # Treat any non-dict entry as a filename string
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
    request: Request, user, comfyui_http_url: str, filename: str, subfolder: str = ""
) -> Optional[str]:
    try:
        file_extension = os.path.splitext(filename)[1] or ".wav"
        local_filename = f"vibevoice_{uuid.uuid4().hex[:8]}{file_extension}"

        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        content_type = mime_map.get(file_extension.lower(), "audio/wav")

        subfolder_param = f"&subfolder={subfolder}" if subfolder else ""
        comfyui_file_url = (
            f"{comfyui_http_url}/view?filename={filename}&type=output{subfolder_param}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(comfyui_file_url) as resp:
                if resp.status == 200:
                    audio_content = await resp.read()

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

                    if file_item and getattr(file_item, "id", None):
                        file_id = str(getattr(file_item, "id", ""))
                        relative_path = request.app.url_path_for(
                            "get_file_content_by_id", id=file_id
                        )
                        return relative_path

                    print("[DEBUG] upload_file_handler returned no file item")
                    return None
                else:
                    print(f"[DEBUG] Failed to download audio: HTTP {resp.status}")
                    return None

    except Exception as e:
        print(f"[DEBUG] Error uploading audio to storage: {str(e)}")
        return None


def get_loaded_models(api_url: str = "http://localhost:11434") -> list[Dict[str, Any]]:
    """Get all currently loaded models in VRAM"""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/ps")
        response.raise_for_status()
        models = response.json().get("models", [])
        return cast(list[Dict[str, Any]], models)
    except requests.RequestException as e:
        print(f"Error fetching loaded models: {e}")
        raise


def unload_all_models(api_url: str = "http://localhost:11434") -> dict[str, bool]:
    """Unload all currently loaded models from VRAM"""
    try:
        loaded_models = get_loaded_models(api_url)
        results: Dict[str, bool] = {}

        for model in loaded_models:
            model_name = model.get("name", "")
            if not model_name:
                continue

            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                )
                results[model_name] = response.status_code == 200
            except requests.RequestException:
                results[model_name] = False

        return results
    except requests.RequestException as e:
        print(f"Error unloading models: {e}")
        return {}


def generate_audio_player_embed(
    audio_url: str,
    text: str,
    speaker_info: str,
    generation_type: str = "Single Speaker",
) -> str:
    """Generate a sleek custom audio player embed with styled controls."""
    # Escape HTML special characters
    safe_text = text.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    safe_speaker = (
        speaker_info.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )
    safe_type = (
        generation_type.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VibeVoice TTS</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            overflow: hidden;
        }}
        body {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .player-container {{
            background: rgba(20, 20, 25, 0.25);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            max-width: 480px;
            width: 100%;
        }}
        .player-header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .player-title {{
            font-size: 20px;
            font-weight: 600;
            color: #f0f0f0;
            margin-bottom: 5px;
            letter-spacing: -0.2px;
        }}
        .player-subtitle {{
            font-size: 10px;
            color: #888;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        
        /* Custom Audio Player */
        .custom-player {{
            margin: 18px 0;
        }}
        .progress-container {{
            width: 100%;
            height: 5px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            cursor: pointer;
            margin-bottom: 14px;
            position: relative;
            overflow: hidden;
        }}
        .progress-bar {{
            height: 100%;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
            width: 0%;
            transition: width 0.1s linear;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .play-btn {{
            width: 44px;
            height: 44px;
            min-width: 44px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
        }}
        .play-btn svg {{
            width: 16px;
            height: 16px;
            fill: currentColor;
        }}
        .play-btn.playing svg {{
            width: 14px;
            height: 14px;
        }}
        .play-btn:hover {{
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }}
        .time-display {{
            font-size: 12px;
            color: #aaa;
            font-variant-numeric: tabular-nums;
            min-width: 85px;
        }}
        .volume-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
        }}
        .volume-btn {{
            width: 28px;
            height: 28px;
            min-width: 28px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #aaa;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
        }}
        .volume-btn svg {{
            width: 14px;
            height: 14px;
            fill: currentColor;
        }}
        .volume-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            color: #ccc;
        }}
        .volume-slider {{
            flex: 1;
            height: 4px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 2px;
            cursor: pointer;
            position: relative;
        }}
        .volume-bar {{
            height: 100%;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 2px;
            width: 100%;
        }}
        
        .info-section {{
            margin-top: 18px;
            padding-top: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .info-item {{
            margin-bottom: 14px;
        }}
        .info-item:last-child {{
            margin-bottom: 0;
        }}
        .info-label {{
            font-size: 10px;
            font-weight: 600;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 6px;
        }}
        .info-content {{
            font-size: 13px;
            color: #ccc;
            line-height: 1.6;
        }}
        
        /* Text content with scrollbar */
        .text-container {{
            max-height: 150px;
            overflow-y: auto;
            font-size: 13px;
            color: #ccc;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            padding-right: 8px;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.25) rgba(255, 255, 255, 0.05);
        }}
        .text-container::-webkit-scrollbar {{
            width: 5px;
        }}
        .text-container::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }}
        .text-container::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.25);
            border-radius: 3px;
        }}
        .text-container::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.35);
        }}
        
        .download-btn {{
            display: inline-block;
            width: 100%;
            padding: 11px 18px;
            background: rgba(255, 255, 255, 0.08);
            color: #e0e0e0;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: 13px;
            transition: all 0.2s;
            margin-top: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .download-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.2);
        }}
        
        audio {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="player-container">
        <div class="player-header">
            <div class="player-title">VibeVoice TTS</div>
            <div class="player-subtitle">{safe_type}</div>
        </div>
        
        <audio id="audioPlayer" preload="auto">
            <source src="{audio_url}" type="audio/wav">
            <source src="{audio_url}" type="audio/mpeg">
        </audio>
        
        <div class="custom-player">
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="controls">
                <button class="play-btn" id="playBtn">
                    <svg viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </button>
                <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
                <div class="volume-container">
                    <button class="volume-btn" id="volumeBtn">
                        <svg viewBox="0 0 24 24" id="volumeIcon">
                            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
                        </svg>
                    </button>
                    <div class="volume-slider" id="volumeSlider">
                        <div class="volume-bar" id="volumeBar"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="info-section">
            <div class="info-item">
                <div class="info-label">Voice Configuration</div>
                <div class="info-content">{safe_speaker}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Text</div>
                <div class="text-container">{safe_text}</div>
            </div>
        </div>
        
        <a href="{audio_url}" download class="download-btn">
            <svg viewBox="0 0 24 24" style="width:14px;height:14px;fill:currentColor;vertical-align:middle;margin-right:6px;">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
            Download Audio
        </a>
    </div>
    
    <script>
        const audio = document.getElementById('audioPlayer');
        const playBtn = document.getElementById('playBtn');
        const progressBar = document.getElementById('progressBar');
        const progressContainer = document.getElementById('progressContainer');
        const timeDisplay = document.getElementById('timeDisplay');
        const volumeSlider = document.getElementById('volumeSlider');
        const volumeBar = document.getElementById('volumeBar');
        const volumeBtn = document.getElementById('volumeBtn');
        const volumeIcon = document.getElementById('volumeIcon');
        
        // SVG icons
        const playIcon = '<path d="M8 5v14l11-7z"/>';
        const pauseIcon = '<path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>';
        const volumeHighIcon = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>';
        const volumeMuteIcon = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';
        const volumeLowIcon = '<path d="M7 9v6h4l5 5V4l-5 5H7z"/>';
        
        // Play/Pause
        playBtn.addEventListener('click', () => {{
            const svg = playBtn.querySelector('svg');
            if (audio.paused) {{
                audio.play();
                svg.innerHTML = pauseIcon;
                playBtn.classList.add('playing');
            }} else {{
                audio.pause();
                svg.innerHTML = playIcon;
                playBtn.classList.remove('playing');
            }}
        }});
        
        // Update progress
        audio.addEventListener('timeupdate', () => {{
            const progress = (audio.currentTime / audio.duration) * 100;
            progressBar.style.width = progress + '%';
            
            const current = formatTime(audio.currentTime);
            const duration = formatTime(audio.duration);
            timeDisplay.textContent = current + ' / ' + duration;
        }});
        
        // Seek
        progressContainer.addEventListener('click', (e) => {{
            const rect = progressContainer.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            audio.currentTime = percent * audio.duration;
        }});
        
        // Volume
        volumeSlider.addEventListener('click', (e) => {{
            const rect = volumeSlider.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            audio.volume = Math.max(0, Math.min(1, percent));
            volumeBar.style.width = (percent * 100) + '%';
            updateVolumeIcon(percent);
        }});
        
        volumeBtn.addEventListener('click', () => {{
            if (audio.volume > 0) {{
                audio.dataset.prevVolume = audio.volume;
                audio.volume = 0;
                volumeBar.style.width = '0%';
                volumeIcon.innerHTML = volumeMuteIcon;
            }} else {{
                const prevVol = audio.dataset.prevVolume || 1;
                audio.volume = prevVol;
                volumeBar.style.width = (prevVol * 100) + '%';
                updateVolumeIcon(prevVol);
            }}
        }});
        
        function updateVolumeIcon(volume) {{
            if (volume === 0) {{
                volumeIcon.innerHTML = volumeMuteIcon;
            }} else if (volume < 0.5) {{
                volumeIcon.innerHTML = volumeLowIcon;
            }} else {{
                volumeIcon.innerHTML = volumeHighIcon;
            }}
        }}
        
        function formatTime(seconds) {{
            if (isNaN(seconds)) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        }}
    </script>
</body>
</html>"""
    return html


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188",
            description="ComfyUI HTTP API endpoint.",
        )
        single_speaker_workflow: str = Field(
            default=DEFAULT_SINGLE_SPEAKER_WORKFLOW,
            description="Full VibeVoice Single Speaker workflow JSON (string).",
        )
        multi_speaker_workflow: str = Field(
            default=DEFAULT_MULTI_SPEAKER_WORKFLOW,
            description="Full VibeVoice Multi-Speaker workflow JSON (string).",
        )
        single_text_node: str = Field(
            default="44",
            description="Node ID for text input in single speaker workflow.",
        )
        single_seed_node: str = Field(
            default="44",
            description="Node ID for seed input in single speaker workflow.",
        )
        multi_text_node: str = Field(
            default="36",
            description="Node ID for text input in multi-speaker workflow.",
        )
        multi_seed_node: str = Field(
            default="36",
            description="Node ID for seed input in multi-speaker workflow.",
        )
        max_wait_time: int = Field(
            default=300,
            description="Max wait time for generation (seconds).",
        )
        unload_ollama_models: bool = Field(
            default=False,
            description="Unload all Ollama models before calling ComfyUI.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL.",
        )
        save_local: bool = Field(
            default=True,
            description="Copy the generated audio to the Open WebUI Storage Backend",
        )

        show_player_embed: bool = Field(
            default=True,
            description="Show the embedded audio player. If false, only returns download link.",
        )

    class UserValves(BaseModel):
        seed: int = Field(
            default=-1,
            description="Seed for audio generation. Use -1 for random seed, or any positive number for reproducible results.",
            ge=-1,
        )

    def __init__(self):
        self.valves = self.Valves()

    async def generate_single_speaker_tts(
        self,
        text: str,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str | HTMLResponse:
        """
        Generate natural, expressive speech from text using voice cloning.

        Creates high-quality, conversational audio that maintains the speaker's identity and prosody.
        Perfect for voiceovers, narration, audiobooks, podcasts, and any content requiring natural-sounding speech.

        Args:
            text: The complete text to convert to speech. Can be any length.

        Example:
            text = "Welcome to today's podcast. We'll be exploring the fascinating world of artificial intelligence and its impact on our daily lives."
        """
        try:
            # Get user valves
            user = Users.get_user_by_id(__user__["id"])
            user_valves = (
                user.valves
                if hasattr(user, "valves") and user.valves
                else self.UserValves()
            )
            if not isinstance(user_valves, self.UserValves):
                user_valves = self.UserValves(**user_valves)
            # Unload Ollama models if requested
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Unloading Ollama models to free VRAM...",
                                "done": False,
                            },
                        }
                    )
                unload_all_models(self.valves.ollama_url)

            # Load workflow
            workflow_str = self.valves.single_speaker_workflow
            if not workflow_str or workflow_str.strip() == "":
                raise ValueError(
                    "Single speaker workflow JSON is not configured in valves"
                )

            workflow = json.loads(workflow_str)

            # Set text input
            if self.valves.single_text_node in workflow:
                workflow[self.valves.single_text_node]["inputs"]["text"] = text
            else:
                raise ValueError(
                    f"Text node {self.valves.single_text_node} not found in workflow"
                )

            # Set seed (random if -1, otherwise use user's value)
            if user_valves.seed == -1:
                import random

                seed = random.randint(0, 2**32 - 1)
            else:
                seed = user_valves.seed

            if self.valves.single_seed_node in workflow:
                workflow[self.valves.single_seed_node]["inputs"]["seed"] = seed

            # Submit to ComfyUI
            client_id = str(uuid.uuid4())
            comfyui_http_url = self.valves.comfyui_api_url.rstrip("/")
            comfyui_ws_url = comfyui_http_url.replace("http", "ws", 1) + "/ws"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generating speech...", "done": False},
                    }
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{comfyui_http_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"ComfyUI API error: {resp.status} - {error_text}"
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

            # Wait for completion
            job_data = await wait_for_completion_ws(
                comfyui_ws_url,
                comfyui_http_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
                __event_emitter__,
            )

            # Extract audio files
            audio_files = extract_audio_files(job_data)
            if not audio_files:
                raise Exception("No audio files generated")

            audio_info = audio_files[0]
            filename = audio_info.get("filename", "")
            subfolder = audio_info.get("subfolder", "")

            # Download to cache if requested
            if self.valves.save_local:
                cache_url = await download_audio_to_storage(
                    __request__, user, comfyui_http_url, filename, subfolder
                )
                if cache_url:
                    audio_url = cache_url
                else:
                    subfolder_param = f"&subfolder={subfolder}" if subfolder else ""
                    audio_url = f"{comfyui_http_url}/view?filename={filename}&type=output{subfolder_param}"
            else:
                subfolder_param = f"&subfolder={subfolder}" if subfolder else ""
                audio_url = f"{comfyui_http_url}/view?filename={filename}&type=output{subfolder_param}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generation complete!", "done": True},
                    }
                )

            # Generate response
            if self.valves.show_player_embed:
                speaker_info = "Single voice clone (pre-loaded in ComfyUI)"
                html_content = generate_audio_player_embed(
                    audio_url, text, speaker_info, "Single Speaker"
                )
                return HTMLResponse(
                    content=html_content, headers={"content-disposition": "inline"}
                )
            else:
                return f"Audio generated successfully!\n\nDownload: {audio_url}"

        except Exception as e:
            error_msg = f"Error generating single speaker TTS: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg

    async def generate_multi_speaker_tts(
        self,
        text: str,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str | HTMLResponse:
        """
                Generate dynamic multi-speaker conversations with distinct cloned voices.

                Creates natural dialogue between up to 4 different speakers, each with their own voice characteristics.
                Perfect for podcasts, interviews, audiobooks with multiple characters, or any conversational content.

                Args:
                    text: Dialogue script where each line starts with [1], [2], [3], or [4] to indicate the speaker, followed by a colon and their dialogue.

                Example:
                    text = '''[1]: Welcome to the AI Ethics Roundtable. I'm your host Sarah.
        [2]: Thanks for having me, Sarah. I'm excited to discuss this important topic.
        [1]: Let's start with your recent paper on algorithmic bias. Can you tell us more?
        [2]: Absolutely. We found that bias can emerge in unexpected ways during training.
        [1]: That's fascinating. How do you think we should address this?'''

                Note: Each number [1] through [4] represents a different speaker with a unique voice.
        """
        try:
            # Get user valves
            user = Users.get_user_by_id(__user__["id"])
            user_valves = (
                user.valves
                if hasattr(user, "valves") and user.valves
                else self.UserValves()
            )
            if not isinstance(user_valves, self.UserValves):
                user_valves = self.UserValves(**user_valves)

            # Unload Ollama models if requested
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Unloading Ollama models to free VRAM...",
                                "done": False,
                            },
                        }
                    )
                unload_all_models(self.valves.ollama_url)

            # Load workflow
            workflow_str = self.valves.multi_speaker_workflow
            if not workflow_str or workflow_str.strip() == "":
                raise ValueError(
                    "Multi-speaker workflow JSON is not configured in valves"
                )

            workflow = json.loads(workflow_str)

            # Set text input
            if self.valves.multi_text_node in workflow:
                workflow[self.valves.multi_text_node]["inputs"]["text"] = text
            else:
                raise ValueError(
                    f"Text node {self.valves.multi_text_node} not found in workflow"
                )

            # Set seed (random if -1, otherwise use user's value)
            if user_valves.seed == -1:
                import random

                seed = random.randint(0, 2**32 - 1)
            else:
                seed = user_valves.seed

            if self.valves.multi_seed_node in workflow:
                workflow[self.valves.multi_seed_node]["inputs"]["seed"] = seed

            # Submit to ComfyUI
            client_id = str(uuid.uuid4())
            comfyui_http_url = self.valves.comfyui_api_url.rstrip("/")
            comfyui_ws_url = comfyui_http_url.replace("http", "ws", 1) + "/ws"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generating speech...", "done": False},
                    }
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{comfyui_http_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"ComfyUI API error: {resp.status} - {error_text}"
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

            # Wait for completion
            job_data = await wait_for_completion_ws(
                comfyui_ws_url,
                comfyui_http_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
                __event_emitter__,
            )

            # Extract audio files
            audio_files = extract_audio_files(job_data)
            if not audio_files:
                raise Exception("No audio files generated")

            audio_info = audio_files[0]
            filename = audio_info.get("filename", "")
            subfolder = audio_info.get("subfolder", "")

            # Download to cache if requested
            if self.valves.save_local:
                cache_url = await download_audio_to_storage(
                    __request__, user, comfyui_http_url, filename, subfolder
                )
                if cache_url:
                    audio_url = cache_url
                else:
                    subfolder_param = f"&subfolder={subfolder}" if subfolder else ""
                    audio_url = f"{comfyui_http_url}/view?filename={filename}&type=output{subfolder_param}"
            else:
                subfolder_param = f"&subfolder={subfolder}" if subfolder else ""
                audio_url = f"{comfyui_http_url}/view?filename={filename}&type=output{subfolder_param}"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generation complete!", "done": True},
                    }
                )

            # Generate response
            if self.valves.show_player_embed:
                speaker_info = (
                    "Multiple voice clones (pre-loaded in ComfyUI) - [1], [2], [3], [4]"
                )
                html_content = generate_audio_player_embed(
                    audio_url, text, speaker_info, "Multi-Speaker"
                )
                return HTMLResponse(
                    content=html_content, headers={"content-disposition": "inline"}
                )
            else:
                return f"Audio generated successfully!\n\nDownload: {audio_url}"

        except Exception as e:
            error_msg = f"Error generating multi-speaker TTS: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return error_msg
