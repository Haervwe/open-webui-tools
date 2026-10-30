"""
title: MiniMax Speech Generator
description: Generate speech with MiniMax and return a playable MP3 audio embed.
author: octo-patch
author_url: https://github.com/octo-patch
version: 1.1.0
required_open_webui_version: 0.9.0
"""

import asyncio
import html
import io
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Literal, Optional, Tuple, Union

import aiohttp
from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from open_webui.models.chats import Chats
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DEFAULT_MODEL = "speech-2.8-hd"
SPEECH_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/t2a_v2",
    "cn_zh": "https://api.minimaxi.com/v1/t2a_v2"
}

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]

# Chat IDs that are never persisted (temporary chats, channels); files are not
# linked to these. Mirrors open_webui.utils.chat_id.is_saved_chat_id.
NON_SAVED_CHAT_ID_PREFIXES = ("temporary:", "local:", "channel:")
TEXT_PREVIEW_CHARS = 1000


class StorageError(Exception):
    """Raised when the generated audio cannot be saved to Open WebUI storage."""


def _script_json(value: Any) -> str:
    """Serialize a value for safe inclusion inside a <script> block."""
    return json.dumps(value).replace("</", "<\\/")


def generate_audio_player_embed(
    audio_url: str, text: str, voice_id: str, model: str
) -> str:
    """Build the styled audio player embed (shared look of the repo's TTS tools).

    The stored file is fetched with the user's token (as in the AceStep tool)
    because the embed iframe is sandboxed and may not send session cookies.
    """
    preview = text if len(text) <= TEXT_PREVIEW_CHARS else text[:TEXT_PREVIEW_CHARS] + "..."
    safe_text = html.escape(preview)
    safe_voice = html.escape(f"{voice_id} · {model}")
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MiniMax Speech</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            overflow: hidden;
        }}
        body {{ display: flex; justify-content: center; align-items: center; padding: 20px; }}
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
        .player-header {{ text-align: center; margin-bottom: 20px; }}
        .player-title {{ font-size: 20px; font-weight: 600; color: #f0f0f0; margin-bottom: 5px; letter-spacing: -0.2px; }}
        .player-subtitle {{ font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.8px; }}
        .custom-player {{ margin: 18px 0; }}
        .progress-container {{
            width: 100%; height: 5px; background: rgba(255, 255, 255, 0.08);
            border-radius: 3px; cursor: pointer; margin-bottom: 14px; overflow: hidden;
        }}
        .progress-bar {{ height: 100%; background: rgba(255, 255, 255, 0.3); border-radius: 3px; width: 0%; transition: width 0.1s linear; }}
        .controls {{ display: flex; align-items: center; gap: 14px; }}
        .play-btn {{
            width: 44px; height: 44px; min-width: 44px; border-radius: 50%;
            background: rgba(255, 255, 255, 0.12); border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff; cursor: pointer; transition: all 0.2s;
            display: flex; align-items: center; justify-content: center; padding: 0;
        }}
        .play-btn svg {{ width: 16px; height: 16px; fill: currentColor; }}
        .play-btn:hover {{ background: rgba(255, 255, 255, 0.2); transform: scale(1.05); }}
        .time-display {{ font-size: 12px; color: #aaa; font-variant-numeric: tabular-nums; min-width: 85px; }}
        .volume-container {{ display: flex; align-items: center; gap: 8px; flex: 1; }}
        .volume-btn {{
            width: 28px; height: 28px; min-width: 28px; border-radius: 50%;
            background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.1);
            color: #aaa; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0;
        }}
        .volume-btn svg {{ width: 14px; height: 14px; fill: currentColor; }}
        .volume-slider {{ flex: 1; height: 4px; background: rgba(255, 255, 255, 0.08); border-radius: 2px; cursor: pointer; }}
        .volume-bar {{ height: 100%; background: rgba(255, 255, 255, 0.3); border-radius: 2px; width: 100%; }}
        .info-section {{ margin-top: 18px; padding-top: 18px; border-top: 1px solid rgba(255, 255, 255, 0.08); }}
        .info-item {{ margin-bottom: 14px; }}
        .info-item:last-child {{ margin-bottom: 0; }}
        .info-label {{ font-size: 10px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }}
        .info-content {{ font-size: 13px; color: #ccc; line-height: 1.6; word-wrap: break-word; }}
        .text-container {{
            max-height: 150px; overflow-y: auto; font-size: 13px; color: #ccc; line-height: 1.6;
            white-space: pre-wrap; word-wrap: break-word; padding-right: 8px;
            scrollbar-width: thin; scrollbar-color: rgba(255, 255, 255, 0.25) rgba(255, 255, 255, 0.05);
        }}
        .download-btn {{
            display: inline-block; width: 100%; padding: 11px 18px;
            background: rgba(255, 255, 255, 0.08); color: #e0e0e0; text-align: center;
            text-decoration: none; border-radius: 8px; font-weight: 500; font-size: 13px;
            margin-top: 16px; border: 1px solid rgba(255, 255, 255, 0.1); cursor: pointer;
        }}
        .download-btn:hover {{ background: rgba(255, 255, 255, 0.15); border-color: rgba(255, 255, 255, 0.2); }}
        audio {{ display: none; }}
    </style>
</head>
<body>
    <div class="player-container">
        <div class="player-header">
            <div class="player-title">MiniMax Speech</div>
            <div class="player-subtitle">Text to Speech</div>
        </div>
        <audio id="audioPlayer" preload="auto"></audio>
        <div class="custom-player">
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="controls">
                <button class="play-btn" id="playBtn">
                    <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                </button>
                <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
                <div class="volume-container">
                    <button class="volume-btn" id="volumeBtn">
                        <svg viewBox="0 0 24 24" id="volumeIcon">
                            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
                        </svg>
                    </button>
                    <div class="volume-slider" id="volumeSlider"><div class="volume-bar" id="volumeBar"></div></div>
                </div>
            </div>
        </div>
        <div class="info-section">
            <div class="info-item">
                <div class="info-label">Voice</div>
                <div class="info-content">{safe_voice}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Text</div>
                <div class="text-container">{safe_text}</div>
            </div>
        </div>
        <a href="#" id="downloadBtn" class="download-btn">
            <svg viewBox="0 0 24 24" style="width:14px;height:14px;fill:currentColor;vertical-align:middle;margin-right:6px;">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
            Download Audio
        </a>
    </div>
    <script>
    (function() {{
        var AUDIO_URL = {_script_json(audio_url)};
        var audio = document.getElementById('audioPlayer');
        var playBtn = document.getElementById('playBtn');
        var progressBar = document.getElementById('progressBar');
        var progressContainer = document.getElementById('progressContainer');
        var timeDisplay = document.getElementById('timeDisplay');
        var volumeSlider = document.getElementById('volumeSlider');
        var volumeBar = document.getElementById('volumeBar');
        var volumeBtn = document.getElementById('volumeBtn');
        var volumeIcon = document.getElementById('volumeIcon');
        var downloadBtn = document.getElementById('downloadBtn');

        var playIcon = '<path d="M8 5v14l11-7z"/>';
        var pauseIcon = '<path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>';
        var volumeHighIcon = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>';
        var volumeMuteIcon = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';
        var volumeLowIcon = '<path d="M7 9v6h4l5 5V4l-5 5H7z"/>';

        function authFetch(url) {{
            var opts = {{ credentials: 'include' }};
            try {{
                var token = localStorage.getItem('token');
                if (token) {{ opts.headers = {{ 'Authorization': 'Bearer ' + token }}; }}
            }} catch (e) {{}}
            return fetch(url, opts);
        }}

        function fetchBlobUrl(url, callback) {{
            authFetch(url).then(function(resp) {{
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.blob();
            }}).then(function(blob) {{
                callback(URL.createObjectURL(blob));
            }}).catch(function() {{ callback(url); }});
        }}

        var blobUrl = null;
        fetchBlobUrl(AUDIO_URL, function(url) {{ blobUrl = url; audio.src = url; }});

        playBtn.addEventListener('click', function() {{
            var svg = playBtn.querySelector('svg');
            if (audio.paused) {{
                audio.play().catch(function(e) {{ console.error(e); }});
                svg.innerHTML = pauseIcon;
            }} else {{
                audio.pause();
                svg.innerHTML = playIcon;
            }}
        }});
        audio.addEventListener('ended', function() {{ playBtn.querySelector('svg').innerHTML = playIcon; }});

        audio.addEventListener('timeupdate', function() {{
            var progress = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
            progressBar.style.width = progress + '%';
            timeDisplay.textContent = formatTime(audio.currentTime) + ' / ' + formatTime(audio.duration);
        }});

        progressContainer.addEventListener('click', function(e) {{
            var rect = progressContainer.getBoundingClientRect();
            if (audio.duration) audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
        }});

        volumeSlider.addEventListener('click', function(e) {{
            var rect = volumeSlider.getBoundingClientRect();
            var percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            audio.volume = percent;
            volumeBar.style.width = (percent * 100) + '%';
            updateVolumeIcon(percent);
        }});

        volumeBtn.addEventListener('click', function() {{
            if (audio.volume > 0) {{
                audio.dataset.prevVolume = audio.volume;
                audio.volume = 0;
                volumeBar.style.width = '0%';
                volumeIcon.innerHTML = volumeMuteIcon;
            }} else {{
                var prevVol = parseFloat(audio.dataset.prevVolume || '1');
                audio.volume = prevVol;
                volumeBar.style.width = (prevVol * 100) + '%';
                updateVolumeIcon(prevVol);
            }}
        }});

        downloadBtn.addEventListener('click', function(e) {{
            e.preventDefault();
            function save(url) {{
                var a = document.createElement('a');
                a.href = url; a.download = 'minimax-speech.mp3';
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
            }}
            if (blobUrl && blobUrl.indexOf('blob:') === 0) {{ save(blobUrl); return; }}
            authFetch(AUDIO_URL).then(function(resp) {{
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.blob();
            }}).then(function(blob) {{
                var dlUrl = URL.createObjectURL(blob);
                save(dlUrl);
                setTimeout(function() {{ URL.revokeObjectURL(dlUrl); }}, 1000);
            }}).catch(function() {{ window.open(AUDIO_URL, '_blank'); }});
        }});

        function updateVolumeIcon(volume) {{
            if (volume === 0) volumeIcon.innerHTML = volumeMuteIcon;
            else if (volume < 0.5) volumeIcon.innerHTML = volumeLowIcon;
            else volumeIcon.innerHTML = volumeHighIcon;
        }}

        function formatTime(seconds) {{
            if (!seconds || isNaN(seconds)) return '0:00';
            var mins = Math.floor(seconds / 60);
            var secs = Math.floor(seconds % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        }}
    }})();
    </script>
</body>
</html>"""


class Tools:
    class Valves(BaseModel):
        MINIMAX_API_KEY: str = Field(
            default="",
            description="MiniMax API key for the selected region.",
            json_schema_extra={"input": {"type": "password"}},
        )
        REGION: Literal["global_en", "cn_zh"] = Field(
            default="global_en", description="MiniMax account region."
        )
        MODEL: str = Field(default=DEFAULT_MODEL, description="MiniMax speech model ID.")
        VOICE_ID: str = Field(
            default="English_expressive_narrator",
            description="MiniMax voice ID to use for speech synthesis.",
        )
        SPEED: float = Field(default=1.0, ge=0.5, le=2.0)
        TIMEOUT_SECONDS: float = Field(default=180.0, ge=1.0)

    class UserValves(BaseModel):
        VOICE_ID: str = Field(
            default="",
            description="Personal MiniMax voice ID. Leave empty to use the admin default.",
        )
        SPEED: Optional[float] = Field(
            default=None,
            ge=0.5,
            le=2.0,
            description="Personal speech speed (0.5-2.0). Leave empty to use the admin default.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def _status(self, emitter: EventEmitter, description: str, done: bool) -> None:
        if emitter:
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    @staticmethod
    def _user_valve(user: Optional[dict], name: str) -> Any:
        valves = (user or {}).get("valves")
        if isinstance(valves, dict):
            return valves.get(name)
        return getattr(valves, name, None)

    async def _save_audio(
        self,
        audio: bytes,
        request: Optional[Request],
        user: Optional[dict],
        metadata: Optional[dict],
    ) -> str:
        """Store the MP3 in Open WebUI and link it to the chat message."""
        if request is None or not (user or {}).get("id"):
            raise StorageError("Open WebUI request context is unavailable.")
        user_model = await Users.get_user_by_id(user["id"])
        if user_model is None:
            raise StorageError("The requesting user could not be resolved.")

        metadata = metadata or {}
        file_metadata = {
            key: metadata[key]
            for key in ("chat_id", "message_id", "session_id")
            if metadata.get(key)
        }
        upload = UploadFile(
            file=io.BytesIO(audio),
            filename=f"minimax_tts_{uuid.uuid4().hex[:8]}.mp3",
            headers={"content-type": "audio/mpeg"},
        )
        try:
            file_item = await upload_file_handler(
                request,
                file=upload,
                metadata=file_metadata,
                process=False,
                user=user_model,
            )
        except HTTPException as exc:
            raise StorageError(
                f"Open WebUI rejected the audio file (HTTP {exc.status_code})."
            ) from None
        if not file_item or not getattr(file_item, "id", None):
            raise StorageError("Open WebUI did not return a file ID.")

        # Link the file to the chat message so it is listed with the chat and
        # readable from shared chats (same as Open WebUI's own image generation).
        chat_id = file_metadata.get("chat_id")
        message_id = file_metadata.get("message_id")
        if chat_id and message_id and not chat_id.startswith(NON_SAVED_CHAT_ID_PREFIXES):
            try:
                await Chats.insert_chat_files(
                    chat_id=chat_id,
                    message_id=message_id,
                    file_ids=[file_item.id],
                    user_id=user_model.id,
                )
            except Exception as exc:
                log.warning("MiniMax TTS: could not link audio to chat (%s)", type(exc).__name__)

        return f"/api/v1/files/{file_item.id}/content"

    async def generate_speech(
        self,
        text: str,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: EventEmitter = None,
    ) -> Union[str, Tuple[HTMLResponse, dict]]:
        """Convert text into speech and display a playable audio clip.

        :param text: The text to speak, containing fewer than 10,000 characters.
        """
        try:
            if not self.valves.MINIMAX_API_KEY.strip():
                raise ValueError("MiniMax API key is not configured.")
            if not text.strip() or len(text) >= 10000:
                raise ValueError("Text must contain 1 to 9,999 characters.")
            model = self.valves.MODEL.strip()
            voice_id = (
                str(self._user_valve(__user__, "VOICE_ID") or "").strip()
                or self.valves.VOICE_ID.strip()
            )
            speed = self._user_valve(__user__, "SPEED")
            if speed is None:
                speed = self.valves.SPEED
            if not model or not voice_id:
                raise ValueError("Configure a MiniMax model and voice ID.")

            await self._status(__event_emitter__, "Generating speech...", False)
            payload = {
                "model": model,
                "text": text,
                "stream": False,
                "output_format": "hex",
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": speed,
                },
                "audio_setting": {"format": "mp3"},
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS),
                headers={
                    "Authorization": f"Bearer {self.valves.MINIMAX_API_KEY.strip()}",
                    "Content-Type": "application/json",
                },
            ) as session:
                async with session.post(
                    SPEECH_ENDPOINTS[self.valves.REGION],
                    json=payload,
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"MiniMax speech request failed (HTTP {response.status})."
                        )
                    try:
                        result = await response.json()
                    except (ValueError, aiohttp.ContentTypeError):
                        raise ValueError("MiniMax returned an invalid JSON response.") from None

            if not isinstance(result, dict):
                raise ValueError("MiniMax returned an invalid response.")
            base_resp = result.get("base_resp")
            if not isinstance(base_resp, dict) or base_resp.get("status_code") != 0:
                code = base_resp.get("status_code") if isinstance(base_resp, dict) else None
                # Only the numeric code is surfaced; provider status_msg is never echoed.
                code_text = (
                    f" (code {code})"
                    if isinstance(code, int) and not isinstance(code, bool)
                    else ""
                )
                raise ValueError(
                    f"MiniMax rejected the speech request{code_text}; check your configuration."
                )
            data = result.get("data")
            if not isinstance(data, dict) or data.get("status") != 2:
                raise ValueError("MiniMax did not return completed audio.")
            audio_hex = data.get("audio")
            if not isinstance(audio_hex, str) or not audio_hex:
                raise ValueError("MiniMax returned no audio.")
            try:
                audio = bytes.fromhex(audio_hex)
            except ValueError:
                raise ValueError("MiniMax returned invalid audio encoding.") from None
            if not audio:
                raise ValueError("MiniMax returned no audio.")

            await self._status(__event_emitter__, "Saving audio to storage...", False)
            audio_url = await self._save_audio(audio, __request__, __user__, __metadata__)
            player_html = generate_audio_player_embed(audio_url, text, voice_id, model)
            await self._status(__event_emitter__, "Speech generation complete.", True)
            return (
                HTMLResponse(content=player_html, headers={"Content-Disposition": "inline"}),
                {
                    "message": "Speech generated successfully and displayed in the audio player.",
                    "audio_url": audio_url,
                    "voice_id": voice_id,
                    "model": model,
                },
            )
        except asyncio.TimeoutError:
            error = "MiniMax speech request timed out. Try a shorter text."
        except aiohttp.ClientError:
            error = "MiniMax speech request failed. Check the connection and API configuration."
        except ValueError as exc:
            error = str(exc)
        except StorageError as exc:
            error = f"Speech was generated but could not be saved to Open WebUI storage: {exc}"
        except Exception as exc:
            log.error("MiniMax TTS failed with an unexpected %s", type(exc).__name__)
            error = "MiniMax speech generation failed unexpectedly."
        await self._status(__event_emitter__, error, True)
        return error
