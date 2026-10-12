"""
title: MiniMax Speech Generator
description: Generate speech with MiniMax and return a playable MP3 audio embed.
author: octo-patch
author_url: https://github.com/octo-patch
version: 1.0.0
required_open_webui_version: 0.11.0
"""

import asyncio
import base64
from typing import Any, Awaitable, Callable, Literal, Optional, Tuple, Union

import aiohttp
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

DEFAULT_MODEL = "speech-2.8-hd"
SPEECH_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/t2a_v2",
    "cn_zh": "https://api.minimaxi.com/v1/t2a_v2"
}

EventEmitter = Optional[Callable[[dict[str, Any]], Awaitable[None]]]


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

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def _status(self, emitter: EventEmitter, description: str, done: bool) -> None:
        if emitter:
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def generate_speech(
        self,
        text: str,
        __event_emitter__: EventEmitter = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """Convert text into speech and display a playable audio clip.

        :param text: The text to speak, containing fewer than 10,000 characters.
        """
        try:
            if not self.valves.MINIMAX_API_KEY.strip():
                raise ValueError("MiniMax API key is not configured.")
            if not text.strip() or len(text) >= 10000:
                raise ValueError("Text must contain 1 to 9,999 characters.")
            if not self.valves.MODEL.strip() or not self.valves.VOICE_ID.strip():
                raise ValueError("Configure a MiniMax model and voice ID.")

            await self._status(__event_emitter__, "Generating speech...", False)
            payload = {
                "model": self.valves.MODEL.strip(),
                "text": text,
                "stream": False,
                "output_format": "hex",
                "voice_setting": {
                    "voice_id": self.valves.VOICE_ID.strip(),
                    "speed": self.valves.SPEED,
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
                raise ValueError(
                    "MiniMax rejected the speech request; check your configuration."
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

            audio_url = "data:audio/mpeg;base64," + base64.b64encode(audio).decode("ascii")
            html = (
                '<audio controls preload="metadata" src="' + audio_url + '">'
                "Your browser does not support audio playback.</audio>"
                '<p><a download="minimax-speech.mp3" href="' + audio_url + '">'
                "Download speech</a></p>"
            )
            await self._status(__event_emitter__, "Speech generation complete.", True)
            return (
                HTMLResponse(content=html, headers={"Content-Disposition": "inline"}),
                "Speech generated successfully and displayed in the audio player.",
            )
        except asyncio.TimeoutError:
            error = "MiniMax speech request timed out. Try a shorter text."
        except aiohttp.ClientError:
            error = "MiniMax speech request failed. Check the connection and API configuration."
        except ValueError as exc:
            error = str(exc)
        await self._status(__event_emitter__, error, True)
        return error
