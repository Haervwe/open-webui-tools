"""Offline tests for the MuAPI media tool."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from muapi_media_tool import (  # noqa: E402
    DEFAULT_API_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    DEFAULT_VIDEO_MODEL,
    Tools,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, json=None):
        self.requests.append(("POST", url, json))
        return next(self.responses)

    def get(self, url):
        self.requests.append(("GET", url, None))
        return next(self.responses)


def test_default_valves_use_current_muapi_endpoints():
    tool = Tools()

    assert tool.valves.API_BASE_URL == DEFAULT_API_BASE_URL
    assert tool.valves.IMAGE_MODEL == DEFAULT_IMAGE_MODEL
    assert tool.valves.VIDEO_MODEL == DEFAULT_VIDEO_MODEL
    assert tool.valves.IMAGE_TO_VIDEO_MODEL == DEFAULT_IMAGE_TO_VIDEO_MODEL


def test_user_valves_override_key_and_models():
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "admin-key"
    user_valves = tool.UserValves(
        MUAPI_API_KEY="user-key",
        IMAGE_MODEL="custom-image-model",
    )

    config = tool._resolve_config({"valves": user_valves})

    assert config["api_key"] == "user-key"
    assert config["image_model"] == "custom-image-model"
    assert config["video_model"] == DEFAULT_VIDEO_MODEL


@pytest.mark.asyncio
async def test_generate_image_polls_nested_muapi_result_shape():
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"id": "image-request-1", "status": "queued"}),
            FakeResponse({"id": "image-request-1", "status": "processing"}),
            FakeResponse(
                {
                    "id": "image-request-1",
                    "status": "completed",
                    "output": {
                        "outputs": ["https://cdn.example.test/image.png"]
                    },
                }
            ),
        ]
    )

    with patch(
        "muapi_media_tool.aiohttp.ClientSession", return_value=session
    ) as client_session, patch(
        "muapi_media_tool.asyncio.sleep", new=AsyncMock()
    ):
        result = await tool.generate_image(
            "a lighthouse",
            aspect_ratio="16:9",
        )

    assert "![Generated image](https://cdn.example.test/image.png)" in result
    assert session.requests[0] == (
        "POST",
        "https://api.muapi.ai/api/v1/nano-banana",
        {"prompt": "a lighthouse", "aspect_ratio": "16:9"},
    )
    assert session.requests[1][1].endswith(
        "/predictions/image-request-1/result"
    )
    assert client_session.call_args.kwargs["headers"] == {
        "x-api-key": "test-key",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_generate_video_submits_supported_options_and_embeds_result():
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"request_id": "video-request-1"}),
            FakeResponse(
                {
                    "status": "succeeded",
                    "output": {"outputs": ["https://cdn.example.test/video.mp4"]},
                }
            ),
        ]
    )

    with patch("muapi_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video(
            "a city at night",
            duration=8,
            aspect_ratio="16:9",
            seed=42,
            high_bitrate=True,
        )

    assert isinstance(result, tuple) and len(result) == 2
    html_response, context = result
    assert "<video" in html_response.body.decode("utf-8")
    assert "https://cdn.example.test/video.mp4" in context
    assert session.requests[0] == (
        "POST",
        "https://api.muapi.ai/api/v1/seedance-2.5-text-to-video",
        {
            "prompt": "a city at night",
            "duration": 8,
            "aspect_ratio": "16:9",
            "seed": 42,
            "high_bitrate": True,
        },
    )


@pytest.mark.asyncio
async def test_generate_video_from_image_uses_attached_http_url():
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"id": "i2v-request-1"}),
            FakeResponse(
                {
                    "status": "completed",
                    "output": {"outputs": ["https://cdn.example.test/i2v.mp4"]},
                }
            ),
        ]
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "animate this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example.test/input.png"},
                },
            ],
        }
    ]

    with patch("muapi_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video_from_image(
            "make the ocean waves move",
            duration=5,
            __messages__=messages,
        )

    assert isinstance(result, tuple)
    assert "https://cdn.example.test/i2v.mp4" in result[1]
    assert session.requests[0] == (
        "POST",
        "https://api.muapi.ai/api/v1/seedance-2.5-image-to-video",
        {
            "prompt": "make the ocean waves move",
            "image_url": "https://cdn.example.test/input.png",
            "duration": 5,
        },
    )


@pytest.mark.asyncio
async def test_missing_key_and_invalid_image_are_reported_without_requests():
    tool = Tools()

    missing_key = await tool.generate_image("a lighthouse")
    assert "MUAPI_API_KEY" in missing_key

    tool.valves.MUAPI_API_KEY = "test-key"
    invalid_image = await tool.generate_video_from_image(
        "animate this", image_url="data:image/png;base64,abc"
    )
    assert "absolute HTTP or HTTPS URL" in invalid_image
