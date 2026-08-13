"""Unit tests for the Atlas Cloud media tool."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from atlascloud_media_tool import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_EDIT_MODEL,
    DEFAULT_VIDEO_MODEL,
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    DEFAULT_AUDIO_TO_VIDEO_MODEL,
    AtlasCloudError,
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

    def post(self, url, json=None, data=None):
        self.requests.append(("POST", url, json or data))
        return next(self.responses)

    def get(self, url):
        self.requests.append(("GET", url, None))
        return next(self.responses)


def test_default_valves_use_current_atlas_models():
    tool = Tools()

    assert tool.valves.API_BASE_URL == "https://api.atlascloud.ai/api/v1"
    assert tool.valves.IMAGE_MODEL == DEFAULT_IMAGE_MODEL
    assert tool.valves.IMAGE_EDIT_MODEL == DEFAULT_IMAGE_EDIT_MODEL
    assert tool.valves.VIDEO_MODEL == DEFAULT_VIDEO_MODEL
    assert tool.valves.IMAGE_TO_VIDEO_MODEL == DEFAULT_IMAGE_TO_VIDEO_MODEL
    assert tool.valves.AUDIO_TO_VIDEO_MODEL == DEFAULT_AUDIO_TO_VIDEO_MODEL


def test_user_valves_override():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "admin-key"
    tool.valves.IMAGE_MODEL = "admin-image-model"

    user_valves = tool.UserValves(
        ATLASCLOUD_API_KEY="user-key",
        IMAGE_MODEL="user-image-model",
    )
    user_dict = {"valves": user_valves}

    config = tool._resolve_config(user_dict)
    assert config["api_key"] == "user-key"
    assert config["image_model"] == "user-image-model"
    assert config["video_model"] == DEFAULT_VIDEO_MODEL


@pytest.mark.asyncio
async def test_missing_api_key_is_reported():
    tool = Tools()
    emitter = AsyncMock()

    result = await tool.generate_image("a lighthouse", __event_emitter__=emitter)

    assert "API key is not configured" in result
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True


@pytest.mark.asyncio
async def test_generate_image_submits_polls_and_formats_outputs():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse(
                {"code": 200, "data": {"id": "prediction-1", "status": "starting"}}
            ),
            FakeResponse({"code": 200, "data": {"status": "processing"}}),
            FakeResponse(
                {
                    "code": 200,
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/image.jpg"],
                    },
                }
            ),
        ]
    )

    with patch(
        "atlascloud_media_tool.aiohttp.ClientSession", return_value=session
    ), patch("atlascloud_media_tool.asyncio.sleep", new=AsyncMock()):
        result = await tool.generate_image("a lighthouse")

    assert "![Generated image](https://cdn.example.test/image.jpg)" in result
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateImage",
        {
            "model": DEFAULT_IMAGE_MODEL,
            "prompt": "a lighthouse",
            "size": "2048*2048",
            "output_format": "jpeg",
        },
    )
    assert session.requests[1][1].endswith("/model/prediction/prediction-1")


@pytest.mark.asyncio
async def test_generate_video_submits_expected_options_and_returns_html_embed():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-video"}}),
            FakeResponse(
                {
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/video.mp4"],
                    }
                }
            ),
        ]
    )

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video(
            "a city at night",
            duration=8,
            resolution="1080p",
            ratio="16:9",
            generate_audio=False,
        )

    # With RETURN_HTML_EMBED=True (default), result is (HTMLResponse, context_str)
    assert isinstance(result, tuple) and len(result) == 2
    html_response, context = result
    html_body = html_response.body.decode("utf-8")
    assert "<video" in html_body
    assert "https://cdn.example.test/video.mp4" in html_body
    assert "https://cdn.example.test/video.mp4" in context
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateVideo",
        {
            "model": DEFAULT_VIDEO_MODEL,
            "prompt": "a city at night",
            "duration": 8,
            "resolution": "1080p",
            "ratio": "16:9",
            "generate_audio": False,
        },
    )


@pytest.mark.asyncio
async def test_generate_video_from_image_submits_correct_payload():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-i2v"}}),
            FakeResponse(
                {
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/i2v.mp4"],
                    }
                }
            ),
        ]
    )

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video_from_image(
            "make the ocean waves move",
            image_url="https://cdn.example.test/source.jpg",
            duration=5,
        )

    assert isinstance(result, tuple)
    html_response, context = result
    assert "https://cdn.example.test/i2v.mp4" in context
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateVideo",
        {
            "model": DEFAULT_IMAGE_TO_VIDEO_MODEL,
            "prompt": "make the ocean waves move",
            "image_url": "https://cdn.example.test/source.jpg",
            "duration": 5,
            "resolution": "720p",
            "ratio": "adaptive",
            "generate_audio": True,
        },
    )


@pytest.mark.asyncio
async def test_edit_image_with_attached_message():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-edit"}}),
            FakeResponse(
                {
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/edited.jpg"],
                    }
                }
            ),
        ]
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "make it sunset"},
                {"type": "image_url", "image_url": {"url": "https://cdn.example.test/input.jpg"}},
            ],
        }
    ]

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.edit_image(
            "make it sunset",
            __messages__=messages,
        )

    assert "![Edited image](https://cdn.example.test/edited.jpg)" in result
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateImage",
        {
            "model": DEFAULT_IMAGE_EDIT_MODEL,
            "prompt": "make it sunset",
            "image_url": "https://cdn.example.test/input.jpg",
            "size": "2048*2048",
            "output_format": "jpeg",
        },
    )


@pytest.mark.asyncio
async def test_failed_prediction_is_reported():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-2"}}),
            FakeResponse({"data": {"status": "failed", "error": "model unavailable"}}),
        ]
    )

    with patch(
        "atlascloud_media_tool.aiohttp.ClientSession", return_value=session
    ), patch("atlascloud_media_tool.asyncio.sleep", new=AsyncMock()):
        result = await tool.generate_image("a lighthouse")

    assert "Atlas Cloud image generation failed: Atlas Cloud generation failed: model unavailable" in result
