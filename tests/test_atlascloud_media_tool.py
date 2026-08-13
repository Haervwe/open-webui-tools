"""Unit tests for the Atlas Cloud media tool."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from atlascloud_media_tool import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
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

    def post(self, url, json):
        self.requests.append(("POST", url, json))
        return next(self.responses)

    def get(self, url):
        self.requests.append(("GET", url, None))
        return next(self.responses)


def test_default_valves_use_current_atlas_models():
    tool = Tools()

    assert tool.valves.API_BASE_URL == "https://api.atlascloud.ai/api/v1"
    assert tool.valves.IMAGE_MODEL == DEFAULT_IMAGE_MODEL
    assert tool.valves.VIDEO_MODEL == DEFAULT_VIDEO_MODEL


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
async def test_generate_video_submits_expected_options_and_formats_output():
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

    assert "[Download generated video](https://cdn.example.test/video.mp4)" in result
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
async def test_failed_prediction_is_reported():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-2"}}),
            FakeResponse({"data": {"status": "failed", "error": "model unavailable"}}),
        ]
    )

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video("a city at night")

    assert "model unavailable" in result


def test_invalid_data_payload_raises_clear_error():
    with pytest.raises(AtlasCloudError, match="invalid response payload"):
        Tools._data({"data": []})


def test_api_error_code_raises_clear_error():
    with pytest.raises(AtlasCloudError, match="insufficient balance"):
        Tools._data({"code": 402, "message": "insufficient balance"})


@pytest.mark.asyncio
async def test_non_object_json_response_raises_clear_error():
    with pytest.raises(AtlasCloudError, match="invalid JSON payload"):
        await Tools._json_response(FakeResponse([]))
