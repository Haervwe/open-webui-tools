"""Focused tests for the zero-key OutageDeck Open WebUI tool."""

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "outagedeck_status_tool.py"
SPEC = importlib.util.spec_from_file_location("outagedeck_status_tool", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    """Async response context manager used by the fake aiohttp session."""

    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self.payload = payload if payload is not None else {}
        self.response_text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, content_type=None):
        return self.payload

    async def text(self):
        return self.response_text


class FakeSession:
    """Capture request arguments and return a configured fake response."""

    def __init__(self, response, calls, timeout=None):
        self.response = response
        self.calls = calls
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs, "timeout": self.timeout})
        return self.response


def session_factory(response, calls):
    """Build a ClientSession-compatible factory for one response."""

    def factory(timeout=None):
        return FakeSession(response, calls, timeout=timeout)

    return factory


def parsed_json(result):
    """Extract the fenced JSON object from a formatted tool response."""
    return json.loads(result.split("```json\n", 1)[1].split("\n```", 1)[0])


@pytest.mark.asyncio
async def test_get_provider_status_uses_fixed_origin_and_campaign_params():
    calls = []
    response = FakeResponse(
        payload={
            "data": {
                "slug": "github",
                "name": "GitHub",
                "currentStatus": {"code": "operational"},
                "services": [{"slug": "github-actions", "status": "operational"}],
                "activeIncidents": [],
            }
        }
    )
    tool = MODULE.Tools()

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.get_provider_status("github")

    assert len(calls) == 1
    assert calls[0]["url"] == "https://outagedeck.com/api/v1/providers/github"
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["params"] == MODULE.CAMPAIGN_PARAMS
    assert calls[0]["headers"]["accept"] == "application/json"
    assert calls[0]["timeout"].total == 15
    assert parsed_json(result)["status"]["code"] == "operational"
    assert "utm_source=open_webui" in result


@pytest.mark.asyncio
async def test_find_providers_validates_filters_and_bounds_output():
    calls = []
    providers = [
        {
            "slug": f"provider-{index}",
            "name": f"Provider {index}",
            "currentStatus": {"code": "operational"},
        }
        for index in range(3)
    ]
    response = FakeResponse(payload={"data": {"count": 3, "providers": providers}})
    tool = MODULE.Tools()
    tool.valves.MAX_PROVIDER_RESULTS = 2

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.find_providers(
            query="provider", status="operational", category="cloud", sort="name"
        )

    assert calls[0]["params"] == {
        "q": "provider",
        "status": "operational",
        "category": "cloud",
        "sort": "name",
        **MODULE.CAMPAIGN_PARAMS,
    }
    output = parsed_json(result)
    assert output["count"] == 2
    assert output["totalMatches"] == 3

    with pytest.raises(ValueError, match="status must be one of"):
        await tool.find_providers(status="offline")


@pytest.mark.asyncio
async def test_list_incidents_sends_exact_filters_and_normalizes_results():
    calls = []
    response = FakeResponse(
        payload={
            "data": {
                "count": 1,
                "page": 2,
                "totalPages": 3,
                "totalIncidents": 5,
                "incidents": [
                    {
                        "slug": "github-api-errors",
                        "title": "API errors",
                        "status": "investigating",
                        "severity": "major",
                    }
                ],
            }
        }
    )
    tool = MODULE.Tools()

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.list_incidents(
            provider_slug="github",
            state="resolved",
            severity="major",
            page=2,
            limit=5,
        )

    assert calls[0]["params"] == {
        "provider": "github",
        "state": "resolved",
        "severity": "major",
        "page": 2,
        "limit": 5,
        **MODULE.CAMPAIGN_PARAMS,
    }
    output = parsed_json(result)
    assert output["incidents"][0]["slug"] == "github-api-errors"
    assert output["totalIncidents"] == 5


@pytest.mark.asyncio
async def test_incident_details_caps_timeline_to_latest_updates():
    calls = []
    response = FakeResponse(
        payload={
            "data": {
                "slug": "github-api-errors",
                "title": "API errors",
                "updates": [
                    {"status": "investigating", "body": "one", "createdAt": "1"},
                    {"status": "monitoring", "body": "two", "createdAt": "2"},
                    {"status": "resolved", "body": "three", "createdAt": "3"},
                ],
            }
        }
    )
    tool = MODULE.Tools()
    tool.valves.MAX_TIMELINE_UPDATES = 2

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.get_incident_details("github-api-errors")

    updates = parsed_json(result)["updates"]
    assert [update["body"] for update in updates] == ["two", "three"]


@pytest.mark.asyncio
async def test_service_status_rejects_unsafe_slug_without_network_access():
    tool = MODULE.Tools()
    with patch.object(MODULE.aiohttp, "ClientSession") as client_session:
        with pytest.raises(ValueError, match="service_slug"):
            await tool.get_service_status("https://example.com/steal")
    client_session.assert_not_called()


@pytest.mark.asyncio
async def test_http_errors_are_model_readable():
    calls = []
    response = FakeResponse(status=404, payload={"error": "not found"})
    tool = MODULE.Tools()

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.get_provider_status("missing-provider")

    assert "failed (404)" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_network_errors_are_model_readable():
    tool = MODULE.Tools()

    class BrokenSession:
        async def __aenter__(self):
            raise aiohttp.ClientConnectionError("offline")

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    with patch.object(MODULE.aiohttp, "ClientSession", return_value=BrokenSession()):
        result = await tool.get_provider_status("github")

    assert "failed (network)" in result
    assert "Could not reach OutageDeck" in result


@pytest.mark.asyncio
async def test_status_emitter_finishes_on_timeout():
    events = []
    tool = MODULE.Tools()

    async def emitter(event):
        events.append(event)

    with patch.object(tool, "_get_json", side_effect=asyncio.TimeoutError):
        with pytest.raises(asyncio.TimeoutError):
            await tool.get_provider_status("github", __event_emitter__=emitter)

    assert events[0]["data"]["done"] is False
    assert events[-1]["data"]["done"] is True
