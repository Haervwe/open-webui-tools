"""Focused tests for the You.com Search Open WebUI tool."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "youcom_search_tool.py"
SPEC = importlib.util.spec_from_file_location("youcom_search_tool", MODULE_PATH)
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

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs, "timeout": self.timeout})
        return self.response


def session_factory(response, calls):
    """Build a ClientSession-compatible factory for one response."""

    def factory(timeout=None):
        return FakeSession(response, calls, timeout=timeout)

    return factory


def sample_payload():
    return {
        "results": {
            "web": [
                {
                    "title": "Open WebUI Docs",
                    "url": "https://docs.openwebui.com/",
                    "description": "Documentation for Open WebUI.",
                    "snippets": ["Open WebUI is a self-hosted UI."],
                },
                {
                    "title": "You.com Search API",
                    "url": "https://you.com/docs/guides/search",
                    "description": "Web Search API guide.",
                },
            ],
            "news": [
                {
                    "title": "Open WebUI 1.0 released",
                    "url": "https://example.com/news/1",
                    "description": "Release coverage.",
                    "page_age": "2026-08-30T00:00:00Z",
                }
            ],
        },
        "metadata": {"query": "open webui", "latency": 0.42},
    }


@pytest.mark.asyncio
async def test_web_search_posts_to_youcom_api_and_formats_results():
    calls = []
    response = FakeResponse(payload=sample_payload())
    tool = MODULE.Tools()
    tool.valves.YDC_API_KEY = "test-key"

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.web_search("open webui")

    assert len(calls) == 1
    assert calls[0]["url"] == MODULE.YOUCOM_API_URL
    assert calls[0]["json"] == {"query": "open webui", "count": 10}
    assert calls[0]["headers"]["X-API-Key"] == "test-key"
    assert calls[0]["timeout"].total == tool.valves.TIMEOUT_SECONDS

    assert "## Search Results" in result
    assert "**Open WebUI Docs**" in result
    assert "https://docs.openwebui.com/" in result
    assert "## News" in result
    assert "**Open WebUI 1.0 released (2026-08-30T00:00:00Z)**" in result


@pytest.mark.asyncio
async def test_web_search_sends_optional_filters_and_overrides():
    calls = []
    response = FakeResponse(payload=sample_payload())
    tool = MODULE.Tools()
    tool.valves.YDC_API_KEY = "test-key"
    tool.valves.FRESHNESS = "week"
    tool.valves.COUNTRY = "US"
    tool.valves.LANGUAGE = "EN"

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        await tool.web_search("open webui", num_results=5, freshness="day")

    sent = calls[0]["json"]
    assert sent["count"] == 5
    assert sent["freshness"] == "day"
    assert sent["country"] == "US"
    assert sent["language"] == "EN"


@pytest.mark.asyncio
async def test_web_search_without_key_returns_setup_message_and_skips_request():
    calls = []
    tool = MODULE.Tools()

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(FakeResponse(payload={}), calls),
    ):
        result = await tool.web_search("open webui")

    assert calls == []
    assert "YDC_API_KEY" in result
    assert "https://you.com/platform/api-keys" in result


@pytest.mark.asyncio
async def test_web_search_reports_http_errors():
    calls = []
    response = FakeResponse(status=401)
    tool = MODULE.Tools()
    tool.valves.YDC_API_KEY = "bad-key"

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.web_search("open webui")

    assert "HTTP 401" in result


@pytest.mark.asyncio
async def test_web_search_reports_no_results():
    calls = []
    response = FakeResponse(payload={"results": {"web": [], "news": []}})
    tool = MODULE.Tools()
    tool.valves.YDC_API_KEY = "test-key"

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        result = await tool.web_search("empty query test")

    assert result == "No results found for the given query."


@pytest.mark.asyncio
async def test_web_search_emits_citations_for_web_results():
    calls = []
    events = []
    response = FakeResponse(payload=sample_payload())
    tool = MODULE.Tools()
    tool.valves.YDC_API_KEY = "test-key"

    async def emitter(event):
        events.append(event)

    with patch.object(
        MODULE.aiohttp,
        "ClientSession",
        session_factory(response, calls),
    ):
        await tool.web_search("open webui", __event_emitter__=emitter)

    citations = [e for e in events if e["type"] == "citation"]
    assert len(citations) == 2
    assert citations[0]["data"]["source"]["url"] == "https://docs.openwebui.com/"
    statuses = [e for e in events if e["type"] == "status"]
    assert any("Searching You.com" in s["data"]["description"] for s in statuses)
    assert statuses[-1]["data"]["done"] is True
