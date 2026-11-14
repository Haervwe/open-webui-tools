"""
title: SpecStream Web Search Tool
description: Web search via the SpecStream Search API: results with titles, links, snippets and publish dates, returned as markdown with citations.
author: specstream
author_url: https://specstream.ai
funding_url: https://github.com/Haervwe/open-webui-tools
requirements: aiohttp
version: 1.0.0
required_open_webui_version: 0.8.0
license: MIT
"""

import re
import aiohttp
from typing import Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

SPECSTREAM_API_URL = "https://api.specstream.ai/v1/web-search"
MAX_SNIPPET_CHARS = 500


def _clean(text: Any, limit: Optional[int] = None) -> str:
    """Flatten untrusted result text for markdown: drop newlines and markdown/link syntax chars."""
    text = re.sub(r"\s+", " ", str(text or ""))
    text = re.sub(r"[*`\[\]<>]", "", text).strip()
    if limit and len(text) > limit:
        text = text[: limit - 1].rstrip() + "..."
    return text


def _safe_link(value: Any) -> str:
    """Return an http(s) URL safe to place on its own markdown line, or "" to drop it."""
    link = str(value or "").strip()
    if not link.lower().startswith(("http://", "https://")):
        return ""
    # Whitespace would let a crafted link start new markdown lines (headings, injected
    # instructions); angle brackets / backticks could open HTML or code spans.
    if re.search(r"[\s<>`]", link):
        return ""
    return link


def _published(value: Any) -> str:
    """Return the YYYY-MM-DD part of a publish date, or "" when there is none."""
    text = str(value or "").strip()
    return text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else ""


def _http_error_message(status: int) -> str:
    if status == 401:
        return (
            f"SpecStream API returned HTTP {status}: the API key was rejected. "
            "Check SPECSTREAM_API_KEY in the tool valves."
        )
    if status in (402, 403):
        return (
            f"SpecStream API returned HTTP {status}: the balance or the key's spending limit is used up. "
            "Check your account at https://specstream.ai."
        )
    if status == 503:
        return f"SpecStream API returned HTTP {status}: search is busy or rate limited. Wait and retry."
    if status >= 500:
        return f"SpecStream API returned HTTP {status}: the SpecStream service is unavailable. Try again later."
    return f"SpecStream API returned HTTP {status}."


async def emit_status(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    description: str,
    done: bool = False,
) -> None:
    """Helper to emit status events."""
    if event_emitter:
        await event_emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )


class Tools:
    class Valves(BaseModel):
        SPECSTREAM_API_KEY: str = Field(
            default="",
            description="SpecStream API key. Create one at https://specstream.ai/signup?src=open-webui-tools",
            json_schema_extra={"input": {"type": "password"}},
        )
        MAX_RESULTS: int = Field(
            default=10,
            description="Maximum number of results to return (1-10)",
            ge=1,
            le=10,
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    async def web_search(
        self,
        query: str,
        num_results: Optional[int] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search the web via the SpecStream Search API and return the top results in markdown.

        Args:
            query: The search query (e.g. "open webui release notes")
            num_results: Number of results to return (overrides the MAX_RESULTS valve, 1-10)

        Returns:
            Markdown-formatted search results with links, snippets and publish dates
        """
        api_key = self.valves.SPECSTREAM_API_KEY
        if not api_key:
            await emit_status(
                __event_emitter__,
                "SpecStream API key not set: add SPECSTREAM_API_KEY in the tool valves",
                done=True,
            )
            return (
                "SpecStream API key is not set. Add your key in the tool's valves "
                "(SPECSTREAM_API_KEY). You can create one at https://specstream.ai/signup?src=open-webui-tools."
            )

        try:
            num = int(num_results) if num_results is not None else self.valves.MAX_RESULTS
        except (TypeError, ValueError):
            num = self.valves.MAX_RESULTS
        num = max(1, min(num, 10))

        await emit_status(__event_emitter__, f"Searching the web for: {query}")

        params = {"q": query, "num": str(num)}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "open-webui-tools/1.0",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    SPECSTREAM_API_URL, params=params, headers=headers
                ) as response:
                    if response.status != 200:
                        error_msg = _http_error_message(response.status)
                        logger.error(error_msg)
                        try:
                            logger.debug("SpecStream error body: %s", (await response.text())[:200])
                        except Exception:
                            pass
                        await emit_status(__event_emitter__, error_msg, done=True)
                        return error_msg
                    data = await response.json()

            results = ((data.get("data") or {}).get("sources") or [])[:num]
            if not results:
                await emit_status(
                    __event_emitter__, "No results found for the query", done=True
                )
                return "No results found for the given query."

            lines = []
            for i, r in enumerate(results, start=1):
                title = _clean(r.get("title"))
                link = _safe_link(r.get("url"))
                description = _clean(r.get("snippet"), MAX_SNIPPET_CHARS)
                published = _published(r.get("published"))
                lines.append(f"{i}. **{title or link or 'Untitled'}**")
                if link:
                    lines.append(f"   {link}")
                if published:
                    lines.append(f"   Published: {published}")
                if description:
                    lines.append(f"   {description}")
                if __event_emitter__ and link:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [description or title],
                                "metadata": [{"source": link}],
                                "source": {"name": title or link, "url": link},
                            },
                        }
                    )

            await emit_status(
                __event_emitter__,
                f"Found {len(results)} results for: {query}",
                done=True,
            )
            return "## Search Results\n\n" + "\n".join(lines)

        except aiohttp.ClientError as e:
            error_msg = f"Network error while calling SpecStream API: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
