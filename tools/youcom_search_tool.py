"""
title: You.com Search Tool
description: Web search via the You.com Search API with snippets, freshness filtering and citation events.
author: mouse-value-add
author_url: https://github.com/mouse-value-add
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.0
license: MIT
"""

from typing import Any, Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

YOUCOM_API_URL = "https://ydc-index.io/v1/search"


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


async def emit_citation(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    document: str,
    source_url: str,
    source_name: str,
) -> None:
    """Helper to emit citation events."""
    if event_emitter:
        await event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [document],
                    "metadata": [{"source": source_url}],
                    "source": {"name": source_name, "url": source_url},
                },
            }
        )


def _format_results(results: list[dict]) -> str:
    """Format web results as a markdown list."""
    lines = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        link = (r.get("url") or "").strip()
        snippet = (r.get("description") or "").strip()
        if not snippet:
            snippets = r.get("snippets") or []
            if snippets:
                snippet = str(snippets[0]).strip()
        if title:
            lines.append(f"{i}. **{title}**")
        else:
            lines.append(f"{i}. {link}")
        if link:
            lines.append(f"   {link}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _format_news(results: list[dict]) -> str:
    """Format news results as a markdown list."""
    lines = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        link = (r.get("url") or "").strip()
        snippet = (r.get("description") or "").strip()
        age = (r.get("page_age") or "").strip()
        header = title or link
        if age:
            header = f"{header} ({age})"
        lines.append(f"{i}. **{header}**")
        if link:
            lines.append(f"   {link}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


class Tools:
    class Valves(BaseModel):
        YDC_API_KEY: str = Field(
            default="",
            description="You.com API key. Get one at https://you.com/platform/api-keys",
            json_schema_extra={"input": {"type": "password"}},
        )
        MAX_RESULTS: int = Field(
            default=10,
            description="Maximum number of web results to return (1-100)",
            ge=1,
            le=100,
        )
        FRESHNESS: str = Field(
            default="",
            description="Optional freshness filter: day, week, month, year, or YYYY-MM-DDtoYYYY-MM-DD",
        )
        COUNTRY: str = Field(
            default="",
            description="Optional country code for results (ISO 3166-1 alpha-2, e.g. US, DE, JP)",
        )
        LANGUAGE: str = Field(
            default="",
            description="Optional language code for results (BCP 47, e.g. EN, DE, JA)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=30,
            description="Request timeout in seconds",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    async def web_search(
        self,
        query: str,
        num_results: Optional[int] = None,
        freshness: Optional[str] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search the web with the You.com Search API and return results in markdown.

        Args:
            query: The search query (e.g. "openai realtime api pricing")
            num_results: Number of web results to return (overrides the MAX_RESULTS valve, 1-100)
            freshness: Optional freshness filter (overrides the FRESHNESS valve): day, week, month, year, or YYYY-MM-DDtoYYYY-MM-DD

        Returns:
            Markdown-formatted search results with links, snippets and sources
        """
        api_key = self.valves.YDC_API_KEY
        if not api_key:
            await emit_status(
                __event_emitter__,
                "You.com API key not set - add YDC_API_KEY in the tool valves",
                done=True,
            )
            return (
                "You.com API key is not set. Add your key in the tool's valves "
                "(YDC_API_KEY). You can get one at https://you.com/platform/api-keys."
            )

        count = num_results or self.valves.MAX_RESULTS
        fresh = freshness or self.valves.FRESHNESS

        await emit_status(__event_emitter__, f"Searching You.com for: {query}")

        payload = {"query": query, "count": count}
        if fresh:
            payload["freshness"] = fresh
        if self.valves.COUNTRY:
            payload["country"] = self.valves.COUNTRY
        if self.valves.LANGUAGE:
            payload["language"] = self.valves.LANGUAGE

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "User-Agent": "open-webui-tools/1.0",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    YOUCOM_API_URL, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        error_msg = (
                            f"You.com API returned HTTP {response.status}. "
                            "Check your API key and quota at https://you.com/platform/api-keys."
                        )
                        logger.error(error_msg)
                        await emit_status(__event_emitter__, error_msg, done=True)
                        return error_msg
                    data = await response.json()

            results = (data.get("results") or {}).get("web") or []
            news = (data.get("results") or {}).get("news") or []

            if not results and not news:
                await emit_status(
                    __event_emitter__, "No results found for the query", done=True
                )
                return "No results found for the given query."

            parts = []

            if results:
                parts.append(f"## Search Results\n\n{_format_results(results[:count])}")
            if news:
                parts.append(f"## News\n\n{_format_news(news[:count])}")

            result = "\n\n".join(parts)

            # Emit citations for the web results
            if __event_emitter__:
                for r in results[:count]:
                    title = (r.get("title") or "").strip()
                    link = (r.get("url") or "").strip()
                    snippet = (r.get("description") or "").strip()
                    if link:
                        await emit_citation(
                            __event_emitter__,
                            snippet or title or link,
                            link,
                            title or link,
                        )

            await emit_status(
                __event_emitter__, "Search completed successfully", done=True
            )
            return result

        except aiohttp.ClientError as e:
            error_msg = f"You.com request failed: {e}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
        except Exception as e:
            error_msg = f"Error performing search: {e}"
            logger.error(error_msg, exc_info=True)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
