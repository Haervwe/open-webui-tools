"""
title: Serply Web Search Tool
description: Google web search via the Serply API - organic results with titles, links and descriptions, returned as markdown with citations.
author: serply
author_url: https://serply.io
funding_url: https://github.com/Haervwe/open-webui-tools
requirements: aiohttp
version: 1.0.1
required_open_webui_version: 0.8.0
license: MIT
"""

import re
import aiohttp
from typing import Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

SERPLY_API_URL = "https://api.serply.io/v1/search"
MAX_DESCRIPTION_CHARS = 500


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


def _http_error_message(status: int) -> str:
    if status in (401, 403):
        return (
            f"Serply API returned HTTP {status}: the API key was rejected. "
            "Check SERPLY_API_KEY in the tool valves."
        )
    if status in (402, 429):
        return (
            f"Serply API returned HTTP {status}: rate limit or credit quota exceeded. "
            "Wait and retry, or check your plan at https://serply.io."
        )
    if status >= 500:
        return f"Serply API returned HTTP {status}: the Serply service is unavailable. Try again later."
    return f"Serply API returned HTTP {status}."


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
        SERPLY_API_KEY: str = Field(
            default="",
            description="Serply API key. Create one at https://serply.io",
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
        Search the web via the Serply API and return the top results in markdown.

        Args:
            query: The search query (e.g. "open webui release notes")
            num_results: Number of results to return (overrides the MAX_RESULTS valve, 1-10)

        Returns:
            Markdown-formatted search results with links and descriptions
        """
        api_key = self.valves.SERPLY_API_KEY
        if not api_key:
            await emit_status(
                __event_emitter__,
                "Serply API key not set - add SERPLY_API_KEY in the tool valves",
                done=True,
            )
            return (
                "Serply API key is not set. Add your key in the tool's valves "
                "(SERPLY_API_KEY). You can create one at https://serply.io."
            )

        try:
            num = int(num_results) if num_results is not None else self.valves.MAX_RESULTS
        except (TypeError, ValueError):
            num = self.valves.MAX_RESULTS
        num = max(1, min(num, 10))

        await emit_status(__event_emitter__, f"Searching the web for: {query}")

        params = {"q": query, "num": str(num)}
        headers = {
            "X-Api-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "open-webui-tools/1.0",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    SERPLY_API_URL, params=params, headers=headers
                ) as response:
                    if response.status != 200:
                        error_msg = _http_error_message(response.status)
                        logger.error(error_msg)
                        try:
                            logger.debug("Serply error body: %s", (await response.text())[:200])
                        except Exception:
                            pass
                        await emit_status(__event_emitter__, error_msg, done=True)
                        return error_msg
                    data = await response.json()

            results = (data.get("results") or [])[:num]
            if not results:
                await emit_status(
                    __event_emitter__, "No results found for the query", done=True
                )
                return "No results found for the given query."

            lines = []
            for i, r in enumerate(results, start=1):
                title = _clean(r.get("title"))
                link = _safe_link(r.get("link") or r.get("url"))
                description = _clean(r.get("description"), MAX_DESCRIPTION_CHARS)
                lines.append(f"{i}. **{title or link or 'Untitled'}**")
                if link:
                    lines.append(f"   {link}")
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
            error_msg = f"Network error while calling Serply API: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
