"""
title: SerpBase Google Search Tool
description: Google web search via the SerpBase API - organic results, featured snippets and AI Overviews in clean JSON. No browser, no CAPTCHAs, no self-hosted instance needed.
author: gefsikatsinelou
author_url: https://github.com/gefsikatsinelou
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.0
license: MIT
"""

import aiohttp
from typing import Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERPBASE_API_URL = "https://api.serpbase.dev/google/search"


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
                    "source": {"name": source_name},
                },
            }
        )


def _format_organic(results: list[dict]) -> str:
    """Format organic results as a markdown list."""
    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        link = r.get("link") or r.get("url") or ""
        snippet = (r.get("snippet") or "").strip()
        if title:
            lines.append(f"{i}. **{title}**")
        else:
            lines.append(f"{i}. {link}")
        if link:
            lines.append(f"   {link}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _format_featured_snippet(fs: dict) -> str:
    """Format the featured snippet answer card."""
    answer = (fs.get("answer") or fs.get("snippet") or "").strip()
    if not answer:
        return ""
    source = fs.get("source") or {}
    title = (source.get("title") or "").strip()
    link = source.get("link") or ""
    out = f"> {answer}"
    if title and link:
        out += f"\n> — [{title}]({link})"
    return out


class Tools:
    class Valves(BaseModel):
        SERPBASE_API_KEY: str = Field(
            default="",
            description="SerpBase API key. Get 100 free searches at https://serpbase.dev (no credit card required)",
            json_schema_extra={"input": {"type": "password"}},
        )
        MAX_RESULTS: int = Field(
            default=10,
            description="Maximum number of organic results to return (1-20)",
            ge=1,
            le=20,
        )
        LANGUAGE: str = Field(
            default="en",
            description="Google interface language (hl), e.g. en, de, es, fr, ja",
        )
        COUNTRY: str = Field(
            default="us",
            description="Google country region (gl), e.g. us, de, uk, jp",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    async def google_search(
        self,
        query: str,
        num_results: Optional[int] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Google via the SerpBase API and return organic results, featured snippets and AI Overviews in markdown.

        Args:
            query: The search query (e.g. "openai realtime api pricing")
            num_results: Number of organic results to return (overrides the MAX_RESULTS valve, 1-20)
            language: Google interface language code (overrides the LANGUAGE valve, e.g. "en")
            country: Google country region code (overrides the COUNTRY valve, e.g. "us")

        Returns:
            Markdown-formatted search results with links and snippets
        """
        api_key = self.valves.SERPBASE_API_KEY
        if not api_key:
            await emit_status(
                __event_emitter__,
                "SerpBase API key not set - add SERPBASE_API_KEY in the tool valves",
                done=True,
            )
            return (
                "SerpBase API key is not set. Add your key in the tool's valves "
                "(SERPBASE_API_KEY). You can get 100 free searches at https://serpbase.dev."
            )

        num = num_results or self.valves.MAX_RESULTS
        hl = language or self.valves.LANGUAGE
        gl = country or self.valves.COUNTRY

        await emit_status(__event_emitter__, f"Searching Google for: {query}")

        payload = {"q": query, "hl": hl, "gl": gl}
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "User-Agent": "open-webui-tools/1.0",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    SERPBASE_API_URL, json=payload, headers=headers, timeout=30
                ) as response:
                    if response.status != 200:
                        error_msg = (
                            f"SerpBase API returned HTTP {response.status}. "
                            "Check your API key and quota at https://serpbase.dev/api-keys."
                        )
                        logger.error(error_msg)
                        await emit_status(__event_emitter__, error_msg, done=True)
                        return error_msg
                    data = await response.json()

            # SerpBase returns HTTP 200 with a business status code even for failures
            if data.get("status") != 0:
                error_msg = (
                    f"SerpBase API error: {data.get('error', 'unknown error')} "
                    f"(status {data.get('status')}). Check your key and quota at "
                    "https://serpbase.dev/dashboard/api-keys."
                )
                logger.error(error_msg)
                await emit_status(__event_emitter__, error_msg, done=True)
                return error_msg

            organic = data.get("organic") or []
            featured = data.get("featured_snippet") or {}
            ai_overview = data.get("ai_overview") or {}

            if not organic and not featured and not ai_overview:
                await emit_status(
                    __event_emitter__, "No results found for the query", done=True
                )
                return "No results found for the given query."

            parts = []

            if ai_overview:
                overview_text = (
                    ai_overview.get("content")
                    or ai_overview.get("answer")
                    or ai_overview.get("text")
                    or ""
                ).strip()
                if overview_text:
                    parts.append(f"## AI Overview\n\n{overview_text}")

            if featured:
                fs_text = _format_featured_snippet(featured)
                if fs_text:
                    parts.append(f"## Featured Snippet\n\n{fs_text}")

            if organic:
                parts.append(f"## Search Results\n\n{_format_organic(organic[:num])}")

            result = "\n\n".join(parts)

            # Emit citations for the organic results
            if __event_emitter__:
                doc_parts = []
                for r in organic[:num]:
                    title = r.get("title", "").strip()
                    link = r.get("link") or r.get("url") or ""
                    snippet = (r.get("snippet") or "").strip()
                    entry = title or link
                    if snippet:
                        entry += f"\n{snippet}"
                    doc_parts.append(entry)
                if doc_parts:
                    await emit_citation(
                        __event_emitter__,
                        "\n\n".join(doc_parts),
                        f"https://www.google.com/search?q={query.replace(' ', '+')}",
                        f"Google search: {query}",
                    )

            await emit_status(
                __event_emitter__,
                f"Found {len(organic)} results for: {query}",
                done=True,
            )
            return result

        except aiohttp.ClientError as e:
            error_msg = f"Network error while calling SerpBase API: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
        except Exception as e:
            error_msg = f"Error during Google search: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, done=True)
            return error_msg
