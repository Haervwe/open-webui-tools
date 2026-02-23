"""
title: Perplexica Search API Tool
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Dict, Literal
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)


class Tools:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="http://host.docker.internal:3001",
            description="Base URL for the Perplexica API",
        )
        FOCUS_MODE: Literal[
            "webSearch",
            "academicSearch",
            "writingAssistant",
            "wolframAlphaSearch",
            "youtubeSearch",
            "redditSearch",
        ] = Field(
            default="webSearch",
            description="Focus mode for search",
        )
        OPTIMIZATION_MODE: Literal["speed", "balanced"] = Field(
            default="balanced",
            description="Search optimization mode: speed (fastest) or balanced (quality)",
        )
        CHAT_MODEL: str = Field(
            default="gpt-4o-mini",
            description="Chat model key as listed in Perplexica providers",
        )
        EMBEDDING_MODEL: str = Field(
            default="text-embedding-3-large",
            description="Embedding model key as listed in Perplexica providers",
        )
        TIMEOUT_SECONDS: int = Field(
            default=300,
            description="Total timeout for Perplexica search in seconds (default: 5 minutes)",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for factual information, current events, or specific topics. Only use this when a search query is explicitly needed or when the user asks for information that requires looking up current or factual data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "web_query": {
                                "type": "string",
                                "description": "The specific search query to look up. Should be a clear, focused search term or question.",
                            }
                        },
                        "required": ["web_query"],
                    },
                },
            }
        ]

    async def _resolve_provider_ids(self, session: aiohttp.ClientSession) -> dict:
        """Fetch /api/providers from Perplexica and resolve provider IDs for the configured models."""
        providers_url = f"{self.valves.BASE_URL.rstrip('/')}/api/providers"
        async with session.get(providers_url) as resp:
            resp.raise_for_status()
            data = await resp.json()

        providers = data.get("providers", [])
        chat_provider_id = None
        embedding_provider_id = None

        for provider in providers:
            pid = provider.get("id", "")
            if not chat_provider_id:
                for m in provider.get("chatModels", []):
                    if m.get("key") == self.valves.CHAT_MODEL:
                        chat_provider_id = pid
                        break
            if not embedding_provider_id:
                for m in provider.get("embeddingModels", []):
                    if m.get("key") == self.valves.EMBEDDING_MODEL:
                        embedding_provider_id = pid
                        break

        if not chat_provider_id:
            raise ValueError(
                f"Chat model '{self.valves.CHAT_MODEL}' not found in any Perplexica provider"
            )
        if not embedding_provider_id:
            raise ValueError(
                f"Embedding model '{self.valves.EMBEDDING_MODEL}' not found in any Perplexica provider"
            )

        return {
            "chat_provider_id": chat_provider_id,
            "embedding_provider_id": embedding_provider_id,
        }

    async def web_search(
        self, query: str, __event_emitter__: Optional[Callable[[Dict], Any]] = None
    ) -> str:
        """Search using the Perplexica API with streaming support."""

        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "status": status,
                            "done": done,
                        },
                    }
                )

        await emit_status(f"Initiating search for: {query}")

        try:
            headers = {"Content-Type": "application/json"}
            url = f"{self.valves.BASE_URL.rstrip('/')}/api/search"

            if self.valves.DEBUG:
                logger.info(f"Perplexica API URL: {url}")
                logger.info(f"Timeout: {self.valves.TIMEOUT_SECONDS}s")

            # Set timeout - total timeout for entire operation
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Auto-resolve provider IDs from Perplexica
                await emit_status("Resolving model providers...")
                resolved = await self._resolve_provider_ids(session)

                if self.valves.DEBUG:
                    logger.info(f"Resolved providers: {resolved}")

                payload = {
                    "chatModel": {
                        "providerId": resolved["chat_provider_id"],
                        "key": self.valves.CHAT_MODEL,
                    },
                    "embeddingModel": {
                        "providerId": resolved["embedding_provider_id"],
                        "key": self.valves.EMBEDDING_MODEL,
                    },
                    "optimizationMode": self.valves.OPTIMIZATION_MODE,
                    "focusMode": self.valves.FOCUS_MODE,
                    "query": query,
                    "history": [],
                    "systemInstructions": None,
                    "stream": True,
                }

                # Clean up request body
                payload = {
                    k: v for k, v in payload.items() if v not in (None, "", "default")
                }

                if self.valves.DEBUG:
                    logger.info(f"Perplexica request payload: {payload}")

                await emit_status(f"Searching for: {query}")

                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                ) as response:
                    if self.valves.DEBUG:
                        logger.info(f"Response status: {response.status}")
                        logger.info(f"Response headers: {dict(response.headers)}")

                    response.raise_for_status()

                    # Handle streaming response
                    sources = []
                    message_chunks = []
                    sources_emitted = False

                    try:
                        async for raw_line in response.content:
                            line = raw_line.decode().strip()
                            if not line:
                                continue

                            try:
                                import json

                                event = json.loads(line)
                            except json.JSONDecodeError as e:
                                if self.valves.DEBUG:
                                    logger.warning(
                                        f"Failed to parse JSON line: {line[:100]}... Error: {e}"
                                    )
                                continue

                            event_type = event.get("type")

                            if self.valves.DEBUG:
                                logger.info(f"Stream event: {event_type}")

                            if event_type == "init":
                                await emit_status("Searching the web...")
                                continue

                            elif event_type == "sources":
                                raw_sources = event.get("data", []) or []

                                if self.valves.DEBUG:
                                    logger.info(
                                        f"Received {len(raw_sources)} raw sources from API"
                                    )

                                for src in raw_sources:
                                    meta = src.get("metadata", {}) or {}
                                    title = (
                                        meta.get("title")
                                        or src.get("title")
                                        or "Untitled source"
                                    )
                                    link = (
                                        meta.get("url")
                                        or meta.get("link")
                                        or meta.get("source")
                                        or src.get("url")
                                        or src.get("link")
                                        or ""
                                    )
                                    link = str(link).strip()
                                    if not (
                                        link.startswith("http://")
                                        or link.startswith("https://")
                                    ):
                                        if self.valves.DEBUG:
                                            logger.warning(
                                                f"Invalid link format, skipping: {link[:100] if link else 'empty'}"
                                            )
                                        link = ""
                                    content = src.get("pageContent", "") or meta.get(
                                        "content", ""
                                    )

                                    if self.valves.DEBUG:
                                        logger.info(
                                            f"Processing source: title={title[:50]}, link={link[:100] if link else 'empty'}, has_content={bool(content)}"
                                        )

                                    # Always add to sources list if we have a valid link
                                    if link:
                                        sources.append(
                                            {
                                                "title": title,
                                                "url": link,
                                                "content": content,
                                            }
                                        )

                                        # Emit citation as source arrives
                                        if __event_emitter__:
                                            await __event_emitter__(
                                                {
                                                    "type": "citation",
                                                    "data": {
                                                        "document": [content or title],
                                                        "metadata": [{"source": link}],
                                                        "source": {
                                                            "name": title,
                                                            "url": link,
                                                        },
                                                    },
                                                }
                                            )

                                if self.valves.DEBUG:
                                    logger.info(
                                        f"Processed {len(sources)} valid sources with links"
                                    )

                                # Emit web_results status with URLs and items matching pipe format
                                if sources and not sources_emitted:
                                    urls = [s["url"] for s in sources]
                                    items = [
                                        {
                                            "title": s["title"],
                                            "url": s["url"],
                                            "link": s["url"],
                                            "source": s["url"],
                                            "snippet": s.get("content", "")[:200]
                                            if s.get("content")
                                            else "",
                                            "favicon": None,
                                        }
                                        for s in sources
                                    ]
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "action": "web_search",
                                                    "description": f"Generating report with {len(sources)} sources...",
                                                    "done": True,
                                                    "urls": urls,
                                                    "items": items,
                                                },
                                            }
                                        )
                                    sources_emitted = True
                                elif not sources and self.valves.DEBUG:
                                    logger.warning("No valid sources with links found")

                            elif event_type == "response":
                                chunk = event.get("data", "")
                                if chunk:
                                    message_chunks.append(chunk)

                            elif event_type == "done":
                                if self.valves.DEBUG:
                                    logger.info("Stream completed")
                                break

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Stream reading timed out after {self.valves.TIMEOUT_SECONDS}s, using partial response"
                        )
                        if self.valves.DEBUG:
                            logger.info(
                                f"Collected {len(message_chunks)} message chunks and {len(sources)} sources before timeout"
                            )

            await emit_status(
                "Search completed successfully", status="complete", done=True
            )

            # Combine message chunks
            message = "".join(message_chunks)
            if not message:
                message = "No response received from Perplexica"

            prefix = "Perplexica Search Results:"
            if message.startswith(prefix):
                message = message[len(prefix) :].lstrip()

            response_text = message
            if sources:
                response_text += "\n\nSources:\n"
                for source in sources:
                    response_text += f"- {source['title']}: {source['url']}\n"

            if self.valves.DEBUG:
                logger.info(f"Final response length: {len(response_text)}")

            return response_text

        except asyncio.TimeoutError:
            error_msg = f"Search timed out after {self.valves.TIMEOUT_SECONDS} seconds"
            logger.error(f"Perplexica timeout: {error_msg}")
            await emit_status(error_msg, status="error", done=True)
            return error_msg
        except aiohttp.ClientResponseError as e:
            error_msg = f"HTTP error: {e.status} {e.message}"
            logger.error(f"Perplexica API error: {error_msg}")
            if self.valves.DEBUG:
                logger.error(
                    f"Response body: {await e.response.text() if hasattr(e, 'response') else 'N/A'}"
                )
            await emit_status(error_msg, status="error", done=True)
            return error_msg
        except Exception as e:
            error_msg = f"Error performing search: {str(e)}"
            logger.error(f"Perplexica search error: {error_msg}", exc_info=True)
            await emit_status(error_msg, status="error", done=True)
            return error_msg
