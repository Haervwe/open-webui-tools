"""
title: Perplexica Search API Tool
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.1
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Dict
import aiohttp


class Tools:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="http://host.docker.internal:3001",
            description="Base URL for the Perplexica API",
        )
        OPTIMIZATION_MODE: str = Field(
            default="balanced",
            description="Search optimization mode (speed or balanced)",
        )
        CHAT_MODEL: str = Field(
            default="llama3.1:latest", description="Default chat model"
        )
        EMBEDDING_MODEL: str = Field(
            default="bge-m3:latest", description="Default embedding model"
        )
        OLLAMA_BASE_URL: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL for Ollama API",
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

    async def web_search(
        self, query: str, __event_emitter__: Optional[Callable[[Dict], Any]] = None
    ) -> str:
        """Search using the Perplexica API."""

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

        # Fixed: Use proper nested structure like the working Pipe
        payload = {
            "focusMode": "webSearch",
            "optimizationMode": self.valves.OPTIMIZATION_MODE,
            "query": query,
            "chatModel": {
                "provider": "ollama",
                "name": self.valves.CHAT_MODEL,  # Changed from "model" to "name"
            },
            "embeddingModel": {
                "provider": "ollama",
                "name": self.valves.EMBEDDING_MODEL,  # Changed from "model" to "name"
            },
            "history": [],  # Changed from None to empty list
        }

        # Fixed: Clean up request body like the working Pipe
        payload = {k: v for k, v in payload.items() if v is not None}
        payload = {k: v for k, v in payload.items() if v != "default"}

        try:
            await emit_status("Sending request to Perplexica API")

            # Fixed: Use aiohttp instead of requests for proper async handling
            headers = {"Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.BASE_URL.rstrip('/')}/api/search",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            # Emit main content as citation
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [result["message"]],
                            "metadata": [{"source": "Perplexica Search"}],
                            "source": {"name": "Perplexica"},
                        },
                    }
                )

            # Emit each source as a citation
            if result.get("sources") and __event_emitter__:
                for source in result["sources"]:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [source["pageContent"]],
                                "metadata": [{"source": source["metadata"]["url"]}],
                                "source": {"name": source["metadata"]["title"]},
                            },
                        }
                    )

            await emit_status(
                "Search completed successfully", status="complete", done=True
            )

            # Format response with citations
            response_text = f"{result['message']}\n\nSources:\n"
            response_text += "- Perplexica Search\n"
            for source in result.get("sources", []):
                response_text += (
                    f"- {source['metadata']['title']}: {source['metadata']['url']}\n"
                )
            return response_text

        except Exception as e:
            error_msg = f"Error performing search: {str(e)}"
            await emit_status(error_msg, status="error", done=True)
            return error_msg
