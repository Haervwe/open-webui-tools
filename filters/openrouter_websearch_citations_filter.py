"""
title: OpenRouter WebSearch Citations Filter
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.0
required_open_webui_version: 0.7.0
description: Enables web search for OpenRouter models by adding plugins and options to the request payload.
"""

from typing import Optional, Dict, Any, Callable, Awaitable, List
from pydantic import BaseModel, Field
import datetime


class Filter:
    class Valves(BaseModel):
        engine: str = Field(
            "auto",
            description="Web search engine: 'native' (provider's built-in), 'exa' (Exa API), or 'auto' (automatic selection)",
            json_schema_extra={"enum": ["auto", "native", "exa"]},
        )
        max_results: int = Field(
            5,
            description="Maximum number of web search results to retrieve (1-10)",
            ge=1,
            le=10,
        )
        search_prompt: str = Field(
            "A web search was conducted on {date}. Incorporate the following web search results into your response.\nIMPORTANT: Cite them using markdown links named using the domain of the source.\nExample: [nytimes.com](https://nytimes.com/some-page).",
            description="Prompt template for incorporating web search results. Use {date} placeholder for current date.",
        )
        search_context_size: str = Field(
            "medium",
            description="Search context size: 'low' (minimal), 'medium' (moderate), 'high' (extensive)",
            json_schema_extra={"enum": ["low", "medium", "high"]},
        )
        emit_url_citations: bool = Field(
            False,
            description="Also emit url_citation annotations as citation events. Open WebUI 0.7+ already shows them as sources, so enabling this duplicates every source; only turn it on for older servers.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjExIiBjeT0iMTEiIHI9IjgiLz48cGF0aCBkPSJtMjEgMjEtNC4zNS00LjM1Ii8+PC9zdmc+"""

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.toggle:
            plugin = {"id": "web"}
            if self.valves.engine != "auto":
                plugin["engine"] = self.valves.engine
            plugin["max_results"] = self.valves.max_results
            search_prompt = self.valves.search_prompt.replace("{date}", datetime.date.today().isoformat())
            plugin["search_prompt"] = search_prompt
            body["plugins"] = [plugin]
            body["web_search_options"] = {"search_context_size": self.valves.search_context_size}
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Web search enabled for this query",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )
        return body

    async def stream(
        self,
        event: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        if not __event_emitter__:
            return event
        for index, choice in enumerate(event.get("choices") or []):
            delta = choice.get("delta") or {}
            annotations = delta.get("annotations")
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                # Open WebUI 0.7+ (utils/middleware.py "Handle delta annotations")
                # already emits a "source" event for every url_citation in
                # choices[0]; emitting it again here only doubles each source's
                # documents/metadata.
                if (
                    not self.valves.emit_url_citations
                    and index == 0
                    and annotation.get("type") == "url_citation"
                    and "url_citation" in annotation
                ):
                    continue

                url: Optional[str] = None
                title: Optional[str] = None
                document: List[str] = []
                metadata: List[Dict[str, Any]] = []

                if annotation.get("type") == "url_citation":
                    url_info = annotation.get("url_citation") or {}
                    url = url_info.get("url")
                    title = url_info.get("title")

                    if title:
                        document.append(title)

                    meta = {"source": url, "title": title}
                    if "start_index" in url_info:
                        meta["start_index"] = url_info["start_index"]
                    if "end_index" in url_info:
                        meta["end_index"] = url_info["end_index"]
                    metadata.append(meta)

                if not document:
                    document.append(str(annotation))
                citation_data: Dict[str, Any] = {
                    "document": document,
                    "metadata": metadata,
                    "source": {
                        "name": title or url or "Citation",
                        "url": url or "",
                    },
                }
                await __event_emitter__({"type": "citation", "data": citation_data})
        return event
