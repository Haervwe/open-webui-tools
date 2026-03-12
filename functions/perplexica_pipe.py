"""
title: Perplexica Pipe
author: haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.3.1
license: MIT
requirements: aiohttp
environment_variables: PERPLEXICA_API_URL
"""

import json
from typing import List, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from open_webui.constants import TASKS
from open_webui.utils.chat import generate_chat_completion
import aiohttp
from open_webui.models.users import User


name = "Perplexica"


class Pipe:
    class Valves(BaseModel):
        enable_perplexica: bool = Field(default=True)
        perplexica_api_url: str = Field(default="http://localhost:3001/api/search")
        perplexica_chat_model: str = Field(
            default="gpt-4o-mini",
            description="Chat model key as listed in Perplexica providers",
        )
        perplexica_embedding_model: str = Field(
            default="text-embedding-3-large",
            description="Embedding model key as listed in Perplexica providers",
        )
        task_model: str = Field(default="gpt-4o-mini")
        max_history_pairs: int = Field(default=12)
        perplexica_timeout_s: int = Field(
            default=120,
            description="Total request timeout in seconds (research can be slow)",
        )

    class UserValves(BaseModel):
        perplexica_sources: Literal["web", "academic", "discussions"] = Field(
            default="web",
            description="Search sources: web, academic, or discussions",
        )
        perplexica_optimization_mode: Literal["speed", "balanced", "quality"] = Field(
            default="balanced",
            description="Search optimization mode: speed (fastest), balanced, or quality (slowest)",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "perplexica_pipe"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.__current_event_emitter__ = None
        self.__request__ = None
        self.__user__ = None
        self.citation = False  # disable automatic citations

    def pipes(self) -> List[dict]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    # ---------- Emit helpers ----------
    async def _emit(self, payload: dict):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(payload)

    async def emit_status(
        self,
        description: str,
        done: bool,
        error: bool = False,
    ):
        data = {"action": "web_search", "description": description, "done": done}
        if error:
            data["error"] = True
        await self._emit({"type": "status", "data": data})

    async def emit_web_results(self, urls: List[str], items: List[dict]):
        """Emit the 'searched N sites' status bubble with clickable source links."""
        await self._emit(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "Searched {{count}} sites",
                    "done": True,
                    "urls": urls,
                    "items": items,
                },
            }
        )

    async def emit_message(self, content: str):
        await self._emit({"type": "message", "data": {"content": content}})

    async def emit_citation(self, title: str, url: str, content: str = ""):
        await self._emit(
            {
                "type": "citation",
                "data": {
                    "document": [content or title or url],
                    "metadata": [
                        {
                            "date_accessed": datetime.utcnow().isoformat() + "Z",
                            "source": title or url,
                            "url": url,
                        }
                    ],
                    "source": {"name": title or url, "url": url},
                },
            }
        )

    # ---------- Main entry ----------
    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
        results=None,
    ) -> Union[str, dict]:
        user_input = self._extract_user_input(body)
        self.__user_valves__ = __user__.pop("valves", None) or self.UserValves()
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        self.__current_event_emitter__ = __event_emitter__

        if __task__ and __task__ != TASKS.DEFAULT:
            response = await generate_chat_completion(
                self.__request__,
                {
                    "model": self.valves.task_model,
                    "messages": body.get("messages"),
                    "stream": False,
                },
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        if not user_input:
            return "No search query provided"

        model = body.get("model", "")
        if "perplexica" not in model.lower() or not self.valves.enable_perplexica:
            return f"Unsupported or disabled search engine for model: {model}"

        stream = bool(body.get("stream"))
        system_instructions = self._extract_system_instructions(body)
        history_pairs = self._build_history_pairs(body.get("messages", []))

        # Signal search start
        await self.emit_status("Searching the web…", done=False)

        response = await self._search_perplexica(
            query=user_input,
            stream=stream,
            system_instructions=system_instructions,
            history_pairs=history_pairs,
        )

        if not stream:
            # Non-stream: sources → citations → web_results → message
            urls: List[str] = []
            items: List[dict] = []
            if isinstance(response, dict) and response.get("sources"):
                for src in response["sources"]:
                    title, link, content, snippet = self._parse_source(src)
                    await self.emit_citation(title, link, content)
                    if link:
                        urls.append(link)
                        items.append(
                            {
                                "title": title,
                                "url": link,
                                "link": link,
                                "source": link,
                                "snippet": snippet,
                                "favicon": None,
                            }
                        )

            await self.emit_web_results(urls, items)

            msg = ""
            if isinstance(response, dict):
                msg = response.get("message", "")
            else:
                msg = str(response)

            return msg

        # Streaming: handled in handler; bare return
        return

    # ---------- Helpers ----------
    def _extract_user_input(self, body: dict) -> str:
        messages = body.get("messages", [])
        if not messages:
            return ""
        last_message = messages[-1]
        if isinstance(last_message.get("content"), list):
            for item in last_message["content"]:
                if item.get("type") == "text":
                    return item.get("text", "")
        return last_message.get("content", "") or ""

    def _extract_system_instructions(self, body: dict) -> str:
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        c.get("text", "") for c in content if c.get("type") == "text"
                    ]
                    return "\n".join([p for p in parts if p])
        return ""

    def _build_history_pairs(self, messages: List[dict]) -> List[List[str]]:
        pairs: List[List[str]] = []
        for m in messages:
            role = m.get("role")
            if role not in ("user", "assistant", "system"):
                continue
            text = ""
            content = m.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text += item.get("text", "")
            elif isinstance(content, str):
                text = content
            if role == "user":
                pairs.append(["human", text])
            elif role == "assistant":
                pairs.append(["assistant", text])
        max_items = self.valves.max_history_pairs * 2
        if len(pairs) > max_items:
            pairs = pairs[-max_items:]
        return pairs

    # ---------- Provider resolver ----------
    async def _resolve_provider_ids(self, session: aiohttp.ClientSession) -> dict:
        """Fetch /api/providers from Perplexica and resolve provider IDs for the configured models."""
        base = self.valves.perplexica_api_url.rstrip("/")
        if base.endswith("/api/search"):
            base = base[: -len("/api/search")]
        providers_url = f"{base}/api/providers"

        async with session.get(providers_url) as resp:
            resp.raise_for_status()
            data = await resp.json()

        providers = data.get("providers", [])
        chat_provider_id = None
        embedding_provider_id = None

        target_chat = self.valves.perplexica_chat_model.lower()
        target_embed = self.valves.perplexica_embedding_model.lower()

        for provider in providers:
            pid = provider.get("id", "")
            if not chat_provider_id:
                for m in provider.get("chatModels", []):
                    aliases = [m.get("key"), m.get("name"), m.get("displayName")]
                    if any(a and str(a).lower() == target_chat for a in aliases):
                        chat_provider_id = pid
                        break
            if not embedding_provider_id:
                for m in provider.get("embeddingModels", []):
                    aliases = [m.get("key"), m.get("name"), m.get("displayName")]
                    if any(a and str(a).lower() == target_embed for a in aliases):
                        embedding_provider_id = pid
                        break

        if not chat_provider_id:
            raise ValueError(
                f"Chat model '{self.valves.perplexica_chat_model}' not found in any Perplexica provider"
            )
        if not embedding_provider_id:
            raise ValueError(
                f"Embedding model '{self.valves.perplexica_embedding_model}' not found in any Perplexica provider"
            )

        return {
            "chat_provider_id": chat_provider_id,
            "embedding_provider_id": embedding_provider_id,
        }

    # ---------- Perplexica search ----------
    async def _search_perplexica(
        self,
        query: str,
        stream: bool,
        system_instructions: str,
        history_pairs: List[List[str]],
    ) -> Union[str, dict]:
        if not self.valves.enable_perplexica:
            return "Perplexica search is disabled"

        headers = {"Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.valves.perplexica_timeout_s)

        try:
            async with aiohttp.ClientSession(
                timeout=timeout, read_bufsize=2**20
            ) as session:
                # Auto-resolve provider IDs via /api/providers
                resolved = await self._resolve_provider_ids(session)

                request_body: Dict[str, Any] = {
                    "chatModel": {
                        "providerId": resolved["chat_provider_id"],
                        "key": self.valves.perplexica_chat_model,
                    },
                    "embeddingModel": {
                        "providerId": resolved["embedding_provider_id"],
                        "key": self.valves.perplexica_embedding_model,
                    },
                    "optimizationMode": self.__user_valves__.perplexica_optimization_mode,
                    "sources": [self.__user_valves__.perplexica_sources],
                    "query": query,
                    "history": history_pairs,
                    "stream": stream,
                }

                if system_instructions:
                    request_body["systemInstructions"] = system_instructions

                async with session.post(
                    self.valves.perplexica_api_url, json=request_body, headers=headers
                ) as resp:
                    resp.raise_for_status()

                    if stream:
                        await self._handle_streaming_response(resp)
                        return  # bare return, no payload
                    else:
                        data = await resp.json()
                        return self._render_non_stream_response(data)

        except aiohttp.ClientResponseError as e:
            await self.emit_status(
                f"Search error: {e.status} {e.message}", done=True, error=True
            )
            return f"HTTP error: {e.status} {e.message}"
        except Exception as e:
            await self.emit_status(f"Search error: {str(e)}", done=True, error=True)
            return f"Error: {str(e)}"

    async def _handle_streaming_response(self, resp: aiohttp.ClientResponse) -> None:
        """
        Perplexica stream event types (newline-delimited JSON):
          {"type": "init",     "data": "Stream connected"}
          {"type": "sources",  "data": [...]}   -- search result documents
          {"type": "response", "data": "chunk"} -- answer text token
          {"type": "done"}                       -- end of stream

        Correct Open WebUI event order:
          1. status(searching, done=False)           ← already emitted before calling this
          2. status(done=True, urls/items)           ← web_results bubble w/ source links
          3. citation × N                            ← individual citation cards
          4. message chunks                          ← streamed answer tokens
        """
        urls: List[str] = []
        items: List[dict] = []
        sources_emitted = False
        got_any_event = False

        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            got_any_event = True

            if etype == "init":
                # Connection confirmed — nothing to show, status already emitted
                continue

            elif etype == "sources":
                # Collect all sources first
                raw_sources = event.get("data") or []
                for src in raw_sources:
                    title, link, content, snippet = self._parse_source(src)
                    if link:
                        if link not in urls:
                            urls.append(link)
                        items.append(
                            {
                                "title": title,
                                "url": link,
                                "link": link,
                                "source": link,
                                "snippet": snippet,
                                "favicon": None,
                            }
                        )

                # 1) emit individual citation cards first (so they exist before the bubble)
                for src in raw_sources:
                    title, link, content, snippet = self._parse_source(src)
                    await self.emit_citation(title, link, content)

                # 2) update the status bubble to "done" with source links
                await self.emit_web_results(urls, items)
                sources_emitted = True

            elif etype == "response":
                # If we somehow got a response before sources, close the status first
                if not sources_emitted:
                    await self.emit_web_results(urls, items)
                    sources_emitted = True

                chunk = event.get("data", "")
                if chunk:
                    await self.emit_message(chunk)

            elif etype == "done":
                # Ensure status is always closed
                if not sources_emitted:
                    await self.emit_web_results(urls, items)
                    sources_emitted = True
                return

        # Stream ended without a "done" event (e.g. SearXNG error killed connection)
        if not sources_emitted:
            if got_any_event:
                # We got init but then a hard error — show error status
                await self.emit_status(
                    "Search failed — SearXNG or upstream error", done=True, error=True
                )
            else:
                await self.emit_web_results([], [])

    def _parse_source(self, src: dict):
        """Extract (title, url, content, snippet) from a Perplexica source document."""
        meta = src.get("metadata", {}) or {}
        title = meta.get("title") or src.get("title") or "Untitled source"
        link = (
            meta.get("url")
            or meta.get("link")
            or meta.get("source")
            or src.get("url")
            or src.get("link")
            or ""
        )
        link = str(link).strip()
        if not (link.startswith("http://") or link.startswith("https://")):
            link = ""
        content = (
            src.get("content", "")
            or src.get("pageContent", "")
            or meta.get("content", "")
        )
        snippet = meta.get("snippet", "") or content[:200]
        return title, link, content, snippet

    def _normalize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized = []
        for src in sources or []:
            title, link, content, snippet = self._parse_source(src)
            normalized.append(
                {
                    "title": title,
                    "url": link,
                    "link": link,
                    "source": link,
                    "content": content,
                }
            )
        return normalized

    def _render_non_stream_response(self, data: Dict[str, Any]) -> dict:
        sources = self._normalize_sources(data.get("sources", []) or [])
        message = data.get("message", "") or "No message available"
        prefix = "Perplexica Search Results:"
        if message.startswith(prefix):
            message = message[len(prefix) :].lstrip()
        return {"message": message, "sources": sources}
