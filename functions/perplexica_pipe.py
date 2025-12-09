"""
title: Perplexica Pipe
author: haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
requirements: aiohttp
environment_variables: PERPLEXICA_API_URL
"""

import json
from typing import List, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass
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
        perplexica_chat_provider: str = Field(
            default="550e8400-e29b-41d4-a716-446655440000"
        )
        perplexica_chat_model: str = Field(default="gpt-4o-mini")
        perplexica_embedding_provider: str = Field(
            default="550e8400-e29b-41d4-a716-446655440000"
        )
        perplexica_embedding_model: str = Field(default="text-embedding-3-large")
        perplexica_focus_mode: Literal[
            "webSearch",
            "academicSearch",
            "writingAssistant",
            "wolframAlphaSearch",
            "youtubeSearch",
            "redditSearch",
        ] = Field(default="webSearch", description="Focus mode for search")
        perplexica_optimization_mode: Literal["speed", "balanced"] = Field(
            default="balanced",
            description="Search optimization mode: speed (fastest) or balanced (quality)",
        )
        task_model: str = Field(default="gpt-4o-mini")
        max_history_pairs: int = Field(default=12)
        perplexica_timeout_ms: int = Field(
            default=1500, description="Perplexica HTTP socket read timeout (ms)"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "perplexica_pipe"
        self.valves = self.Valves()
        self.__current_event_emitter__ = None
        self.__request__ = None
        self.__user__ = None
        self.citation = False  # disable automatic citations

    def pipes(self) -> List[dict]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    # ---------- Emit helpers ----------
    async def emit_status_basic(
        self,
        description: str,
        done: bool,
        error: bool = False,
        action: str = "web_search",
    ):
        data = {"action": action, "description": description, "done": done}
        if error:
            data["error"] = True
        payload = {"type": "status", "data": data}
        print("[DEBUG] emit_status_basic", payload)
        if not self.__current_event_emitter__:
            return
        await self.__current_event_emitter__(payload)

    async def emit_web_results(
        self, urls: List[str], items: List[dict], action: str = "web_search"
    ):
        data = {
            "action": action,
            "description": "Searched {{count}} sites",
            "done": True,
            "urls": urls,
            "items": items,
        }
        payload = {"type": "status", "data": data}
        print("[DEBUG] emit_web_results", payload)
        if not self.__current_event_emitter__:
            return
        await self.__current_event_emitter__(payload)

    async def emit_message(self, message: str, is_stream: bool = False):
        payload = {
            "type": "message",
            "data": {"content": message, "is_stream": is_stream},
        }
        print("[DEBUG] emit_message", payload)
        if not self.__current_event_emitter__:
            return
        await self.__current_event_emitter__(payload)

    async def emit_citation(self, title: str, url: str, content: str = ""):
        payload = {
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
        print("[DEBUG] emit_citation", payload)
        if not self.__current_event_emitter__:
            return
        await self.__current_event_emitter__(payload)

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

        # Start status
        await self.emit_status_basic(
            "Searching the web", done=False, action="web_search"
        )

        response = await self._search_perplexica(
            query=user_input,
            stream=stream,
            system_instructions=system_instructions,
            history_pairs=history_pairs,
        )

        if not stream:
            urls: List[str] = []
            items: List[dict] = []
            if isinstance(response, dict) and response.get("sources"):
                for src in response["sources"]:
                    await self.emit_citation(
                        src.get("title", ""), src.get("url", ""), src.get("content", "")
                    )
                    if src.get("url"):
                        urls.append(src["url"])
                    items.append(
                        {"title": src.get("title", ""), "url": src.get("url", "")}
                    )
            if isinstance(response, dict):
                if response.get("message"):
                    await self.emit_message(response["message"], is_stream=False)
                await self.emit_web_results(urls, items, action="web_search")
                return response.get("message", "")
            else:
                await self.emit_message(str(response), is_stream=False)
                await self.emit_web_results(urls, items, action="web_search")
                return str(response)

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

        request_body: Dict[str, Any] = {
            "chatModel": {
                "providerId": self.valves.perplexica_chat_provider,
                "key": self.valves.perplexica_chat_model,
            },
            "embeddingModel": {
                "providerId": self.valves.perplexica_embedding_provider,
                "key": self.valves.perplexica_embedding_model,
            },
            "optimizationMode": self.valves.perplexica_optimization_mode,
            "focusMode": self.valves.perplexica_focus_mode,
            "query": query,
            "history": history_pairs,
            "systemInstructions": system_instructions or None,
            "stream": stream,
        }

        request_body = {
            k: v for k, v in request_body.items() if v not in (None, "", "default")
        }

        headers = {"Content-Type": "application/json"}
        print("[DEBUG] request_body", request_body)

        timeout = aiohttp.ClientTimeout(
            total=None, sock_read=self.valves.perplexica_timeout_ms / 1000
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
            await self.emit_status_basic(
                f"Error: {e.status} {e.message}",
                done=True,
                error=True,
                action="web_search",
            )
            return f"HTTP error: {e.status} {e.message}"
        except Exception as e:
            await self.emit_status_basic(
                f"Error: {str(e)}", done=True, error=True, action="web_search"
            )
            return f"Error: {str(e)}"

    async def _handle_streaming_response(self, resp: aiohttp.ClientResponse) -> None:
        buffer: List[str] = []
        web_results_emitted = False
        urls: List[str] = []
        items: List[dict] = []

        def add_source(src: dict):
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
                link = ""  # invalidate non-http(s)
            content = src.get("pageContent", "") or meta.get("content", "")
            snippet = meta.get("snippet", "") or content[:200]

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
            # items without a link are ignored for web_results; still return for citations
            return title, link, content

        async for raw_line in resp.content:
            line = raw_line.decode().strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print("[DEBUG] skip malformed line", line)
                continue

            etype = event.get("type")
            print("[DEBUG] stream event", etype, event)
            if etype == "init":
                continue
            if etype == "sources":
                sources = event.get("data", []) or []
                for src in sources:
                    title, link, content = add_source(src)
                    await self.emit_citation(title, link, content)
                if not web_results_emitted and urls:
                    await self.emit_web_results(urls, items, action="web_search")
                    web_results_emitted = True
            elif etype == "response":
                chunk = event.get("data", "")
                if chunk:
                    buffer.append(chunk)
                    await self.emit_message(chunk, is_stream=True)
            elif etype == "done":
                if not web_results_emitted and urls:
                    await self.emit_web_results(urls, items, action="web_search")
                    web_results_emitted = True
                return

        if not web_results_emitted and urls:
            await self.emit_web_results(urls, items, action="web_search")
        return

    def _normalize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized = []
        for src in sources or []:
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
            content = src.get("pageContent", "") or meta.get("content", "")
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
