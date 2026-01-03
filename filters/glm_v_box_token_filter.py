"""
title: GLM V Box Token Filter
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.0
description: Removes <|begin_of_box|> and <|end_of_box|> tokens from GLM V model responses in both streaming and outlet.
"""

from typing import Optional, Dict, Any, Callable, Awaitable, List, Union
from pydantic import BaseModel
import re


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.box_pattern = re.compile(r"<\|begin_of_box\|>|<\|end_of_box\|>")

    def _clean_text(self, text: str) -> str:
        """Remove box tokens from text."""
        return self.box_pattern.sub("", text)

    async def stream(
        self,
        event: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Filter streaming responses to remove box tokens."""
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta and delta["content"]:
                delta["content"] = self._clean_text(delta["content"])
        return event

    async def outlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Filter completed responses to remove box tokens."""
        messages = body.get("messages", [])
        for message in messages:
            if "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    message["content"] = self._clean_text(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            item["text"] = self._clean_text(item["text"])
        return body
