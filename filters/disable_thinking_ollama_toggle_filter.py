"""
title: Disable Thinking Ollama Toggle
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.1
required_open_webui_version: 0.8.10
description: Filter that disables or minimizes Ollama reasoning/thinking based on the model name. Also handles OpenAI-compatible backends (llama.cpp / llama-swap) via chat_template_kwargs and reasoning_effort.
"""

from typing import Optional, Dict, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import logging

name = "disable_ollama_reasoning"
logger = logging.getLogger(name)

if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.set_name(name)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxtYXNrIGlkPSJsaW5lTWRTcGVlZFR3b3RvbmVMb29wMCI+PHBhdGggZmlsbD0iI2ZmZiIgZmlsbC1vcGFjaXR5PSIwIiBzdHJva2U9IiNmZmYiIHN0cm9rZS1kYXNoYXJyYXk9IjU2IiBzdHJva2UtZGFzaG9mZnNldD0iNTYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIyIiBkPSJNNSAxOXYwYy0wLjMgMCAtMC41OSAtMC4xNSAtMC43NCAtMC40MWMtMC44IC0xLjM0IC0xLjI2IC0yLjkxIC0xLjI2IC00LjU5YzAgLTQuOTcgNC4wMyAtOSA5IC05YzQuOTcgMCA5IDQuMDMgOSA5YzAgMS42OCAtMC40NiAzLjI1IC0xLjI2IDQuNTljLTAuMTUgMC4yNiAtMC40NCAwLjQxIC0wLjc0IDAuNDFaIj48YW5pbWF0ZSBmaWxsPSJmcmVlemUiIGF0dHJpYnV0ZU5hbWU9ImZpbGwtb3BhY2l0eSIgYmVnaW49IjAuM3MiIGR1cj0iMC4xNXMiIHZhbHVlcz0iMDswLjMiLz48YW5pbWF0ZSBmaWxsPSJmcmVlemUiIGF0dHJpYnV0ZU5hbWU9InN0cm9rZS1kYXNob2Zmc2V0IiBkdXI9IjAuNnMiIHZhbHVlcz0iNTY7MCIvPjwvcGF0aD48ZyB0cmFuc2Zvcm09InJvdGF0ZSgtMTAwIDEyIDE0KSI+PHBhdGggZD0iTTEyIDE0QzEyIDE0IDEyIDE0IDEyIDE0QzEyIDE0IDEyIDE0IDEyIDE0QzEyIDE0IDEyIDE0IDEyIDE0QzEyIDE0IDEyIDE0IDEyIDE0WiI+PGFuaW1hdGUgZmlsbD0iZnJlZXplIiBhdHRyaWJ1dGVOYW1lPSJkIiBiZWdpbj0iMC40cyIgZHVyPSIwLjJzIiB2YWx1ZXM9Ik0xMiAxNEMxMiAxNCAxMiAxNCAxMiAxNEMxMiAxNCAxMiAxNCAxMiAxNEMxMiAxNCAxMiAxNCAxMiAxNEMxMiAxNCAxMiAxNCAxMiAxNFo7TTE2IDE0QzE2IDE2LjIxIDE0LjIxIDE4IDEyIDE4QzkuNzkgMTggOCAxNi4yMSA4IDE0QzggMTEuNzkgMTIgMCAxMiAwQzEyIDAgMTYgMTEuNzkgMTYgMTRaIi8+PC9wYXRoPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Ik0xMiAxNEMxMiAxNCAxMiAxNCAxMiAxNEMxMiAxNCAxMiAxNCAxMiAxNEMxMiAxNCAxMiAxNCAxMiAxNEMxMiAxNCAxMiAxNCAxMiAxNFoiPjxhbmltYXRlIGZpbGw9ImZyZWV6ZSIgYXR0cmlidXRlTmFtZT0iZCIgYmVnaW49IjAuNHMiIGR1cj0iMC4ycyIgdmFsdWVzPSJNMTIgMTRDMTIgMTQgMTIgMTQgMTIgMTRDMTIgMTQgMTIgMTQgMTIgMTRDMTIgMTQgMTIgMTQgMTIgMTRDMTIgMTQgMTIgMTQgMTIgMTRaO00xNCAxNEMxNCAxNS4xIDEzLjEgMTYgMTIgMTZDMTAuOSAxNiAxMCAxNS4xIDEwIDE0QzEwIDEyLjkgMTIgNCAxMiA0QzEyIDQgMTQgMTIuOSAxNCAxNFoiLz48L3BhdGg+PGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiBiZWdpbj0iMC40cyIgZHVyPSI2cyIgcmVwZWF0Q291bnQ9ImluZGVmaW5pdGUiIHR5cGU9InJvdGF0ZSIgdmFsdWVzPSItMTAwIDEyIDE0OzQ1IDEyIDE0OzQ1IDEyIDE0OzQ1IDEyIDE0OzIwIDEyIDE0OzEwIDEyIDE0OzAgMTIgMTQ7MzUgMTIgMTQ7NDUgMTIgMTQ7NTUgMTIgMTQ7NTAgMTIgMTQ7MTUgMTIgMTQ7LTIwIDEyIDE0Oy0xMDAgMTIgMTQiLz48L2c+PC9tYXNrPjxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0iY3VycmVudENvbG9yIiBtYXNrPSJ1cmwoI2xpbmVNZFNwZWVkVHdvdG9uZUxvb3AwKSIvPjwvc3ZnPg=="""
        # Track accumulated content to check for opening tags
        self.accumulated_content = {}

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_id = body.get("model", "") or ""
        is_gpt_oss = "gpt-oss" in model_id
        owned_by = (__model__ or {}).get("owned_by")
        if owned_by is None:
            owned_by = "ollama" if "options" in body else None

        if owned_by == "ollama":
            # OWUI only creates body["options"] for Ollama-owned models
            options = body.setdefault("options", {})
            if is_gpt_oss:
                options["reasoning_effort"] = "minimal"
                options["think"] = "minimal"
            else:
                options["think"] = False
        else:
            # OpenAI-compatible backends (llama.cpp, llama-swap, vLLM): the body is
            # forwarded as-is, so use the template kwargs these servers understand
            template_kwargs = body.get("chat_template_kwargs")
            if not isinstance(template_kwargs, dict):
                template_kwargs = {}
            if is_gpt_oss:
                body["reasoning_effort"] = "low"
                template_kwargs["reasoning_effort"] = "low"
            else:
                template_kwargs["enable_thinking"] = False
            body["chat_template_kwargs"] = template_kwargs
        return body
