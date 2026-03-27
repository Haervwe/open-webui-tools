"""
title: Planner v3
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 3.4.0
required_open_webui_version: 0.8.12
"""

import ast
import hashlib
import json
import logging
import os
import re
import time
import uuid
import html as html_module
from uuid import uuid4
from typing import Callable, Awaitable, Any, Optional, Union, Generator, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import Request, UploadFile
from starlette.datastructures import Headers
import io

from open_webui.utils.chat import (
    generate_chat_completion as generate_raw_chat_completion,
)
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import process_tool_result
from open_webui.models.models import Models
from open_webui.models.users import Users
from open_webui.models.files import Files
from open_webui.models.chats import Chats
from open_webui.routers.files import upload_file_handler


from typing import Dict


# --- Pydantic Models ---
class ToolFunctionModel(BaseModel):
    name: str
    arguments: str
    description: str = ""


class ToolCallEntryModel(BaseModel):
    id: str
    function: ToolFunctionModel


class TaskStateModel(BaseModel):
    status: str
    description: str = ""


# For subagent_history, keep as dict for now (complex key)

ToolCallDict = Dict[str, ToolCallEntryModel]


# --- New Agent Models ---
class AgentDefinition(BaseModel):
    id: str
    name: str
    description: str
    system_message: str
    features: Dict[str, bool] = Field(default_factory=dict)
    type: str = "builtin"  # "builtin" or "terminal"
    temperature: Optional[float] = None
    model_id: Optional[str] = None
    builtin_model_override: Optional[Dict[str, Any]] = None


class SubagentTaskResponse(BaseModel):
    task_id: str
    status: str = "completed"
    called_tools: list[dict] = Field(default_factory=list)
    result: str
    note: Optional[str] = None


name = "Planner"


def setup_logger():
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
    return logger


logger = setup_logger()

# ---------------------------------------------------------------------------
# Utility Classes
# ---------------------------------------------------------------------------


class Utils:

    # Regex patterns for cleaning agent XML/thinking tags
    THINK_OPEN_PATTERN = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>|\|begin_of_thought\|",
        re.IGNORECASE,
    )
    THINK_CLOSE_PATTERN = re.compile(
        r"</(think|thinking|reason|reasoning|thought|Thought)>|\|end_of_thought\|",
        re.IGNORECASE,
    )
    THINKING_TAG_CLEANER_PATTERN = re.compile(
        r"</?(?:think|thinking|reason|reasoning|thought|Thought)>|\|begin_of_thought\||\|end_of_thought\|",
        re.IGNORECASE,
    )

    @staticmethod
    def distill_history_for_llm(messages: list) -> list:
        """
        Cleans and normalizes message history for LLM consumption.
        - Flattens list-based content (OpenAI/WebUI format) to string.
        - Strips all UI-only artifacts (<details> tags for reasoning, tools, status).
        """
        distilled = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 1. Flatten content if it's a list (Open WebUI / OpenAI structured format)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)

            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            # 2. Strip ALL UI artifacts (reasoning, tool_calls, status, state)
            # We use a broad regex to catch all <details> blocks as they are UI-specific.
            content = re.sub(
                r"<details.*?>.*?</details>", "", content, flags=re.DOTALL
            ).strip()

            if content:
                distilled.append({"role": role, "content": content})
        return distilled

    @staticmethod
    def extract_xml_tool_calls(text: str) -> tuple[ToolCallDict, str]:
        """
        Extracts tool calls from <tool_call>...</tool_call> XML blocks in the text.
        Returns:
            tuple[ToolCallDict, str]: (tool_calls_dict, cleaned_text)
        """
        tool_calls_dict: ToolCallDict = {}
        xml_count = 0
        cleaned_text = text
        if "<tool_call>" in text:
            # Match <tool_call> blocks, allowing for unclosed tags at the end of string
            xml_matches = re.finditer(
                r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.DOTALL
            )
            for match in xml_matches:
                tc_data = match.group(1).strip()
                # Support <function name="..."> or <function=...>
                func_match = re.search(
                    r'<function(?:\s*=\s*|\s+name\s*=\s*)"?([^>"\s]+)"?>', tc_data
                )
                if func_match:
                    func_name = func_match.group(1).strip()
                    kwargs = {}
                    # Support <parameter name="..."> or <parameter=...>
                    param_matches = re.findall(
                        r'<parameter(?:\s*=\s*|\s+name\s*=\s*)"?([^>"\s]+)"?>(.*?)(?=</parameter>|<parameter|<function|</tool_call>|$)',
                        tc_data,
                        re.DOTALL,
                    )
                    for p_name, p_val in param_matches:
                        kwargs[p_name.strip()] = Utils.clean_thinking(p_val.strip())

                    tool_calls_dict[f"xml_{xml_count}"] = {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(kwargs),
                        },
                    }
                else:
                    try:
                        data = json.loads(tc_data)
                        if isinstance(data, dict) and "name" in data:
                            tool_calls_dict[f"xml_{xml_count}"] = {
                                "id": str(uuid.uuid4()),
                                "type": "function",
                                "function": {
                                    "name": data["name"],
                                    "arguments": json.dumps(
                                        data.get(
                                            "arguments", data.get("parameters", {})
                                        )
                                    ),
                                },
                            }
                    except Exception:
                        pass
                xml_count += 1
            # Clean up all tool_call blocks, including unclosed ones
            cleaned_text = re.sub(
                r"<tool_call>.*?(?:</tool_call>|$)", "", cleaned_text, flags=re.DOTALL
            ).strip()
        return tool_calls_dict, cleaned_text



    @staticmethod
    def clean_thinking(text: str) -> str:
        """
        Remove ALL thinking/reasoning content and tags from text.
        Intended for the final USER-FACING CONTENT area.
        """
        if not text:
            return ""
        # 1. Remove complete pairs
        pattern = re.compile(
            r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
            r"|"
            r"\|begin_of_thought\|.*?\|end_of_thought\|",
            re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(pattern, "", text)

        # 2. Hide tool calls from display
        text = Utils.hide_tool_calls(text)

        # 3. Handle unclosed tags (usually at start of stream)
        text = re.sub(
            r"<(?:think|thinking|reason|reasoning|thought|Thought)>.*?(?=<tool_call>|$)",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"\|begin_of_thought\|.*?(?=<tool_call>|$)",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # 4. Remove any stray tags
        text = Utils.THINKING_TAG_CLEANER_PATTERN.sub("", text)
        return text.strip()

    @staticmethod
    def hide_tool_calls(text: str) -> str:
        """
        Remove <tool_call> blocks but preserve all other text.
        Intended for the THINKING TRACE area.
        """
        if not text:
            return ""
        return re.sub(
            r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.DOTALL
        ).strip()

    @staticmethod
    def middle_truncate(text: str, max_chars: int) -> str:
        """
        Truncate text from the middle if it exceeds max_chars.
        Args:
            text (str): The input text.
            max_chars (int): Maximum allowed characters.
        Returns:
            str: Truncated text if needed.
        """
        if not text or len(text) <= max_chars:
            return text
        half = max_chars // 2
        return (
            text[:half]
            + f"\n\n... [TRUNCATED {len(text) - max_chars} CHARS] ...\n\n"
            + text[-half:]
        )

    @staticmethod
    def _parse_sse_events(buffer: str) -> tuple[list[dict[str, Any]], str, bool]:
        """
        Parse SSE events from a buffer string.
        Args:
            buffer (str): SSE event buffer.
        Returns:
            tuple[list[dict], str, bool]: (events, remaining_buffer, done)
        """
        events = []
        done = False
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                stripped = line.strip()
                if stripped.startswith("data:"):
                    data_lines.append(stripped[5:].lstrip())
            if not data_lines:
                continue
            payload = "\n".join(data_lines).strip()
            if not payload:
                continue
            if payload == "[DONE]":
                done = True
                break
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    events.append(parsed)
            except Exception:
                continue
        return events, buffer, done

    @staticmethod
    def _extract_stream_events(event_payload: dict[str, Any]) -> Any:
        """
        Extracts stream events from an event payload dict.
        Args:
            event_payload (dict): Event payload.
        Yields:
            dict: Event dicts for reasoning, content, or tool_calls.
        """
        choices = event_payload.get("choices", [])
        if not choices:
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta", {}) or {}
        # Reasoning
        for rk in ["reasoning", "reasoning_content", "thinking"]:
            rv = delta.get(rk)
            if rv:
                yield {"type": "reasoning", "text": rv}
        # Content
        cv = delta.get("content")
        if cv:
            yield {"type": "content", "text": cv}
        # Tool Calls
        tc = delta.get("tool_calls")
        if tc:
            yield {"type": "tool_calls", "data": tc}

    @staticmethod
    async def get_streaming_completion(
        request: Any, form_data: dict[str, Any], user: Any
    ) -> Any:
        """
        Wrapper to turn raw streaming response into an event generator.
        """
        form_data["stream"] = True
        try:
            # v3 parity: some versions of OWUI expect the user model, others the dict.
            response = await generate_raw_chat_completion(request, form_data, user=user)

            # 1. Handle StreamingResponse (standard case)
            if hasattr(response, "body_iterator"):
                sse_buffer = ""
                async for chunk in response.body_iterator:
                    decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                    sse_buffer += decoded
                    events, sse_buffer, done = Utils._parse_sse_events(sse_buffer)
                    for event_payload in events:
                        for event in Utils._extract_stream_events(event_payload):
                            yield event
                    if done:
                        break
                return

            # 2. Handle non-streaming dict responses (fallback)
            if isinstance(response, dict):
                content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content:
                    yield {"type": "content", "text": content}
                return

            # 3. Handle potential error responses (Starlette/FastAPI Response objects)
            if hasattr(response, "body"):
                body_bytes = (
                    await response.body()
                    if callable(getattr(response, "body", None))
                    else getattr(response, "body", b"")
                )
                try:
                    body_json = json.loads(body_bytes)
                    error_detail = body_json.get("error", {}).get(
                        "message", str(body_json)
                    )
                    yield {"type": "error", "text": f"LLM Provider Error: {error_detail}"}
                    return
                except:
                    yield {
                        "type": "error",
                        "text": f"LLM Error (Status {getattr(response, 'status_code', 'unknown')}): {body_bytes.decode('utf-8', 'ignore')}",
                    }
                    return

            # 4. Fallback for strings
            if isinstance(response, str):
                yield {"type": "content", "text": response}
                return

            raise ValueError(f"Response does not support streaming: {type(response)}")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "text": str(e)}

    @staticmethod
    def resolve_references(text: str, results: dict[str, str]) -> str:
        """
        Replace @task_id references with their full content, skipping matches inside <details> blocks.
        Args:
            text (str): Input text.
            results (dict[str, str]): Mapping of task_id to content.
        Returns:
            str: Text with references resolved.
        """
        """Replace @task_id references with their full content, skipping matches inside <details> blocks."""
        if not text or not isinstance(text, str):
            return text

        # Split by <details> tags to avoid replacing inside them
        parts = re.split(r"(<details.*?</details>)", text, flags=re.DOTALL)
        pattern = r"@([a-zA-Z0-9_-]+)"

        for i in range(len(parts)):
            # Only process parts that are NOT <details> blocks
            if not parts[i].startswith("<details"):
                matches = re.findall(pattern, parts[i])
                for match in matches:
                    if match in results:
                        parts[i] = parts[i].replace(f"@{match}", results[match])

        return "".join(parts)

    @staticmethod
    def resolve_dict_references(
        data: Any,
        results: dict[str, str],
        skip_keys: list[str] = ["task_id", "task_ids", "related_tasks"],
    ) -> Any:
        """
        Recursively resolve @task_id references in strings within a dict or list.
        Args:
            data (Any): Data structure (str, list, dict).
            results (dict[str, str]): Mapping of task_id to content.
            skip_keys (list): Keys to skip when resolving.
        Returns:
            Any: Data with references resolved.
        """
        """Recursively resolve @task_id references in strings within a dict or list."""
        if isinstance(data, str):
            if "@" not in data:
                return data
            for tid, result in results.items():
                ref = f"@{tid}"
                if ref in data:
                    data = data.replace(ref, result)
            return data
        elif isinstance(data, list):
            return [
                Utils.resolve_dict_references(item, results, skip_keys) for item in data
            ]
        elif isinstance(data, dict):
            # Resolve values but skip keys that identify tasks
            return {
                k: (
                    v
                    if k in skip_keys
                    else Utils.resolve_dict_references(v, results, skip_keys)
                )
                for k, v in data.items()
                if v is not None
            }
        return data

    @staticmethod
    def extract_json_array(text: str) -> list:
        """
        Extract the first valid JSON array from text, handled redundantly for robustness.
        Args:
            text (str): Input text possibly containing a JSON array.
        Returns:
            list: Extracted JSON array or empty list.
        """
        # 1. Clean thinking tags
        text = Utils.clean_thinking(text)

        # 2. Extract from markdown if present
        markdown_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if markdown_match:
            text = markdown_match.group(1)

        # 3. Basic cleanup
        text = text.strip()

        # 4. Try finding the first [ or {
        start_obj = text.find("{")
        start_arr = text.find("[")

        if start_obj == -1 and start_arr == -1:
            return []

        start = (
            start_obj
            if (start_arr == -1 or (start_obj != -1 and start_obj < start_arr))
            else start_arr
        )

        # 5. Use raw_decode to find the first valid JSON
        decoder = json.JSONDecoder()
        remaining_text = text[start:]

        try:
            obj, _ = decoder.raw_decode(remaining_text)
            logger.debug(f"JSON object decoded: {type(obj)}")
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "tasks" in obj:
                return obj["tasks"]
            if isinstance(obj, dict):
                return [obj]  # Wrap single task if returned as object
        except Exception as e:
            logger.warning(f"JSON extraction failed: {e}")

        # 6. Fallback: regex search for anything with brackets
        try:
            array_match = re.search(r"\[.*\]", text, re.DOTALL)
            if array_match:
                return json.loads(array_match.group(0))
        except Exception as e:
            logger.warning(f"JSON array extraction failed path 2: {e}")

        return []

    @staticmethod
    def parse_tool_arguments(args_str: str) -> dict[str, Any]:
        """Parses tool arguments string into a dictionary, with fallbacks."""
        try:
            return json.loads(args_str or "{}")
        except:
            try:
                return ast.literal_eval(args_str)
            except:
                return {}


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------


class PlannerState:
    def __init__(self, global_history: dict[str, Any] = None):
        self._tasks: dict[str, TaskStateModel] = {}
        self._results: dict[str, str] = {}
        self._subagent_history: dict[Any, Any] = (
            global_history if global_history is not None else {}
        )

    @property
    def tasks(self) -> dict[str, TaskStateModel]:
        return self._tasks

    @property
    def results(self) -> dict[str, str]:
        return self._results

    @property
    def subagent_history(self) -> dict[Any, Any]:
        return self._subagent_history

    def update_task(self, task_id: str, status: str, description: str = None) -> None:
        if task_id not in self._tasks:
            self._tasks[task_id] = TaskStateModel(status="pending", description="")
        self._tasks[task_id].status = status
        if description:
            self._tasks[task_id].description = description

    def store_result(self, task_id: str, result: str) -> None:
        self._results[task_id] = result

    def get_history(self, chat_id: str, sub_task_id: str, model_id: str) -> list[Any]:
        key = (chat_id, sub_task_id, model_id)
        return self._subagent_history.get(key, [])

    def set_history(
        self, chat_id: str, sub_task_id: str, model_id: str, messages: list[Any]
    ) -> None:
        key = (chat_id, sub_task_id, model_id)
        self._subagent_history[key] = messages


# ---------------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------------


class UIRenderer:
    def __init__(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
        event_call: Optional[Callable[[dict[str, Any]], Awaitable[Any]]] = None,
    ):
        self.emitter = event_emitter
        self.call = event_call

    @staticmethod
    def _base_theme_js() -> str:
        """Returns a JS snippet that reads the current OWUI theme and builds a `col` object."""
        return """
      const isDark = document.documentElement.classList.contains('dark');
      const col = isDark
        ? { bg: 'var(--color-gray-950)', panel: 'var(--color-gray-900)',
            border: 'var(--color-gray-700)', text: 'var(--color-white)',
            sub: 'var(--color-gray-400)', input: 'var(--color-gray-800)',
            inputBorder: 'var(--color-gray-600)',
            btn: 'var(--color-gray-800)', btnBorder: 'var(--color-gray-600)', btnText: 'var(--color-gray-200)',
            btnPrimary: 'var(--color-gray-100)', btnPrimaryText: 'var(--color-gray-900)',
            overlay: 'rgba(0,0,0,0.7)' }
        : { bg: 'var(--color-gray-100)', panel: 'var(--color-gray-50)',
            border: 'var(--color-gray-200)', text: 'var(--color-gray-900)',
            sub: 'var(--color-gray-500)', input: 'var(--color-white)',
            inputBorder: 'var(--color-gray-300)',
            btn: 'var(--color-gray-200)', btnBorder: 'var(--color-gray-300)', btnText: 'var(--color-gray-700)',
            btnPrimary: 'var(--color-gray-900)', btnPrimaryText: 'var(--color-white)',
            overlay: 'rgba(0,0,0,0.4)' };"""

    def build_ask_user_js(
        self,
        prompt_text: str,
        placeholder: str = "Type your response...",
        timeout_s: int = 120,
    ) -> str:
        p = json.dumps(prompt_text)
        ph = json.dumps(placeholder)
        return f"""return (function() {{
  return new Promise((resolve) => {{
{self._base_theme_js()}
    var _timer, _cd;
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
    const panel = document.createElement('div');
    panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:440px;padding:28px;display:flex;flex-direction:column;gap:20px;`;
    const titleEl = document.createElement('div'); titleEl.textContent = 'Input Required'; titleEl.style.cssText = `font-size:18px;font-weight:700;color:${{col.text}};`; panel.appendChild(titleEl);
    const msgEl = document.createElement('div'); msgEl.textContent = {p}; msgEl.style.cssText = `font-size:14px;color:${{col.sub}};line-height:1.5;`; panel.appendChild(msgEl);
    const input = document.createElement('input'); input.placeholder = {ph}; input.style.cssText = `background:${{col.input}};border:1px solid ${{col.inputBorder}};color:${{col.text}};padding:12px 16px;border-radius:12px;font-size:14px;outline:none;focus:border-blue-500;`; panel.appendChild(input);
    const countdown = document.createElement('div'); countdown.style.cssText = `font-size:12px;color:${{col.sub}};text-align:center;`; panel.appendChild(countdown);
    const footer = document.createElement('div'); footer.style.cssText = 'display:flex;gap:10px;';
    const makeBtn = (label, primary) => {{ const b = document.createElement('button'); b.textContent = label; b.style.cssText = `flex:1;padding:12px 18px;border-radius:9999px;font-size:14px;font-weight:600;cursor:pointer;border:1px solid ${{primary ? 'transparent' : col.btnBorder}};background:${{primary ? col.btnPrimary : col.btn}};color:${{primary ? col.btnPrimaryText : col.btnText}};transition:opacity 0.15s;`; b.onmouseenter = () => b.style.opacity='0.85'; b.onmouseleave = () => b.style.opacity='1'; return b; }};
    const submitBtn = makeBtn('Submit', true); const skipBtn = makeBtn('Skip', false);
    submitBtn.onclick = () => {{ if(!input.value.trim()) return; clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'accept', value: input.value.trim()}})); }};
    skipBtn.onclick = () => {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }};
    input.onkeydown = (e) => {{ if(e.key === 'Enter') submitBtn.onclick(); if(e.key === 'Escape') skipBtn.onclick(); }};
    footer.appendChild(submitBtn); footer.appendChild(skipBtn);
    panel.appendChild(footer); overlay.appendChild(panel); document.body.appendChild(overlay); input.focus();
    let remaining = {timeout_s};
    _cd = setInterval(() => {{ remaining--; countdown.textContent = `Auto-skips in ${{remaining}}s`; if(remaining <= 0) {{ clearInterval(_cd); }} }}, 1000);
    _timer = setTimeout(() => {{ clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }}, {timeout_s * 1000});
    function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
  }});
}})()"""

    def build_give_options_js(
        self,
        prompt_text: str,
        choices: list,
        context: str = "",
        timeout_s: int = 120,
        allow_custom: bool = True,
    ) -> str:
        p = json.dumps(prompt_text)
        cx = json.dumps(context)
        ch = json.dumps(choices)
        alc = "true" if allow_custom else "false"
        return f"""return (function() {{
  return new Promise((resolve) => {{
{self._base_theme_js()}
    var _timer, _cd;
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
    const panel = document.createElement('div');
    panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:480px;padding:28px;display:flex;flex-direction:column;gap:18px;`;
    const titleEl = document.createElement('div'); titleEl.textContent = {p}; titleEl.style.cssText = `font-size:18px;font-weight:700;color:${{col.text}};`; panel.appendChild(titleEl);
    const ctx = {cx};
    if (ctx) {{ const ctxEl = document.createElement('div'); ctxEl.textContent = ctx; ctxEl.style.cssText = `font-size:13px;color:${{col.sub}};line-height:1.4;`; panel.appendChild(ctxEl); }}
    
    const grid = document.createElement('div'); grid.style.cssText = 'display:flex;flex-direction:column;gap:8px;';
    const CHOICES = {ch};
    CHOICES.forEach(c => {{
      const b = document.createElement('button');
      b.textContent = c;
      b.style.cssText = `padding:12px 18px;border-radius:12px;font-size:14px;font-weight:600;cursor:pointer;border:1px solid ${{col.border}};background:${{col.btn}};color:${{col.text}};text-align:left;transition:opacity 0.15s;`;
      b.onmouseenter = () => b.style.opacity = '0.8';
      b.onmouseleave = () => b.style.opacity = '1';
      b.onclick = () => {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'accept', value: c}})); }};
      grid.appendChild(b);
    }});
    panel.appendChild(grid);

    if ({alc}) {{
      const customContainer = document.createElement('div');
      customContainer.style.cssText = 'display:flex;flex-direction:column;gap:8px;margin-top:8px;';
      const customLabel = document.createElement('div');
      customLabel.textContent = 'Other / Custom Input:';
      customLabel.style.cssText = `font-size:12px;color:${{col.sub}};font-weight:600;`;
      customContainer.appendChild(customLabel);
      
      const inputWrapper = document.createElement('div');
      inputWrapper.style.cssText = 'display:flex;gap:8px;';
      
      const customInput = document.createElement('input');
      customInput.placeholder = 'Type custom option...';
      customInput.style.cssText = `flex:1;background:${{col.input}};border:1px solid ${{col.inputBorder}};color:${{col.text}};padding:10px 14px;border-radius:10px;font-size:14px;outline:none;`;
      
      const customBtn = document.createElement('button');
      customBtn.textContent = '➔';
      customBtn.style.cssText = `padding:0 15px;border-radius:10px;background:${{col.btnPrimary}};color:${{col.btnPrimaryText}};border:none;cursor:pointer;font-weight:bold;`;
      
      customBtn.onclick = () => {{
        const val = customInput.value.trim();
        if (val) {{
          clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'accept', value: val}}));
        }}
      }};
      
      customInput.onkeydown = (e) => {{ if(e.key === 'Enter') customBtn.onclick(); }};
      
      inputWrapper.appendChild(customInput);
      inputWrapper.appendChild(customBtn);
      customContainer.appendChild(inputWrapper);
      panel.appendChild(customContainer);
    }}

    const countdown = document.createElement('div'); countdown.style.cssText = `font-size:12px;color:${{col.sub}};text-align:center;`; panel.appendChild(countdown);
    const footer = document.createElement('div'); footer.style.cssText = 'display:flex;gap:10px;margin-top:10px;';
    const makeBtn = (label) => {{ const b = document.createElement('button'); b.textContent = label; b.style.cssText = `flex:1;padding:10px 16px;border-radius:9999px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid ${{col.btnBorder}};background:${{col.btn}};color:${{col.btnText}};transition:opacity 0.15s;`; b.onmouseenter = () => b.style.opacity='0.8'; b.onmouseleave = () => b.style.opacity='1'; return b; }};
    const skipBtn = makeBtn('Skip'); const skipAllBtn = makeBtn('Skip All');
    skipBtn.onclick = () => {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }};
    skipAllBtn.onclick = () => {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'skip_all', value: ''}})); }};
    footer.appendChild(skipBtn); footer.appendChild(skipAllBtn);
    panel.appendChild(footer);
    overlay.appendChild(panel); document.body.appendChild(overlay);
    overlay.onclick = (e) => {{ if(e.target===overlay) {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }} }};
    let remaining = {timeout_s};
    _cd = setInterval(() => {{ remaining--; countdown.textContent = `Auto-skips in ${{remaining}}s`; if(remaining <= 0) {{ clearInterval(_cd); }} }}, 1000);
    _timer = setTimeout(() => {{ clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }}, {timeout_s * 1000});
    function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
  }});
}})()"""

    def build_continue_cancel_js(self, context_msg: str, timeout_s: int = 300) -> str:
        msg = json.dumps(context_msg)
        return f"""return (function() {{
  return new Promise((resolve) => {{
{self._base_theme_js()}
    var _timer, _cd;
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
    const panel = document.createElement('div');
    panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:440px;padding:28px;display:flex;flex-direction:column;gap:20px;text-align:center;`;
    const icon = document.createElement('div'); icon.textContent = '⏱️'; icon.style.cssText = 'font-size:36px;'; panel.appendChild(icon);
    const titleEl = document.createElement('div'); titleEl.textContent = 'Iteration Limit Reached'; titleEl.style.cssText = `font-size:18px;font-weight:700;color:${{col.text}};`; panel.appendChild(titleEl);
    const msgEl = document.createElement('div'); msgEl.textContent = {msg}; msgEl.style.cssText = `font-size:14px;color:${{col.sub}};line-height:1.5;`; panel.appendChild(msgEl);
    const countdown = document.createElement('div'); countdown.style.cssText = `font-size:12px;color:${{col.sub}};`; panel.appendChild(countdown);
    const footer = document.createElement('div'); footer.style.cssText = 'display:flex;gap:10px;';
    const makeBtn = (label, primary) => {{ const b = document.createElement('button'); b.textContent = label; b.style.cssText = `flex:1;padding:12px 18px;border-radius:9999px;font-size:14px;font-weight:600;cursor:pointer;border:1px solid ${{primary ? 'transparent' : col.btnBorder}};background:${{primary ? col.btnPrimary : col.btn}};color:${{primary ? col.btnPrimaryText : col.btnText}};transition:opacity 0.15s;`; b.onmouseenter = () => b.style.opacity='0.85'; b.onmouseleave = () => b.style.opacity='1'; return b; }};
    const continueBtn = makeBtn('Continue', true); const cancelBtn = makeBtn('Cancel', false);
    continueBtn.onclick = () => {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'continue'}})); }};
    cancelBtn.onclick = () => {{ clearTimeout(_timer); clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'cancel'}})); }};
    footer.appendChild(continueBtn); footer.appendChild(cancelBtn);
    panel.appendChild(footer); overlay.appendChild(panel); document.body.appendChild(overlay);
    let remaining = {timeout_s};
    _cd = setInterval(() => {{ remaining--; countdown.textContent = `Auto-cancels in ${{remaining}}s`; if(remaining <= 0) {{ clearInterval(_cd); }} }}, 1000);
    _timer = setTimeout(() => {{ clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'cancel'}})); }}, {timeout_s * 1000});
    function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
  }});
}})()"""

    def build_plan_approval_js(self, tasks: list, timeout_s: int = 600) -> str:
        ts = json.dumps(tasks)
        return f"""
    return (function() {{
      return new Promise((resolve) => {{
    {self._base_theme_js()}
        let _timer;
        const overlay = document.createElement('div');
        overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
        const panel = document.createElement('div');
        panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:520px;max-height:90vh;padding:32px;display:flex;flex-direction:column;gap:24px;`;
        
        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;gap:12px;flex-shrink:0;';
        const icon = document.createElement('div'); icon.textContent = '📋'; icon.style.cssText = 'font-size:24px;';
        const title = document.createElement('div'); title.textContent = 'Review Proposed Plan'; title.style.cssText = `font-size:20px;font-weight:800;color:${{col.text}};letter-spacing:-0.4px;`;
        header.appendChild(icon); header.appendChild(title); panel.appendChild(header);

        const scrollContainer = document.createElement('div');
        scrollContainer.style.cssText = 'overflow-y:auto;flex:1;display:flex;flex-direction:column;gap:12px;padding-right:8px;';
        
        const tasksData = {ts};
        tasksData.forEach((t, i) => {{
            const card = document.createElement('div');
            card.style.cssText = `background:${{col.input}};border:1px solid ${{col.inputBorder}};border-radius:12px;padding:12px 16px;display:flex;gap:12px;align-items:flex-start;`;
            
            const num = document.createElement('div');
            num.textContent = i + 1;
            num.style.cssText = `width:24px;height:24px;background:${{col.btnPrimary}};color:${{col.btnPrimaryText}};border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:bold;flex-shrink:0;margin-top:2px;`;
            
            const content = document.createElement('div');
            content.style.cssText = 'display:flex;flex-direction:column;gap:4px;';
            const tid = document.createElement('div'); tid.textContent = t.task_id; tid.style.cssText = `font-size:11px;font-weight:bold;color:${{col.sub}};text-transform:uppercase;`;
            const desc = document.createElement('div'); desc.textContent = t.description; desc.style.cssText = `font-size:14px;color:${{col.text}};line-height:1.4;`;
            
            content.appendChild(tid); content.appendChild(desc);
            card.appendChild(num); card.appendChild(content);
            scrollContainer.appendChild(card);
        }});
        panel.appendChild(scrollContainer);

        const inputContainer = document.createElement('div');
        inputContainer.style.cssText = 'display:flex;flex-direction:column;gap:10px;flex-shrink:0;';
        const inputLabel = document.createElement('div'); inputLabel.textContent = 'Feedback (optional):'; inputLabel.style.cssText = `font-size:12px;font-weight:700;color:${{col.sub}};text-transform:uppercase;letter-spacing:0.5px;`;
        const feedbackInput = document.createElement('textarea');
        feedbackInput.placeholder = 'e.g., "Add a step to check for X" or "Skip the second task"';
        feedbackInput.style.cssText = `background:${{col.input}};border:1px solid ${{col.inputBorder}};color:${{col.text}};padding:14px;border-radius:14px;font-size:14px;outline:none;min-height:70px;resize:none;transition:border-color 0.2s;`;
        feedbackInput.onfocus = () => feedbackInput.style.borderColor = 'var(--color-blue-500)';
        feedbackInput.onblur = () => feedbackInput.style.borderColor = col.inputBorder;
        inputContainer.appendChild(inputLabel); inputContainer.appendChild(feedbackInput); panel.appendChild(inputContainer);

        const footer = document.createElement('div');
        footer.style.cssText = 'display:flex;gap:12px;flex-shrink:0;';
        
        const makeBtn = (label, primary) => {{
            const b = document.createElement('button');
            b.textContent = label;
            b.style.cssText = `flex:1;padding:14px 20px;border-radius:9999px;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.2s;border:1px solid ${{primary ? 'transparent' : col.btnBorder}};background:${{primary ? col.btnPrimary : col.btn}};color:${{primary ? col.btnPrimaryText : col.btnText}};`;
            b.onmouseenter = () => {{ b.style.opacity='0.9'; b.style.transform='translateY(-1px)'; }};
            b.onmouseleave = () => {{ b.style.opacity='1'; b.style.transform='translateY(0)'; }};
            return b;
        }};

        const acceptBtn = makeBtn('Accept Plan', true);
        const feedbackBtn = makeBtn('Send Feedback', false);

        acceptBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'accept'}})); }};
        feedbackBtn.onclick = () => {{
            const val = feedbackInput.value.trim();
            if (val) {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'feedback', value: val}})); }}
            else {{ acceptBtn.onclick(); }}
        }};

        footer.appendChild(acceptBtn); footer.appendChild(feedbackBtn); panel.appendChild(footer);

        const countdown = document.createElement('div');
        countdown.style.cssText = `font-size:11px;color:${{col.sub}};text-align:center;margin-top:-12px;flex-shrink:0;`;
        panel.appendChild(countdown);

        overlay.appendChild(panel); document.body.appendChild(overlay);
        feedbackInput.focus();

        let remaining = {timeout_s};
        const _cd = setInterval(() => {{
            remaining--;
            countdown.textContent = `Auto-accepting in ${{remaining}}s`;
            if(remaining <= 0) {{ clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'accept'}})); }}
        }}, 1000);

        _timer = setTimeout(() => {{ clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'accept'}})); }}, {timeout_s * 1000});

        function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
      }});
    }})()"""

    async def emit_status(self, message: str, done: bool = False) -> None:
        await self.emitter(
            {"type": "status", "data": {"description": message, "done": done}}
        )

    async def emit_replace(self, content: str) -> None:
        await self.emitter({"type": "replace", "data": {"content": content}})

    def build_tool_call_details(
        self,
        call_id: str,
        name: str,
        arguments: str,
        done: bool = False,
        result: Any = None,
    ) -> str:
        """Constructs the HTML for a tool call to be embedded in the chat message (v3 parity)."""
        args_escaped = html_module.escape(arguments)
        if done:
            result_text = (
                result
                if isinstance(result, str)
                else json.dumps(result or "", ensure_ascii=False)
            )
            result_escaped = html_module.escape(
                json.dumps(result_text, ensure_ascii=False)
            )
            return (
                f'<details type="tool_calls" done="true" id="{call_id}" '
                f'name="{name}" arguments="{args_escaped}" '
                f'result="{result_escaped}">\n'
                f"<summary>Tool Executed</summary>\n</details>\n"
            )
        return (
            f'<details type="tool_calls" done="false" id="{call_id}" '
            f'name="{name}" arguments="{args_escaped}">\n'
            f"<summary>Executing...</summary>\n</details>\n"
        )

    async def emit_html_embed(self, planner_state: dict[str, Any]) -> None:
        html = self._generate_html_embed(planner_state)
        await self.emitter({"type": "embeds", "data": {"embeds": [html]}})

    def _generate_html_embed(self, planner_state: dict[str, Any]) -> str:


        embed_id = "pe-" + hashlib.md5(str(time.monotonic()).encode()).hexdigest()[:8]

        status_colors = {
            "pending": "#9ca3af",
            "in_progress": "#60a5fa",
            "completed": "#10b981",
            "failed": "#ef4444",
        }
        check_icon = {
            "pending": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle></svg>',
            "in_progress": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
            "completed": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
            "failed": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
        }

        tasks_html = ""
        for task_id, task_info in planner_state.items():
            # Use property access for TaskStateModel
            status = getattr(task_info, "status", "pending")
            safe_tid = html_module.escape(task_id)
            safe_desc = html_module.escape(getattr(task_info, "description", ""))
            color = status_colors.get(status, status_colors["pending"])
            icon = check_icon.get(status, "")
            tasks_html += f"""
            <div class="pe-card" style="margin-bottom:12px;padding:16px;border-left:4px solid {color};border-radius:12px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                    <div style="display:flex;align-items:center;gap:8px;color:{color};">
                        {icon}
                        <strong class="pe-title" style="font-size:14px;font-weight:600;letter-spacing:0.3px;">{safe_tid}</strong>
                    </div>
                    <span style="font-size:10px;font-weight:700;padding:4px 10px;border-radius:99px;text-transform:uppercase;letter-spacing:1px;color:{color};background:rgba(128,128,128,0.15);">
                        {status.replace("_", " ")}
                    </span>
                </div>
                <div class="pe-desc" style="font-size:13px;line-height:1.5;font-weight:400;padding-left:26px;">
                    {safe_desc}
                </div>
            </div>"""

        if not tasks_html:
            tasks_html = '<div class="pe-empty" style="padding:16px;text-align:center;font-size:13px;font-style:italic;border-radius:12px;border:1px dashed rgba(128,128,128,0.3);">Formulating Plan...</div>'

        theme_script = f"""<script>
(function(){{
  var ID='{embed_id}';
  function rd(){{try{{return window.parent.document;}}catch(e){{return document;}}}}
  function parseRgb(s){{
    var m=(s||'').match(/rgba?[(]([0-9]+)[, ]+([0-9]+)[, ]+([0-9]+)/);
    return m?[+m[1],+m[2],+m[3]]:null;
  }}
  function clamp(v){{return Math.max(0,Math.min(255,Math.round(v)));}}
  function adj(c,n){{return[clamp(c[0]+n),clamp(c[1]+n),clamp(c[2]+n)];}}
  function rgb(c){{return'rgb('+c[0]+','+c[1]+','+c[2]+')';}}
  function luma(c){{return(0.299*c[0]+0.587*c[1]+0.114*c[2])/255;}}
  function applyTheme(){{
    var el=document.getElementById(ID);
    if(!el)return;
    var r=rd();
    var bodyBg=window.parent.getComputedStyle(r.body).backgroundColor;
    var base=parseRgb(bodyBg);
    if(!base||luma(base)===0&&bodyBg.indexOf('rgba')>-1){{
      bodyBg=window.parent.getComputedStyle(r.documentElement).backgroundColor;
      base=parseRgb(bodyBg);
    }}
    if(!base)base=[17,24,39];
    var dark=luma(base)<0.5;
    var outerBg =rgb(adj(base, dark?10:-6));
    var cardBg  =rgb(adj(base, dark?22:-14));
    var borderC =rgb(adj(base, dark?38:-24));
    var titleC  =dark?'#f1f5f9':'#0f172a';
    var subC    =dark?'#94a3b8':'#64748b';
    var descC   =dark?'#cbd5e1':'#475569';
    el.style.background=outerBg;
    el.style.borderColor=borderC;
    el.style.boxShadow=dark?'0 4px 20px rgba(0,0,0,0.5)':'0 4px 12px rgba(0,0,0,0.1)';
    var h3=el.querySelector('h3');if(h3)h3.style.color=titleC;
    var sub=el.querySelector('.pe-subtitle');if(sub)sub.style.color=subC;
    el.querySelectorAll('.pe-card').forEach(function(c){{c.style.background=cardBg;}});
    el.querySelectorAll('.pe-title').forEach(function(t){{t.style.color=titleC;}});
    el.querySelectorAll('.pe-desc').forEach(function(d){{d.style.color=descC;}});
    el.querySelectorAll('.pe-empty').forEach(function(e){{
      e.style.color=subC;
      e.style.borderColor=rgb(adj(base,dark?45:-30));
    }});
  }}
  applyTheme();
  setTimeout(applyTheme,150);
  setTimeout(applyTheme,600);
  try{{
    var r=rd();
    var obs=new MutationObserver(applyTheme);
    obs.observe(r.documentElement,{{attributes:true,attributeFilter:['class','style','data-theme']}});
    if(r.body)obs.observe(r.body,{{attributes:true,attributeFilter:['class','style']}});
  }}catch(e){{}}
}})();
</script>"""

        html = (
            "<style>html,body{margin:0;padding:0;background:transparent!important}</style>\n"
            f'<div id="{embed_id}" style="background:#1e293b;border:1px solid #334155;'
            "border-radius:20px;padding:28px;margin:6px;"
            "font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;"
            'box-shadow:0 4px 20px rgba(0,0,0,0.5);">\n'
            '  <div style="display:flex;flex-direction:column;align-items:center;'
            'text-align:center;gap:12px;margin-bottom:24px;">\n'
            '    <div style="font-size:32px;">🧠</div>\n'
            "    <div>\n"
            '      <h3 style="margin:0;color:#f1f5f9;font-size:18px;font-weight:800;'
            'letter-spacing:-0.2px;">Planner Subagents</h3>\n'
            '      <p class="pe-subtitle" style="margin:4px 0 0 0;font-size:12px;'
            'color:#94a3b8;font-weight:500;">Live Execution State</p>\n'
            "    </div>\n"
            "  </div>\n"
            f'  <div style="display:flex;flex-direction:column;gap:4px;">\n    {tasks_html}\n  </div>\n'
            f"</div>\n{theme_script}"
        )
        return html


# ---------------------------------------------------------------------------
# Tool Management
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(
        self,
        valves: Any,
        user: Any,
        request: Any = None,
        pipe_metadata: dict[str, Any] = None,
        model_knowledge: Optional[list[dict]] = None,
        planner_features: Optional[dict[str, Any]] = None,
    ):
        self.valves = valves
        self.user = user
        self.request = request
        self.pipe_metadata = pipe_metadata or {}
        self.model_knowledge = model_knowledge
        self.planner_features = planner_features or {}
        self.subagent_tools_cache = {}

    def get_filtered_builtin_tools(
        self,
        valves: Any,
        user_valves: Any,
        model_info: dict[str, Any],
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolves built-in tools for the planner, filtering out those handled by subagents.
        Only adds tools if they were originally present in the planner model's features.
        """
        if not self.planner_features:
            return {}

        active_features = {}
        # Mapping of feature to subagent enable valve
        feature_map = {
            "web_search": "ENABLE_WEB_SEARCH_AGENT",
            "image_generation": "ENABLE_IMAGE_GENERATION_AGENT",
            "code_interpreter": "ENABLE_CODE_INTERPRETER_AGENT",
            "knowledge": "ENABLE_KNOWLEDGE_AGENT",
        }

        for feature, valve_name in feature_map.items():
            # If the feature was originally present in the planner model
            if self.planner_features.get(feature):
                # If the corresponding subagent is NOT enabled, the planner keeps the tool
                if not getattr(valves, valve_name, True):
                    active_features[feature] = True
                # Special case for knowledge: rule says only to planner if not present in subagent
                if feature == "knowledge" and not getattr(
                    valves, "ENABLE_KNOWLEDGE_AGENT", True
                ):
                    active_features["knowledge"] = True

        if not active_features:
            return {}

        try:
            return get_builtin_tools(
                self.request,
                extra_params,
                features=active_features,
                model=model_info,
            )
        except Exception as e:
            logger.error(f"Failed to load filtered built-in tools: {e}")
            return {}

    def get_tools_spec(
        self, user: Any, user_valves: Any, available_tasks: list[Any] = None
    ) -> list[dict[str, Any]]:
        plan_mode = user_valves.PLAN_MODE
        truncation = user_valves.TASK_RESULT_TRUNCATION
        user_input_enabled = user_valves.ENABLE_USER_INPUT_TOOLS
        # 1. Virtual Subagents: Always included if enabled in valves
        virtual_model_ids = []
        if getattr(self.valves, "ENABLE_IMAGE_GENERATION_AGENT", True):
            virtual_model_ids.append("image_gen_agent")
        if getattr(self.valves, "ENABLE_WEB_SEARCH_AGENT", True):
            virtual_model_ids.append("web_search_agent")
        if getattr(self.valves, "ENABLE_KNOWLEDGE_AGENT", True):
            virtual_model_ids.append("knowledge_agent")
        if getattr(self.valves, "ENABLE_CODE_INTERPRETER_AGENT", True):
            virtual_model_ids.append("code_interpreter_agent")

        # Terminal agent requires BOTH enablement and a terminal session
        if getattr(
            self.valves, "ENABLE_TERMINAL_AGENT", True
        ) and self.pipe_metadata.get("terminal_id"):
            virtual_model_ids.append("terminal_agent")

        # 2. Extra Subagents: from SUBAGENT_MODELS list
        subagents_list = (
            self.valves.SUBAGENT_MODELS.split(",")
            if self.valves.SUBAGENT_MODELS
            else []
        )

        # Merge, unique, preserve order
        final_subagents = []
        for vid in virtual_model_ids:
            if vid not in final_subagents:
                final_subagents.append(vid)
        for sid in subagents_list:
            sid = sid.strip()
            if sid and sid not in final_subagents:
                final_subagents.append(sid)

        subagents_list = final_subagents

        tools_spec = []

        # schemas for IDs (no longer using enum to allow dynamic task creation)
        task_id_schema = {
            "type": "string",
            "description": "ID identifying the conversation thread (e.g. 'task_1').",
        }

        if plan_mode:
            tools_spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": "update_state",
                        "description": "Add a new task to the plan or modify an existing one's status or description.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": task_id_schema,
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                        "failed",
                                    ],
                                    "description": "New status for the task. Use 'pending' or 'in_progress' to roll back, or 'completed'/'failed' for manual marking.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of the task. Required when adding a new task, optional when updating.",
                                },
                            },
                            "required": ["task_id", "status"],
                        },
                    },
                }
            )

        # call_subagent: use generic task_id schema
        sub_task_id_schema = {
            "type": "string",
            "description": "ID identifying the conversation thread (e.g. 'task_1').",
        }

        tools_spec.append(
            {
                "type": "function",
                "function": {
                    "name": "call_subagent",
                    "description": "Call a specialized subagent model to perform a task. Returns the output from the model. Using the same task_id continues the same conversation thread (thread persistence).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "description": "The ID of the model to use",
                                "enum": (
                                    subagents_list if subagents_list else ["__none__"]
                                ),
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Detailed instructions for the subagent. Use '@task_id' (e.g. '@task_1') as a TEXT REPLACEMENT MACRO — it will be automatically replaced with the LAST complete subagent response text for that task_id.",
                            },
                            "task_id": sub_task_id_schema,
                            "related_tasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of previously completed Task IDs whose results you need contextually available to the subagent.",
                            },
                        },
                        "required": ["model_id", "prompt", "task_id"],
                    },
                },
            }
        )

        if truncation:
            tools_spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_task_result",
                        "description": "Read the FULL, untruncated raw text result of a completed task verbatim. Use this when the result shown in the call_subagent response was truncated.",
                        "parameters": {
                            "type": "object",
                            "properties": {"task_id": task_id_schema},
                            "required": ["task_id"],
                        },
                    },
                }
            )

        # review_tasks
        tools_spec.append(
            {
                "type": "function",
                "function": {
                    "name": "review_tasks",
                    "description": "Spawn an invisible LLM cross-review over massive task results using a custom prompt, saving your own context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_ids": {"type": "array", "items": {"type": "string"}},
                            "prompt": {
                                "type": "string",
                                "description": "Instructions on what to review or extract from these given task IDs",
                            },
                            "review_id": {
                                "type": "string",
                                "description": "Optional virtual ID to reference this review in subsequent tasks (e.g. 'review_1'). Macros @review_1 will be available.",
                            },
                        },
                        "required": ["task_ids", "prompt"],
                    },
                },
            }
        )

        if user_input_enabled:
            tools_spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "description": "Ask the user for free-form text input. Returns the text, or a skip/skip-all sentinel.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt_text": {
                                    "type": "string",
                                    "description": "The question or request to present to the user",
                                },
                                "placeholder": {
                                    "type": "string",
                                    "description": "Optional hint text for the input field",
                                },
                            },
                            "required": ["prompt_text"],
                        },
                    },
                }
            )
            tools_spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": "give_options",
                        "description": "Present the user with a list of choices and wait for their selection. Returns the chosen option, or a skip/skip-all sentinel.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt_text": {
                                    "type": "string",
                                    "description": "The question or prompt to display",
                                },
                                "choices": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of options to present to the user",
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Optional background context to show beneath the title",
                                },
                                "allow_custom": {
                                    "type": "boolean",
                                    "description": "Whether to allow the user to provide a custom 'Other' input. Defaults to true.",
                                },
                            },
                            "required": ["prompt_text", "choices"],
                        },
                    },
                }
            )

        return tools_spec

    async def get_planner_tools_dict(
        self, body: dict[str, Any], user_valves: Any
    ) -> dict[str, Any]:
        """Resolve external tools and filtered built-in tools for the planner."""
        tool_ids = body.get("metadata", {}).get("toolIds", [])
        extra_params = {
            "chat_id": self.pipe_metadata.get("chat_id") or body.get("chat_id"),
            "tool_ids": tool_ids,
        }

        # 1. Resolve external tools
        tools_dict = {}
        if tool_ids:
            tools_dict = await get_tools(self.request, tool_ids, self.user, extra_params)

        # 2. Resolve filtered built-in tools
        # We need model_info for get_builtin_tools model context
        app_models = getattr(self.request.app.state, "MODELS", {})
        planner_model_id = self.valves.PLANNER_MODEL
        planner_info = app_models.get(planner_model_id, {})

        builtin_tools = self.get_filtered_builtin_tools(
            self.valves, user_valves, planner_info, extra_params
        )
        if builtin_tools:
            tools_dict.update(builtin_tools)

        return tools_dict

    async def get_subagent_tools(
        self,
        model_id: str,
        virtual_agents: dict[str, AgentDefinition],
        app_models: dict[str, Any],
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Fetches tools for subagents, ensuring builtin tools (image gen, search, etc.) are loaded for virtual agents."""
        if model_id in self.subagent_tools_cache:
            return self.subagent_tools_cache[model_id]

        # 1. Virtual Agents Handling
        if model_id in virtual_agents:
            va = virtual_agents[model_id]
            va_model_id = va.model_id

            # Object resolution: Dict-first, then DB-fallback (parity with v3 better robustness)
            va_runtime_info = app_models.get(va_model_id)
            va_db_info = Models.get_model_by_id(va_model_id)

            va_model_obj = va_runtime_info or va_db_info

            s_tools_dict = {}
            if va.type == "terminal":
                try:
                    terminal_id = self.pipe_metadata.get("terminal_id")
                    s_tools_dict = await get_terminal_tools(
                        self.request, terminal_id, self.user, extra_params
                    )
                except Exception as e:
                    logger.error(f"Failed to load terminal tools for {model_id}: {e}")
            elif va.type == "builtin":
                # Important: merge the va override with the actual model info to ensure id/name exist
                builtin_model = va.builtin_model_override or {}
                if not builtin_model and va_model_obj:
                    builtin_model = va_model_obj
                elif builtin_model and va_model_obj:
                    # If we have both, we ensure the ID is carried over
                    if hasattr(va_model_obj, "id"):
                        builtin_model["id"] = va_model_obj.id
                    elif isinstance(va_model_obj, dict):
                        builtin_model["id"] = va_model_obj.get("id", va_model_id)

                try:
                    # Use features from va config and the merged model
                    s_tools_dict = get_builtin_tools(
                        self.request,
                        extra_params,
                        features=va.features,
                        model=builtin_model,
                    )
                    logger.info(
                        f"Loaded {len(s_tools_dict)} builtin tools for subagent {model_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load builtin tools for {model_id}: {e}")

            s_tools = (
                [
                    {"type": "function", "function": t["spec"]}
                    for t in s_tools_dict.values()
                ]
                if s_tools_dict
                else None
            )
            result = {
                "dict": s_tools_dict,
                "specs": s_tools,
                "system_message": va.system_message,
                "actual_model": va_model_id,
                "temperature_override": va.temperature,
            }
            self.subagent_tools_cache[model_id] = result
            return result

        # Regular subagent logic
        model_info = app_models.get(model_id, {})
        model_db_info = Models.get_model_by_id(model_id)
        model_system_message = ""
        subagent_tool_ids = []
        p_features = {}

        if model_db_info:
            meta = (
                model_db_info.meta.model_dump()
                if hasattr(model_db_info.meta, "model_dump")
                else model_db_info.meta
            )
            params = (
                model_db_info.params.model_dump()
                if hasattr(model_db_info.params, "model_dump")
                else model_db_info.params
            )

            if isinstance(meta, dict):
                subagent_tool_ids.extend(meta.get("toolIds", []))
                if isinstance(meta.get("features"), dict):
                    p_features.update(meta["features"])

            if isinstance(params, dict):
                subagent_tool_ids.extend(
                    params.get("toolIds", []) or params.get("tools", [])
                )
                model_system_message = params.get("system", "") or ""
                if isinstance(params.get("features"), dict):
                    p_features.update(params["features"])

        if model_info:
            info_meta = model_info.get("info", {}).get("meta", {})
            info_params = model_info.get("info", {}).get("params", {})
            subagent_tool_ids.extend(info_meta.get("toolIds", []))
            subagent_tool_ids.extend(
                info_params.get("toolIds", []) or info_params.get("tools", [])
            )
            if not model_system_message:
                model_system_message = info_params.get("system", "") or ""
            if isinstance(info_meta.get("features"), dict):
                p_features.update(info_meta["features"])
            if isinstance(info_params.get("features"), dict):
                p_features.update(info_params["features"])

        s_tools_dict = {}
        if subagent_tool_ids:
            s_tools_dict = await get_tools(
                self.request, list(set(subagent_tool_ids)), self.user, extra_params
            )

        # v3 parity: Inject builtin tools for regular subagents
        try:
            builtin_tools = get_builtin_tools(
                self.request, extra_params, features=p_features, model=model_info
            )
            if builtin_tools:
                s_tools_dict.update(builtin_tools)
        except Exception as e:
            logger.error(f"Failed to load built-in tools for subagent {model_id}: {e}")

        result = {
            "dict": s_tools_dict,
            "specs": [
                {"type": "function", "function": t["spec"]}
                for t in s_tools_dict.values()
            ],
            "system_message": model_system_message,
        }
        self.subagent_tools_cache[model_id] = result
        return result


class PromptBuilder:
    @staticmethod
    def build_subagent_check_prompt(
        task_id: str,
        model_id: str,
        response_text: str,
        task_description: str = "",
        planner_input: str = "",
    ) -> str:
        return f"""You are a quality control judge for an AI subagent ({model_id}) fulfilling a specific task ({task_id}).
Your goal is to verify if the subagent's FINAL response is complete, accurate, and correctly references all generated assets.

### CONTEXT:
- **Planner's Goal/Input for Subagent**: {planner_input}
- **Task Description**: {task_description}

### CRITERIA:
1. **Task Completion**: Does the response actually answer the planner's prompt or fulfill the assigned task?
2. **Asset Referencing**: If the subagent generated images, files, or used search tools, are the URLs, paths, or source links explicitly provided in the response? (Crucial for the main planner to see them).
3. **Clarity**: Is the response well-formatted and easy for the main planner to synthesize?

### RESPONSE TO VERIFY:
---
{response_text}
---

### INSTRUCTIONS:
- You MUST return a JSON object with two fields: "action" and "feedback".
- If the response is EXCELLENT, set "action" to "approve" and "feedback" to "Lacks nothing.".
- If the response is LACKING, set "action" to "redo" and provide a detailed instruction in "feedback".
- The feedback for redo MUST be explicit: "REDO: [Reasons]. Please provide your FULL final response again, including all required links and fixes."
- Be strict but fair. Do not ask for prose if the task was technical.
"""

    @staticmethod
    def build_system_prompt(
        valves: Any,
        user_valves: Any,
        tools_spec: list,
        metadata: dict = None,
        mode: str = "execute",
        messages: list = None,
    ) -> str:
        """Construct the full system prompt with tools and mandatory rules dynamically."""
        full_prompt = valves.SYSTEM_PROMPT
        plan_mode = user_valves.PLAN_MODE
        truncation = user_valves.TASK_RESULT_TRUNCATION
        user_input = user_input_enabled = user_valves.ENABLE_USER_INPUT_TOOLS
        subagents_list = (
            valves.SUBAGENT_MODELS.split(",") if valves.SUBAGENT_MODELS else []
        )

        metadata = metadata or {}
        pipe_meta = metadata.get("__metadata__", {})
        terminal_id = pipe_meta.get("terminal_id")

        # Follow-up Detection (v3.3 pattern)
        # We consider it a follow-up if there are more than 1 user messages in the history.
        is_follow_up = False
        if messages:
            user_msg_count = sum(1 for m in messages if m.get("role") == "user")
            if user_msg_count > 1:
                is_follow_up = True

        # UI Parity: Always doc virtual agents if enabled in valves, regardless of model features
        if getattr(valves, "ENABLE_IMAGE_GENERATION_AGENT", True):
            if "image_gen_agent" not in subagents_list:
                subagents_list.append("image_gen_agent")
        if getattr(valves, "ENABLE_WEB_SEARCH_AGENT", True):
            if "web_search_agent" not in subagents_list:
                subagents_list.append("web_search_agent")
        if getattr(valves, "ENABLE_KNOWLEDGE_AGENT", True):
            if "knowledge_agent" not in subagents_list:
                subagents_list.append("knowledge_agent")
        if getattr(valves, "ENABLE_CODE_INTERPRETER_AGENT", True):
            if "code_interpreter_agent" not in subagents_list:
                subagents_list.append("code_interpreter_agent")
        if getattr(valves, "ENABLE_TERMINAL_AGENT", True) and terminal_id:
            if "terminal_agent" not in subagents_list:
                subagents_list.append("terminal_agent")

        # Build subagent descriptions from the list
        subagent_descriptions = ""
        # Virtual Agent descriptions (v3 parity)
        va_descs = {
            "image_gen_agent": "- ID: image_gen_agent (Name: Image Generation Agent)\n  Description: Built-in image generation and editing subagent. Can generate and edit images from text prompts. Always return the image URLs or file paths in your final response so the planner can use them.",
            "web_search_agent": "- ID: web_search_agent (Name: Web Search Agent)\n  Description: Built-in web search and research subagent. Can search the web for information and fetch content from URLs. Synthesize and return the relevant information clearly in your response.",
            "knowledge_agent": "- ID: knowledge_agent (Name: Knowledge Agent)\n  Description: Built-in knowledge, notes, chat history, and user memory retrieval subagent. Can search and read notes, knowledge bases, user memory, and past conversations.",
            "code_interpreter_agent": "- ID: code_interpreter_agent (Name: Code Interpreter Agent)\n  Description: Built-in code interpreter subagent. Can generate content in ANY language (HTML, CSS, JS, Python, shell scripts, JSON, etc.) and execute Python in a sandboxed Jupyter environment. Use this for ALL coding, scripting, content generation, and computation tasks.",
            "terminal_agent": "- ID: terminal_agent (Name: Terminal Agent)\n  Description: Built-in terminal subagent. Can execute commands, read/write files, and interact with the system terminal. Use this for system-level tasks, and file manipulation.",
        }

        desc_list = []
        for m in subagents_list:
            m = m.strip()
            if not m:
                continue
            if m in va_descs:
                desc_list.append(va_descs[m])
            else:
                desc_list.append(f"- ID: {m}")

        if desc_list:
            subagent_descriptions = "\n".join(desc_list)
        else:
            subagent_descriptions = "None configured."

        tools_doc = ""
        if plan_mode:
            tools_doc = (
                "1. `update_state(task_id: str, status: str, description: str)`: Add a new task to the plan or modify an existing one. Use this to:\n"
                "   - **Add Tasks**: If you discover new subgoals during execution.\n"
                "   - **Rollback or Retry**: Move a task back to 'pending' or 'in_progress' if a retry is needed or if a subagent failed but can be corrected.\n"
                "   - **Manual Completion**: Mark a task as 'completed' or 'failed' ONLY when the task did NOT involve calling a subagent. `call_subagent` already handles state transitions automatically.\n"
                "   - **Avoid Redundancy**: Do NOT call `update_state` to set 'in_progress' or 'completed' for tasks you are delegating via `call_subagent`; observe the 'status' field in the tool response instead.\n"
                "   - **Constraints**: Always provide a `description` when adding a new `task_id`. For updates, the `description` is optional.\n"
                "2. `call_subagent(model_id: str, prompt: str, task_id: str, related_tasks: list[str])`: Use this to delegate a subtask to a specialized model.\n"
                "   - **Task Status Lifecycle**: Starting a `call_subagent` will automatically set the task to 'in_progress' ONLY if it is currently 'pending' or 'failed'. Success automatically marks the task as 'completed'. Check the `status` field in the response.\n"
                "   - **Threading & Context**: The `task_id` identifies the conversation thread with the subagent. To **continue or follow up** on a previous interaction, you MUST use the **same** `task_id`. To start a **fresh** conversation, use a **new** `task_id`.\n"
                "   - **@task_id Text Replacement**: When you write `@task_id` (e.g., `@task_1`) in a **prompt** or your **final response**, it will be **automatically replaced** with the LAST complete subagent response text for that task_id.\n"
                "   - **Raw Task ID (no @)**: Use the plain ID (`task_1`) in tool parameters. NEVER prefix with @ in parameter fields.\n"
                "   - **CRITICAL — `related_tasks` for cross-task data passing**: Subagents are ISOLATED — they CANNOT see any other task's results unless you explicitly pass them. When a subagent needs data produced by a DIFFERENT task, you MUST list that task's raw ID in the `related_tasks` array.\n"
            )
            tool_idx = 3
        else:
            tools_doc = (
                "1. `call_subagent(model_id: str, prompt: str, task_id: str, related_tasks: list[str])`: Use this to delegate a subtask to a specialized model.\n"
                "   - **Threading & Context**: The `task_id` identifies the conversation thread with the subagent. To **continue or follow up** on a previous interaction, use the **same** `task_id`. To start a **fresh** conversation, use a **new** `task_id`.\n"
                "   - **@task_id Text Replacement**: When you write `@task_id` (e.g., `@task_1`) in a **prompt** or your **final response**, it will be **automatically replaced** with the LAST complete subagent response text for that task_id.\n"
                "   - **Raw Task ID (no @)**: Use the plain ID (`task_1`) in tool parameters. NEVER prefix with @ in parameter fields.\n"
                "   - **CRITICAL — `related_tasks` for cross-task data passing**: Subagents are ISOLATED — they CANNOT see any other task's results unless you explicitly pass them. When a subagent needs data produced by a DIFFERENT task, you MUST list that task's raw ID in the `related_tasks` array.\n"
            )
            tool_idx = 2

        if truncation:
            tools_doc += f"{tool_idx}. `read_task_result(task_id: str)`: Read the FULL, untruncated raw text result of a completed task verbatim. Use this when the result shown in the call_subagent response was truncated.\n"
            tool_idx += 1

        tools_doc += f"{tool_idx}. `review_tasks(task_ids: list, prompt: str)`: Spawn a specialized LLM reviewer to evaluate and synthesize results from multiple tasks using a custom prompt.\n"

        mandatory_rules = (
            "### MANDATORY RULES:\n"
            "- Delegate work to subagents using `call_subagent` for complex analysis, generation, or reasoning.\n"
            "- **CODING RULE**: For COMPLEX coding, scripting, calculation, or data-processing task, delegate to a `code_interpreter_agent` or equivalent if available and the task is complex. NEVER use web_search_agent or knowledge_agent or any unrelated subagent for code. ALWAYS provide FULL code in the final output, either by writing it yourself or using @task_id substitution tags.\n"
            "- **ALWAYS pass `related_tasks`** when a subagent needs results from previous tasks.\n"
            "- Use `@task_id` references in your final response to include large previous outputs.\n"
            "- Final Output is YOUR responsibility. Make it look good.\n"
            "- If any subagent response is incomplete, missing details, or lacks required assets, you MUST follow up with additional subagent calls or clarifying prompts until the answer is complete and all requirements are met. Using the same task id makes sure the subagent can continue from where it left off.\n"
            "- For any assets (images, files, data, etc.), ensure that links, relative/absolute paths, or URLs are always provided in the output so downstream consumers can access them.\n"
        )
        if truncation:
            mandatory_rules += "\n- **RESULT TRUNCATION**: The `result` field in responses may be truncated. The FULL output is available via `@task_id` or `read_task_result(task_id)`."

        # 1. Build dynamic blocks
        if mode == "execute":
            full_prompt += f"\n\n### BUILT-IN TOOLS:\n{tools_doc}"
            full_prompt += f"\n\n{mandatory_rules}"

        # 2. Add Mode-Specific Guidelines
        if mode == "plan":
            full_prompt += """
\n### PLANNING PHASE - ACTIVE
Analyze the request and decompose it into a series of logical, executable tasks using the available subagents.
- **Goal**: Create a step-by-step roadmap to fulfill the user's core objective.
- **Output Schema**: Return STRICTLY a JSON object: `{"tasks": [{"task_id": "task_1", "description": "..."}, ...]}`.
- **Task Granularity**: Each task should be an atomic, actionable step (e.g., "Use web_search_agent to find X", "Analyze the results of task_1 to do Y", "Use code_interpreter_agent to build Z").
- **Subagent Selection**: Describe tasks in terms of the available subagent capabilities if no subagent has the proper capabilities assume the orchestrator has it.
- **Constraint**: Return ONLY the raw JSON object. NO prose, NO explanations, NO greetings.
"""
        elif mode == "execute":
            full_prompt += """
\n### EXECUTION PHASE - ACTIVE
Your objective is to fulfill the request by executing the established plan.
- Use 'call_subagent' for all task delegation.
- Use 'update_state' to add or modify tasks and track progress manually if needed.
- Synthesis: After ALL tasks are finished, provide a clean final response leveraging @task_id macros.
"""

        if user_input_enabled:
            full_prompt += (
                "\n### USER INTERACTION:\n"
                "User Interaction Tools (ask_user, give_options) are ACTIVE. "
                "Use them whenever you need user input or a choice from the user. "
                "NEVER ask the user a question in plain text — ALWAYS use the appropriate tool instead."
            )

        # 4. Context Synchronization (Follow-up logic)
        if is_follow_up:
            full_prompt += """
\n### CONTEXT SYNCHRONIZATION
This turn is a continuation of an active conversation. 
Your internal task state (PlannerState) has been restored from the previous turn's state file. 
Analyze the cleaned history to synchronize with the current project status, generated assets, and established logic. 
Do not repeat research or generations already present in the history.
Use @task_id references to build upon previous work.
"""

        # 5. Available Subagents
        full_prompt += f"\n\n### AVAILABLE SUBAGENTS:\n{subagent_descriptions}\n"

        return full_prompt


# ---------------------------------------------------------------------------
# Subagent Management
# ---------------------------------------------------------------------------


class SubagentManager:
    VIRTUAL_AGENTS: Dict[str, AgentDefinition] = {
        "image_gen_agent": AgentDefinition(
            id="image_gen_agent",
            name="Image Generation Agent",
            description="Built-in image generation and editing subagent. Can generate and edit images from text prompts.",
            system_message=(
                "You are a specialized image generation subagent. Your role is to generate or edit images based on the user's prompt. "
                "Use the generate_image tool for creating new images and edit_image for modifying existing ones. "
                "Always return the image URLs or file paths in your final response so the planner can use them."
            ),
            features={"image_generation": True},
            type="builtin",
            builtin_model_override={
                "info": {
                    "meta": {
                        "builtinTools": {
                            "time": False,
                            "knowledge": False,
                            "chats": False,
                            "memory": False,
                            "web_search": False,
                            "image_generation": True,
                            "code_interpreter": False,
                            "notes": False,
                            "channels": False,
                        }
                    }
                }
            },
        ),
        "web_search_agent": AgentDefinition(
            id="web_search_agent",
            name="Web Search Agent",
            description="Built-in web search subagent. Can search the web and fetch URL content.",
            system_message=(
                "You are a specialized web search and research subagent. Your role is to search the web for information and fetch content from URLs. "
                "Use search_web to find relevant results and fetch_url to retrieve full page content. "
                "Synthesize and return the relevant information clearly in your response."
            ),
            features={"web_search": True},
            type="builtin",
            builtin_model_override={
                "info": {
                    "meta": {
                        "builtinTools": {
                            "time": True,
                            "knowledge": False,
                            "chats": False,
                            "memory": False,
                            "web_search": True,
                            "image_generation": False,
                            "code_interpreter": False,
                            "notes": False,
                            "channels": False,
                        }
                    }
                }
            },
        ),
        "knowledge_agent": AgentDefinition(
            id="knowledge_agent",
            name="Knowledge Agent",
            description="Built-in knowledge, notes, chat history, and user memory retrieval subagent.",
            system_message=(
                "You are a specialized knowledge retrieval subagent. Your role is to search through notes, knowledge bases, user memory, and chat history. "
                "Use the available search and retrieval tools to find the information requested. "
                "Return the relevant findings clearly and completely in your response."
            ),
            features={
                "knowledge": True,
                "chats": True,
                "memory": True,
                "notes": True,
                "channels": True,
            },
            type="builtin",
            builtin_model_override={
                "info": {
                    "meta": {
                        "builtinTools": {
                            "time": False,
                            "knowledge": True,
                            "chats": True,
                            "memory": True,
                            "web_search": False,
                            "image_generation": False,
                            "code_interpreter": False,
                            "notes": True,
                            "channels": True,
                        }
                    }
                }
            },
        ),
        "code_interpreter_agent": AgentDefinition(
            id="code_interpreter_agent",
            name="Code Interpreter Agent",
            description="Built-in code interpreter subagent. Can generate content in ANY language and execute Python. Executes Python code and returns results. The code_interpreter tool is moved here exclusively.",
            system_message=(
                "You are a specialized code and content generation subagent. "
                "You can generate content in ANY language — HTML, CSS, JavaScript, Python, shell scripts, JSON, YAML, and more.\n"
                "### FILE HANDLING:\n"
                "- If the user provides a 'file:///' URI, this is the absolute local path on the backend server. Open it directly in your Python code.\n"
                "- If you see a relative link like '/api/v1/files/uuid', and you need to access it in your environment, use 'curl -O {OPEN_WEBUI_URL}/api/v1/files/uuid' (prepend the base URL).\n"
                "- Use relative links (e.g. '/api/v1/files/uuid') when creating HTML artifacts or UI-facing references.\n"
                "### BEST PRACTICES:\n"
                "- For HTML/CSS/JS or any web content: output the FULL, complete, self-contained content DIRECTLY — "
                "do NOT wrap it in Python code that generates it. Return it as-is so the planner can use it immediately.\n"
                "- For computation, data processing, or tasks that need execution: use the code_interpreter tool.\n"
                "- Output ONLY the final, complete, working content unless the user explicitly asks for explanations.\n"
                "- Do NOT add prose, commentary, or markdown outside of code blocks unless asked.\n"
                "- Return generated file paths, URLs, or raw content in your response so the planner can use them.\n"
                "- If the task requires a file to be created, return its absolute path in your response.\n"
                "- For visualizations or plots, save them to a file and return the path."
            ),
            features={"code_interpreter": True},
            type="builtin",
            temperature=0.1,
            builtin_model_override={
                "info": {
                    "meta": {
                        "builtinTools": {
                            "time": False,
                            "knowledge": False,
                            "chats": False,
                            "memory": False,
                            "web_search": False,
                            "image_generation": False,
                            "code_interpreter": True,
                            "notes": False,
                            "channels": False,
                        }
                    }
                }
            },
        ),
        "terminal_agent": AgentDefinition(
            id="terminal_agent",
            name="Terminal Agent",
            description="Built-in terminal subagent. Can execute commands, read/write files, and interact with the system terminal.",
            system_message=(
                "You are a specialized terminal subagent. Your role is to execute terminal commands, read and write files, and perform system operations.\n"
                "### FILE HANDLING:\n"
                "- If the user provides a 'file:///' URI, this is the absolute local path on the backend server. Use it directly in your commands.\n"
                "- If you see a relative link like '/api/v1/files/uuid', and you need to access it via <curl/wget>, prepend the base URL: '{OPEN_WEBUI_URL}/api/v1/files/uuid'.\n"
                "- If you see a relative link like '/files/uuid', and you need to access it, try to find it in the current working directory or subdirectories.\n"
                "### BEST PRACTICES:\n"
                "- Use 'ls -F' to distinguish directories from files.\n"
                "- Use 'cat' or 'tail' to read files. Avoid 'vi' or other interactive editors.\n"
                "- If a command produces too much output, use 'grep' or 'head' to filter it.\n"
                "- Always state your reasoning before running a command."
            ),
            features={},
            type="terminal",
            builtin_model_override={
                "info": {
                    "meta": {
                        "builtinTools": {
                            "time": False,
                            "knowledge": False,
                            "chats": False,
                            "memory": False,
                            "web_search": False,
                            "image_generation": False,
                            "code_interpreter": False,
                            "notes": False,
                            "channels": False,
                        }
                    }
                }
            },
        ),
    }

    def __init__(
        self,
        ui: UIRenderer,
        state: PlannerState,
        metadata: dict[str, Any],
        valves: Any,
        base_url: str = "",
        model_knowledge: Optional[list[dict]] = None,
    ):
        self.ui = ui
        self.state = state
        self.metadata = metadata
        self.valves = valves
        self.base_url = base_url
        self.model_knowledge = model_knowledge

        # Resolve virtual agents with fallback model (v3 parity better robustness)
        self.virtual_agents = {
            vid: va.model_copy(
                update={"model_id": va.model_id or valves.PLANNER_MODEL}
            )
            for vid, va in self.VIRTUAL_AGENTS.items()
        }

    async def _verify_subagent_response(
        self,
        task_id: str,
        model_id: str,
        response_text: str,
        valves,
        user_obj,
        body,
        task_description: str = "",
        planner_input: str = "",
    ) -> tuple[bool, str]:
        """
        Uses a judge model to verify if the subagent's response is complete and correctly references assets.
        Returns: (is_approved, redo_instruction)
        """
        await self.ui.emit_status(f"Verifying {task_id}...")
        judge_model = valves.SUBAGENT_CHECK_MODEL or valves.PLANNER_MODEL
        check_prompt = PromptBuilder.build_subagent_check_prompt(
            task_id, model_id, response_text, task_description, planner_input
        )

        check_body = {
            **body,
            "model": judge_model,
            "messages": [{"role": "system", "content": check_prompt}],
            "stream": False,
            "tools": None,
            "tool_choice": None,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "verification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["approve", "redo"]},
                            "feedback": {"type": "string"},
                        },
                        "required": ["action", "feedback"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        # Call the judge model (non-streaming for simplicity of parsing)
        max_retries = 1
        current_retry = 0

        while current_retry <= max_retries:
            judge_response_chunks = []
            try:
                async for event in Utils.get_streaming_completion(
                    self.metadata.get("__request__"), check_body, user_obj
                ):
                    if event["type"] == "content":
                        judge_response_chunks.append(event["text"])
                    elif event["type"] == "error":
                        logger.error(f"Subagent verification error: {event['text']}")
                        return True, ""  # Fail open on judge error
            except Exception as e:
                logger.error(f"Subagent verification exception: {e}")
                return True, ""

            raw_judge_text = "".join(judge_response_chunks).strip()
            try:
                # We use extract_json_array's logic prefix but handle it as a dict for this specific case
                # or just simple json.loads if it's strict
                data = json.loads(raw_judge_text)
                action = data.get("action", "approve").lower()
                feedback = data.get("feedback", "")

                if action == "approve":
                    return True, ""
                return False, feedback
            except Exception as e:
                if current_retry < max_retries:
                    logger.warning(
                        f"Failed to parse judge JSON: {e}. Retrying... Raw: {raw_judge_text}"
                    )
                    # Corrective prompt
                    check_body["messages"].append(
                        {"role": "assistant", "content": raw_judge_text}
                    )
                    check_body["messages"].append(
                        {
                            "role": "user",
                            "content": "SYSTEM: Your previous response was not a valid JSON object. Please return ONLY the JSON object following the schema strictly.",
                        }
                    )
                    current_retry += 1
                    continue
                else:
                    logger.warning(f"Failed to parse judge JSON after retries: {e}. Raw: {raw_judge_text}")
                    # Fallback to simple string check if JSON parsing fails for some reason
                    if "REDO:" in raw_judge_text.upper():
                        return False, raw_judge_text
                    return True, ""

    async def call_subagent(
        self,
        model_id: str,
        prompt: str,
        task_id: str,
        related_tasks: list[str],
        chat_id: str,
        valves,
        body: dict,
        user_valves,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Main entry point for subagent execution.
        Resolves task persistence, prepares context, and runs the tool-calling loop.
        """
        # UI Parity: Emit "Consulting..." immediately upon entry to subagent phase
        await self.ui.emit_status(f"Consulting {model_id}...")

        # 1. Prepare Context
        context = await self._prepare_subagent_context(
            model_id, prompt, task_id, related_tasks, chat_id, valves, body
        )

        # 2. Execute Loop
        result = await self._execute_subagent_loop(
            context, task_id, model_id, chat_id, valves, body, user_valves, planner_input=prompt
        )

        return result

    async def _prepare_subagent_context(
        self,
        model_id: str,
        prompt: str,
        task_id: str,
        related_tasks: list[str],
        chat_id: str,
        valves,
        body: dict,
    ) -> dict[str, Any]:
        """Resolves model info, tools, and constructs the system prompt and initial history."""
        user_obj = self.metadata.get("__user_obj__")
        registry = ToolRegistry(
            valves,
            user_obj,
            self.metadata.get("__request__"),
            self.metadata.get("__metadata__"),
            model_knowledge=self.model_knowledge,
        )

        # Fix: Fetch app_models from request.app.state instead of __event_call__
        app_models = getattr(self.metadata.get("__request__").app.state, "MODELS", {})
        sub_info = await registry.get_subagent_tools(
            model_id,
            self.virtual_agents,
            app_models,
            self.metadata,
        )

        actual_model = sub_info.get("actual_model", model_id)
        sub_temp_override = sub_info.get("temperature_override")

        sub_sys = sub_info.get("system_message", "")
        if not sub_sys:
            sub_sys = f"You are a specialized subagent acting as {model_id}. Follow the prompt directly and accurately using your tools."

        # Inject base_url into subagent instructions (Terminal/Coding agents only if they use the placeholders)
        sub_sys = sub_sys.replace("{OPEN_WEBUI_URL}", self.base_url)

        # Subagent background execution rules (v3 parity)
        sub_sys += (
            "\n\nCRITICAL CONTEXT: You are running as a headless subagent entirely in the background. "
            "DO NOT return markdown elements that rely on Open WebUI UI embeds for tool generation output. "
            "Any tools you use (like generate_image, search, etc.) will return URLs, base64 data, or raw paths exactly as returned by tools. "
            "You MUST always return these raw HTML references, URLs, files, or images as relative or absolute paths, or direct URLs, unconditionally in your final reply so the main planner can use them. "
            "If you generate or reference any assets (images, files, data, audio, etc.), it is CRITICAL to include the correct link, path, or URL in your output. "
            "If you are missing any required asset links or references, you MUST follow up or retry until all assets are accessible via explicit links or paths. "
        )

        if related_tasks:
            for rt in related_tasks:
                rid = rt.lstrip("@")
                if rid in self.state.results:
                    sub_sys += f"\n\n--- RESULTS FROM PREVIOUS TASK {rt} ---\n{self.state.results[rid]}\n--- END OF {rt} ---\n"

        history = self.state.get_history(chat_id, task_id, model_id)
        if history:
            # v3 pacing: truncating history in sub-threads is risky for OpenRouter providers
            # who expect the full tool call chain. We'll skip aggressive truncation for now.
            if history[0]["role"] == "system":
                history[0]["content"] = sub_sys
            history.append({"role": "user", "content": prompt})
        else:
            history = [
                {"role": "system", "content": sub_sys},
                {"role": "user", "content": prompt},
            ]

        return {
            "actual_model": actual_model,
            "model_id": model_id,
            "temp_override": sub_temp_override,
            "history": history,
            "tools_specs": sub_info.get("specs"),
            "tools_dict": sub_info.get("dict", {}),
            "user_obj": user_obj,
        }

    async def _execute_subagent_loop(
        self,
        context: dict[str, Any],
        task_id: str,
        model_id: str,
        chat_id: str,
        valves,
        body: dict,
        user_valves,
        planner_input: str = "",
    ) -> dict[str, Any]:
        """Main tool-calling loop for the subagent."""
        sub_final_answer_chunks = []
        sub_called_tools = []
        sub_iteration = 0
        max_sub_iters = valves.MAX_SUBAGENT_ITERATIONS or 100
        history = context["history"]

        while True:
            sub_iteration += 1

            # (A) Iteration Limit Check
            if not await self._handle_subagent_iteration_limit(
                sub_iteration, max_sub_iters, task_id, model_id, user_valves
            ):
                sub_final_answer_chunks.append("[Subagent stopped at iteration limit.]")
                break

            # (B) Execute Turn
            turn_result = await self._execute_subagent_turn(context, body)
            sub_content = turn_result["content"]
            sub_tc_dict = turn_result["tool_calls"]
            raw_content = turn_result["raw_content"]

            if sub_content:
                sub_final_answer_chunks.append(sub_content)

            if not sub_tc_dict:
                # Loop Terminal - Final Answer found
                
                # Subagent Judge Verification (v3 parity extension)
                if valves.ENABLE_SUBAGENT_CHECK:
                    current_answer = "".join(sub_final_answer_chunks).strip()
                    if current_answer:
                        task_desc = self.state.tasks.get(task_id, TaskStateModel(status="pending")).description
                        is_approved, redo_instruction = await self._verify_subagent_response(
                            task_id,
                            model_id,
                            current_answer,
                            valves,
                            context["user_obj"],
                            body,
                            task_description=task_desc,
                            planner_input=planner_input,
                        )
                        if not is_approved:
                            history.append({"role": "user", "content": redo_instruction})
                            await self.ui.emit_status(
                                f"Refining {model_id} response for {task_id}..."
                            )
                            # Reset chunks to avoid double output if the model doesn't repeat everything
                            # but the prompt asks for FULL response, so we should clear them.
                            sub_final_answer_chunks = []
                            continue

                # Reasoning Promotion (v3 parity): Only if NO content AND NO tool calls
                if not "".join(sub_final_answer_chunks).strip() and turn_result.get(
                    "reasoning"
                ):
                    reasoning_text = f"Thinking: {turn_result['reasoning']}"
                    sub_final_answer_chunks.append(reasoning_text)
                    sub_content = reasoning_text

                history.append({"role": "assistant", "content": sub_content or ""})
                break

            if turn_result.get("error") and not sub_tc_dict:
                # If a provider error occurred and no tools were found, 
                # we still append to history then break to avoid infinite loops.
                history.append({"role": "assistant", "content": sub_content or ""})
                break

            # (C) Process Tool Calls
            tool_calls_list = list(sub_tc_dict.values())
            history.append(
                {
                    "role": "assistant",
                    "content": sub_content or "",  # Some providers prefer "" over None with tools
                    "tool_calls": tool_calls_list,
                }
            )

            for stc in tool_calls_list:
                stc_name = stc["function"]["name"]
                stc_args_str = stc["function"]["arguments"]
                call_id = stc.get("id")

                stc_args_obj = Utils.parse_tool_arguments(stc_args_str)

                target_tool = context["tools_dict"].get(stc_name)
                if target_tool:
                    await self._execute_subagent_tool_call(
                        stc_name,
                        stc_args_str,
                        stc_args_obj,
                        call_id,
                        context,
                        history,
                        sub_called_tools,
                        model_id,
                    )
                else:
                    if await self._handle_missing_tool(
                        stc_name, stc_args_str, call_id, context, history, sub_called_tools, model_id
                    ):
                        break

        final_result = "\n".join(sub_final_answer_chunks).strip()
        self.state.store_result(task_id, final_result)
        self.state.set_history(chat_id, task_id, model_id, history)

        result_preview = (
            Utils.middle_truncate(final_result, valves.TASK_RESULT_LIMIT)
            if user_valves.TASK_RESULT_TRUNCATION
            else final_result
        )
        structured_response = {
            "task_id": task_id,
            "status": "completed",
            "called_tools": sub_called_tools,
            "result": result_preview,
        }
        if (
            user_valves.TASK_RESULT_TRUNCATION
            and len(final_result) > valves.TASK_RESULT_LIMIT
        ):
            structured_response["note"] = (
                f"IMPORTANT: Result was truncated. Use @{task_id} in prompts for literal text replacement, or call read_task_result('{task_id}') to read the complete output."
            )
        else:
            structured_response["note"] = (
                f"Use @{task_id} in prompts for literal text replacement."
            )

        return {
            "task_id": task_id,
            "result": json.dumps(structured_response, ensure_ascii=False),
        }

    async def _handle_subagent_iteration_limit(
        self, iteration: int, max_iters: int, task_id: str, model_id: str, user_valves
    ) -> bool:
        """Handles iteration limits by prompting the user or stopping if not in YOLO mode."""
        if user_valves.YOLO_MODE or max_iters <= 0 or iteration <= max_iters:
            return True

        if self.metadata.get("__event_call__"):
            try:
                ctx_msg = f"Subagent '{model_id}' ({task_id}) has reached {iteration - 1} tool-call iterations. Continue?"
                js = self.ui.build_continue_cancel_js(ctx_msg, timeout_s=300)
                raw = await self.metadata["__event_call__"](
                    {"type": "execute", "data": {"code": js}}
                )
                raw_str = (
                    raw
                    if isinstance(raw, str)
                    else (
                        (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                    )
                )
                try:
                    res_json = (
                        json.loads(raw_str)
                        if isinstance(raw_str, str) and raw_str.startswith("{")
                        else {"action": "cancel", "value": raw_str}
                    )
                except:
                    res_json = {"action": "cancel", "value": str(raw_str)}
                return res_json.get("action") == "continue"
            except Exception as e:
                logger.error(f"Iteration limit modal error: {e}")
                return False
        return False

    async def _execute_subagent_turn(
        self, context: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any]:
        """Single LLM turn for the subagent, including streaming and XML tool interception."""
        content_chunks = []
        reasoning_chunks = []
        tc_dict = {}
        error_occurred = False

        actual_model = context["actual_model"]
        model_id = context.get("model_id", "unknown")
        history = context["history"]
        tools_specs = context["tools_specs"]
        temp_override = context.get("temp_override")
        user_obj = context["user_obj"]

        sub_body = {
            **body,
            "model": actual_model,
            "messages": history,
            "tools": tools_specs,
            "metadata": self.metadata.get("__metadata__", {}),
        }

        # Inject model knowledge explicitly for knowledge_agent or generally if appropriate
        if model_id == "knowledge_agent" and self.model_knowledge:
            sub_body["metadata"]["knowledge"] = self.model_knowledge
            # Parity with commit 0f0ba7d: ensure __model_knowledge__ is available for tool calls
            sub_body["metadata"]["__model_knowledge__"] = self.model_knowledge

        if temp_override is not None:
            sub_body["temperature"] = temp_override

        # Reasoning state
        reasoning_buffer = ""
        reasoning_start_time = None

        async for event in Utils.get_streaming_completion(
            self.metadata.get("__request__"), sub_body, user_obj
        ):
            etype = event["type"]
            if etype == "error":
                err = f"Agent Error: {event.get('text', 'Unknown stream error')}"
                content_chunks.append(f"\n\n> [!CAUTION]\n> {err}\n\n")
                logger.error(err)
                error_occurred = True
                break
            elif etype == "reasoning":
                piece = event.get("text", "")
                if piece:
                    if reasoning_start_time is None:
                        reasoning_start_time = time.monotonic()
                    reasoning_chunks.append(piece)
                    reasoning_buffer += piece
            elif etype == "content":
                if reasoning_buffer:
                    reasoning_buffer = ""
                text = event["text"]
                content_chunks.append(text)
            elif etype == "tool_calls":
                if reasoning_buffer:
                    reasoning_buffer = ""
                for tc in event["data"]:
                    idx = tc["index"]
                    if idx not in tc_dict:
                        tc_dict[idx] = {
                            "id": tc.get("id") or f"call_{uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": tc["function"].get("name", ""),
                                "arguments": "",
                            },
                        }
                    if "name" in tc["function"] and tc["function"]["name"]:
                        tc_dict[idx]["function"]["name"] = tc["function"]["name"]
                    if "arguments" in tc["function"]:
                        tc_dict[idx]["function"]["arguments"] += tc["function"][
                            "arguments"
                        ]

        raw_content = "".join(content_chunks)
        content = Utils.clean_thinking(raw_content)
        reasoning = "".join(reasoning_chunks)

        # Intercept hallucinated XML <tool_call> in both content and reasoning (v3 parity)
        if not tc_dict:
            tc_dict_content, content = Utils.extract_xml_tool_calls(content)
            tc_dict_reasoning, reasoning = Utils.extract_xml_tool_calls(reasoning)
            # Merge both dicts
            # Merge both dicts (preserving native calls)
            tc_dict = {**tc_dict, **tc_dict_content, **tc_dict_reasoning}

        return {
            "content": content,
            "raw_content": raw_content,
            "tool_calls": tc_dict,
            "reasoning": reasoning,
            "error": error_occurred,
        }

    # --- Subagent Helpers ---

    async def _execute_subagent_tool_call(
        self,
        name: str,
        args_str: str,
        args_obj: dict,
        call_id: str,
        context: dict,
        history: list,
        sub_called_tools: list,
        model_id: str,
    ) -> None:
        """Executes a single subagent tool call and updates history/states."""
        target_tool = context["tools_dict"].get(name)
        try:
            # UI Parity: Emit status and tool call details
            await self.ui.emit_status(f"[Subagent: {model_id}] Executing {name}...")
            # We don't strictly need tc_tag here as in planner, 
            # but we follow v3's internal logging/status pattern.

            tc_res = await target_tool["callable"](**args_obj)
            tc_return = process_tool_result(
                self.metadata.get("__request__"),
                name,
                tc_res,
                target_tool.get("type", ""),
                False,
                self.metadata.get("__metadata__"),
                context["user_obj"],
            )
            res_str = tc_return[0] if len(tc_return) > 0 else str(tc_res)

            truncated_args = {
                k: (str(v)[:80] + "..." if len(str(v)) > 80 else str(v))
                for k, v in args_obj.items()
            }
            sub_called_tools.append(
                {"tool": name, "arguments": truncated_args, "success": True}
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": str(res_str),
                }
            )
        except Exception as e:
            logger.error(f"[Subagent: {model_id}] Error executing {name}: {e}")
            await self.ui.emit_status(f"[Subagent: {model_id}] Error executing {name}: {e}")
            sub_called_tools.append({"tool": name, "arguments": {}, "success": False})
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": f"Error: {e}",
                }
            )

        
    async def _handle_missing_tool( 
        self,
        name: str,
        args_str: str,
        call_id: str,
        context: dict,
        history: list,
        sub_called_tools: list,
        model_id: str,
    ) -> bool: # Returns True if it should break the loop due to repeated failures
        error_msg = f"Tool {name} not found."
        available_tools = list(context["tools_dict"].keys())
        if available_tools:
            error_msg += f" Available tools for this subagent: {', '.join(available_tools)}."

        logger.warning(f"[Subagent: {model_id}] {error_msg} Arguments: {args_str}")
        await self.ui.emit_status(f"[Subagent: {model_id}] Attempted unknown tool: {name}")

        sub_called_tools.append({"tool": name, "arguments": {}, "success": False})

        # Safety break logic
        consecutive_failures = 0
        for h in reversed(history):
            if h.get("role") == "tool" and "not found" in h.get("content", ""):
                if h.get("name") == name:
                    consecutive_failures += 1
                else: break
            else: break

        if consecutive_failures >= 3:
            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": f"Error: {error_msg} Repeated failures detected. Please stop or try a different approach.",
            })
            await self.ui.emit_status(f"[Subagent: {model_id}] Stopping due to repeated tool failures: {name}")
            return True

        history.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": f"Error: {error_msg}",
        })
        return False


# Planner Engine (Main Logic)
# ---------------------------------------------------------------------------


class InternalToolExecutor:
    """Encapsulates the logic for built-in planner tools."""

    def __init__(self, engine: "PlannerEngine"):
        self.engine = engine
        self.state = engine.state
        self.ui = engine.ui
        self.subagents = engine.subagents
        self.metadata = engine.metadata

    async def update_state(self, args: dict, user_valves: Any) -> str:
        self.state.update_task(
            args["task_id"],
            args["status"],
            args.get("description"),
        )
        if user_valves.PLAN_MODE:
            await self.ui.emit_html_embed(self.state.tasks)
        return f"State updated for {args['task_id']}"

    async def call_subagent(
        self, args: dict, chat_id: str, valves: Any, body: dict, user_valves: Any
    ) -> str:
        if user_valves.PLAN_MODE:
            current_status = self.state.tasks.get(
                args["task_id"], TaskStateModel(status="pending")
            ).status
            if current_status in ["pending", "failed"]:
                self.state.update_task(args["task_id"], "in_progress")
                await self.ui.emit_html_embed(self.state.tasks)

        res_dict = await self.subagents.call_subagent(
            args["model_id"],
            args["prompt"],
            args["task_id"],
            args.get("related_tasks", []),
            chat_id,
            valves,
            body,
            user_valves,
            self.metadata,
        )
        self.state.update_task(args["task_id"], "completed")
        if user_valves.PLAN_MODE:
            await self.ui.emit_html_embed(self.state.tasks)
        return res_dict["result"]

    async def ask_user(self, args: dict, valves: Any) -> str:
        js = self.ui.build_ask_user_js(
            args.get("prompt_text"),
            args.get("placeholder", "Type here..."),
            valves.USER_INPUT_TIMEOUT,
        )
        raw = await self.metadata["__event_call__"](
            {"type": "execute", "data": {"code": js}}
        )
        if (
            isinstance(raw, dict)
            and "result" not in raw
            and "value" not in raw
            and "action" in raw
        ):
            res_json = raw
        else:
            raw_str = (
                raw
                if isinstance(raw, str)
                else ((raw.get("result") or raw.get("value") or "{}") if raw else "{}")
            )
            try:
                res_json = (
                    json.loads(raw_str)
                    if isinstance(raw_str, str) and raw_str.startswith("{")
                    else {"action": "accept", "value": raw_str}
                )
            except:
                res_json = {"action": "accept", "value": str(raw_str)}
        return (
            f"User: {res_json.get('value')}"
            if res_json.get("action") == "accept"
            else "User skipped."
        )

    async def give_options(self, args: dict, valves: Any) -> str:
        allow_custom = args.get("allow_custom", True)
        js = self.ui.build_give_options_js(
            args.get("prompt_text"),
            args.get("choices", []),
            args.get("context", ""),
            valves.USER_INPUT_TIMEOUT,
            allow_custom=allow_custom,
        )
        raw = await self.metadata["__event_call__"](
            {"type": "execute", "data": {"code": js}}
        )
        if (
            isinstance(raw, dict)
            and "result" not in raw
            and "value" not in raw
            and "action" in raw
        ):
            res_json = raw
        else:
            raw_str = (
                raw
                if isinstance(raw, str)
                else ((raw.get("result") or raw.get("value") or "{}") if raw else "{}")
            )
            try:
                res_json = (
                    json.loads(raw_str)
                    if isinstance(raw_str, str) and raw_str.startswith("{")
                    else {"action": "accept", "value": raw_str}
                )
            except:
                res_json = {"action": "accept", "value": str(raw_str)}
        return (
            f"User selected: {res_json.get('value')}"
            if res_json.get("action") == "accept"
            else "User skipped."
        )

    def read_task_result(self, args: dict) -> str:
        rid = args.get("task_id", "").lstrip("@")
        return self.state.results.get(rid, f"Task {rid} not found.")

    async def review_tasks(self, args: dict, valves: Any, body: dict, user_obj: Any) -> str:
        rt_ids, rt_prompt = args.get("task_ids", []), args.get("prompt", "")
        if not rt_ids or not rt_prompt:
            return "Error: must specify task_ids and prompt."

        await self.ui.emit_status("Reviewing tasks cross-reference...")
        review_sys = "You are a specialized review subagent. Synthesize the following task results logically."
        for rx in rt_ids:
            clean_rx = rx.lstrip("@")
            if clean_rx in self.state.results:
                review_sys += f"\n\n--- RESULTS FROM TASK {rx} ---\n{self.state.results[clean_rx]}\n--- END OF {rx} ---\n"

        review_body = {
            **body,
            "model": valves.REVIEW_MODEL or valves.PLANNER_MODEL,
            "messages": [
                {"role": "system", "content": review_sys},
                {"role": "user", "content": rt_prompt},
            ],
            "metadata": self.metadata.get("__metadata__", {}),
        }
        res_chunks = []
        async for ev in Utils.get_streaming_completion(
            self.metadata.get("__request__"), review_body, user_obj
        ):
            if ev["type"] == "content":
                res_chunks.append(ev["text"])

        final_review = "".join(res_chunks)

        # Virtual ID Support (v3 parity extension)
        review_id = args.get("review_id")
        if not review_id:
            # Generate a generic ID based on existing reviews
            review_count = sum(
                1 for k in self.state.results.keys() if k.startswith("review_")
            )
            review_id = f"review_{review_count + 1}"

        clean_rid = review_id.lstrip("@")
        self.state.results[clean_rid] = final_review

        return f"[Review {clean_rid}]:\n{final_review}"


class PlannerEngine:
    def __init__(
        self,
        ui: UIRenderer,
        state: PlannerState,
        subagents: SubagentManager,
        registry: ToolRegistry,
        metadata: dict[str, Any],
        model_knowledge: Optional[list[dict]] = None,
    ):
        self.ui = ui
        self.state = state
        self.subagents = subagents
        self.registry = registry
        self.metadata = metadata
        self.model_knowledge = model_knowledge
        self.tools = InternalToolExecutor(self)

    async def run(self, chat_id: str, valves: Any, user_valves: Any, body: dict, files: list = None) -> Union[str, Generator, AsyncGenerator]:
        """Main entry point for the planner engine. Orchestrates planning, execution, and verification."""
        
        # 0. State Recovery (v3.3)
        # We attempt to restore previous turn's state BEFORE clearing.
        await self._recover_state_from_files(body, chat_id, files)
        
        user_obj = self.metadata.get("__user_obj__")

        # 1. Phase 1: Planning
        if user_valves.PLAN_MODE:
            await self._phase_planning(valves, user_valves, user_obj, body)

        # 2. Phase 2: Execution & Verification Loop
        final_answer = await self._phase_execution_loop(
            chat_id, valves, user_valves, user_obj, body
        )
        
        # 3. State Persistence (v3.3)
        # Save state to file and attach to response
        await self._save_state_to_file(chat_id)

        return final_answer

    async def _recover_state_from_files(self, body: dict, chat_id: str, current_files: list = None) -> None:
        """
        Scans attached files (current and history) for the latest state file.
        Prioritizes the absolute latest JSON state file found.
        """
        state_file = None
        
        # 1. First, check files attached directly to this message
        if current_files:
            for f in reversed(current_files):
                name = f.get("name", f.get("filename", ""))
                if "planner_state" in name and name.endswith(".json"):
                    state_file = f
                    break
        
        # 2. If not found in current message, look back through history (explicit DB query)
        if not state_file:
            chat_id = self.metadata.get("__chat_id__")
            if chat_id:
                logger.info(f"Performing deep history scan for chat {chat_id} via database...")
                chat_obj = Chats.get_chat_by_id(chat_id)
                if chat_obj and hasattr(chat_obj, "chat"):
                    messages_map = chat_obj.chat.get("history", {}).get("messages", {})
                    # Standard Open WebUI history traversal
                    current_id = chat_obj.chat.get("history", {}).get("currentId")
                    
                    visited = set()
                    while current_id and current_id not in visited:
                        visited.add(current_id)
                        msg = messages_map.get(current_id)
                        if not msg:
                            break
                        
                        msg_files = msg.get("files")
                        if msg_files:
                            for f in reversed(msg_files):
                                name = f.get("name", f.get("filename", ""))
                                if "planner_state" in name and name.endswith(".json"):
                                    state_file = f
                                    logger.info(f"Successfully recovered state from DB history: {name}")
                                    break
                        
                        if state_file:
                            break
                        current_id = msg.get("parentId")
            
            # Fallback to body messages if DB query failed or chat_id missing
            if not state_file:
                messages = body.get("messages", [])
                for message in reversed(messages):
                    msg_files = message.get("files")
                    if msg_files:
                        for f in reversed(msg_files):
                            name = f.get("name", f.get("filename", ""))
                            if "planner_state" in name and name.endswith(".json"):
                                state_file = f
                                break
                    if state_file:
                        break

        if not state_file:
            logger.info("No planner state file found in current turn, database, or history.")
            return

        try:
            file_id = state_file.get("file_id") or state_file.get("id")
            if not file_id:
                logger.warning(f"State file found but missing ID: {state_file}")
                return
            
            # Use global Files model (Open WebUI)
            # Defensive check: if Files is somehow None or not available, log it
            if Files is None:
                logger.error("Global 'Files' model table is None - cannot recover state.")
                return
                
            file_obj = Files.get_file_by_id(file_id)
            if not file_obj:
                logger.warning(f"State file with ID {file_id} not found in database.")
                return
            
            file_path = getattr(file_obj, 'path', None)
            if not file_path and hasattr(file_obj, 'meta'):
                file_path = file_obj.meta.get('path')
            
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                    
                # Restore tasks
                if "tasks" in data:
                    self.state.tasks.clear()
                    for tid, tdata in data["tasks"].items():
                        self.state.update_task(tid, tdata.get("status", "pending"), tdata.get("description", ""))
                
                # Restore results
                if "results" in data:
                    self.state.results.clear()
                    self.state.results.update(data["results"])
                
                # Restore subagent history
                if "subagent_history" in data:
                    self.state.subagent_history.clear()
                    for key_str, history in data["subagent_history"].items():
                        try:
                            # Reconstruct tuple key (chat_id, sub_task_id, model_id)
                            parts = key_str.split("::")
                            if len(parts) == 3:
                                tuple_key = tuple(parts)
                                self.state.subagent_history[tuple_key] = history
                        except Exception as parse_err:
                            logger.error(f"Failed to parse subagent history key {key_str}: {parse_err}")
                    logger.info(f"Restored {len(self.state.subagent_history)} sub-conversation threads.")
                
                status_msg = f"Recovered state from {state_file.get('name', 'latest file')}"
                logger.info(status_msg)
                await self.ui.emit_status(status_msg)
            else:
                logger.error(f"State file path not found or inaccessible: {file_path}")
        except Exception as e:
            logger.error(f"Critical error during state recovery: {e}")

    async def _save_state_to_file(self, chat_id: str):
        """Serializes current state and emits as a file attachment via upload_file_handler."""
        emitter = self.metadata.get("__event_emitter__")
        request = self.metadata.get("__request__")
        user = self.metadata.get("__user_obj__")
        
        if not emitter or not request or not user:
            return

        try:
            # Convert tuple keys (chat_id, sub_task_id, model_id) to strings for JSON
            serialized_history = {}
            for tuple_key, history in self.state.subagent_history.items():
                if isinstance(tuple_key, (list, tuple)) and len(tuple_key) == 3:
                    key_str = "::".join(map(str, tuple_key))
                    serialized_history[key_str] = history

            state_data = {
                "tasks": {tid: {"status": t.status, "description": t.description} for tid, t in self.state.tasks.items()},
                "results": self.state.results,
                "subagent_history": serialized_history
            }
            
            filename = f"planner_state_{chat_id}.json"
            content = json.dumps(state_data, ensure_ascii=False).encode('utf-8')
            
            # Use in-memory file with proper headers to set content_type
            file_upload = UploadFile(
                file=io.BytesIO(content), 
                filename=filename,
                headers=Headers({"content-type": "application/json"})
            )
            
            # Call sync as in working examples and confirmed in source
            file_item = upload_file_handler(
                request=request, 
                file=file_upload, 
                metadata={}, 
                process=False, 
                user=user
            )
            
            if file_item:
                file_id = getattr(file_item, "id", None)
                if file_id:
                    file_info = {
                        "file_id": str(file_id),
                        "name": filename
                    }
                    
                    # 1. Update the internal list for this turn
                    internal_files = self.metadata.get("__files__")
                    if isinstance(internal_files, list):
                        internal_files.append(file_info)

                    # 2. Synchronize directly with the database for multi-turn persistence
                    target_chat_id = self.metadata.get("__chat_id__") or chat_id
                    target_msg_id = self.metadata.get("__message_id__")
                    
                    if target_chat_id and target_msg_id:
                        try:
                            # Direct DB update (same method used in Doodle Paint)
                            Chats.add_message_files_by_id_and_message_id(
                                target_chat_id, 
                                target_msg_id, 
                                [file_info]
                            )
                            logger.info(f"Successfully persisted state file to chat {target_chat_id} message {target_msg_id}")
                        except Exception as db_err:
                            logger.error(f"Failed to synchronize state file with database: {db_err}")

                    # 3. Emit event for immediate UI feedback
                    await emitter({
                        "type": "files",
                        "data": {
                            "files": [file_info]
                        }
                    })
        except Exception as e:
            logger.error(f"Failed to save state to file: {e}")

    async def _phase_planning(self, valves, user_valves, user_obj, body):
        """Generates the initial task list based on the user prompt."""
        await self.ui.emit_status("Planning...")
        if user_valves.PLAN_MODE:
            await self.ui.emit_html_embed(self.state.tasks)

        # Prepare context
        distilled_history = Utils.distill_history_for_llm(body.get("messages", []))
        plan_sys = PromptBuilder.build_system_prompt(
            valves,
            user_valves,
            [],
            metadata=self.metadata,
            mode="plan",
            messages=distilled_history,
        )
        messages = [{"role": "system", "content": plan_sys}] + distilled_history

        json_retries = 0
        max_json_retries = 1

        while True:
            plan_body = {
                **body,
                "model": valves.PLANNER_MODEL,
                "messages": messages,
                "tools": None,
                "tool_choice": None,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "plan",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "tasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "task_id": {"type": "string"},
                                            "description": {"type": "string"},
                                        },
                                        "required": ["task_id", "description"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["tasks"],
                            "additionalProperties": False,
                        },
                    },
                },
                "metadata": self.metadata.get("__metadata__", {}),
            }

            # Inject model knowledge if knowledge_agent is NOT present
            if not getattr(valves, "ENABLE_KNOWLEDGE_AGENT", True) and self.model_knowledge:
                plan_body["metadata"]["knowledge"] = self.model_knowledge
                plan_body["metadata"]["__model_knowledge__"] = self.model_knowledge

            plan_chunks = []
            async for event in Utils.get_streaming_completion(
                self.metadata.get("__request__"), plan_body, user_obj
            ):
                etype = event["type"]
                if etype == "content":
                    plan_chunks.append(event["text"])
                elif etype == "error":
                    err = f"Planning failed: {event.get('text', 'LLM Error')}"
                    await self.ui.emit_status(err, True)
                    logger.error(err)
                    break

            raw_plan_text = "".join(plan_chunks)
            plan_json = Utils.extract_json_array(raw_plan_text)
            self.state.tasks.clear()
            for task in plan_json:
                tid, desc = task.get("task_id"), task.get("description")
                if tid and desc:
                    self.state.update_task(tid, "pending", desc)

            if not self.state.tasks:
                if json_retries < max_json_retries:
                    logger.warning(
                        f"No tasks parsed from plan. Retrying... Raw: {raw_plan_text}"
                    )
                    messages.append({"role": "assistant", "content": raw_plan_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": "SYSTEM: No valid tasks were found in your response. Please provide the plan strictly as a JSON object with a 'tasks' array.",
                        }
                    )
                    json_retries += 1
                    continue
                else:
                    logger.warning("No tasks parsed from plan after retries. Using fallback.")
                    self.state.update_task("task_1", "pending", "Process user request")

            await self.ui.emit_status(
                f"Plan formed with {len(self.state.tasks)} tasks."
            )

            if user_valves.PLAN_MODE:
                await self.ui.emit_html_embed(self.state.tasks)

            # Plan Approval logic (Ignored in YOLO mode)
            if (
                user_valves.ENABLE_PLAN_APPROVAL
                and not user_valves.YOLO_MODE
                and self.metadata.get("__event_call__")
            ):
                try:
                    tasks_data = [
                        {"task_id": tid, "description": t.description}
                        for tid, t in self.state.tasks.items()
                    ]
                    js = self.ui.build_plan_approval_js(tasks_data)
                    raw = await self.metadata["__event_call__"](
                        {"type": "execute", "data": {"code": js}}
                    )
                    raw_str = (
                        raw
                        if isinstance(raw, str)
                        else (
                            (raw.get("result") or raw.get("value") or "{}")
                            if raw
                            else "{}"
                        )
                    )
                    try:
                        res_json = (
                            json.loads(raw_str)
                            if isinstance(raw_str, str) and raw_str.startswith("{")
                            else {"action": "accept", "value": raw_str}
                        )
                    except:
                        res_json = {"action": "accept", "value": str(raw_str)}

                    if res_json.get("action") == "feedback":
                        feedback = res_json.get("value", "")
                        await self.ui.emit_status(
                            f"Adjusting plan based on feedback..."
                        )
                        # Append feedback to messages for re-planning
                        messages.append(
                            {"role": "assistant", "content": "".join(plan_chunks)}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"SYSTEM: User provided feedback on the proposed plan: {feedback}. Please provide an updated plan JSON array.",
                            }
                        )
                        json_retries = 0  # Reset retry counter for new feedback
                        continue
                except Exception as e:
                    logger.error(f"Plan approval error: {e}")
            break

    async def _phase_execution_loop(
        self, chat_id: str, valves: Any, user_valves: Any, user_obj: Any, body: dict
    ):
        """Main loop for execution and periodic verification (v3 parity loop stability)."""
        planner_iteration = 0
        judge_retries = 0
        total_emitted = ""

        # Prepare context
        distilled_history = Utils.distill_history_for_llm(body.get("messages", []))
        exec_sys = PromptBuilder.build_system_prompt(
            valves,
            user_valves,
            self.registry.get_tools_spec(
                user_obj, user_valves, list(self.state.tasks.keys())
            ),
            metadata=self.metadata,
            mode="execute",
            messages=distilled_history,
        )
        exec_history = [{"role": "system", "content": exec_sys}] + distilled_history
        tasks_serializable = {
            tid: t.model_dump(mode="json") if hasattr(t, "model_dump") else t.dict()
            for tid, t in self.state.tasks.items()
        }
        # Use more explicit formatting similar to v3 for the established plan
        if user_valves.PLAN_MODE:
            exec_history.append(
                {
                    "role": "system",
                    "content": f"Here is the established plan. Do not deviate from it. Execute the steps logically:\n{json.dumps(tasks_serializable)}",
                }
            )
            await self.ui.emit_status(
                f"Starting execution for {len(self.state.tasks)} tasks..."
            )

        external_tools_dict = await self.registry.get_planner_tools_dict(
            body, user_valves
        )

        while True:
            planner_iteration += 1
            await self.ui.emit_status("Working...")

            # (A) Iteration Limit Check
            if not await self._handle_iteration_limit(
                planner_iteration, valves, user_valves
            ):
                break

            # (B) Execute Turn
            turn_result = await self._execute_planner_turn(
                exec_history, total_emitted, valves, user_obj, body, user_valves
            )
            content, tc_dict, turn_emitted = (
                turn_result["content"],
                turn_result["tool_calls"],
                turn_result["total_emitted"],
            )
            turn_start_base = turn_result["turn_start_base"]

            # Reasoning Promotion (v3 parity): Only if NO content AND NO tool calls (prevent double content)
            if not content and not tc_dict and turn_result.get("reasoning"):
                content = f"Thinking: {turn_result['reasoning']}"
                # Revert to turn_start_base to delete the thinking block from UI (v3 parity)
                total_emitted = turn_start_base + content
                await self.ui.emit_replace(total_emitted)
            elif content or tc_dict or turn_result.get("reasoning"):
                # Ensure total_emitted is updated even if content is empty (v3 parity fix for reasoning stripping)
                total_emitted = turn_emitted

            if not tc_dict:
                # (C) Verification Phase (Judge)
                if user_valves.PLAN_MODE:
                    should_continue, total_emitted = await self._phase_verification(
                        exec_history,
                        content,
                        total_emitted,
                        judge_retries,
                        valves,
                        user_valves,
                        user_obj,
                        body,
                    )
                    if should_continue:
                        # Ensure UI reflects the judge's decision to continue before the next planner turn
                        await self.ui.emit_replace(total_emitted)
                        judge_retries += 1
                        continue

                # Final completion: resolve references in the last content, emit and return full process history
                resolved_content = Utils.resolve_references(content, self.state.results)
                # Replace only the last content segment in total_emitted with the resolved version
                if content and resolved_content != content:
                    # Replace the last occurrence of content in total_emitted
                    idx = total_emitted.rfind(content)
                    if idx != -1:
                        total_emitted = (
                            total_emitted[:idx]
                            + resolved_content
                            + total_emitted[idx + len(content) :]
                        )
                await self.ui.emit_status("Completed", True)
                await self.ui.emit_replace(total_emitted)
                return total_emitted

            # (D) Handle Tool Calls
            tool_calls_list = list(tc_dict.values())
            exec_history.append(
                {
                    "role": "assistant",
                    "content": content or "",  # Some providers prefer "" over None with tools
                    "tool_calls": tool_calls_list,
                }
            )
            total_emitted = await self._handle_tool_calls(
                tool_calls_list,
                exec_history,
                total_emitted,
                external_tools_dict,
                chat_id,
                valves,
                user_valves,
                user_obj,
                body,
            )

    async def _handle_iteration_limit(
        self, iteration: int, valves: Any, user_valves: Any
    ) -> bool:
        """Prompts user if iteration limit reached."""
        if (
            user_valves.YOLO_MODE
            or valves.MAX_PLANNER_ITERATIONS <= 0
            or iteration <= valves.MAX_PLANNER_ITERATIONS
        ):
            return True
        if self.metadata.get("__event_call__"):
            try:
                js = self.ui.build_continue_cancel_js(
                    f"The planner has reached {iteration - 1} iterations. Continue?",
                    timeout_s=300,
                )
                raw = await self.metadata["__event_call__"](
                    {"type": "execute", "data": {"code": js}}
                )
                raw_str = (
                    raw
                    if isinstance(raw, str)
                    else (
                        (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                    )
                )
                try:
                    res_json = (
                        json.loads(raw_str)
                        if isinstance(raw_str, str) and raw_str.startswith("{")
                        else {"action": "cancel", "value": raw_str}
                    )
                except:
                    res_json = {"action": "cancel", "value": str(raw_str)}
                if res_json.get("action") == "continue":
                    return True
                await self.ui.emit_status("Planner stopped by user.", True)
            except Exception as e:
                logger.error(f"Iteration limit modal error: {e}")
                return False
        return False

    async def _execute_planner_turn(
        self,
        exec_history: list,
        total_emitted: str,
        valves: Any,
        user_obj: Any,
        body: dict,
        user_valves: Any,
    ) -> dict:
        """Performs a single LLM turn for the planner with live reasoning and clean content streaming."""
        tc_dict, content_chunks, reasoning_chunks = {}, [], []
        tools = self.registry.get_tools_spec(
            user_obj, user_valves, list(self.state.tasks.keys())
        )

        planner_body = {
            **body,
            "model": valves.PLANNER_MODEL,
            "messages": exec_history,
            "tools": tools,
            "metadata": self.metadata.get("__metadata__", {}),
        }

        # Inject model knowledge if knowledge_agent is NOT present
        if not getattr(valves, "ENABLE_KNOWLEDGE_AGENT", True) and self.model_knowledge:
            planner_body["metadata"]["knowledge"] = self.model_knowledge
            planner_body["metadata"]["__model_knowledge__"] = self.model_knowledge

        # Reasoning state for live emission (v3 parity)
        reasoning_buffer = ""
        reasoning_start_time = None
        total_emitted_base = total_emitted
        error_occurred = False

        async for event in Utils.get_streaming_completion(
            self.metadata.get("__request__"), planner_body, user_obj
        ):
            etype = event["type"]

            if etype == "reasoning":
                piece = event.get("text", "")
                if piece:
                    if reasoning_start_time is None:
                        reasoning_start_time = time.monotonic()
                    reasoning_chunks.append(piece)
                    reasoning_buffer += piece
                    # Clean tool calls and stray tags from reasoning display
                    display = Utils.hide_tool_calls(reasoning_buffer)
                    display = Utils.THINKING_TAG_CLEANER_PATTERN.sub("", display)
                    display = "\n".join(
                        f"> {l}" if not l.startswith(">") else l
                        for l in display.splitlines()
                    )
                    await self.ui.emit_replace(
                        total_emitted_base
                        + '\n\n<details type="reasoning" done="false">\n<summary>Thinking...</summary>\n'
                        + display
                        + "\n</details>\n\n"
                    )

            elif etype in ["content", "tool_calls"]:
                if reasoning_buffer:
                    # Seal the reasoning block with tool calls stripped
                    dur = (
                        round(time.monotonic() - reasoning_start_time)
                        if reasoning_start_time
                        else 1
                    )
                    display = Utils.hide_tool_calls(reasoning_buffer)
                    display = Utils.THINKING_TAG_CLEANER_PATTERN.sub("", display)
                    if display.strip():
                        display = "\n".join(
                            f"> {l}" if not l.startswith(">") else l
                            for l in display.splitlines()
                        )
                        total_emitted_base += (
                            f'\n\n<details type="reasoning" done="true" duration="{dur}">\n<summary>Thought for {dur} seconds</summary>\n'
                            + display
                            + "\n</details>\n\n"
                        )
                    reasoning_buffer = ""
                    await self.ui.emit_replace(
                        total_emitted_base + "".join(content_chunks)
                    )

                if etype == "content":
                    text = event["text"]
                    content_chunks.append(text)
                    # Clean thinking tags from content stream
                    display_content = Utils.clean_thinking("".join(content_chunks))
                    await self.ui.emit_replace(total_emitted_base + display_content)
                elif etype == "tool_calls":
                    for tc in event["data"]:
                        idx = tc["index"]
                        if idx not in tc_dict:
                            tc_dict[idx] = {
                                "id": tc.get("id") or f"call_{uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": tc["function"].get("name", ""),
                                    "arguments": "",
                                },
                            }
                        if "name" in tc["function"] and tc["function"]["name"]:
                            tc_dict[idx]["function"]["name"] = tc["function"]["name"]
                        if "arguments" in tc["function"]:
                            tc_dict[idx]["function"]["arguments"] += tc["function"][
                                "arguments"
                            ]

            elif etype == "error":
                error_msg = f"LLM Error: {event.get('text', 'Unknown failure')}"
                await self.ui.emit_status(error_msg, True)
                await self.ui.emit_replace(
                    total_emitted + f"\n\n> [!CAUTION]\n> {error_msg}"
                )
                error_occurred = True
                break

        if reasoning_buffer:  # Final seal if no content/tools followed
            dur = (
                round(time.monotonic() - reasoning_start_time)
                if reasoning_start_time
                else 1
            )
            display = Utils.hide_tool_calls(reasoning_buffer)
            display = Utils.THINKING_TAG_CLEANER_PATTERN.sub("", display)
            if display.strip():
                display = "\n".join(
                    f"> {l}" if not l.startswith(">") else l
                    for l in display.splitlines()
                )
                total_emitted_base += (
                    f'\n\n<details type="reasoning" done="true" duration="{dur}">\n<summary>Thought for {dur} seconds</summary>\n'
                    + display
                    + "\n</details>\n\n"
                )

        content, reasoning = "".join(content_chunks), "".join(reasoning_chunks)

        # XML Interception (v3 parity, DRY)
        tc_dict_content, content = Utils.extract_xml_tool_calls(content)
        tc_dict_reasoning, reasoning = Utils.extract_xml_tool_calls(reasoning)

        # Merge native and XML tool calls
        tc_dict = {**tc_dict, **tc_dict_content, **tc_dict_reasoning}

        return {
            "content": content,
            "tool_calls": tc_dict,
            "reasoning": reasoning,
            "raw_content": "".join(content_chunks),
            "total_emitted": total_emitted_base + content,
            "turn_start_base": total_emitted,
            "error": error_occurred,
        }

    async def _handle_tool_calls(
        self,
        tool_calls: list,
        exec_history: list,
        total_emitted: str,
        external_tools_dict: dict,
        chat_id: str,
        valves: Any,
        user_valves: Any,
        user_obj: Any,
        body: dict,
    ) -> str:
        """Executes tool calls while providing immediate UI feedback via Details tags."""

        for tc in sorted(tool_calls, key=lambda x: str(x.get("id", ""))):
            func_name, args_str, call_id = (
                tc["function"]["name"],
                tc["function"]["arguments"],
                tc.get("id", str(uuid4())),
            )

            args = Utils.parse_tool_arguments(args_str)

            resolved_args = Utils.resolve_dict_references(args, self.state.results)

            # UI Restoration: Build and emit tool call details (v3 parity)
            tc_tag = self.ui.build_tool_call_details(
                call_id, func_name, args_str, done=False
            )
            total_emitted += "\n\n" + tc_tag
            await self.ui.emit_replace(total_emitted)

            tool_res = ""
            try:
                match func_name:
                    case "update_state":
                        tool_res = await self.tools.update_state(resolved_args, user_valves)
                    case "call_subagent":
                        tool_res = await self.tools.call_subagent(
                            resolved_args, chat_id, valves, body, user_valves
                        )
                    case "ask_user":
                        tool_res = await self.tools.ask_user(resolved_args, valves)
                    case "give_options":
                        tool_res = await self.tools.give_options(resolved_args, valves)
                    case "read_task_result":
                        tool_res = self.tools.read_task_result(resolved_args)
                    case "review_tasks":
                        tool_res = await self.tools.review_tasks(
                            resolved_args, valves, body, user_obj
                        )
                    case name if name in external_tools_dict:
                        tool_data = external_tools_dict[name]
                        allowed_keys = (
                            tool_data.get("spec", {})
                            .get("parameters", {})
                            .get("properties", {})
                            .keys()
                        )
                        filtered_args = {
                            k: v for k, v in resolved_args.items() if k in allowed_keys
                        }
                        res = await tool_data["callable"](**filtered_args)
                        tc_return = process_tool_result(
                            self.metadata.get("__request__"),
                            name,
                            res,
                            tool_data.get("type", ""),
                            False,
                            self.metadata.get("__metadata__"),
                            user_obj,
                        )
                        tool_res = tc_return[0] if len(tc_return) > 0 else str(res)
                    case _:
                        # Tool not found: Log, and provide better feedback to LLM
                        error_msg = f"Tool {func_name} not found."

                        available_tools = ["call_subagent", "review_tasks"]
                        if user_valves.PLAN_MODE:
                            available_tools.append("update_state")
                        if user_valves.TASK_RESULT_TRUNCATION:
                            available_tools.append("read_task_result")
                        if user_valves.ENABLE_USER_INPUT_TOOLS:
                            available_tools.extend(["ask_user", "give_options"])

                        available_tools.extend(list(external_tools_dict.keys()))
                        error_msg += (
                            f" Available tools for the current planner session: "
                            f"{', '.join(available_tools)}."
                        )

                        logger.warning(f"[Planner] {error_msg}")
                        await self.ui.emit_status(f"Attempted unknown tool: {func_name}")
                        tool_res = f"Error: {error_msg}"
            except Exception as e:
                logger.error(f"[Planner] Error executing {func_name}: {e}")
                await self.ui.emit_status(f"Error executing {func_name}")
                tool_res = f"Error: {e}"

            exec_history.append(
                {
                    "role": "tool",
                    "content": tool_res,
                    "tool_call_id": call_id,
                    "name": func_name,
                }
            )

            # v3 parity: switch to evaluating status after tool results are in
            await self.ui.emit_status("Planner evaluating...")

            # Replace the "Executing..." tag with "Done" tag in the total_emitted string
            done_tag = self.ui.build_tool_call_details(
                call_id, func_name, args_str, done=True, result=tool_res
            )
            total_emitted = total_emitted.replace(tc_tag, done_tag)
            await self.ui.emit_replace(total_emitted)

        return total_emitted


    async def _phase_verification(
        self,
        exec_history: list,
        content: str,
        total_emitted: str,
        retries: int,
        valves: Any,
        user_valves: Any,
        user_obj: Any,
        body: dict,
    ) -> tuple[bool, str]:
        """Judge (Phase 3) verification of task states with live reasoning and structured output."""
        unresolved = [
            tid
            for tid, info in self.state.tasks.items()
            if info.status not in ["completed", "failed"]
        ]
        can_retry = user_valves.YOLO_MODE or (retries < valves.JUDGE_RETRY_LIMIT)

        if not unresolved or not can_retry:
            return False, total_emitted

        await self.ui.emit_status("Verifying task states...")
        judge_msg = (
            f"SYSTEM: Review the conversation. The following tasks are not marked as completed: {', '.join(unresolved)}. "
            "Respond with a JSON object (no extra text) containing:\n"
            "1. 'updates': array of {'task_id': string, 'status': 'completed'|'failed', 'description': string}\n"
            "2. 'follow_up_prompt': string - instructions to continue if tasks are genuinely incomplete, or empty string if all resolved."
        )

        if not exec_history or exec_history[-1].get("content") != content:
            exec_history.append({"role": "assistant", "content": content})

        judge_body = {
            **body,
            "model": valves.REVIEW_MODEL or valves.PLANNER_MODEL,
            "messages": exec_history + [{"role": "user", "content": judge_msg}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task_id": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": ["completed", "failed"],
                                        },
                                        "description": {"type": "string"},
                                    },
                                    "required": ["task_id", "status", "description"],
                                    "additionalProperties": False,
                                },
                            },
                            "follow_up_prompt": {"type": "string"},
                        },
                        "required": ["updates", "follow_up_prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            "metadata": self.metadata.get("__metadata__", {}),
        }

        max_judge_retries = 1
        current_judge_retry = 0

        while current_judge_retry <= max_judge_retries:
            judge_chunks = []
            reasoning_chunks = []
            reasoning_start_time = time.monotonic()

            async for ev in Utils.get_streaming_completion(
                self.metadata.get("__request__"), judge_body, user_obj
            ):
                etype = ev["type"]
                if etype == "reasoning":
                    reasoning_chunks.append(ev.get("text", ""))
                elif etype == "content":
                    judge_chunks.append(ev["text"])

            try:
                # Clean thinking tags from judge output before JSON parsing
                raw = Utils.clean_thinking("".join(judge_chunks))
                reasoning = "".join(reasoning_chunks)

                brace_start = raw.find("{")
                if brace_start != -1:
                    judge_res = json.loads(raw[brace_start:])
                    updated = False
                    for upd in judge_res.get("updates", []):
                        tid, status = upd.get("task_id"), upd.get("status")
                        if tid in self.state.tasks and status in ["completed", "failed"]:
                            self.state.update_task(tid, status, upd.get("description"))
                            updated = True

                    if updated:
                        await self.ui.emit_html_embed(self.state.tasks)

                    follow_up = judge_res.get("follow_up_prompt", "").strip()
                    still_incomplete = [
                        tid
                        for tid, info in self.state.tasks.items()
                        if info.status not in ["completed", "failed"]
                    ]

                    if still_incomplete and follow_up:
                        exec_history.append(
                            {
                                "role": "user",
                                "content": f"SYSTEM: The following tasks are still incomplete: {', '.join(still_incomplete)}. {follow_up}",
                            }
                        )

                        # If judge proposed a follow-up, show its reasoning as a "Thought" block (v3/v4 hybrid)
                        if reasoning.strip():
                            dur = round(time.monotonic() - reasoning_start_time)
                            display = "\n".join(
                                f"> {l}" if not l.startswith(">") else l
                                for l in reasoning.splitlines()
                            )
                            total_emitted += (
                                f'\n\n<details type="reasoning" done="true" duration="{dur}">\n<summary>Judge Verification Feedback</summary>\n'
                                + display
                                + "\n</details>\n\n"
                            )

                        await self.ui.emit_status("Continuing based on judge feedback...")
                        return True, total_emitted
                    break
                else:
                    raise ValueError(f"No JSON brace found in judge response: {raw}")
            except Exception as e:
                if current_judge_retry < max_judge_retries:
                    logger.warning(f"Judge verification parsing failed: {e}. Retrying...")
                    judge_body["messages"].append({"role": "assistant", "content": "".join(judge_chunks)})
                    judge_body["messages"].append(
                        {
                            "role": "user",
                            "content": "SYSTEM: Your verdict was not a valid JSON. Please return strictly a JSON object following the schema.",
                        }
                    )
                    current_judge_retry += 1
                    continue
                else:
                    logger.warning(f"Judge verification failed after retries: {e}")
                    break

        if (
            exec_history
            and exec_history[-1]["role"] == "assistant"
            and not exec_history[-1].get("tool_calls")
        ):
            exec_history.pop()
        return False, total_emitted


# ---------------------------------------------------------------------------
# Pipe (Open WebUI Manifold)
# ---------------------------------------------------------------------------


class Pipe:
    class Valves(BaseModel):
        PLANNER_MODEL: str = Field(
            default="",
            description="Mandatoy. The main model driving the planner, works Best with a Base Model (not workspace presets) | (must support Tool Calling and Structured Outputs and only native tool calling is supported) ",
        )
        OPEN_WEBUI_URL: str = Field(
            default="",
            description="The base URL of your Open WebUI instance (e.g. http://localhost:3000). Used for absolute file links in subagents.",
        )
        SUBAGENT_MODELS: str = Field(
            default="",
            description="Comma-separated list of model IDs available to be queried as subagents works best with Workspace Model presets | only native tool calling is supported",
        )
        TEMPERATURE: float = Field(
            default=0.7, description="Temperature for the planner agent"
        )
        TASK_RESULT_LIMIT: int = Field(
            default=4000,
            description="Character limit for subagent results before middle-truncation occurs.",
        )
        REVIEW_MODEL: str = Field(
            default="",
            description="Model used for review_tasks , works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)",
        )
        ENABLE_SUBAGENT_CHECK: bool = Field(
            default=False,
            description="Enable a judge model to verify subagent responses for task completion and correct asset referencing BEFORE returning to the planner.",
        )
        SUBAGENT_CHECK_MODEL: str = Field(
            default="",
            description="Model used for subagent verification (leave blank to use the planner model)",
        )
        SYSTEM_PROMPT: str = Field(
            default="""You are an advanced agentic Planner. You have the ability to formulate a plan, act on it by delegating tasks to specialized subagents or using tools, and track your progress.
Your goal is to fulfill the user's request.""",
            description="System Prompt for the planner agent",
        )
        USER_INPUT_TIMEOUT: int = Field(
            default=120,
            description="Timeout in seconds for user-input modal responses (ask_user / give_options). After this time the input is auto-skipped.",
        )
        MAX_PLANNER_ITERATIONS: int = Field(
            default=25,
            description="Maximum planner loop iterations before asking the user to continue or cancel. Set to 0 to disable.",
        )
        MAX_SUBAGENT_ITERATIONS: int = Field(
            default=25,
            description="Maximum tool-call iterations per subagent thread before asking the user to continue or cancel. Set to 0 to disable.",
        )
        JUDGE_RETRY_LIMIT: int = Field(
            default=1,
            description="Maximum number of judge verification (only on PLAN mode) retries when tasks are still incomplete. If YOLO mode is enabled, this is unlimited.",
        )
        ENABLE_TERMINAL_AGENT: bool = Field(
            default=True,
            description="Enable terminal subagent (only active when a terminal is attached to the request)",
        )
        TERMINAL_AGENT_MODEL: str = Field(
            default="",
            description="Model for the terminal agent, works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)",
        )
        ENABLE_IMAGE_GENERATION_AGENT: bool = Field(
            default=True, description="Enable built-in image generation subagent"
        )
        IMAGE_GENERATION_AGENT_MODEL: str = Field(
            default="",
            description="Model for the image generation agent , works Best with a Base Model (not workspace presets) |(leave blank to use the planner model)",
        )
        ENABLE_WEB_SEARCH_AGENT: bool = Field(
            default=True, description="Enable built-in web search subagent"
        )
        WEB_SEARCH_AGENT_MODEL: str = Field(
            default="",
            description="Model for the web search agent , works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)",
        )
        ENABLE_KNOWLEDGE_AGENT: bool = Field(
            default=True,
            description="Enable built-in knowledge, notes, and chat retrieval subagent",
        )
        KNOWLEDGE_AGENT_MODEL: str = Field(
            default="",
            description="Model for the knowledge agent , works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)",
        )
        ENABLE_CODE_INTERPRETER_AGENT: bool = Field(
            default=True,
            description="Enable built-in code interpreter subagent. Executes Python code and returns results. The code_interpreter tool is moved here exclusively.",
        )
        CODE_INTERPRETER_AGENT_MODEL: str = Field(
            default="",
            description="Model for the code interpreter agent, works best with a Base Model (not workspace presets) | (leave blank to use the planner model)",
        )
        CODE_INTERPRETER_TEMPERATURE: float = Field(
            default=0.1,
            description="Temperature for the code interpreter subagent. Low values (0.0-0.2) produce more deterministic, accurate code.",
        )

    class UserValves(BaseModel):
        PLAN_MODE: bool = Field(
            default=True,
            description="Enable Plan Mode with visual task state tracking (HTML plan embed, state updates, completion verification). When disabled, the agent delegates to subagents directly without structured planning overhead.",
        )
        ENABLE_USER_INPUT_TOOLS: bool = Field(
            default=True,
            description="Allow the planner to call ask_user and give_options to request clarification or choices from you during execution. Disable to let the planner run fully autonomously.",
        )
        YOLO_MODE: bool = Field(
            default=False,
            description="YOLO: disable all iteration limits for both the planner and subagents. The planner will run until it naturally finishes with no Continue/Cancel interruptions.",
        )
        TASK_RESULT_TRUNCATION: bool = Field(
            default=True,
            description="Enable middle-truncation for subagent task results to save context.",
        )
        ENABLE_PLAN_APPROVAL: bool = Field(
            default=False,
            description="Enable manual plan approval. After planning, you will be asked to Accept or provide Feedback (Ignored in YOLO mode).",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request = None,
        __metadata__: dict = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[None]] = None,
        __files__: list = None,
        __chat_id__: str = None,
        __message_id__: str = None,
        **kwargs,
    ) -> Union[str, Generator, AsyncGenerator]:
        """
        Main pipe entry point for Open WebUI.
        """
        # Ensure metadata is present even if not passed
        __metadata__ = __metadata__ or body.get("metadata", {})
        
        self.user_valves = (
            __user__.pop("valves", None)
            if isinstance(__user__, dict)
            else getattr(__user__, "valves", None)
        ) or self.UserValves()
        ui = UIRenderer(__event_emitter__, __event_call__)
        # Ensure __files__ is a list to handle new attachments during the turn
        if __files__ is None:
            __files__ = []

        # Consistent with v3 fallback logic
        pipe_metadata = __metadata__ or body.get("metadata", {}) or {}
        chat_id = (
            __chat_id__
            or pipe_metadata.get("chat_id")
            or body.get("chat_id")
            or body.get("id")
            or "default"
        )
        message_id = __message_id__ or pipe_metadata.get("message_id")

        # Resolve full user object (v3 parity)
        user_obj = Users.get_user_by_id(__user__.get("id"))

        # Comprehensive metadata for engine components (v3 parity + internal objects)
        metadata = {
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": pipe_metadata,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__user_obj__": user_obj,
            "__files__": __files__,
            "__chat_id__": chat_id,
            "__message_id__": message_id,
        }

        # Resolve base URL (Valve -> Env -> Request)
        base_url = self.valves.OPEN_WEBUI_URL
        if not base_url:
            base_url = os.environ.get("WEBUI_URL", "")
        if not base_url and __request__:
            base_url = str(__request__.base_url).rstrip("/")

        # Extract model knowledge and features for tool management
        model_knowledge = pipe_metadata.get("knowledge") or pipe_metadata.get(
            "model_knowledge"
        )
        app_models = getattr(__request__.app.state, "MODELS", {})
        planner_model_id = self.valves.PLANNER_MODEL
        planner_info = app_models.get(planner_model_id, {})
        planner_features = (
            planner_info.get("info", {}).get("meta", {}).get("features", {})
        )

        registry = ToolRegistry(
            self.valves,
            user_obj,
            __request__,
            pipe_metadata,
            model_knowledge=model_knowledge,
            planner_features=planner_features,
        )
        state = PlannerState()
        subagents = SubagentManager(
            ui,
            state,
            metadata,
            self.valves,
            base_url=base_url,
            model_knowledge=model_knowledge,
        )
        engine = PlannerEngine(
            ui, state, subagents, registry, metadata, model_knowledge=model_knowledge
        )

        return await engine.run(chat_id, self.valves, self.user_valves, body, __files__)
