"""
title: Planner v3
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 3.1.0
required_open_webui_version: 0.8.10
"""

import asyncio
import logging
import json
import re
import ast
import html as html_module
from uuid import uuid4
from typing import Callable, Awaitable, Any, Optional

from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.tools import get_tools, get_builtin_tools
from open_webui.utils.middleware import process_tool_result
from open_webui.models.models import Models
from open_webui.models.users import Users
from pydantic import BaseModel, Field

name = "Planner V3"

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

THINK_OPEN_PATTERN = re.compile(
    r"<(?:think|thinking|reason|reasoning|thought|Thought)>|\|begin_of_thought\|",
    re.IGNORECASE,
)
THINK_CLOSE_PATTERN = re.compile(
    r"</(?:think|thinking|reason|reasoning|thought|Thought)>|\|end_of_thought\|",
    re.IGNORECASE,
)

def clean_thinking_tags(message: str) -> str:
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL | re.IGNORECASE,
    )
    return re.sub(pattern, "", message).strip()

class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: Any
    __model__: str
    __request__: Any

    class Valves(BaseModel):
        PLANNER_MODEL: str = Field(
            default="", 
            description="The main model driving the planner (leave blank for current default model)"
        )
        SUBAGENT_MODELS: str = Field(
            default="", 
            description="Comma-separated list of model IDs available to be queried as subagents"
        )
        TEMPERATURE: float = Field(
            default=0.7, 
            description="Temperature for the planner agent"
        )
        SYSTEM_PROMPT: str = Field(
            default="""You are an advanced agentic Planner. You have the ability to formulate a plan, act on it by delegating tasks to specialized subagents or using tools, and track your progress.
Your goal is to fulfill the user's request.

You have access to the following built-in special tools:
1. `update_state(task_id: str, status: str, description: str)`: Use this to track the tasks you are working on. Call this tool whenever you start, update, or finish a logical step. "status" should be 'pending', 'in_progress', 'completed', or 'failed'.
2. `call_subagent(model_id: str, prompt: str, task_id: str)`: Use this to delegate a subtask to a specialized model. If you need it to read results from a previous subagent task, you can pass the previous task's ID as a reference string starting with "@" (e.g., "@task_1") in the prompt. It will be substituted with the full output of that task automatically.

You must:
- BE STRICT WITH STATE STRUCTURE. Do NOT lump your entire plan into a single task. Break your plan down into small, distinct steps and create a separate `update_state` entry for EACH step as 'pending' before executing them. 
- Methodically execute the steps, using `call_subagent` for complex analysis, generation, or reasoning steps.
- As you finish each small step, call `update_state` again to mark that specific task as 'completed'. This allows for small complete checks and visibility as you progress.
- Once the objective is complete, compile the final result. If your final result needs to include large outputs from previous subagents, simply include the "@task_id" reference in your final text response. It will be replaced automatically.
- Do not make up information. Use your tools.

Available Subagent Models:
{subagent_models}
""",
            description="System Prompt for the planner agent"
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def emit_status(self, message: str, done: bool = False):
        await self.__current_event_emitter__({
            "type": "status",
            "data": {"description": message, "done": done},
        })

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_replace(self, content: str):
        await self.__current_event_emitter__(
            {"type": "replace", "data": {"content": content}}
        )

    def _generate_html_embed(self, planner_state: dict) -> str:
        tasks_html = ""
        for task_id, task_info in planner_state.items():
            status = task_info.get("status", "pending")
            desc = task_info.get("description", "")
            
            status_colors = {
                "pending": "rgba(156, 163, 175, 1)",
                "in_progress": "rgba(59, 130, 246, 1)",
                "completed": "rgba(16, 185, 129, 1)",
                "failed": "rgba(239, 68, 68, 1)"
            }
            color = status_colors.get(status, "rgba(156, 163, 175, 1)")
            
            check_icon = {
                "pending": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle></svg>''',
                "in_progress": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>''',
                "completed": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>''',
                "failed": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'''
            }
            icon = check_icon.get(status, "")
            
            tasks_html += f'''
            <div style="margin-bottom: 12px; padding: 16px; background: rgba(0,0,0,0.15); border-left: 4px solid {color}; border-radius: 12px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: {color};">
                        {icon}
                        <strong style="color: #f8fafc; font-size: 14px; font-weight: 600; letter-spacing: 0.3px;">{task_id}</strong>
                    </div>
                    <span style="font-size: 10px; font-weight: 700; padding: 4px 10px; border-radius: 99px; background: {color.replace(', 1)', ', 0.15)')}; color: {color}; text-transform: uppercase; letter-spacing: 1px;">
                        {status.replace("_", " ")}
                    </span>
                </div>
                <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5; font-weight: 400; padding-left: 26px;">
                    {desc}
                </div>
            </div>
            '''

        if not tasks_html:
            tasks_html = '''
            <div style="padding: 16px; text-align: center; color: rgba(255,255,255,0.4); font-size: 13px; font-style: italic; background: rgba(0,0,0,0.1); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1);">
                Awaiting active plan formation...
            </div>
            '''

        html = f'''
        <div class="planner-embed" style="background: rgba(20, 20, 25, 0.7); backdrop-filter: blur(24px); border: 1px solid rgba(255,255,255,0.06); box-shadow: 0 30px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1); border-radius: 24px; padding: 24px; margin: 16px 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
                <div style="font-size: 26px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));">🧠</div>
                <div>
                    <h3 style="margin: 0; color: #fff; font-size: 18px; font-weight: 800; letter-spacing: -0.2px;">Planner Subagents</h3>
                    <p style="margin: 4px 0 0 0; font-size: 12px; color: #94a3b8; font-weight: 500;">Live Execution State</p>
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 4px;">
                {tasks_html}
            </div>
        </div>
        '''
        return html

    async def emit_html_embed(self, planner_state: dict):
        html = self._generate_html_embed(planner_state)
        await self.__current_event_emitter__({
            "type": "embeds",
            "data": {"embeds": [html]}
        })

    def resolve_action_references(self, text: str, action_results: dict[str, str]) -> str:
        """Replace @task_id references with their full content."""
        if not text or not isinstance(text, str):
            return text
        
        pattern = r"@([a-zA-Z0-9_-]+)"
        matches = re.findall(pattern, text)
        for match in matches:
            if match in action_results:
                replacement = action_results[match]
                text = text.replace(f"@{match}", replacement)
        return text

    def resolve_dict_references(self, params: dict, action_results: dict[str, str]) -> dict:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str):
                resolved[k] = self.resolve_action_references(v, action_results)
            elif isinstance(v, dict):
                resolved[k] = self.resolve_dict_references(v, action_results)
            elif isinstance(v, list):
                resolved_list = []
                for item in v:
                    if isinstance(item, str):
                        resolved_list.append(self.resolve_action_references(item, action_results))
                    elif isinstance(item, dict):
                        resolved_list.append(self.resolve_dict_references(item, action_results))
                    else:
                        resolved_list.append(item)
                resolved[k] = resolved_list
            else:
                resolved[k] = v
        return resolved

    # ── Tool calling details emission ──────────────────────────────────────────

    def _build_tool_call_details(
        self,
        call_id: str,
        name: str,
        arguments: str,
        done: bool = False,
        result=None,
    ) -> str:
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

    # ── Streaming completion ──────────────────────────────────────────────────

    def _parse_sse_events(self, buffer: str) -> tuple[list[dict], str, bool]:
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
            except json.JSONDecodeError:
                pass

        return events, buffer, done

    def _extract_stream_events(self, event_payload: dict):
        choices = event_payload.get("choices", [])
        if not choices:
            return

        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta", {}) or {}

        reasoning_keys = ["reasoning", "reasoning_content", "thinking"]
        for reasoning_key in reasoning_keys:
            reasoning_text = delta.get(reasoning_key)
            if isinstance(reasoning_text, str) and reasoning_text:
                yield {"type": "reasoning", "text": reasoning_text}

        content = delta.get("content")
        if isinstance(content, str) and content:
            yield {"type": "content", "text": content}

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            yield {"type": "tool_calls", "data": tool_calls}

    # Removed manual tool instruction injection based on user constraint

    async def get_streaming_completion(self, messages, model_id: str, body: dict, tools: list = None):
        form_data = {**body}
        form_data["model"] = model_id
        form_data["messages"] = messages
        form_data["temperature"] = self.valves.TEMPERATURE
        form_data["stream"] = True
        
        if "tools" in form_data:
            del form_data["tools"]

        if tools:
            form_data["tools"] = tools

        response = await generate_chat_completion(
            self.__request__,
            form_data,
            user=self.__user__,
            bypass_filter=True,
            bypass_system_prompt=True,
        )

        if not hasattr(response, "body_iterator"):
            raise ValueError("Response does not support streaming")

        sse_buffer = ""
        async for chunk in response.body_iterator:
            decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            sse_buffer += decoded
            events, sse_buffer, done = self._parse_sse_events(sse_buffer)
            for event_payload in events:
                for event in self._extract_stream_events(event_payload):
                    yield event
            if done:
                break

    # ── Main Pipe Loop ─────────────────────────────────────────────────────────

    async def pipe(
        self, 
        body: dict, 
        __user__: dict, 
        __task__=None, 
        __tools__=None, 
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __event_call__: Callable[[Any], Awaitable[None]] = None,
        __request__=None
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__event_call__ = __event_call__
        self.__user__ = Users.get_user_by_id(__user__.get("id"))
        self.__request__ = __request__
        self.__model__ = body.get("model", "")

        messages = body.get("messages", [])
        if not messages:
            await self.emit_message("No messages provided.")
            return "No messages provided."

        model_id = self.valves.PLANNER_MODEL
        subagents_str = self.valves.SUBAGENT_MODELS
        subagents_list = [m.strip() for m in subagents_str.split(",") if m.strip()]
        
        # Retrieve model descriptions dynamically from app state
        app_models = getattr(self.__request__.app.state, "MODELS", {})
        subagent_descriptions = []
        for m in subagents_list:
            model_info = app_models.get(m, {})
            model_name = model_info.get("name", m)
            meta = model_info.get("info", {}).get("meta", {})
            desc = meta.get("description", "No description available")
            subagent_descriptions.append(f"- ID: {m} (Name: {model_name})\n  Description: {desc}")
        
        system_prompt = self.valves.SYSTEM_PROMPT.replace(
            "{subagent_models}", 
            "\n".join(subagent_descriptions) if subagent_descriptions else "None specified in config"
        )

        sys_message = {"role": "system", "content": system_prompt}
        planner_messages = [sys_message] + messages

        metadata = body.get("metadata", {})
        tool_ids = metadata.get("tool_ids", [])
        if not tool_ids:
            tool_ids = body.get("tool_ids", [])
            
        extra_params = {
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": metadata,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
        }
            
        tools_dict = {}
        if tool_ids:
            tools_dict = await get_tools(self.__request__, tool_ids, self.__user__, extra_params)

        # Preload and cache subagent tools to avoid redundant DB calls on every tool execution
        subagent_tools_cache = {}
        for sub_model in subagents_list:
            subagent_tool_ids = []
            model_info = app_models.get(sub_model, {})
            model_db_info = Models.get_model_by_id(sub_model)
            model_system_message = ""
            
            if model_db_info:
                meta = model_db_info.meta.model_dump() if hasattr(model_db_info.meta, "model_dump") else model_db_info.meta
                params = model_db_info.params.model_dump() if hasattr(model_db_info.params, "model_dump") else model_db_info.params
                if isinstance(meta, dict):
                    subagent_tool_ids.extend(meta.get("toolIds", []))
                if isinstance(params, dict):
                    subagent_tool_ids.extend(params.get("toolIds", []) or params.get("tools", []))
                    model_system_message = params.get("system", "") or ""
                    
            if model_info:
                info_meta = model_info.get("info", {}).get("meta", {})
                info_params = model_info.get("info", {}).get("params", {})
                subagent_tool_ids.extend(info_meta.get("toolIds", []))
                subagent_tool_ids.extend(info_params.get("toolIds", []) or info_params.get("tools", []))
                if not model_system_message:
                    model_system_message = info_params.get("system", "") or ""
                
            p_features = dict(body.get("features", {}))
            if model_db_info:
                meta = model_db_info.meta.model_dump() if hasattr(model_db_info.meta, "model_dump") else model_db_info.meta
                params = model_db_info.params.model_dump() if hasattr(model_db_info.params, "model_dump") else model_db_info.params

                for obj in [meta, params, model_info.get("info", {}).get("meta", {}), model_info.get("info", {}).get("params", {})]:
                    if isinstance(obj, dict):
                        if isinstance(obj.get("features"), dict):
                            p_features.update(obj["features"])
                        for f_id in obj.get("defaultFeatureIds", []):
                            p_features[f_id] = True
                            
            combined_tool_ids = list(set([*tool_ids, *subagent_tool_ids]))
            s_tools_dict = {}
            if combined_tool_ids:
                s_tools_dict = await get_tools(self.__request__, combined_tool_ids, self.__user__, extra_params)
                
            try:
                builtin_tools = get_builtin_tools(self.__request__, extra_params, features=p_features, model=model_info)
                if builtin_tools:
                    s_tools_dict.update(builtin_tools)
            except Exception as e:
                logging.error(f"Failed to load built-in tools for subagent {sub_model}: {e}")
                
            s_tools = [{"type": "function", "function": t["spec"]} for t in s_tools_dict.values()] if s_tools_dict else None
            subagent_tools_cache[sub_model] = {
                "dict": s_tools_dict,
                "specs": s_tools,
                "system_message": model_system_message
            }

        planner_state = {}
        action_results = {}
        total_emitted = ""

        # Emit initial html embed state
        await self.emit_html_embed(planner_state)

        tools_spec = [
            {
                "type": "function",
                "function": {
                    "name": "update_state",
                    "description": "Track task progress by providing a task id, status, and description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "Unique identifier for the task"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"]},
                            "description": {"type": "string", "description": "Short description of the task"}
                        },
                        "required": ["task_id", "status", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_subagent",
                    "description": "Call a specialized subagent model to perform a task. Returns the output from the model.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_id": {"type": "string", "description": "The ID of the model to use"},
                            "prompt": {"type": "string", "description": "Detailed instructions for the subagent"},
                            "task_id": {"type": "string", "description": "Unique ID to assign to this subagent task for tracking and future @task_id referencing"}
                        },
                        "required": ["model_id", "prompt", "task_id"]
                    }
                }
            }
        ]

        # Add external tools to tools_spec
        for tool_name, tool_data in tools_dict.items():
            if "spec" in tool_data:
                tools_spec.append({"type": "function", "function": tool_data["spec"]})

        while True:
            try:
                import time
                content_chunks = []
                tool_calls_dict = {}
                reasoning_buffer = ""
                reasoning_start_time = None
                
                async for event in self.get_streaming_completion(planner_messages, model_id, body=body, tools=tools_spec):
                    event_type = event["type"]
                    
                    if event_type == "reasoning":
                        reasoning_piece = event.get("text", "")
                        if reasoning_piece:
                            if reasoning_start_time is None:
                                reasoning_start_time = time.monotonic()
                            reasoning_buffer += reasoning_piece
                            display = "\n".join(
                                f"> {line}" if not line.startswith(">") else line
                                for line in reasoning_buffer.splitlines()
                            )
                            await self.emit_replace(
                                total_emitted
                                + '<details type="reasoning" done="false">\n'
                                + "<summary>Thinking...</summary>\n"
                                + display
                                + "\n</details>\n\n"
                            )
                        
                    elif event_type in ["content", "tool_calls"]:
                        if reasoning_buffer:
                            reasoning_duration = (
                                round(time.monotonic() - reasoning_start_time)
                                if reasoning_start_time else 1
                            )
                            display = "\n".join(
                                f"> {line}" if not line.startswith(">") else line
                                for line in reasoning_buffer.splitlines()
                            )
                            total_emitted += (
                                f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                + display
                                + "\n</details>\n\n"
                            )
                            reasoning_buffer = ""
                            await self.emit_replace(total_emitted)

                        if event_type == "content":
                            text = event["text"]
                            content_chunks.append(text)
                            total_emitted += text
                            await self.emit_message(text)
                        elif event_type == "tool_calls":
                            for tc in event["data"]:
                                idx = tc["index"]
                                if idx not in tool_calls_dict:
                                    tool_calls_dict[idx] = {
                                        "id": tc.get("id"),
                                        "function": {"name": tc["function"].get("name", ""), "arguments": ""}
                                    }
                                if "name" in tc["function"] and tc["function"]["name"]:
                                    tool_calls_dict[idx]["function"]["name"] = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_calls_dict[idx]["function"]["arguments"] += tc["function"]["arguments"]

                if reasoning_buffer:
                    reasoning_duration = (
                        round(time.monotonic() - reasoning_start_time)
                        if reasoning_start_time else 1
                    )
                    display = "\n".join(
                        f"> {line}" if not line.startswith(">") else line
                        for line in reasoning_buffer.splitlines()
                    )
                    total_emitted += (
                        f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                        f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                        + display
                        + "\n</details>\n\n"
                    )
                    reasoning_buffer = ""
                    await self.emit_replace(total_emitted)

                final_content = "".join(content_chunks)

                # Intercept hallucinated `<tool_call>` XML emitted into the body
                if not tool_calls_dict and "<tool_call>" in final_content:
                    xml_tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', final_content, re.DOTALL)
                    for idx, tc_data in enumerate(xml_tool_calls):
                        func_match = re.search(r'<function=([^>]+)>', tc_data)
                        if func_match:
                            func_name = func_match.group(1).strip()
                            kwargs = {}
                            param_matches = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', tc_data, re.DOTALL)
                            for p_name, p_val in param_matches:
                                kwargs[p_name.strip()] = p_val.strip()
                                
                            tool_calls_dict[idx] = {
                                "id": str(uuid4()),
                                "function": {
                                    "name": func_name,
                                    "arguments": json.dumps(kwargs)
                                }
                            }
                        else:
                            try:
                                data = json.loads(tc_data.strip())
                                if isinstance(data, dict) and "name" in data:
                                    tool_calls_dict[idx] = {
                                        "id": str(uuid4()),
                                        "function": {
                                            "name": data["name"],
                                            "arguments": json.dumps(data.get("arguments", data.get("parameters", {})))
                                        }
                                    }
                            except:
                                pass
                    final_content = re.sub(r'<tool_call>.*?</tool_call>', '', final_content, flags=re.DOTALL).strip()

                # Append message to history
                assistant_message = {"role": "assistant", "content": final_content}
                if tool_calls_dict:
                    tool_calls_list = list(tool_calls_dict.values())
                    assistant_message["tool_calls"] = tool_calls_list
                planner_messages.append(assistant_message)

                if not tool_calls_dict:
                    # Final response from planner - stream final resolved context if any
                    resolved_content = self.resolve_action_references(final_content, action_results)
                    if resolved_content != final_content:
                        # Resend the cleaned up and fully replaced final message
                        total_emitted = total_emitted.replace(final_content, resolved_content)
                        await self.emit_replace(total_emitted)
                    break

                # Execute sequential/parallel tools from dict
                tool_calls = list(tool_calls_dict.values())
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments_str = tool_call["function"]["arguments"]
                    call_id = tool_call.get("id", str(uuid4()))
                    
                    try:
                        args = json.loads(arguments_str)
                    except Exception:
                        try:
                            args = ast.literal_eval(arguments_str)
                        except:
                            args = {}

                    args = self.resolve_dict_references(args, action_results)

                    # Emit "Executing..." tag
                    executing_tag = self._build_tool_call_details(call_id, function_name, arguments_str, done=False)
                    if not total_emitted.endswith("\n"):
                        total_emitted += "\n"
                    total_emitted += executing_tag
                    await self.emit_replace(total_emitted)

                    tool_result_str = ""
                    
                    if function_name == "update_state":
                        task_id = args.get("task_id", "")
                        status = args.get("status", "pending")
                        desc = args.get("description", "")
                        if task_id:
                            planner_state[task_id] = {
                                "status": status,
                                "description": desc
                            }
                            await self.emit_html_embed(planner_state)
                            # Add tiny sleep to prevent race conditions with frontend embeds renderer resetting
                            await asyncio.sleep(0.1)
                            await self.emit_replace(total_emitted)
                            tool_result_str = f"State updated for {task_id}"
                    
                    elif function_name == "call_subagent":
                        sub_model = args.get("model_id", "")
                        sub_prompt = args.get("prompt", "")
                        sub_task_id = args.get("task_id", "")
                        
                        await self.emit_status(f"[Subagent: {sub_model}] Executing {sub_task_id}...", False)
                        
                        cached = subagent_tools_cache.get(sub_model, {})
                        sub_sys = cached.get("system_message", "")
                        if not sub_sys:
                            sub_sys = f"You are a specialized subagent acting as {sub_model}. Follow the prompt directly and accurately using your tools."
                            
                        # Ensure subagents understand they aren't interacting with the UI directly
                        sub_sys += "\n\nCRITICAL CONTEXT: You are running as a headless subagent entirely in the background. DO NOT return markdown elements that rely on Open WebUI UI embeds for tool generation output. Any tools you use (like generate_image, search, etc.) will return URLs, base64 data, or raw paths. You MUST return these raw HTML references, URLs, or file paths unconditionally in your final reply so the main planner can use them."
                            
                        sub_messages = [
                            {"role": "system", "content": sub_sys},
                            {"role": "user", "content": sub_prompt}
                        ]
                        
                        try:
                            sub_tools_dict = cached.get("dict", {})
                            sub_tools = cached.get("specs", None)
                            
                            sub_final_answer_chunks = []
                            
                            while True:
                                sub_tc_dict = {}
                                
                                async for sub_event in self.get_streaming_completion(sub_messages, sub_model, body=body, tools=sub_tools):
                                    event_type = sub_event["type"]
                                    if event_type == "content":
                                        sub_final_answer_chunks.append(sub_event["text"])
                                    elif event_type == "tool_calls":
                                        for tc in sub_event["data"]:
                                            idx = tc["index"]
                                            if idx not in sub_tc_dict:
                                                sub_tc_dict[idx] = {
                                                    "id": tc.get("id"),
                                                    "function": {"name": tc["function"].get("name", ""), "arguments": ""}
                                                }
                                            if "name" in tc["function"] and tc["function"]["name"]:
                                                sub_tc_dict[idx]["function"]["name"] = tc["function"]["name"]
                                            if "arguments" in tc["function"]:
                                                sub_tc_dict[idx]["function"]["arguments"] += tc["function"]["arguments"]
                                                    
                                sub_content = "".join(sub_final_answer_chunks)
                                sub_final_answer_chunks = [] # Reset for next cycle if tool calls happen
                                
                                # Intercept hallucinated `<tool_call>` XML for subagents
                                if not sub_tc_dict and "<tool_call>" in sub_content:
                                    xml_tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', sub_content, re.DOTALL)
                                    for idx, tc_data in enumerate(xml_tool_calls):
                                        func_match = re.search(r'<function=([^>]+)>', tc_data)
                                        if func_match:
                                            func_name = func_match.group(1).strip()
                                            kwargs = {}
                                            param_matches = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', tc_data, re.DOTALL)
                                            for p_name, p_val in param_matches:
                                                kwargs[p_name.strip()] = p_val.strip()
                                                
                                            sub_tc_dict[idx] = {
                                                "id": str(uuid4()),
                                                "function": {
                                                    "name": func_name,
                                                    "arguments": json.dumps(kwargs)
                                                }
                                            }
                                        else:
                                            try:
                                                data = json.loads(tc_data.strip())
                                                if isinstance(data, dict) and "name" in data:
                                                    sub_tc_dict[idx] = {
                                                        "id": str(uuid4()),
                                                        "function": {
                                                            "name": data["name"],
                                                            "arguments": json.dumps(data.get("arguments", data.get("parameters", {})))
                                                        }
                                                    }
                                            except:
                                                pass
                                    sub_content = re.sub(r'<tool_call>.*?</tool_call>', '', sub_content, flags=re.DOTALL).strip()
                                
                                if not sub_tc_dict:
                                    # No more tools, we have the final answer!
                                    sub_final_answer_chunks.append(sub_content)
                                    break
                                
                                # Process subagent tool calls
                                tool_calls_list = list(sub_tc_dict.values())
                                sub_messages.append({"role": "assistant", "content": sub_content, "tool_calls": tool_calls_list})
                                
                                for stc in tool_calls_list:
                                    stc_name = stc["function"]["name"]
                                    stc_args_str = stc["function"]["arguments"]
                                    call_id = stc.get("id", str(uuid4()))
                                    
                                    try:
                                        stc_args_obj = json.loads(stc_args_str)
                                    except:
                                        try:
                                            stc_args_obj = ast.literal_eval(stc_args_str)
                                        except:
                                            stc_args_obj = {}
                                            
                                    target_tool = sub_tools_dict.get(stc_name)
                                    if target_tool:
                                        tc_res = await target_tool["callable"](**stc_args_obj)
                                        tc_return = process_tool_result(self.__request__, stc_name, tc_res, target_tool.get("type", ""), False, body.get("metadata",{}), self.__user__)
                                        res_str = tc_return[0] if len(tc_return) > 0 else str(tc_res)
                                        
                                        sub_messages.append({
                                            "role": "tool",
                                            "tool_call_id": call_id,
                                            "name": stc_name,
                                            "content": str(res_str)
                                        })
                                    else:
                                        err_res = f"Error: Tool {stc_name} not found"
                                        sub_messages.append({
                                            "role": "tool",
                                            "tool_call_id": call_id,
                                            "name": stc_name,
                                            "content": err_res
                                        })

                            tool_result_str = "".join(sub_final_answer_chunks)
                            if sub_task_id:
                                action_results[sub_task_id] = tool_result_str
                                
                        except Exception as e:
                            logger.error(f"Subagent error: {e}")
                            tool_result_str = f"Error calling subagent: {e}"

                    elif function_name in tools_dict:
                        tool_data = tools_dict[function_name]
                        try:
                            allowed_keys = tool_data.get("spec", {}).get("parameters", {}).get("properties", {}).keys()
                            filtered_args = {k: v for k, v in args.items() if k in allowed_keys}
                            
                            res = await tool_data["callable"](**filtered_args)
                            tool_return = process_tool_result(self.__request__, function_name, res, tool_data.get("type", ""), False, body.get("metadata", {}), self.__user__)
                            tool_result_str = tool_return[0] if len(tool_return) > 0 else str(res)
                        except Exception as e:
                            tool_result_str = f"Error executing tool {function_name}: {e}"
                    else:
                        tool_result_str = f"Error: Function {function_name} is not recognized or available."

                    # Context Truncation
                    context_result_str = tool_result_str
                    if len(context_result_str) > 2000:
                        ref_id = args.get("task_id", function_name)
                        context_result_str = context_result_str[:1800] + f"\\n...[TRUNCATED. Full output saved. Use '@{ref_id}' to reference the full output later]..."

                    # Complete executing tag in the UI stream
                    total_emitted = total_emitted.replace(executing_tag, "")
                    done_tag = self._build_tool_call_details(
                        call_id, function_name, arguments_str, done=True, result=context_result_str
                    )
                    
                    if not total_emitted.endswith("\n"):
                        total_emitted += "\n"
                    total_emitted += done_tag
                    await self.emit_replace(total_emitted)

                    planner_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": context_result_str
                    })

            except Exception as e:
                logger.error(f"Planner loop error: {e}")
                err_str = f"\n\n**Error:** {str(e)}"
                total_emitted += err_str
                await self.emit_message(err_str)
                break

        await self.emit_status("Planner execution complete.", True)
        return total_emitted
