"""
title: Planner v3
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 3.8.0
required_open_webui_version: 0.8.12

Features:
- **Autonomous Multi-Agent Orchestration**: Dynamic task decomposition and delegation to specialized subagents.
- **Agentic Planning & Self-Correction**: Iterative planning with user-in-the-loop approval and feedback cycles.
- **Parallel Execution (v3.15+)**: Concurrent execution of tool calls and subagent tasks using `asyncio.gather` for significantly faster performance.
- **Native Open WebUI Integration**:
    - **Skills Support**: Automatic resolution and inclusion of user skills in the system prompt.
    - **Terminal Integration**: Direct interactive terminal access via `terminal_agent`.
    - **Native Tool Parity**: Mirrors Open WebUI's logic for built-in tools (web search, image gen, knowledge, code interpreter).
- **Specialized Virtual Subagents**: `web_search_agent`, `image_gen_agent`, `knowledge_agent`, `code_interpreter_agent`, and `terminal_agent`.
- **Interactive UI Components**:
    - **Plan Approval Modal**: Review and modify plans before execution.
    - **Real-time Progress Tracker**: High-fidelity HTML/JS visualization of the execution state.
    - **Step-by-Step User Interaction**: `ask_user` and `give_options` tools for middle-of-turn feedback.
- **Robust State Persistence**: Automatic saving/recovery of task states and history across chat turns via JSON attachments.
- **High-Performance Architecture**: `QueueEmitter` for non-blocking, throttled UI updates and MCP resilience patches.

Requirements (Open WebUI parity):
- Native function calling only: models must use OpenAI-style ``tools`` on the API body.
  Non-native tool modes are not supported.
- Built-in virtual subagents (image_gen_agent, web_search_agent, knowledge_agent,
  code_interpreter_agent, terminal_agent) use fixed role features so the pipe works
  without duplicating workspace toggles; disable a role via planner valves if needed.
- Models listed in SUBAGENT_MODELS / workspace terminal lists use the same tool
  surface as their saved workspace model (toolIds, MCP, terminal, and builtins only
  when native FC and meta.capabilities.builtin_tools allow).

Known Issues:
- Filter for subagents are not tested.
- Parallel subagents in the same environment that manage files on the same environment (Code intepreter / terminal agents)
  concurrently may inherently cause issues.
- Parallel subagents in local environments that require loading and unloading another external models (Local LLMs + Comfyui)risk OOM errors.
"""
import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
import copy
import html as html_module
from uuid import uuid4
from typing import (
    Callable,
    Awaitable,
    Any,
    Optional,
    Union,
    Generator,
    AsyncGenerator,
    Dict,
    List,
)
from pydantic import BaseModel, Field
from fastapi import Request, UploadFile
from starlette.datastructures import Headers
import io

from open_webui.utils.chat import (
    generate_chat_completion as generate_raw_chat_completion,
)
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import (
    process_tool_result,
    add_file_context,
    chat_completion_files_handler,
    get_system_oauth_token,
)
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.access_control import has_connection_access
from open_webui.utils.misc import is_string_allowed
from open_webui.utils.headers import include_user_info_headers
from open_webui.env import (
    ENABLE_FORWARD_USER_INFO_HEADERS,
    FORWARD_SESSION_INFO_HEADER_CHAT_ID,
    FORWARD_SESSION_INFO_HEADER_MESSAGE_ID,
)
from open_webui.models.models import Models
from open_webui.models.users import Users
from open_webui.models.files import Files
from open_webui.models.chats import Chats
from open_webui.models.skills import Skills
from open_webui.routers.files import upload_file_handler

# --- ComfyUI Parallelism Patch (v14) ---
# Monkeypatch ComfyUI router functions to use unique client IDs per call,
# resolving WebSocket deadlocks when multiple subagents generate images for the same user.
try:
    import open_webui.routers.images as images_router

    def _apply_comfy_patch():
        for func_name in ["comfyui_create_image", "comfyui_edit_image"]:
            orig_func = getattr(images_router, func_name, None)
            if orig_func and not hasattr(orig_func, "_patched"):

                async def patched_func(*args, **kwargs):
                    # client_id is the 3rd positional argument (model, payload, client_id, ...)
                    new_args = list(args)
                    if len(new_args) >= 3:
                        new_args[2] = f"{new_args[2]}_{uuid.uuid4().hex[:8]}"
                    elif "client_id" in kwargs:
                        kwargs["client_id"] = (
                            f"{kwargs['client_id']}_{uuid.uuid4().hex[:8]}"
                        )
                    return await orig_func(*new_args, **kwargs)

                patched_func._patched = True
                setattr(images_router, func_name, patched_func)

    _apply_comfy_patch()
except Exception as e:
    logging.error(f"Failed to apply ComfyUI parallelism patch: {e}")


# --- MCP Parallelism & Resilience Patch (v15) ---
# Monkeypatch MCPClient to use unique identifiers per connection,
# similar to ComfyUI, avoiding session collisions and status leaks.
try:
    from open_webui.utils.mcp.client import MCPClient

    _orig_mcp_connect = MCPClient.connect

    async def _patched_mcp_connect(self, url: str, headers: Optional[dict] = None):
        # 1. Provide a unique identity to the session via headers.
        # This helps backend servers isolate concurrent requests from the same user/agent.
        new_headers = dict(headers) if headers else {}
        uid = uuid.uuid4().hex[:8]
        new_headers.setdefault("X-MCP-Client-ID", f"planner_{uid}")

        # Some SSE/HTTP implementations use User-Agent or custom headers for pinning.
        orig_ua = new_headers.get("User-Agent", "OpenWebUI-MCP-Client")
        new_headers["User-Agent"] = f"{orig_ua} (Planner-{uid})"

        # 2. Call original connect with isolated headers
        return await _orig_mcp_connect(self, url, new_headers)

    if not hasattr(MCPClient, "_patched_mcp"):
        MCPClient.connect = _patched_mcp_connect
        MCPClient._patched_mcp = True
except Exception as e:
    logging.error(f"Failed to apply MCP parallelism patch: {e}")

# v3.6: Suppress extremely noisy MCP validation warnings that can saturate the logs and block the receive loop.
# The mcp library uses loguru, so standard logging.getLogger doesn't work.
try:
    from loguru import logger as loguru_logger
    import sys

    # Add a custom filter to handle 'Failed to validate notification' errors from the MCP SDK
    # These are caused by non-standard notifications (like notifications/message) from some servers.
    def _mcp_log_filter(record):
        if (
            "mcp.shared.session" in record.get("name", "")
            and "Failed to validate notification" in record["message"]
        ):
            return False
        return True

    # Note: Accessing loguru's global state is invasive, but necessary for the 'Function' environment.
    # OpenWebUI usually has its own loguru config; we attempt to add our filter if it exists.
    if hasattr(loguru_logger, "remove") and hasattr(loguru_logger, "add"):
        # v3.6.8: Use remove() + add() instead of configure() to avoid wiping existing OWUI handlers.
        # We try to find any existing stderr handler to replace, or just add ours.
        try:
            loguru_logger.add(sys.stderr, filter=_mcp_log_filter, level="DEBUG")
        except Exception:
            # Fallback if add fails
            if hasattr(loguru_logger, "configure"):
                loguru_logger.configure(
                    handlers=[{"sink": sys.stderr, "filter": _mcp_log_filter}]
                )
except Exception as e:
    logging.debug(f"Failed to apply loguru filter: {e}")


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


class PlannerContext(BaseModel):
    """Unified context for the planner during a single turn."""

    request: Request
    user: Any
    metadata: dict[str, Any]
    event_emitter: Any
    event_call: Optional[Any] = None
    valves: Any
    user_valves: Any
    body: dict[str, Any]
    planner_info: dict[str, Any]
    model_knowledge: Optional[list[dict]] = None
    chat_id: str
    message_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# --- New Helper Classes ---
class QueueEmitter:
    """Non-blocking event emitter that pushes to an asyncio.Queue for background processing."""

    def __init__(self, queue: asyncio.Queue, engine=None):
        self.queue = queue
        self.engine = engine
        self.last_emit_time = 0.0
        self._lock = asyncio.Lock()  # v3.6.8: Protect shared mutable state
        self.throttle_interval = 0.05  # 50ms (20fps)

    async def __call__(self, event: dict) -> None:
        # v3.15: Intercept native Open WebUI tool file events to populate the engine state trackers dynamically.
        # This allows built-in tools (like generate_image) to survive replace wipes and stream termination.
        if event.get("type") == "chat:message:files" and self.engine:
            files = event.get("data", {}).get("files", [])
            if files:
                # 1. Protect against local emit replace wipes
                async with self.engine._files_lock:
                    self.engine.produced_files.extend(files)

                # 2. Mutate global metadata context to securely write state into the Open WebUI core backend DB at exit
                metadata_files = self.engine.metadata.get("__files__")
                if isinstance(metadata_files, list):
                    existing_urls = {
                        f.get("url") for f in metadata_files if isinstance(f, dict)
                    }
                    for f in files:
                        if isinstance(f, dict) and f.get("url") not in existing_urls:
                            metadata_files.append(f)

                    # v3.15: CRITICAL FIX. `__files__` is sometimes a detached list instantiated loosely by
                    # Open WebUI's `functions.py`. We MUST forcibly inject it back into `__metadata__["files"]`
                    # so the main router saves it during the websocket stream clean-up phase.
                    raw_metadata_obj = self.engine.metadata.get("__metadata__")
                    if isinstance(raw_metadata_obj, dict):
                        raw_metadata_obj["files"] = metadata_files

        # v3.6: Aggressive throttling for status updates in the producer phase.
        # This keeps the queue from bloating with redundant progress messages.
        is_status = event.get("type") == "status"
        # v3.6.11: Exempt 'done=True' status updates from throttling to ensure
        # UI state consistency (especially for grouped tool calls).
        is_done_status = is_status and event.get("data", {}).get("done") is True

        if is_status and not is_done_status:
            async with self._lock:
                now = time.monotonic()
                if now - self.last_emit_time < self.throttle_interval:
                    return
                self.last_emit_time = now
                # v3.15: Put nowait inside lock for atomic throttle-and-push
                try:
                    self.queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass
            return

        # Non-blocking fire-and-forget for non-throttled events
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            # Under extreme pressure, drop status updates to protect event loop
            if not is_status:
                await self.queue.put(event)  # Block for important events (content)
            else:
                pass

    async def emit_status(self, text: str, done: bool = False):
        await self({"type": "status", "data": {"description": text, "done": done}})


class UIState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    total_emitted: str
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)


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


def merge_workspace_model_dict(
    app_models: dict[str, Any], model_id: str
) -> dict[str, Any]:
    """Merge app.state.MODELS entry with DB model for OWUI-shaped workspace config (tool + builtin parity)."""
    # v3.15: Rely on app_models (app.state.MODELS) which already contains merged workspace state.
    # ELIMINATE synchronous DB hit for Models.get_model_by_id.
    m = copy.deepcopy(app_models.get(model_id) or {"id": model_id, "info": {}})
    if "info" not in m:
        m["info"] = {}
    return m


def workspace_feature_flags(model: dict[str, Any]) -> dict[str, bool]:
    """Features saved on the model (meta + params), same sources OWUI merges for workspace models."""
    feats: dict[str, bool] = {}
    info = model.get("info") or {}
    for key in ("meta", "params"):
        block = info.get(key) or {}
        if isinstance(block.get("features"), dict):
            for fk, fv in block["features"].items():
                feats[fk] = bool(fv)
    return feats


async def planner_merge_mcp_tools_from_ids(
    request: Request,
    user: Any,
    tool_ids: list[str],
    metadata: dict[str, Any],
    extra_params: dict[str, Any],
    event_emitter: Any = None,
    mcp_handler: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    (get_tools in OWUI does not implement MCP).
    """
    out: dict[str, Any] = {}
    if not tool_ids or not request:
        return out

    oauth_token = None
    try:
        oauth_token = await get_system_oauth_token(request, user)
    except Exception:
        pass

    resolved_servers = set()
    for tool_id in tool_ids:
        if not isinstance(tool_id, str):
            continue

        # v15: Improve resolution to handle unprefixed IDs or mcp: prefix
        server_id = None
        if tool_id.startswith("server:mcp:"):
            server_id = tool_id[len("server:mcp:") :]
        elif tool_id.startswith("mcp:"):
            server_id = tool_id[len("mcp:") :]
            # v3.6.8: Optimize by skipping unknown IDs faster if we have many connections
            mcp_connections = getattr(
                request.app.state.config, "TOOL_SERVER_CONNECTIONS", []
            )
            for server_connection in mcp_connections:
                if server_connection.get("type", "") == "mcp":
                    sid = server_connection.get("info", {}).get("id")
                    if not sid:
                        continue
                    if tool_id == sid or tool_id.startswith(f"{sid}_"):
                        server_id = sid
                        break

        if not server_id:
            logger.debug(
                f"[Planner] tool_id '{tool_id}' not resolved as MCP. Prefix check failed."
            )
            continue

        if server_id in resolved_servers:
            continue
        resolved_servers.add(server_id)

        try:
            mcp_server_connection = None
            for server_connection in request.app.state.config.TOOL_SERVER_CONNECTIONS:
                if (
                    server_connection.get("type", "") == "mcp"
                    and server_connection.get("info", {}).get("id") == server_id
                ):
                    mcp_server_connection = server_connection
                    break

            if not mcp_server_connection:
                logger.error(
                    "MCP server with id %s not found in connections. Available server IDs: %s",
                    server_id,
                    [
                        s.get("info", {}).get("id")
                        for s in request.app.state.config.TOOL_SERVER_CONNECTIONS
                        if s.get("type") == "mcp"
                    ],
                )
                continue

            if not has_connection_access(user, mcp_server_connection):
                logger.warning(
                    "Access denied to MCP server %s for user %s",
                    server_id,
                    getattr(user, "id", ""),
                )
                continue

            auth_type = mcp_server_connection.get("auth_type", "")
            headers: dict[str, str] = {}
            if auth_type == "bearer":
                headers["Authorization"] = (
                    f"Bearer {mcp_server_connection.get('key', '')}"
                )
            elif auth_type == "none":
                pass
            elif auth_type == "session":
                tok = getattr(getattr(request, "state", None), "token", None)
                creds = getattr(tok, "credentials", None) if tok else None
                if creds:
                    headers["Authorization"] = f"Bearer {creds}"
            elif auth_type == "system_oauth":
                if oauth_token:
                    headers["Authorization"] = (
                        f"Bearer {oauth_token.get('access_token', '')}"
                    )
            elif auth_type == "oauth_2.1":
                try:
                    splits = server_id.split(":")
                    sid = splits[-1] if len(splits) > 1 else server_id
                    mgr = getattr(request.app.state, "oauth_client_manager", None)
                    if mgr:
                        ot = await mgr.get_oauth_token(user.id, f"mcp:{sid}")
                        if ot:
                            headers["Authorization"] = (
                                f"Bearer {ot.get('access_token', '')}"
                            )
                except Exception as e:
                    logger.error("OAuth token for MCP: %s", e)

            connection_headers = mcp_server_connection.get("headers", None)
            if connection_headers and isinstance(connection_headers, dict):
                for key, value in connection_headers.items():
                    headers[key] = value

            if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                headers = include_user_info_headers(headers, user)
                # v3.15: Support both original and double-underscore prefixed session keys
                # Check both metadata (inner) and extra_params (wide) for resilience
                cid = None
                if isinstance(metadata, dict):
                    cid = metadata.get("chat_id") or metadata.get("__chat_id__")
                if not cid and isinstance(extra_params, dict):
                    cid = extra_params.get("chat_id") or extra_params.get("__chat_id__")

                if cid:
                    headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = cid

                mid = None
                if isinstance(metadata, dict):
                    mid = metadata.get("message_id") or metadata.get("__message_id__")
                if not mid and isinstance(extra_params, dict):
                    mid = extra_params.get("message_id") or extra_params.get("__message_id__")

                if mid:
                    headers[FORWARD_SESSION_INFO_HEADER_MESSAGE_ID] = mid

            if mcp_handler is not None:
                client = await mcp_handler(
                    url=mcp_server_connection.get("url", ""),
                    headers=headers if headers else None,
                )
            else:
                # Fallback to local client (original behavior, legacy or fallback)
                client = MCPClient()
                await client.connect(
                    url=mcp_server_connection.get("url", ""),
                    headers=headers if headers else None,
                )

            # v3.6: Ensure each client has a lock to prevent concurrent tool calls on the same connection,
            # which can cause deadlocks in some MCP servers or client implementations.
            if not hasattr(client, "_call_lock"):
                client._call_lock = asyncio.Lock()

            function_name_filter_list = mcp_server_connection.get("config", {}).get(
                "function_name_filter_list", ""
            )
            if isinstance(function_name_filter_list, str):
                function_name_filter_list = function_name_filter_list.split(",")

            # v3.6: Serialize list_tool_specs to avoid transport-level deadlocks
            # when multiple subagents initialize tools for the same server in parallel.
            async with client._call_lock:
                tool_specs = await client.list_tool_specs()

            def make_tool_function(mcp_client, function_name, sid):
                async def tool_function(**kwargs):
                    try:
                        logger.debug(
                            f"[MCP] Calling '{function_name}' on server '{sid}' with args: {kwargs}"
                        )
                        # v3.6: Use the per-client lock to serialize calls to this server.
                        async with mcp_client._call_lock:
                            result = await mcp_client.call_tool(
                                function_name,
                                function_args=kwargs,
                            )
                        logger.debug(
                            f"[MCP] Raw result from '{function_name}' on '{sid}': {result}"
                        )

                        # MCP returns CallToolResult with 'content' list
                        if hasattr(result, "content") and result.content:
                            texts = []
                            for c in result.content:
                                if hasattr(c, "text") and c.text:
                                    texts.append(c.text)
                                elif hasattr(c, "image"):
                                    texts.append(
                                        f"[Image Content: {c.image[:50]}...]"
                                        if isinstance(c.image, str)
                                        else "[Image Content]"
                                    )
                                else:
                                    texts.append(str(c))

                            final_res = "\n".join(texts)
                            logger.debug(
                                f"[MCP] Unpacked result (len {len(final_res)}): {final_res[:100]}..."
                            )
                            return final_res

                        if hasattr(result, "isError") and result.isError:
                            return f"MCP Error from {sid}: {result}"

                        return str(result)
                    except Exception as e:
                        logger.error(
                            f"Failed to call MCP tool '{function_name}' on '{sid}': {e}",
                            exc_info=True,
                        )
                        return f"Error calling MCP tool: {e}"

                return tool_function

            for tool_spec in tool_specs:
                if function_name_filter_list:
                    if not is_string_allowed(
                        tool_spec["name"], function_name_filter_list
                    ):
                        continue
                tool_function = make_tool_function(client, tool_spec["name"], server_id)
                out[f'{server_id}_{tool_spec["name"]}'] = {
                    "spec": {
                        **tool_spec,
                        "name": f'{server_id}_{tool_spec["name"]}',
                    },
                    "callable": tool_function,
                    "type": "mcp",
                    "client": client,
                    "direct": False,
                }
        except Exception as e:
            logger.debug("MCP tool load failed for %s: %s", tool_id, e)
            if event_emitter:
                try:
                    await event_emitter(
                        {
                            "type": "chat:message:error",
                            "data": {
                                "error": {
                                    "content": f"Failed to connect to MCP server '{server_id}'"
                                }
                            },
                        }
                    )
                except Exception:
                    pass
            continue

    return out


async def apply_native_completion_file_prep(
    request: Any,
    body: dict[str, Any],
    user: Any,
    workspace_model: dict[str, Any],
    pipe_metadata: dict[str, Any],
    event_emitter: Any,
    has_builtin_tools_in_payload: bool,
) -> None:
    """
    Mirror middleware: add_file_context when native FC + builtins are used;
    then chat_completion_files_handler when file_context capability is enabled.
    Mutates body in place (messages + metadata).
    """
    if not request or not user or not body:
        return

    async def _noop_emitter(_ev: dict) -> None:
        return None

    emitter = event_emitter if event_emitter is not None else _noop_emitter

    info_params = (workspace_model.get("info") or {}).get("params") or {}
    fc = info_params.get("function_calling", "native")
    caps = ((workspace_model.get("info") or {}).get("meta") or {}).get(
        "capabilities"
    ) or {}
    builtin_tools_enabled = caps.get("builtin_tools", True)
    chat_id = pipe_metadata.get("chat_id")
    msgs = body.get("messages")
    if (
        msgs is not None
        and fc == "native"
        and builtin_tools_enabled
        and body.get("tools")
        and has_builtin_tools_in_payload
    ):
        # v3.6: Copy messages to avoid mutating the caller's body dict in place
        body["messages"] = add_file_context(copy.deepcopy(msgs), chat_id, user)

    file_context_enabled = caps.get("file_context", True)
    if not file_context_enabled:
        return

    udump = user.model_dump() if hasattr(user, "model_dump") else {}
    extra = {
        "__event_emitter__": emitter,
        "__metadata__": pipe_metadata,
        "__user__": udump,
        "__request__": request,
    }
    try:
        body, _flags = await chat_completion_files_handler(request, body, extra, user)
    except Exception as e:
        logger.error("chat_completion_files_handler (planner parity): %s", e)


def unpack_terminal_tools_result(result: Any) -> tuple[dict, Any]:
    """
    Open WebUI get_terminal_tools returns (tools_dict, system_prompt) on success
    but a bare dict on failure paths (not found, access denied, empty specs).
    """
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    if isinstance(result, dict):
        return result, None
    return {}, None


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
        - Strips UI-only artifacts (<details> tags for reasoning, status).
        - Preserves semantic tool results from <details type="tool_calls">.
        """
        distilled = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Preserve tool-role messages verbatim — they contain tool results, not UI markup
            if role == "tool":
                if content:
                    distilled.append(
                        {
                            "role": "tool",
                            "content": content,
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "name": msg.get("name", ""),
                        }
                    )
                continue

            # 1. Flatten content if it's a list (Open WebUI / OpenAI structured format)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)

            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            # 2. Strip ONLY reasoning/thinking/status blocks — these are pure UI decoration
            content = re.sub(
                r'<details\s+type="(?:reasoning|status|state)".*?>.*?</details>',
                "",
                content,
                flags=re.DOTALL,
            ).strip()

            # 3. For tool_calls blocks: strip the HTML wrapper but preserve the semantic content
            def replace_tool_details(match):
                block = match.group(0)
                # v3.5: If this is the inline "Execution Plan", strip it completely from history
                if 'id="plan"' in block:
                    return ""

                name_m = re.search(r'name="([^"]+)"', block)
                result_m = re.search(r'result="([^"]*)"', block)
                if name_m and result_m:
                    r_text = html_module.unescape(result_m.group(1))
                    try:
                        # result is double-encoded JSON in the attribute
                        r_data = json.loads(r_text)
                        if isinstance(r_data, str):
                            r_text = (
                                r_data[:200] + "..." if len(r_data) > 200 else r_data
                            )
                        else:
                            r_text = (
                                str(r_data)[:200] + "..."
                                if len(str(r_data)) > 200
                                else str(r_data)
                            )
                    except Exception:
                        pass
                    return f"[Tool: {name_m.group(1)} → {r_text}]"
                return ""

            content = re.sub(
                r'<details\s+type="tool_calls".*?>.*?</details>',
                replace_tool_details,
                content,
                flags=re.DOTALL,
            ).strip()

            if content:
                # v3.6: Robustly clean any stray reasoning tags or prefixes from history to avoid confusion
                distilled.append(
                    {"role": role, "content": Utils.clean_thinking(content)}
                )
            elif msg.get("tool_calls"):
                # Always preserve assistant turns with tool_calls even if content is empty
                distilled.append(
                    {"role": role, "content": "", "tool_calls": msg.get("tool_calls")}
                )

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
        Handles both XML-style tags and common text prefixes (Thinking:, etc.).
        """
        if not text:
            return ""
        # 1. Remove complete pairs (case-insensitive)
        text = re.sub(
            r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"\|begin_of_thought\|.*?\|end_of_thought\|",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # 2. Handle unclosed tags but PROTECT tool calls (stop before them)
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

        # 2. Hide tool calls from display
        text = Utils.hide_tool_calls(text)

        # 3. Robustly remove reasoning prefixes (v3.6: handles common "Thinking: " style outputs)
        # We use a regex to ensure we don't leave stray colons, dots, or spaces.
        text = re.sub(
            r"(?i)^(?:Thinking|Thought|Reasoning|Analysis)[:\s.*-]*\s*",
            "",
            text,
            flags=re.MULTILINE,
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
    def clean_ui_artifacts(text: str) -> str:
        """
        Strips UI-specific HTML components (like <details>, <iframe, <thinking>)
        that may have been added by Open WebUI's result processing.
        Preserves model reasoning if it's not wrapped in UI-only tags.
        """
        if not text or not isinstance(text, str):
            return text

        # 1. Strip reasoning/status details — these are pure UI decoration
        text = re.sub(
            r'<details\s+type="(?:reasoning|status|state)".*?>.*?</details>',
            "",
            text,
            flags=re.DOTALL,
        )
        # 2. Strip thinking/thought tags - redundant with clean_thinking but safe here
        text = re.sub(
            r"<(?:thinking|thought)>.*?</(?:thinking|thought)>",
            "",
            text,
            flags=re.DOTALL,
        )

        return text.strip()

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
                    decoded = (
                        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                    )
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
                    yield {
                        "type": "error",
                        "text": f"LLM Provider Error: {error_detail}",
                    }
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

        # Use a safe regex to find @task_id only if NOT preceded by word characters or dots (like in an email)
        for i in range(len(parts)):
            if not parts[i].startswith("<details"):
                for tid, result in results.items():
                    # Safer regex: must not follow a word character or dot, followed by the exact ID and a word boundary
                    pattern = rf"(?<![\w.-])@{re.escape(tid)}\b"
                    # Escape backslashes in the replacement string for re.sub
                    safe_result = result.replace("\\", "\\\\")
                    parts[i] = re.sub(pattern, safe_result, parts[i])

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
                # Safer regex: must not follow a word character or dot, followed by the exact ID and a word boundary
                pattern = rf"(?<![\w.-])@{re.escape(tid)}\b"
                # Escape backslashes in the replacement string for re.sub
                safe_result = result.replace("\\", "\\\\")
                data = re.sub(pattern, safe_result, data)
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
        self._subagent_metadata: dict[Any, Any] = {}

    @property
    def tasks(self) -> dict[str, TaskStateModel]:
        return self._tasks

    @property
    def results(self) -> dict[str, str]:
        return self._results

    @property
    def subagent_history(self) -> dict[Any, Any]:
        return self._subagent_history

    @property
    def subagent_metadata(self) -> dict[Any, Any]:
        return self._subagent_metadata

    def update_task(self, task_id: str, status: str, description: str = None) -> None:
        if not task_id or not isinstance(task_id, str) or not task_id.strip():
            return
        if task_id not in self._tasks:
            self._tasks[task_id] = TaskStateModel(status="pending", description="")
        self._tasks[task_id].status = status
        if description:
            self._tasks[task_id].description = description

    def store_result(self, task_id: str, result: str) -> None:
        self._results[task_id] = result

    def get_history(self, chat_id: str, sub_task_id: str, model_id: str) -> list[Any]:
        # v3.15: Use JSON array as a robust composite key (escapes colons in model/task IDs)
        key = json.dumps([chat_id, sub_task_id, model_id], ensure_ascii=False)
        # Check new key format
        if key in self._subagent_history:
            return self._subagent_history[key]
        # Legacy fallback (colon-separated)
        legacy_key = f"{chat_id}:{sub_task_id}:{model_id}"
        return self._subagent_history.get(legacy_key, [])

    def set_history(
        self, chat_id: str, sub_task_id: str, model_id: str, messages: list[Any]
    ) -> None:
        # v3.15: Standardize on JSON array keys for robustness
        key = json.dumps([chat_id, sub_task_id, model_id], ensure_ascii=False)
        self._subagent_history[key] = messages

    def get_metadata(self, chat_id: str, sub_task_id: str, model_id: str) -> dict[str, Any]:
        # v3.15: Use JSON array as a robust composite key (escapes colons in model/task IDs)
        key = json.dumps([chat_id, sub_task_id, model_id], ensure_ascii=False)
        return self._subagent_metadata.get(key, {})

    def set_metadata(
        self, chat_id: str, sub_task_id: str, model_id: str, metadata: dict[str, Any]
    ) -> None:
        # v3.15: Standardize on JSON array keys for robustness
        key = json.dumps([chat_id, sub_task_id, model_id], ensure_ascii=False)
        self._subagent_metadata[key] = metadata


class UIRenderer:
    def __init__(
        self,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
        event_call: Optional[Callable[[dict[str, Any]], Awaitable[Any]]] = None,
        ui_queue: Optional[asyncio.Queue] = None,
    ):
        self._real_emitter = event_emitter
        # v3.6.8 Structural Overhaul: Use QueueEmitter if queue provided, else fallback to direct emitter.
        if ui_queue is not None:
            self.emitter = QueueEmitter(ui_queue)
        else:
            # Fallback (direct emission via provided event_emitter)
            # If no queue is provided, QueueEmitter would orphaned. We wrap the original emitter instead.
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
        event = {"type": "status", "data": {"description": message, "done": done}}
        # v3.6.12 Handle terminal events synchronously to avoid pending drops if the worker cancels instantly
        if done and hasattr(self, "_real_emitter") and self._real_emitter:
            await self._real_emitter(event)
        else:
            await self.emitter(event)

    async def emit_files(self, files: list) -> None:
        event = {"type": "chat:message:files", "data": {"files": files}}
        if hasattr(self, "_real_emitter") and self._real_emitter:
            await self._real_emitter(event)
        else:
            await self.emitter(event)

    async def emit_replace(self, content: str) -> None:
        await self.emitter({"type": "replace", "data": {"content": content}})

    def build_tool_call_details(
        self,
        call_id: str,
        name: str,
        arguments: str,
        done: bool = False,
        result: str = "",
        files: list = None,
        embeds: list = None,
        duration: str = "",
    ) -> str:
        """Constructs the HTML for a tool call to be embedded in the chat message (v3 parity)."""
        args_escaped = html_module.escape(arguments)
        result_str = str(result) if result is not None else ""
        res_escaped = html_module.escape(result_str)

        # v3.15: CRITICAL FIX. We must escape JSON strings for HTML attributes (parity with plan block).
        files_escaped = html_module.escape(json.dumps(files)) if files else ""
        embeds_escaped = (
            html_module.escape(json.dumps(embeds, ensure_ascii=False)) if embeds else ""
        )

        # Default summary for the expanded view
        default_summary = "Tool Executed"
        if not done:
            default_summary = "Executing\u2026"

        return (
            f'<details type="tool_calls" done="{"true" if done else "false"}" id="{call_id}" '
            f'name="{name}" arguments="{args_escaped}" result="{res_escaped}" '
            f'files="{files_escaped}" embeds="{embeds_escaped}" duration="{duration}">\n'
            f"<summary>{default_summary}</summary>\n</details>\n"
        )

    def get_html_status_block(self, planner_state: dict[str, Any]) -> str:
        """Generates the inline HTML block for the execution plan (v3.5 integrated style)."""
        html_content = self._generate_html_embed(planner_state)

        # Wrap in details type="tool_calls" with embeds attribute to trigger integrated UI
        # Result attribute contains a placeholder string serialized for the frontend
        embeds_json = json.dumps([html_content], ensure_ascii=False)
        embeds_escaped = html_module.escape(embeds_json)

        return (
            f'<details type="tool_calls" done="true" id="plan" '
            f'name="Execution Plan" '
            f'arguments="&quot;&quot;" '
            f'result="&quot;Plan updated.&quot;" '
            f'embeds="{embeds_escaped}">\n'
            f"<summary>Execution Plan</summary>\n</details>\n"
        )

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

class StatePersistence:
    """Handles loading and saving planner state across turns via JSON files attached to the chat."""

    def __init__(
        self,
        context: PlannerContext,
        state: PlannerState,
        ui: UIRenderer,
    ):
        self.context = context
        self.state = state
        self.ui = ui
        self.metadata = context.metadata
        self.request = context.request
        self.user = context.user

    async def recover_from_files(self, body: dict, chat_id: str, current_files: list = None) -> None:
        """Exact same logic as the old _recover_state_from_files, but now in its own class."""
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
            chat_id = self.context.chat_id
            if chat_id:
                logger.info(
                    f"Performing deep history scan for chat {chat_id} via database..."
                )
                chat_obj = await asyncio.to_thread(Chats.get_chat_by_id, chat_id)
                if chat_obj and hasattr(chat_obj, "chat"):
                    messages_map = chat_obj.chat.get("history", {}).get("messages", {})
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
                                    logger.info(
                                        f"Successfully recovered state from DB history: {name}"
                                    )
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
            logger.info(
                "No planner state file found in current turn, database, or history."
            )
            return

        try:
            file_id = state_file.get("file_id") or state_file.get("id")
            if not file_id:
                logger.warning(f"State file found but missing ID: {state_file}")
                return

            if Files is None:
                logger.error(
                    "Global 'Files' model table is None - cannot recover state."
                )
                return

            file_obj = await asyncio.to_thread(Files.get_file_by_id, file_id)
            if not file_obj:
                logger.warning(f"State file with ID {file_id} not found in database.")
                return

            file_path = getattr(file_obj, "path", None)
            if not file_path and hasattr(file_obj, "meta"):
                file_path = file_obj.meta.get("path")

            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)

                # Restore tasks
                if "tasks" in data:
                    self.state.tasks.clear()
                    for tid, tdata in data["tasks"].items():
                        self.state.update_task(
                            tid,
                            tdata.get("status", "pending"),
                            tdata.get("description", ""),
                        )

                # Restore results
                if "results" in data:
                    self.state.results.clear()
                    self.state.results.update(data["results"])

                # Restore subagent history
                if "subagent_history" in data:
                    self.state.subagent_history.clear()
                    for key_str, history in data["subagent_history"].items():
                        try:
                            parts = None
                            if key_str.startswith("[") and key_str.endswith("]"):
                                parts = json.loads(key_str)
                            elif ":" in key_str:
                                parts = key_str.split(":", 2)
                            else:
                                parts = (
                                    ast.literal_eval(key_str)
                                    if isinstance(key_str, str)
                                    else key_str
                                )

                            if isinstance(parts, (list, tuple)) and len(parts) == 3:
                                norm_key = json.dumps([chat_id, parts[1], parts[2]], ensure_ascii=False)
                                self.state.subagent_history[norm_key] = history
                            else:
                                self.state.subagent_history[str(key_str)] = history
                        except Exception as e:
                            logger.error(
                                f"Failed to reconstruct history key {key_str}: {e}"
                            )

                # Restore subagent metadata
                if "subagent_metadata" in data:
                    self.state.subagent_metadata.clear()
                    for key_str, metadata in data["subagent_metadata"].items():
                        try:
                            parts = None
                            if key_str.startswith("[") and key_str.endswith("]"):
                                parts = json.loads(key_str)
                            elif ":" in key_str:
                                parts = key_str.split(":", 2)
                            else:
                                parts = (
                                    ast.literal_eval(key_str)
                                    if isinstance(key_str, str)
                                    else key_str
                                )

                            if isinstance(parts, (list, tuple)) and len(parts) == 3:
                                norm_key = json.dumps([chat_id, parts[1], parts[2]], ensure_ascii=False)
                                self.state.subagent_metadata[norm_key] = metadata
                            else:
                                self.state.subagent_metadata[str(key_str)] = metadata
                        except Exception as e:
                            logger.error(
                                f"Failed to reconstruct metadata key {key_str}: {e}"
                            )

                status_msg = (
                    f"Recovered state from {state_file.get('name', 'latest file')}"
                )
                logger.info(status_msg)
                await self.ui.emit_status(status_msg)
            else:
                logger.warning(f"State file path not found or invalid: {file_path}")
        except Exception as e:
            logger.error(f"Failed to recover planner state: {e}", exc_info=True)

    async def save_to_file(self, chat_id: str) -> None:
        """Exact same logic as the old _save_state_to_file."""
        emitter = self.context.event_emitter
        request = self.request
        user = self.user

        if not emitter or not request or not user:
            return

        try:
            state_data = {
                "tasks": {
                    tid: {"status": t.status, "description": t.description}
                    for tid, t in self.state.tasks.items()
                },
                "results": self.state.results,
                "subagent_history": self.state.subagent_history,
                "subagent_metadata": self.state.subagent_metadata,
            }

            filename = f"planner_state_{chat_id}.json"
            content = json.dumps(state_data, ensure_ascii=False).encode("utf-8")

            file_upload = UploadFile(
                file=io.BytesIO(content),
                filename=filename,
                headers=Headers({"content-type": "application/json"}),
            )

            file_item = upload_file_handler(
                request=request, file=file_upload, metadata={}, process=False, user=user
            )

            if file_item:
                file_id = getattr(file_item, "id", None)
                if file_id:
                    file_info = {"file_id": str(file_id), "name": filename}

                    # 1. Update the internal list for this turn
                    internal_files = self.metadata.get("__files__")
                    if isinstance(internal_files, list):
                        internal_files.append(file_info)

                    # 2. Synchronize directly with the database for multi-turn persistence
                    target_chat_id = self.context.chat_id or chat_id
                    target_msg_id = self.context.message_id

                    if target_chat_id and target_msg_id:
                        try:
                            # Direct DB update
                            await asyncio.to_thread(
                                Chats.add_message_files_by_id_and_message_id,
                                target_chat_id,
                                target_msg_id,
                                [file_info],
                            )
                            logger.info(
                                f"Successfully persisted state file to chat {target_chat_id} message {target_msg_id}"
                            )
                        except Exception as db_err:
                            logger.error(
                                f"Failed to synchronize state file with database: {db_err}"
                            )

                    # 3. Emit event for immediate UI feedback
                    await emitter(
                        {"type": "chat:message:files", "data": {"files": [file_info]}}
                    )

                    return file_info
        except Exception as e:
            logger.error(f"Failed to save state to file: {e}")
        return None


class MCPHub:
    """Manages MCP client lifecycle, connection deduplication, and cleanup."""

    def __init__(self, valves: Any):
        self.valves = valves
        self.mcp_clients: dict[str, dict[str, Any]] = {}
        self._mcp_lock: asyncio.Lock = asyncio.Lock()
        self._connecting_mcp: dict[str, asyncio.Future] = {}

    async def get_or_create_client(
        self, url: str, headers: Optional[dict] = None
    ) -> "StreamableHTTPMCPClient":
        headers_json = json.dumps(headers, sort_keys=True) if headers else "{}"
        conn_key = hashlib.sha256(f"{url}|{headers_json}".encode()).hexdigest()

        async with self._mcp_lock:
            if conn_key in self.mcp_clients:
                return self.mcp_clients[conn_key]["proxy"]

            if conn_key in self._connecting_mcp:
                fut = self._connecting_mcp[conn_key]
            else:
                fut = asyncio.get_running_loop().create_future()
                self._connecting_mcp[conn_key] = fut

                async def _connect(ck=conn_key, f=fut):
                    try:
                        client = StreamableHTTPMCPClient(
                            url,
                            headers,
                            timeout=int(self.valves.SUBAGENT_TIMEOUT),
                        )
                        await client.connect()
                        async with self._mcp_lock:
                            self.mcp_clients[ck] = {"proxy": client}
                            self._connecting_mcp.pop(ck, None)
                        if not f.done():
                            f.set_result(True)
                    except Exception as e:
                        async with self._mcp_lock:
                            self._connecting_mcp.pop(ck, None)
                        if not f.done():
                            f.set_exception(e)

                asyncio.create_task(_connect(), name=f"MCP-Connect-{conn_key[:8]}")

        await fut
        return self.mcp_clients[conn_key]["proxy"]

    async def stop(self) -> None:
        if self._connecting_mcp:
            logger.info(f"Cleaning up {len(self._connecting_mcp)} pending MCP connections...")
            for fut in list(self._connecting_mcp.values()):
                if not fut.done():
                    fut.cancel()
            self._connecting_mcp.clear()

        if self.mcp_clients:
            logger.info(f"Disconnecting {len(self.mcp_clients)} MCP clients...")
            for info in self.mcp_clients.values():
                await info["proxy"].disconnect()
            self.mcp_clients.clear()


class StreamableHTTPMCPClient:
    """
    Per-server MCP client for Streamable HTTP transport.
    Each call is an independent cancellable task — no worker queue.
    """

    def __init__(self, url: str, headers: Optional[dict], timeout: int = 120):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._client: Optional[MCPClient] = None
        self._call_lock = asyncio.Lock()  # kept for ToolRegistry compatibility

    async def connect(self):
        async with asyncio.timeout(60.0):
            self._client = MCPClient()
            await self._client.connect(url=self.url, headers=self.headers)

    async def list_tool_specs(self) -> list[dict]:
        return await self._call_with_timeout(self._client.list_tool_specs)

    async def call_tool(self, name: str, function_args: Optional[dict] = None) -> Any:
        return await self._call_with_timeout(
            self._client.call_tool, name, function_args=function_args or {}
        )

    async def _call_with_timeout(self, coro_fn, *args, **kwargs):
        """
        Wraps each call in an explicit Task so task.cancel() propagates
        correctly through anyio checkpoints, bypassing asyncio.timeout
        unreliability on SSE receive loops.
        """

        async def _inner():
            return await coro_fn(*args, **kwargs)

        task = asyncio.create_task(_inner())
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            raise asyncio.TimeoutError(
                f"MCP call to {self.url} timed out after {self.timeout}s"
            )
        except Exception:
            if not task.done():
                task.cancel()
            raise

    async def disconnect(self):
        if self._client:
            try:
                async with asyncio.timeout(3.0):
                    await self._client.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Tool Management
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self, context: PlannerContext):
        self.context = context
        self.engine: Optional["PlannerEngine"] = None
        self.valves = context.valves
        self.user = context.user
        self.request = context.request
        self.wide_metadata = context.metadata
        self.pipe_metadata = context.metadata.get("__metadata__", {})
        self.model_knowledge = context.model_knowledge
        self.planner_info = context.planner_info
        self.planner_features = (
            self.planner_info.get("info", {}).get("meta", {}).get("features", {})
        )
        self.subagent_tools_cache = {}
        self.user_skills = context.metadata.get("__user_skills__", {})
        self._resolution_lock = asyncio.Lock()

    async def _resolve_model_skills(
        self, model_info: Any, extra_params: dict
    ) -> tuple[list[str], str]:
        """
        Replicate Open WebUI skill discovery logic using cached user skills.
        Returns a list of skill IDs and a formatted system prompt fragment.
        """
        skill_ids = []
        skill_prompt = ""

        if not model_info:
            return [], ""

        # v3.15: Handle model info (dict/BaseModel) consistently with cached skills
        if isinstance(model_info, dict):
            # Try to resolve skillIds from either top-level or info.meta
            meta = model_info.get("info", {}).get("meta", {})
            if not meta:
                meta = model_info  # Direct metadata dict
            model_skill_ids = meta.get("skillIds", [])
        else:
            meta = (
                model_info.meta.model_dump()
                if hasattr(model_info.meta, "model_dump")
                else model_info.meta
            )
            model_skill_ids = meta.get("skillIds", []) if isinstance(meta, dict) else []

        if not model_skill_ids:
            return [], ""

        try:
            # v3.15: Optimized skill resolution.
            # Use pre-fetched self.user_skills (id -> skill) to eliminate DB hits.
            available_skills = []
            for sid in model_skill_ids:
                skill = self.user_skills.get(sid)
                if skill:
                    available_skills.append(skill)
                    skill_ids.append(sid)

            if available_skills:
                skill_descriptions = ""
                for skill in available_skills:
                    skill_descriptions += f"<skill>\n<name>{skill.name}</name>\n<description>{skill.description or ''}</description>\n</skill>\n"

                if skill_descriptions:
                    skill_prompt = (
                        f"<available_skills>\n{skill_descriptions}</available_skills>"
                    )

        except Exception as e:
            logger.error(f"Error resolving skills: {e}")

        return skill_ids, skill_prompt

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
        # Always proceed if skills are present, otherwise check for features
        if not self.planner_features and not extra_params.get("__skill_ids__"):
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

        if not active_features and not extra_params.get("__skill_ids__"):
            return {}

        try:
            # v3 parity: Explicitly control which built-in tool categories are enabled
            # to prevent polluting the planner with redundant core tools (time, notes, etc.)
            # handled by subagents.
            # Base on live MODELS entry so capabilities/knowledge match Open WebUI core;
            # view_skill is controlled only by extra_params["__skill_ids__"] (see OWUI get_builtin_tools).
            if self.planner_info:
                m_info = copy.deepcopy(self.planner_info)
            else:
                m_info = copy.deepcopy(model_info or {})
            if "info" not in m_info:
                m_info["info"] = {}
            if "meta" not in m_info["info"]:
                m_info["info"]["meta"] = {}
            meta = m_info["info"]["meta"]
            if self.model_knowledge:
                mk = self.model_knowledge
                meta["knowledge"] = (
                    copy.deepcopy(mk) if isinstance(mk, list) else [copy.deepcopy(mk)]
                )
            elif isinstance(model_info, dict):
                pipe_meta = model_info.get("info", {}).get("meta", {})
                if pipe_meta.get("knowledge") is not None:
                    meta.setdefault("knowledge", copy.deepcopy(pipe_meta["knowledge"]))

            # Start by disabling everything, then enable only active features and skills
            builtin_tools_config = {
                "time": False,
                "knowledge": active_features.get("knowledge", False),
                "chats": False,
                "memory": active_features.get(
                    "memory", False
                ),  # Usually handled by OWUI, but for parity
                "web_search": active_features.get("web_search", False),
                "image_generation": active_features.get("image_generation", False),
                "code_interpreter": active_features.get("code_interpreter", False),
                "notes": False,
                "channels": False,
            }

            m_info["info"]["meta"]["builtinTools"] = builtin_tools_config

            return get_builtin_tools(
                self.request,
                extra_params,
                features=active_features,
                model=m_info,
            )
        except Exception as e:
            logger.error(f"Failed to load filtered built-in tools: {e}")
            return {}

    def get_tools_spec(
        self, available_tasks: list[Any] = None
    ) -> list[dict[str, Any]]:
        user_valves = self.context.user_valves
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

        # 3. Workspace Terminal Overrides: from WORKSPACE_TERMINAL_MODELS list
        workspace_terminal_models = [
            m.strip()
            for m in self.valves.WORKSPACE_TERMINAL_MODELS.split(",")
            if m.strip()
        ]
        for mid in workspace_terminal_models:
            if mid and mid not in final_subagents:
                final_subagents.append(mid)

        subagents_list = final_subagents

        tools_spec = []

        # schemas for IDs (no longer using enum to allow dynamic task creation)
        task_id_schema = {
            "type": "string",
            "description": "REQUIRED: Unique identifier for the task/thread (e.g. 'task_research').",
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
            "description": "REQUIRED: Unique identifier for this subagent thread (e.g. 'task_research').",
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
                                "description": "Detailed instructions for the subagent. If you need data from previous tasks, list them in 'related_tasks' instead of using macros here.",
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

    async def get_planner_tools_dict(self) -> dict[str, Any]:
        """Resolve external tools and filtered built-in tools for the planner."""
        user_valves = self.context.user_valves
        tool_ids = (
            self.pipe_metadata.get("toolIds")
            or self.pipe_metadata.get("tool_ids")
            or []
        )
        logger.debug(
            f"[Planner] Resolving tools for planner. pipe_metadata keys: {list(self.pipe_metadata.keys())}, tool_ids: {tool_ids}"
        )
        extra_params = {
            "chat_id": self.context.chat_id,
            "tool_ids": tool_ids,
            "__user__": (
                self.user.model_dump()
                if hasattr(self.user, "model_dump")
                else (self.user.__dict__ if self.user else {})
            ),
            "__metadata__": self.context.metadata,
        }

        # 1. Resolve external tools (DB + OpenAPI servers) and MCP (not in get_tools)
        seen_ids: set[str] = set()
        ordered_tool_ids: list[str] = []
        for t in tool_ids or []:
            if t and t not in seen_ids:
                seen_ids.add(t)
                ordered_tool_ids.append(t)

        tools_dict: dict[str, Any] = {}
        if ordered_tool_ids:
            tools_dict = await get_tools(
                self.request, ordered_tool_ids, self.user, extra_params
            )
        app_models = getattr(self.request.app.state, "MODELS", {})
        model_ws = merge_workspace_model_dict(app_models, self.context.body.get("model", "") or "")
        extra_mcp = {
            **extra_params,
            "__model__": model_ws,
            "__messages__": self.context.body.get("messages", []),
            "__files__": self.pipe_metadata.get("files") or [],
        }
        mcp_tools = await planner_merge_mcp_tools_from_ids(
            self.request,
            self.user,
            ordered_tool_ids,
            self.pipe_metadata,
            extra_mcp,
            None,
            self.engine.mcp_hub.get_or_create_client,
        )
        if mcp_tools:
            tools_dict.update(mcp_tools)

        # 2. Resolve filtered built-in tools
        # For the planner, configuration (skills/tools) comes from its own model info
        skill_ids, skill_prompt = await self._resolve_model_skills(
            self.planner_info, extra_params
        )
        if skill_prompt:
            # For now, we set __skill_ids__ so view_skill is available.
            extra_params["__skill_ids__"] = skill_ids

        builtin_tools = self.get_filtered_builtin_tools(
            self.valves, user_valves, self.pipe_metadata, extra_params
        )
        if builtin_tools:
            tools_dict.update(builtin_tools)

        # Terminal fallback for Planner
        enable_planner_terminal = user_valves.ENABLE_PLANNER_TERMINAL_ACCESS
        is_terminal_agent_present = (
            self.valves.ENABLE_TERMINAL_AGENT or self.valves.WORKSPACE_TERMINAL_MODELS
        )

        terminal_id = self.pipe_metadata.get("terminal_id")
        if terminal_id and (enable_planner_terminal or not is_terminal_agent_present):
            try:
                raw_term = await get_terminal_tools(
                    self.request, terminal_id, self.user, extra_params
                )
                t_tools, _t_sys = unpack_terminal_tools_result(raw_term)
                if t_tools:
                    tools_dict.update(t_tools)
            except Exception as e:
                logger.error(f"Failed to load terminal tools for planner: {e}")

        return tools_dict

    async def get_subagent_tools(
        self,
        model_id: str,
        virtual_agents: dict[str, AgentDefinition],
        app_models: dict[str, Any],
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Fetches tools for subagents, ensuring builtin tools (image gen, search, etc.) are loaded for virtual agents."""
        # v3.15: Shallow copy extra_params to prevent cross-turn / parallel side effects during resolution
        extra_params = copy.copy(extra_params)

        # v3.6: Lazy-init resolution lock for robustness against stale instances
        if not hasattr(self, "_resolution_lock"):
            self._resolution_lock = asyncio.Lock()

        # v3.6.8: Double-checked locking. If cache hit, return immediately without lock.
        if model_id in self.subagent_tools_cache:
            return self.subagent_tools_cache[model_id]

        async with self._resolution_lock:
            # Re-check after acquiring lock
            if model_id in self.subagent_tools_cache:
                return self.subagent_tools_cache[model_id]

            # v3.6.8: Use SUBAGENT_TIMEOUT valve (default 1200s).
            # Separate lock wait from resolution timeout to avoid cascading timeouts.
            async with asyncio.timeout(float(self.valves.SUBAGENT_TIMEOUT)):
                # Ensure __user__ is present and non-empty for view_skill/builtin tools parity
                if not extra_params.get("__user__"):
                    if self.user:
                        extra_params["__user__"] = (
                            self.user.model_dump()
                            if hasattr(self.user, "model_dump")
                            else (self.user.__dict__ if self.user else {})
                        )
                    else:
                        # Deep fallback: attempt to extract from pipe_metadata or request state if available
                        extra_params["__user__"] = self.wide_metadata.get(
                            "__user__"
                        ) or getattr(self.request.state, "user", {})

                # Ensure __user__ is a dict (standard Open WebUI requirement for parity)
                if not isinstance(extra_params.get("__user__"), dict):
                    extra_params["__user__"] = {}

                # 1. Virtual Agents Handling
                if model_id in virtual_agents:
                    va = virtual_agents[model_id]
                    va_model_id = va.model_id

                    # Object resolution: Dict-first, then app_models fallback (ELIMINATE direct DB hit)
                    va_runtime_info = app_models.get(va_model_id)
                    va_model_obj = va_runtime_info or {"id": va_model_id}

                    s_tools_dict = {}
                    terminal_sys = ""
                    if va.type == "terminal":
                        try:
                            terminal_id = self.pipe_metadata.get("terminal_id")
                            raw_term = await get_terminal_tools(
                                self.request, terminal_id, self.user, extra_params
                            )
                            s_tools_dict, terminal_sys = unpack_terminal_tools_result(
                                raw_term
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to load terminal tools for {model_id}: {e}"
                            )
                    elif va.type == "builtin":
                        if va.builtin_model_override:
                            builtin_model = copy.deepcopy(va.builtin_model_override)
                        elif isinstance(va_model_obj, dict):
                            builtin_model = copy.deepcopy(va_model_obj)
                        else:
                            builtin_model = merge_workspace_model_dict(
                                app_models, va_model_id
                            )
                        if va.builtin_model_override and va_model_obj:
                            if hasattr(va_model_obj, "id"):
                                builtin_model["id"] = va_model_obj.id
                            elif isinstance(va_model_obj, dict):
                                builtin_model["id"] = va_model_obj.get(
                                    "id", va_model_id
                                )

                        if model_id == "knowledge_agent" and self.model_knowledge:
                            bm_info = builtin_model.setdefault("info", {})
                            bm_meta = bm_info.setdefault("meta", {})
                            mk = self.model_knowledge
                            bm_meta["knowledge"] = (
                                copy.deepcopy(mk)
                                if isinstance(mk, list)
                                else [copy.deepcopy(mk)]
                            )

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
                            logger.error(
                                f"Failed to load builtin tools for {model_id}: {e}"
                            )

                    s_tools = (
                        [
                            {"type": "function", "function": t["spec"]}
                            for t in s_tools_dict.values()
                        ]
                        if s_tools_dict
                        else None
                    )

                    final_sys = va.system_message or ""
                    if terminal_sys:
                        final_sys = f"{final_sys.rstrip()}\n\n{terminal_sys}"

                    result = {
                        "dict": s_tools_dict,
                        "specs": s_tools,
                        "system_message": final_sys,
                        "actual_model": va_model_id,
                        "temperature_override": va.temperature,
                    }
                    self.subagent_tools_cache[model_id] = result
                    return result

                # Regular subagent: workspace model + exact toolIds/MCP, then builtins only per OWUI gates
                model_ws = merge_workspace_model_dict(app_models, model_id)
                model_system_message = ""
                subagent_tool_ids: list[str] = []
                logger.debug(
                    f"[Planner] Resolving subagent tools for {model_id}. Registry tracker: {self.engine and self.engine.mcp_hub is not None}"
                )

                # v3.15: Optimized metadata extraction from model_ws (no DB model_db_info)
                meta = model_ws.get("info", {}).get("meta", {})
                params = model_ws.get("info", {}).get("params", {})
                if isinstance(meta, dict):
                    subagent_tool_ids.extend(
                        meta.get("toolIds") or meta.get("tool_ids") or []
                    )
                if isinstance(params, dict):
                    subagent_tool_ids.extend(
                        params.get("toolIds")
                        or params.get("tool_ids")
                        or params.get("tools")
                        or []
                    )
                    model_system_message = params.get("system", "") or ""

                info_meta = model_ws.get("info", {}).get("meta", {})
                info_params = model_ws.get("info", {}).get("params", {})
                subagent_tool_ids.extend(
                    info_meta.get("toolIds") or info_meta.get("tool_ids") or []
                )
                subagent_tool_ids.extend(
                    info_params.get("toolIds")
                    or info_params.get("tool_ids")
                    or info_params.get("tools")
                    or []
                )
                if not model_system_message:
                    model_system_message = info_params.get("system", "") or ""

                seen_sub: set[str] = set()
                ordered_sub_ids: list[str] = []
                for tid in subagent_tool_ids:
                    if tid and tid not in seen_sub:
                        seen_sub.add(tid)
                        ordered_sub_ids.append(tid)

                s_tools_dict: dict[str, Any] = {}
                if ordered_sub_ids:
                    try:
                        assigned_tools = await get_tools(
                            self.request, ordered_sub_ids, self.user, extra_params
                        )
                        if assigned_tools:
                            s_tools_dict.update(assigned_tools)
                    except Exception as e:
                        logger.error(
                            f"Failed to load assigned tools for subagent {model_id}: {e}"
                        )

                extra_mcp_sub = {
                    **extra_params,
                    "__model__": model_ws,
                    "__messages__": [],
                    "__files__": self.pipe_metadata.get("files") or [],
                }
                # v3.15: Extra parameters for MCP: Model-specific tools and file context parity
                extra_mcp_sub = {
                    **extra_params,
                    "__model__": model_ws,
                    "__messages__": [],
                    "__files__": self.pipe_metadata.get("files") or [],
                }
                # v3.15: Pass the inner metadata and the wide extra_mcp_sub context.
                # headers will now be correctly derived in planner_merge_mcp_tools_from_ids by checking extra_mcp_sub.
                mcp_sub = await planner_merge_mcp_tools_from_ids(
                    self.request,
                    self.user,
                    ordered_sub_ids,
                    self.pipe_metadata,
                    extra_mcp_sub,
                    extra_params.get("__event_emitter__"),
                    mcp_handler=self.engine.mcp_hub.get_or_create_client if self.engine and self.engine.mcp_hub else None,
                )
                if mcp_sub:
                    s_tools_dict.update(mcp_sub)

                skill_ids, skill_prompt = await self._resolve_model_skills(
                    model_ws, extra_params
                )
                if skill_prompt:
                    model_system_message = (
                        f"{model_system_message}\n\n{skill_prompt}"
                        if model_system_message
                        else skill_prompt
                    )
                    extra_params["__skill_ids__"] = skill_ids

                workspace_terminal_models = [
                    m.strip()
                    for m in self.valves.WORKSPACE_TERMINAL_MODELS.split(",")
                    if m.strip()
                ]
                if model_id in workspace_terminal_models:
                    terminal_id = self.pipe_metadata.get("terminal_id")
                    if terminal_id:
                        try:
                            raw_term = await get_terminal_tools(
                                self.request, terminal_id, self.user, extra_params
                            )
                            terminal_tools, terminal_sys = unpack_terminal_tools_result(
                                raw_term
                            )
                            if terminal_tools:
                                s_tools_dict.update(terminal_tools)
                            if terminal_sys:
                                model_system_message = (
                                    f"{model_system_message}\n\n{terminal_sys}"
                                    if model_system_message
                                    else terminal_sys
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to load terminal tools for subagent {model_id}: {e}"
                            )

                info_params_ws = model_ws.get("info", {}).get("params") or {}
                function_calling = info_params_ws.get("function_calling", "native")
                caps_ws = (model_ws.get("info", {}).get("meta") or {}).get(
                    "capabilities"
                ) or {}
                builtin_tools_enabled = caps_ws.get("builtin_tools", True)
                features_ws = workspace_feature_flags(model_ws)

                if function_calling == "native" and builtin_tools_enabled:
                    try:
                        builtin_tools = get_builtin_tools(
                            self.request,
                            extra_params,
                            features=features_ws,
                            model=model_ws,
                        )
                        for name, td in builtin_tools.items():
                            if name not in s_tools_dict:
                                s_tools_dict[name] = td
                    except Exception as e:
                        logger.error(
                            f"Failed to load built-in tools for subagent {model_id}: {e}"
                        )
                elif function_calling != "native":
                    logger.warning(
                        "Subagent model %s has non-native function_calling; builtins skipped (planner requires native FC).",
                        model_id,
                    )

                result = {
                    "dict": s_tools_dict,
                    "specs": (
                        [
                            {"type": "function", "function": t["spec"]}
                            for t in s_tools_dict.values()
                        ]
                        if s_tools_dict
                        else None
                    ),
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
        skill_prompt: str = "",
        terminal_sys: str = "",
    ) -> str:
        """Construct the full system prompt with tools and mandatory rules dynamically."""
        full_prompt = valves.SYSTEM_PROMPT
        if skill_prompt:
            full_prompt = f"{full_prompt}\n\n{skill_prompt}"
        if terminal_sys:
            full_prompt = f"{full_prompt}\n\n{terminal_sys}"
        plan_mode = user_valves.PLAN_MODE
        truncation = user_valves.TASK_RESULT_TRUNCATION
        user_input = user_input_enabled = user_valves.ENABLE_USER_INPUT_TOOLS
        subagents_list = (
            valves.SUBAGENT_MODELS.split(",") if valves.SUBAGENT_MODELS else []
        )
        # Inclusion of workspace terminal agents in descriptions
        workspace_terminal_models = [
            m.strip() for m in valves.WORKSPACE_TERMINAL_MODELS.split(",") if m.strip()
        ]
        for mid in workspace_terminal_models:
            if mid and mid not in subagents_list:
                subagents_list.append(mid)

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
                label = ""
                name = ""
                desc = ""

                # v3.6.10: Fetch model name and description from Open WebUI state/DB
                request = (metadata or {}).get("__request__")
                if request:
                    app_models = getattr(request.app.state, "MODELS", {})
                    m_info = app_models.get(m)
                    if m_info:
                        name = m_info.get("name", "")
                        info = m_info.get("info", {})
                        desc = info.get("meta", {}).get("description", "") or info.get(
                            "description", ""
                        )

                # Fallback to DB if not in app state (e.g. workspace presets not yet loaded)
                if not name or not desc:
                    try:
                        db_model = Models.get_model_by_id(m)
                        if db_model:
                            if not name:
                                name = db_model.name
                            if not desc:
                                meta = db_model.meta
                                if isinstance(meta, dict):
                                    desc = meta.get("description", "")
                                elif hasattr(meta, "description"):
                                    desc = meta.description
                                elif hasattr(meta, "model_dump"):
                                    desc = meta.model_dump().get("description", "")
                    except Exception:
                        pass  # Ignore DB failures during prompt construction

                workspace_terminal_models = [
                    v.strip()
                    for v in valves.WORKSPACE_TERMINAL_MODELS.split(",")
                    if v.strip()
                ]
                if m in workspace_terminal_models:
                    label = " (Terminal Access)"

                entry = f"- ID: {m}{label}"
                if name:
                    entry += f" (Name: {name})"
                if desc:
                    entry += f"\n  Description: {desc}"
                desc_list.append(entry)

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
                "   - **NO REDUNDANT UPDATES**: NEVER call `update_state` to change a task's status to `in_progress` or `completed` if you are using `call_subagent` for that task. `call_subagent` manages these transitions internally. Doing so wastes tokens and execution turns.\n"
                "   - **Constraints**: Always provide a `description` when adding a new `task_id`. For updates, the `description` is optional.\n"
                "2. `call_subagent(model_id: str, prompt: str, task_id: str, related_tasks: list[str])`: Use this to delegate a subtask to a specialized model.\n"
                "   - **Task Status Lifecycle**: Starting a `call_subagent` will automatically set the task to 'in_progress' ONLY if it is currently 'pending' or 'failed'. Success automatically marks the task as 'completed'. Check the `status` field in the response.\n"
                "   - **Threading & Context**: The `task_id` identifies the conversation thread with the subagent. To **continue or follow up** on a previous interaction, you MUST use the **same** `task_id`. To start a **fresh** conversation, use a **new** `task_id`.\n"
                "   - **@task_id Text Replacement (Orchestrator Macro)**: Writing `@task_id` (e.g., `@task_research`) in your **FINAL response** to the user will automatically insert the full text result of that task. **USE THIS to avoid re-typing or redundantly summarizing large code blocks, data, or technical reports.** Do NOT use these macros in subagent prompts — use `related_tasks` for that purpose.\n"
                "   - **Raw Task ID (no @ or :)**: Use the plain ID (`task_research`) in tool parameters. NEVER prefix with @ or : in parameter fields or when defining task IDs.\n"
                "   - **CRITICAL — `related_tasks` for cross-task data passing**: Subagents are ISOLATED — they CANNOT see any other task's results unless you explicitly pass them. When a subagent needs data produced by a DIFFERENT task, you MUST list that task's raw ID in the `related_tasks` array. **The task must already have a result stored (completed status) for this to work.**\n"
            )
            tool_idx = 3
        else:
            tools_doc = (
                "1. `call_subagent(model_id: str, prompt: str, task_id: str, related_tasks: list[str])`: Use this to delegate a subtask to a specialized model.\n"
                "   - **Threading & Context**: The `task_id` identifies the conversation thread with the subagent. To **continue or follow up** on a previous interaction, use the **same** `task_id`. To start a **fresh** conversation, use a **new** `task_id`.\n"
                "   - **@task_id Text Replacement (Orchestrator Macro)**: Writing `@task_id` (e.g., `@task_research`) in your **FINAL response** to the user will automatically insert the full text result of that task. **USE THIS to avoid re-typing or redundantly summarizing large code blocks, data, or technical reports.** Do NOT use these macros in subagent prompts — use `related_tasks` for that purpose.\n"
                "   - **Raw Task ID (no @ or :)**: Use the plain ID (`task_research`) in tool parameters. NEVER prefix with @ or : in parameter fields or when defining task IDs.\n"
                "   - **CRITICAL — `related_tasks` for cross-task data passing**: Subagents are ISOLATED — they CANNOT see any other task's results unless you explicitly pass them. When a subagent needs data produced by a DIFFERENT task, you MUST list that task's raw ID in the `related_tasks` array. **The task must already have a result stored (completed status) for this to work.**\n"
            )
            tool_idx = 2

        if truncation:
            tools_doc += f"{tool_idx}. `read_task_result(task_id: str)`: Read the FULL, untruncated raw text result of a completed task verbatim. Use this when the result shown in the call_subagent response was truncated.\n"
            tool_idx += 1

        tools_doc += f"{tool_idx}. `review_tasks(task_ids: list, prompt: str)`: Spawn a specialized LLM reviewer to evaluate and synthesize results from multiple tasks using a custom prompt.\n"

        mandatory_rules = (
            "### MANDATORY RULES:\n"
            "- Delegate work to subagents using `call_subagent` for complex analysis, generation, or reasoning.\n"
            "- **CODING RULE**: For COMPLEX coding, scripting, calculation, or data-processing task, delegate to a `code_interpreter_agent` or equivalent. NEVER use web_search_agent or knowledge_agent for code. **ALWAYS provide FULL code in the final output by using @task_id substitution tags** (e.g., '@task_coding') to avoid truncation or partial copying.\n"
            "- **ALWAYS pass `related_tasks`** when a subagent needs results from previous tasks. Subagents CANNOT see other task results without this.\n"
            "- **ID CONSISTENCY**: You MUST use the EXACT `task_id` values you defined during the PLANNING phase. NEVER prepend task IDs with colons (:) or @ symbols in tool calls.\n"
            "- **ANTIGRAVITY MACROS**: Use `@task_id` references in your FINAL response to include large previous outputs, code, or reports. **DO NOT summarize or manually copy-paste large technical results** if a macro can include the original verbatim.\n"
            "- Final Output is YOUR responsibility. Make it look professional and clean.\n"
            "- If any subagent response is incomplete, missing details, or lacks required assets, you MUST follow up with additional subagent calls or clarifying prompts until the answer is complete and all requirements are met.\n"
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
- **Output Schema**: Return STRICTLY a JSON object: `{"tasks": [{"task_id": "task_1_research", "description": "..."}, ...]}`.
- **Task Granularity**: Each task should be an atomic, actionable step (e.g., "Use web_search_agent to find X", "Analyze the results of task_1_research to do Y", "Use code_interpreter_agent to build Z").
- **Subagent Selection**: Describe tasks in terms of the available subagent capabilities if no subagent has the proper capabilities assume the orchestrator has it.
- **Constraint**: Return ONLY the raw JSON object. NO prose, NO explanations, NO greetings. DO NOT prepend task_id values with colons (:) or @ symbols.
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
                "- If you see a relative link like '/api/v1/files/uuid', and you need to access it in your environment, use 'curl --cookie \"token={OPEN_WEBUI_TOKEN}\" -O {OPEN_WEBUI_URL}/api/v1/files/uuid' (prepend the base URL and pass the token as a cookie).\n"
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
                "- If you see a relative link like '/api/v1/files/uuid', and you need to access it via <curl/wget>, prepend the base URL and pass the token as a cookie: 'curl --cookie \"token={OPEN_WEBUI_TOKEN}\" -O {OPEN_WEBUI_URL}/api/v1/files/uuid'.\n"
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
        context: PlannerContext,
        ui: UIRenderer,
        state: PlannerState,
        registry: ToolRegistry = None,
        engine: "PlannerEngine" = None,
    ):
        self.context = context
        self.ui = ui
        self.state = state
        self.metadata = context.metadata
        self.valves = context.valves
        self.user_valves = context.user_valves
        self.request = context.request
        self.user = context.user
        self.model_knowledge = context.model_knowledge
        # Resolve base URL (Valve -> Env -> Request)
        self.base_url = self.valves.OPEN_WEBUI_URL
        if not self.base_url:
            self.base_url = os.environ.get("WEBUI_URL", "")
        if not self.base_url:
            self.base_url = str(context.request.base_url).rstrip("/")
        self.queue_emitter = ui.emitter  # Already a QueueEmitter in structural overhaul
        self.registry = registry
        self.engine = engine
        # We use the subagent_tools_cache from the registry if available
        self.subagent_tools_cache = {}

        # Resolve virtual agents with fallback model and valve overrides
        AGENT_MODEL_VALVES = {
            "image_gen_agent": "IMAGE_GENERATION_AGENT_MODEL",
            "web_search_agent": "WEB_SEARCH_AGENT_MODEL",
            "knowledge_agent": "KNOWLEDGE_AGENT_MODEL",
            "code_interpreter_agent": "CODE_INTERPRETER_AGENT_MODEL",
            "terminal_agent": "TERMINAL_AGENT_MODEL",
        }

        self.virtual_agents = {}
        for vid, va in self.VIRTUAL_AGENTS.items():
            valve_field = AGENT_MODEL_VALVES.get(vid)
            valve_value = getattr(self.valves, valve_field, "") if valve_field else ""
            resolved_model = (
                valve_value.strip()
                if isinstance(valve_value, str) and valve_value.strip()
                else (va.model_id or self.valves.PLANNER_MODEL)
            )

            # Wire temperature overrides (e.g., for code interpreter)
            temp = va.temperature
            if vid == "code_interpreter_agent" and hasattr(
                self.valves, "CODE_INTERPRETER_TEMPERATURE"
            ):
                temp = self.valves.CODE_INTERPRETER_TEMPERATURE

            self.virtual_agents[vid] = va.model_copy(
                update={"model_id": resolved_model, "temperature": temp}
            )

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
                    self.context.request, check_body, user_obj
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
                    logger.warning(
                        f"Failed to parse judge JSON after retries: {e}. Raw: {raw_judge_text}"
                    )
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
        # UI Parity: Removed individual "Consulting..." here as requested by user.
        # It's now handled by consolidation in the parent or by tool-specific status below.

        # 1. Prepare Context
        context = await self._prepare_subagent_context(
            model_id, prompt, task_id, related_tasks, chat_id, valves, body
        )

        # 2. Execute Loop
        result = await self._execute_subagent_loop(
            context,
            task_id,
            model_id,
            chat_id,
            valves,
            body,
            user_valves,
            planner_input=prompt,
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
        user_obj = self.context.user
        registry = self.registry or ToolRegistry(self.context)

        # Fix: Fetch app_models from request.app.state instead of __event_call__
        app_models = getattr(self.context.request.app.state, "MODELS", {})
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

        # Extract Authentication Token from Request (v3.5 extension)
        # Support both Authorization header and Cookie: token=<JWT>
        request = self.context.request
        token = ""
        if request and hasattr(request, "headers"):
            # 1. Try Cookie header (Open WebUI backend standard)
            cookie_header = request.headers.get("cookie", "")
            token_match = re.search(r"token=([^;]+)", cookie_header)
            if token_match:
                token = token_match.group(1).strip()

            # 2. Fallback to Authorization: Bearer
            if not token:
                auth_header = request.headers.get("authorization", "")
                if auth_header.lower().startswith("bearer "):
                    token = auth_header[7:].strip()

        # Perform token substitution in system prompt
        # Internal substitution intended for subagents (terminal/code interpreter)
        # to access Open WebUI files/API via curl/wget using {OPEN_WEBUI_TOKEN}.
        sub_sys = sub_sys.replace("{OPEN_WEBUI_TOKEN}", token)

        # Subagent background execution rules (v3 parity)
        sub_sys += (
            "\n\nCRITICAL CONTEXT: You are running as a headless subagent entirely in the background. "
            "DO NOT return markdown elements that rely on Open WebUI UI embeds for tool generation output. "
            "Any tools you use (like generate_image, search, etc.) will return URLs, base64 data, or raw paths exactly as returned by tools. "
            "You MUST always return these raw HTML references, URLs, files, or images as relative or absolute paths, or direct URLs, unconditionally in your final reply so the main planner can use them. "
            "If you generate or reference any assets (images, files, data, audio, etc.), it is CRITICAL to include the correct link, path, or URL in your output. "
            "If you are missing any required asset links or references, you MUST follow up or retry until all assets are accessible via explicit links or paths. "
        )

        history = self.state.get_history(chat_id, task_id, model_id)

        # Sticky Related Tasks (v3.5 extension):
        # Once a task is linked to a session, we track its version (hash) and re-inject if updated.
        # v3.15: Standardize on JSON array keys for metadata session tracking
        metadata = self.state.get_metadata(chat_id, task_id, model_id)
        if "linked_tasks" not in metadata:
            metadata["linked_tasks"] = {}
        self.state.set_metadata(chat_id, task_id, model_id, metadata)

        linked_tasks = metadata["linked_tasks"]

        # Current batch + Persisted batch
        task_set = set([rt.lstrip("@:") for rt in (related_tasks or []) if rt])
        task_set.update(linked_tasks.keys())

        results_injection = []
        for rid in task_set:
            if rid in self.state.results:
                raw_res = self.state.results[rid]
                res_hash = hashlib.md5(raw_res.encode("utf-8")).hexdigest()

                # Check if we should inject: new task OR updated result
                if linked_tasks.get(rid) != res_hash:
                    # Provide updated context
                    header = f"--- UPDATED RESULTS FROM TASK {rid} ---"
                    if linked_tasks.get(rid) is None:
                        header = f"--- RESULTS FROM RELATED TASK {rid} ---"

                    results_injection.append(
                        f"\n\n{header}\n{raw_res}\n--- END OF {rid} ---\n"
                    )
                    linked_tasks[rid] = res_hash

        injection_content = "".join(results_injection)
        actual_prompt = prompt + injection_content

        # Perform token substitution in the actual prompt
        # Allows user/planner to pass wildcard to subagents (terminal/code interpreter).
        actual_prompt = actual_prompt.replace("{OPEN_WEBUI_TOKEN}", token)

        if history:
            # v3 pacing: truncating history in sub-threads is risky for OpenRouter providers
            # who expect the full tool call chain. We'll skip aggressive truncation for now.
            if history[0]["role"] == "system":
                history[0]["content"] = sub_sys
            history.append({"role": "user", "content": actual_prompt})
        else:
            history = [
                {"role": "system", "content": sub_sys},
                {"role": "user", "content": actual_prompt},
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
        sub_called_tools = {}
        sub_iteration = 0
        max_sub_iters = valves.MAX_SUBAGENT_ITERATIONS or 100
        history = context["history"]

        while True:
            sub_iteration += 1
            logger.debug(
                f"[Subagent: {model_id}] Entering loop iteration {sub_iteration}..."
            )

            # (A) Iteration Limit Check
            can_continue, new_max = await self._handle_subagent_iteration_limit(
                sub_iteration, max_sub_iters, task_id, model_id, valves, user_valves
            )
            max_sub_iters = new_max
            if not can_continue:
                sub_final_answer_chunks.append("[Subagent stopped at iteration limit.]")
                break

            sub_history_lock = asyncio.Lock()

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
                if user_valves.ENABLE_SUBAGENT_CHECK:
                    current_answer = "".join(sub_final_answer_chunks).strip()
                    if current_answer:
                        task_desc = self.state.tasks.get(
                            task_id, TaskStateModel(status="pending")
                        ).description
                        is_approved, redo_instruction = (
                            await self._verify_subagent_response(
                                task_id,
                                model_id,
                                current_answer,
                                valves,
                                context["user_obj"],
                                body,
                                task_description=task_desc,
                                planner_input=planner_input,
                            )
                        )
                        if not is_approved:
                            history.append(
                                {"role": "user", "content": redo_instruction}
                            )
                            await self.ui.emit_status(
                                f"Refining {model_id} response for {task_id}..."
                            )
                            # Reset chunks to avoid double output if the model doesn't repeat everything
                            # but the prompt asks for FULL response, so we should clear them.
                            sub_final_answer_chunks = []
                            continue

                # Reasoning Promotion (v3 parity): Only if NO content AND NO tool calls
                if not sub_content.strip() and turn_result.get("reasoning"):
                    # v3.6: Clean promoted reasoning to avoid leaking tags or prefixes into the UI content area
                    reasoning_text = Utils.clean_thinking(turn_result["reasoning"])
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
                    "content": sub_content
                    or "",  # Some providers prefer "" over None with tools
                    "tool_calls": tool_calls_list,
                }
            )

            if valves.PARALLEL_SUBAGENT_EXECUTION:
                # Parallel subagent tool execution
                tasks = [
                    self._execute_single_subagent_tool_call(
                        stc,
                        context,
                        history,
                        sub_called_tools,
                        model_id,
                        task_id,
                        sub_history_lock,
                    )
                    for stc in tool_calls_list
                ]
                # v3.6: Capture anchor BEFORE gather to avoid race condition if history grows
                history_tail_start = len(history)
                await asyncio.gather(*tasks)

                # Re-sort the history tail to match original tool_calls order
                call_id_order = {
                    tc.get("id"): i for i, tc in enumerate(tool_calls_list)
                }
                tool_tail = history[history_tail_start:]
                tool_tail.sort(
                    key=lambda m: call_id_order.get(m.get("tool_call_id", ""), 999)
                )
                history[history_tail_start:] = tool_tail
            else:
                # Sequential subagent tool execution
                for stc in tool_calls_list:
                    await self._execute_single_subagent_tool_call(
                        stc,
                        context,
                        history,
                        sub_called_tools,
                        model_id,
                        task_id,
                        sub_history_lock,
                    )

        await self.ui.emit_status(f"Agent {model_id} completed {task_id}.")

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

    async def _execute_single_subagent_tool_call(
        self,
        stc: dict,
        context: dict,
        history: list,
        sub_called_tools: dict[str, int],
        model_id: str,
        task_id: str,
        history_lock: Optional[asyncio.Lock] = None,
    ) -> None:
        """Helper to execute a single subagent tool call."""
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
                task_id,
                history_lock,
            )
        else:
            await self._handle_missing_tool(
                stc_name,
                stc_args_str,
                call_id,
                context,
                history,
                sub_called_tools,
                model_id,
                history_lock,
            )

    async def _handle_subagent_iteration_limit(
        self,
        iteration: int,
        max_iters: int,
        task_id: str,
        model_id: str,
        valves,
        user_valves,
    ) -> tuple[bool, int]:
        """Handles iteration limits by prompting the user or stopping if not in YOLO mode."""
        if user_valves.YOLO_MODE or max_iters <= 0 or iteration <= max_iters:
            return True, max_iters

        if self.context.event_call:
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

                if res_json.get("action") == "continue":
                    # Add original valve amount to the limit
                    extension = valves.MAX_SUBAGENT_ITERATIONS or 10
                    return True, max_iters + extension
                return False, max_iters
            except Exception as e:
                logger.error(f"Iteration limit modal error: {e}")
                return False, max_iters
        return False, max_iters

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

        request = self.context.request
        app_models = getattr(request.app.state, "MODELS", {}) if request else {}
        workspace_model = merge_workspace_model_dict(app_models, actual_model)
        has_bi = any(
            (v.get("type") == "builtin") for v in context.get("tools_dict", {}).values()
        )
        await apply_native_completion_file_prep(
            request,
            sub_body,
            user_obj,
            workspace_model,
            self.metadata.get("__metadata__", {}),
            self.context.event_emitter,
            has_bi,
        )

        # Reasoning state
        reasoning_buffer = ""
        reasoning_start_time = None

        # v3.6: Add heartbeat and trace logging for LLM turn
        turn_start = time.monotonic()
        logger.debug(
            f"[Subagent: {model_id}] Starting streaming LLM turn. Model: {actual_model}"
        )

        try:
            chunk_count = 0
            last_heartbeat = turn_start
            # v3.6.8: Wrap the entire iterator to ensure it's closed even on CancelledError/TimeoutError.
            stream_gen = Utils.get_streaming_completion(
                self.context.request, sub_body, user_obj
            )
            try:
                # v3.6.8: Use SUBAGENT_TIMEOUT valve (default 1200s) for long reasoning models.
                async with asyncio.timeout(float(self.valves.SUBAGENT_TIMEOUT)):
                    async for event in stream_gen:
                        chunk_count += 1
                        now = time.monotonic()
                        if now - last_heartbeat >= 10.0:
                            logger.debug(
                                f"[Subagent: {model_id}] Alive... ({chunk_count} chunks received, {now-turn_start:.1f}s elapsed)"
                            )
                            last_heartbeat = now
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
                                        "id": tc.get("id")
                                        or f"call_{uuid4().hex[:12]}",
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"].get("name", ""),
                                            "arguments": "",
                                        },
                                    }
                                if "name" in tc["function"] and tc["function"]["name"]:
                                    tc_dict[idx]["function"]["name"] = tc["function"][
                                        "name"
                                    ]
                                if "arguments" in tc["function"]:
                                    tc_dict[idx]["function"]["arguments"] += tc[
                                        "function"
                                    ]["arguments"]
            finally:
                # v3.6.8: Explicitly close the generator to avoid connection leaks on timeout/cancel
                if hasattr(stream_gen, "aclose"):
                    await stream_gen.aclose()
        except asyncio.TimeoutError:
            logger.error(
                f"[Subagent: {model_id}] LLM Turn timed out after 300s. Potential provider hang."
            )
            error_occurred = True
        except Exception as e:
            logger.error(f"[Subagent: {model_id}] Error during LLM turn: {e}")
            error_occurred = True

        raw_content = "".join(content_chunks)
        content = Utils.clean_thinking(raw_content)
        reasoning = "".join(reasoning_chunks)

        logger.debug(
            f"[Subagent: {model_id}] LLM turn completed in {time.monotonic()-turn_start:.1f}s. Content len: {len(content)}"
        )

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
        sub_called_tools: dict[str, int],
        model_id: str,
        task_id: str,
        history_lock: Optional[asyncio.Lock] = None,
    ) -> None:
        """Executes a single subagent tool call and updates history/states."""
        target_tool = context["tools_dict"].get(name)
        user_obj = self.context.user
        tc_files = []
        res_content = ""
        try:
            # UI Parity: Emit status and tool call details
            await self.ui.emit_status(f"[{task_id}] Executing {name}...")

            # Inject metadata variables for tools that support them (v3.5 robustness)
            # This ensures builtin tools like generate_image can emit files/status.
            allowed_keys = (
                target_tool.get("spec", {})
                .get("parameters", {})
                .get("properties", {})
                .keys()
            )
            filtered_args = {k: v for k, v in args_obj.items() if k in allowed_keys}

            # Context variables required by many built-in tools
            context_vars = {
                "__request__": self.context.request,
                "__user__": self.metadata.get("__user__"),
                "__event_emitter__": self.queue_emitter,
                "__event_call__": self.context.event_call,
                "__chat_id__": self.context.chat_id
                or self.metadata.get("chat_id"),
                "__message_id__": self.context.message_id
                or self.metadata.get("message_id"),
                "__files__": self.metadata.get("__files__"),
                "__metadata__": self.metadata.get("__metadata__"),
            }
            # v3.15: Safe Signature-based Injection
            # We inject special context variables only if they are in the tool's signature.
            try:
                import inspect
                sig = inspect.signature(target_tool["callable"])
                for k, v in context_vars.items():
                    if v is not None and k in sig.parameters and k not in filtered_args:
                        filtered_args[k] = v
            except Exception as e:
                logger.warning(f"Signature inspection failed for subagent tool {name}: {e}")

            # Execute tool: configurable timeout (default 1200s)
            logger.debug(f"[Subagent: {model_id}] Calling tool {name}...")
            async with asyncio.timeout(float(self.valves.SUBAGENT_TIMEOUT)):
                res = await target_tool["callable"](**filtered_args)

            # v3.15: Robust tool result processing parity with main planner engine.
            # We use process_tool_result to handle file uploads, base64 conversion,
            # and rich object decomposition.
            tc_return = process_tool_result(
                self.context.request,
                name,
                res,
                target_tool.get("type", ""),
                False,
                self.metadata.get("__metadata__"),
                user_obj,
            )

            # Step 1: Unpack result, files, and embeds
            tc_files = []
            tc_embeds = []
            r_val = tc_return
            if isinstance(tc_return, tuple):
                r_val, tuple_files, tc_embeds = tc_return
                if tuple_files:
                    tc_files.extend(tuple_files)

            # Step 2: Extract clean history content for the LLM (v3.15)
            # We respect UI components by extracting the 'message' field.
            if isinstance(r_val, dict):
                res_content = (
                    r_val.get("message")
                    or r_val.get("description")
                    or json.dumps(r_val, ensure_ascii=False)
                )
            else:
                res_content = str(r_val) if r_val is not None else ""

            # Step 3: Emit embeds to the UI (v3.15 parity: Metadata-based only if not part of main UI)
            if tc_embeds and self.queue_emitter:
                await self.queue_emitter(
                    {
                        "type": "embeds",
                        "data": {
                            "embeds": tc_embeds,
                        },
                    }
                )

            # Accumulate files for final consolidated persistence sync
            if tc_files and self.engine:
                async with self.engine._files_lock:
                    # Update internal metadata list for this turn
                    current_files = self.metadata.get("__files__")
                    if isinstance(current_files, list):
                        # Avoid duplicates in the local metadata list. Handle both 'id', 'file_id', and 'url'.
                        existing_ids = {
                            f.get("id") or f.get("file_id") or f.get("url")
                            for f in current_files
                            if isinstance(f, dict)
                        }
                        for f in tc_files:
                            if isinstance(f, dict):
                                fid = f.get("id") or f.get("file_id") or f.get("url")
                                if fid and fid not in existing_ids:
                                    current_files.append(f)
                    
                    # v3.15: Main planner tool files are also tracked in produced_files for final sync
                    self.engine.produced_files.extend(tc_files)

                    logger.debug(
                        f"[Subagent: {model_id}] Tool {name} returned {len(tc_files)} files. Tracked in engine."
                    )

            # Update UI to "Done" status for THIS tool call (For logs/potential future use)
            done_tag = self.ui.build_tool_call_details(
                call_id, name, args_str, done=True, result=res_content, files=tc_files
            )

            async with history_lock or asyncio.Lock():
                sub_called_tools[name] = sub_called_tools.get(name, 0) + 1
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": res_content,
                    }
                )
        except Exception as e:
            logger.error(f"[Subagent: {model_id}] Error executing {name}: {e}")
            await self.ui.emit_status(
                f"[Subagent: {model_id}] Error executing {name}: {e}"
            )
            async with history_lock or asyncio.Lock():
                sub_called_tools[name] = sub_called_tools.get(name, 0) + 1
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
        sub_called_tools: dict[str, int],
        model_id: str,
        history_lock: Optional[asyncio.Lock] = None,
    ) -> bool:  # Returns True if it should break the loop due to repeated failures
        error_msg = f"Tool {name} not found."
        available_tools = list(context["tools_dict"].keys())
        if available_tools:
            error_msg += (
                f" Available tools for this subagent: {', '.join(available_tools)}."
            )

        logger.warning(f"[Subagent: {model_id}] {error_msg} Arguments: {args_str}")
        await self.ui.emit_status(
            f"[Subagent: {model_id}] Attempted unknown tool: {name}"
        )

        async with history_lock or asyncio.Lock():
            sub_called_tools[name] = sub_called_tools.get(name, 0) + 1

        # Safety break logic
        consecutive_failures = 0
        for h in reversed(history):
            if h.get("role") == "tool" and "not found" in h.get("content", ""):
                if h.get("name") == name:
                    consecutive_failures += 1
                else:
                    break
            else:
                break

        if consecutive_failures >= 3:
            async with history_lock or asyncio.Lock():
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": f"Error: {error_msg} Repeated failures detected. Please stop or try a different approach.",
                    }
                )
            await self.ui.emit_status(
                f"[Subagent: {model_id}] Stopping due to repeated tool failures: {name}"
            )
            return True

        async with history_lock or asyncio.Lock():
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": f"Error: {error_msg}",
                }
            )
        return False


# Planner Engine (Main Logic)
# ---------------------------------------------------------------------------


class InternalToolExecutor:
    """Encapsulates the logic for built-in planner tools."""

    def __init__(self, context: PlannerContext, engine: "PlannerEngine"):
        self.context = context
        self.engine = engine
        self.ui = engine.ui
        self.metadata = context.metadata

    @property
    def state(self) -> "PlannerState":
        return self.engine.state

    @property
    def subagents(self) -> "SubagentManager":
        return self.engine.subagents

    async def update_state(
        self, updates: dict[str, Any], user_valves: Any = None
    ) -> str:
        """Updates the planner's internal state and refreshes the UI summary block."""
        if "task_id" in updates:
            tid = updates.get("task_id", "").lstrip("@:")
            if not tid:
                return "Error: task_id is required and cannot be empty or ':'."
            # Handle single task update (tool call format)
            self.state.update_task(
                tid, updates.get("status"), updates.get("description")
            )
        else:
            # Handle batch update (internal format)
            updated_count = 0
            for tid, tdata in updates.items():
                if isinstance(tdata, dict):
                    clean_tid = tid.lstrip("@:")
                    if clean_tid:
                        self.state.update_task(
                            clean_tid, tdata.get("status"), tdata.get("description")
                        )
                        updated_count += 1
            if updated_count == 0:
                return "Error: No valid task IDs found in update payload."

        # Refresh the integrated inline plan summary
        self.engine.current_plan_html = self.ui.get_html_status_block(self.state.tasks)
        await self.engine._emit_replace(getattr(self.engine, "total_emitted", ""))
        return "Plan updated."

    async def call_subagent(
        self, args: dict, chat_id: str, valves: Any, body: dict, user_valves: Any
    ) -> str:
        task_id = args.get("task_id", "").lstrip("@:")
        if not task_id:
            return "Error: task_id is required for subagent calls."
        if not args.get("model_id") or not args.get("prompt"):
            return "Error: model_id and prompt are required for subagent calls."

        if user_valves.PLAN_MODE:
            current_status = self.state.tasks.get(
                task_id, TaskStateModel(status="pending")
            ).status
            if current_status in ["pending", "failed"]:
                self.state.update_task(task_id, "in_progress")
                self.engine.current_plan_html = self.ui.get_html_status_block(
                    self.state.tasks
                )
                await self.engine._emit_replace(
                    getattr(self.engine, "total_emitted", "")
                )

        sanitized_related = []
        missing_tasks = []
        for rt in args.get("related_tasks", []):
            if rt is None:
                continue
            clean_rt = str(rt).lstrip("@:")
            if not clean_rt:
                continue
            if clean_rt not in self.state.results:
                missing_tasks.append(rt)
            else:
                sanitized_related.append(clean_rt)

        if missing_tasks:
            return f"Error: The following related tasks do not exist or have no results yet: {', '.join(missing_tasks)}. Ensure you only list completed tasks in 'related_tasks'."

        # Track the subagent call as an active task for cancellation
        sub_task = asyncio.create_task(
            self.subagents.call_subagent(
                args["model_id"],
                args["prompt"],
                task_id,
                sanitized_related,
                chat_id,
                valves,
                body,
                user_valves,
                self.metadata,
            )
        )
        self.engine.active_tasks.append(sub_task)

        try:
            res_dict = await sub_task
        finally:
            if sub_task in self.engine.active_tasks:
                self.engine.active_tasks.remove(sub_task)

        # v3.15: Intermediate persistence sync removed.
        # Subagent files are already synced via self.engine.produced_files track.

        self.state.update_task(task_id, "completed")
        if user_valves.PLAN_MODE:
            self.engine.current_plan_html = self.ui.get_html_status_block(
                self.state.tasks
            )
            await self.engine._emit_replace(getattr(self.engine, "total_emitted", ""))
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
        rid = args.get("task_id", "").lstrip("@:")
        return self.state.results.get(rid, f"Task {rid} not found.")

    async def review_tasks(
        self, args: dict, valves: Any, body: dict, user_obj: Any
    ) -> str:
        rt_ids, rt_prompt = args.get("task_ids", []), args.get("prompt", "")
        if not rt_ids or not rt_prompt:
            return "Error: must specify task_ids and prompt."

        await self.ui.emit_status("Reviewing tasks cross-reference...")

        # Track the review call as an active task
        async def run_review():
            review_sys = "You are a specialized review subagent. Synthesize the following task results logically."
            for rx in rt_ids:
                clean_rx = rx.lstrip("@:")
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
                self.context.request, review_body, user_obj
            ):
                if ev["type"] == "content":
                    res_chunks.append(ev["text"])

            final_review = "".join(res_chunks)
            return final_review

        review_task = asyncio.create_task(run_review())
        self.engine.active_tasks.append(review_task)

        try:
            final_review = await review_task
        finally:
            if review_task in self.engine.active_tasks:
                self.engine.active_tasks.remove(review_task)

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
        context: PlannerContext,
        ui: UIRenderer,
        state: PlannerState,
        subagents: SubagentManager,
        registry: ToolRegistry,
    ):
        self.context = context
        self.ui = ui
        self.state = state
        self.subagents = subagents
        self.registry = registry
        self.metadata = context.metadata
        self.valves = context.valves
        self.model_knowledge = context.model_knowledge

        self.state_persistence = StatePersistence(context, state, ui)
        self.mcp_hub = MCPHub(context.valves)
        self.tools = InternalToolExecutor(context, self)
        self.current_plan_html = ""
        self.total_emitted = ""  # Trace current Turn body only
        self.active_tasks: list[asyncio.Task] = []
        # v3.15: Consolidated file persistence. Accumulate files and sync at the end.
        self.produced_files = []
        self._files_lock = asyncio.Lock()


    async def stop(self) -> None:
        if self.active_tasks:
            logger.info(f"Cancelling {len(self.active_tasks)} active subagent tasks...")
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait(self.active_tasks, timeout=2.0)
            except Exception as e:
                logger.error(f"Error during subagent task cancellation: {e}")
            self.active_tasks.clear()

        await self.mcp_hub.stop()

    def get_current_files(self) -> list:
        files_map = {}
        for f in self.metadata.get("__files__", []):
            if isinstance(f, dict):
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid:
                    files_map[fid] = f
        for f in getattr(self, "produced_files", []):
            if isinstance(f, dict):
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid:
                    files_map[fid] = f
        return list(files_map.values()) if files_map else None

    async def _emit_replace(self, content: str) -> None:
        """Helper to emit an inline replace event with the current plan prepended.
        v3.15: CRITICAL. We MUST include 'files' in the replace payload if we want them to persist
        during structural DOM refreshes. Otherwise, Svelte will unmount the FileList component.
        """
        data = {"content": self.current_plan_html + content}
        if self.produced_files:
            data["files"] = self.produced_files
        await self.ui.emitter({"type": "replace", "data": data})

    async def _emit_message(self, delta: str) -> None:
        """Helper to emit an append-only message event (efficient for content deltas)."""
        await self.ui.emitter({"type": "message", "data": {"content": delta}})

    async def run(
        self,
        chat_id: str,
        valves: Any,
        user_valves: Any,
        body: dict,
        files: list = None,
    ) -> AsyncGenerator[str, None]:
        """Main entry point for the planner engine. Orchestrates planning, execution, and verification."""

        # v3.5: Initialize turn body track
        self.total_emitted = ""

        # 0. State Recovery (v3.3)
        # We attempt to restore previous turn's state BEFORE clearing.
        await self.state_persistence.recover_from_files(body, chat_id, files)

        user_obj = self.context.user

        # 1. Phase 1: Planning
        if user_valves.PLAN_MODE:
            await self._phase_planning(valves, user_valves, user_obj, body)
            yield ""  # Heartbeat

        # 2. Phase 2: Execution & Verification Loop
        if not user_valves.PLAN_MODE:
            await self.ui.emit_status("Working...")

        # v3.5: Execution Loop Phase with real-time UI streaming
        async for delta in self._phase_execution_loop(
            chat_id, valves, user_valves, user_obj, body
        ):
            if isinstance(delta, str):
                final_answer = (
                    delta  # Final string returned at the end of generator turn
                )
                # v3.6: Yield heartbeat to keep the SSE stream alive
                yield ""
            else:
                # Handle non-string deltas if any (future-proofing)
                yield ""

        # v3.3: Final State Persistence
        # Save state to file and attach to response. This adds a state file to produced_files.
        await self.state_persistence.save_to_file(chat_id)

        # v3.15: Final consolidated file persistence sync.
        # Performs a single DB update for ALL produced files from tools and state saving.
        # We ensure target chat/message IDs are resolved from session metadata.
        target_chat_id = self.context.chat_id or chat_id
        target_msg_id = self.context.message_id

        if target_chat_id and target_msg_id:
            async with self._files_lock:
                if self.produced_files:
                    try:
                        # Convert to pure IDs if necessary for robust DB insertion
                        sync_files = []
                        for f in self.produced_files:
                            fid = (
                                f.get("id") or f.get("file_id")
                                if isinstance(f, dict)
                                else None
                            )
                            if fid:
                                # Ensure we have the full structure if possible
                                sync_files.append(f)
                            elif isinstance(f, str):
                                # If it's just an ID string
                                sync_files.append({"id": f})

                        if sync_files:
                            await asyncio.to_thread(
                                Chats.add_message_files_by_id_and_message_id,
                                target_chat_id,
                                target_msg_id,
                                sync_files,
                            )
                            logger.info(
                                f"[Planner] Final sync: Committed {len(sync_files)} files to DB."
                            )
                    except Exception as e:
                        logger.error(f"Failed consolidated file sync: {e}")

        # v3.5: Final response and asset persistence sync
        full_content = self.current_plan_html + (final_answer or "")

        # v3.15: Robust asset synchronization.
        # We merge files from the DB (canonical), our internal `produced_files` accumulator,
        # and the session metadata to ensure the UI local store is fully synchronized
        # even if the DB fetch is delayed or incomplete.
        final_files_map = {}

        # 1. Start with metadata files (from initial load or tool calls in this turn)
        for f in self.metadata.get("__files__", []):
            if isinstance(f, dict):
                fid = f.get("id") or f.get("file_id") or f.get("url")
                if fid:
                    final_files_map[fid] = f

        # 2. Add produced files from this turn (state files, subagent tool outputs)
        async with self._files_lock:
            for f in self.produced_files:
                if isinstance(f, dict):
                    fid = f.get("id") or f.get("file_id") or f.get("url")
                    if fid:
                        final_files_map[fid] = f

        # 3. Pull latest from DB as the ultimate source of truth
        if target_chat_id and target_msg_id:
            try:
                full_message = await asyncio.to_thread(
                    Chats.get_message_by_id_and_message_id,
                    target_chat_id,
                    target_msg_id,
                )
                if full_message:
                    for f in full_message.get("files", []):
                        if isinstance(f, dict):
                            fid = f.get("id") or f.get("file_id") or f.get("url")
                            if fid:
                                final_files_map[fid] = f
                        elif isinstance(f, str):
                            final_files_map[f] = {"id": f}
            except Exception as e:
                logger.error(f"Failed to fetch final files from DB: {e}")

        final_files = list(final_files_map.values())

        # v3.6: Explicitly emit the final files list one last time to sync the UI local store.
        if final_files and self.ui:
            await self.ui.emit_files(final_files)

        # To prevent Open WebUI's native completion handler from overwriting our files
        # when it saves the final message state, we spawn a background task that sleeps
        # briefly and then enforces the true file list.
        if final_files and target_chat_id and target_msg_id:

            async def _delayed_sync():
                await asyncio.sleep(2.0)
                try:
                    await asyncio.to_thread(
                        Chats.add_message_files_by_id_and_message_id,
                        target_chat_id,
                        target_msg_id,
                        final_files,
                    )
                    logger.info(
                        f"[Planner] Delayed final sync: Restored {len(final_files)} files to DB."
                    )
                except Exception as e:
                    logger.error(f"[Planner] Delayed sync failed: {e}")

            asyncio.create_task(_delayed_sync())

        # We yield the final content as the last token
        # v3.15: Terminal status emission. 
        # We call this BEFORE yielding the final content to ensure the UI captures it.
        await self.ui.emit_status("Task completed.", done=True)

        # We yield the final content as the last token
        # For Pipes, this is the most stable way to ensure the final save includes all text.
        yield full_content
        return

    async def _phase_planning(self, valves, user_valves, user_obj, body):
        """Generates the initial task list based on the user prompt."""
        await self.ui.emit_status("Planning...")
        if user_valves.PLAN_MODE:
            # v3.5: Initialize and emit the plan block (initially empty)
            self.current_plan_html = self.ui.get_html_status_block(self.state.tasks)
            await self._emit_replace("")

        # Prepare context
        distilled_history = Utils.distill_history_for_llm(body.get("messages", []))
        # v3.5: Support skill and terminal prompt injection for planner
        # Temporary extra_params for skill/terminal resolution
        tmp_params = {"__user__": self.metadata.get("__user__", {})}
        # Resolve skills from pipe metadata for the planner
        skill_ids, skill_prompt = await self.registry._resolve_model_skills(
            self.registry.pipe_metadata, tmp_params
        )

        terminal_sys = ""
        enable_planner_terminal = user_valves.ENABLE_PLANNER_TERMINAL_ACCESS
        is_terminal_agent_present = (
            valves.ENABLE_TERMINAL_AGENT or valves.WORKSPACE_TERMINAL_MODELS
        )
        terminal_id = self.metadata.get("__metadata__", {}).get("terminal_id")

        if terminal_id and (enable_planner_terminal or not is_terminal_agent_present):
            try:
                raw_term = await get_terminal_tools(
                    self.context.request,
                    terminal_id,
                    self.context.user,
                    tmp_params,
                )
                _, terminal_sys = unpack_terminal_tools_result(raw_term)
            except Exception:
                pass

        plan_sys = PromptBuilder.build_system_prompt(
            valves,
            user_valves,
            [],
            metadata=self.metadata,
            mode="plan",
            messages=distilled_history,
            skill_prompt=skill_prompt,
            terminal_sys=terminal_sys,
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
            if (
                not getattr(valves, "ENABLE_KNOWLEDGE_AGENT", True)
                and self.model_knowledge
            ):
                plan_body["metadata"]["knowledge"] = self.model_knowledge
                plan_body["metadata"]["__model_knowledge__"] = self.model_knowledge

            request = self.context.request
            app_models = getattr(request.app.state, "MODELS", {}) if request else {}
            workspace_model = merge_workspace_model_dict(
                app_models, body.get("model", "") or ""
            )
            await apply_native_completion_file_prep(
                request,
                plan_body,
                user_obj,
                workspace_model,
                self.metadata.get("__metadata__", {}),
                self.context.event_emitter,
                False,
            )

            plan_chunks = []
            async for event in Utils.get_streaming_completion(
                self.context.request, plan_body, user_obj
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

            # After successful parsing, show the final formulated plan
            if self.state.tasks:
                if user_valves.PLAN_MODE:
                    self.current_plan_html = self.ui.get_html_status_block(
                        self.state.tasks
                    )
                    await self._emit_replace("")

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
                    logger.warning(
                        "No tasks parsed from plan after retries. Using fallback."
                    )
                    self.state.update_task(
                        "main_task", "pending", "Process user request"
                    )

            await self.ui.emit_status(
                f"Plan formed with {len(self.state.tasks)} tasks."
            )

            # Plan Approval logic (Ignored in YOLO mode)
            if (
                user_valves.ENABLE_PLAN_APPROVAL
                and not user_valves.YOLO_MODE
                and self.context.event_call
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
    ) -> AsyncGenerator[str, None]:
        """Main loop for execution and periodic verification (v3 parity loop stability)."""
        planner_iteration = 0
        judge_retries = 0
        max_planner_iters = valves.MAX_PLANNER_ITERATIONS or 20
        total_emitted = ""

        # Prepare context
        distilled_history = Utils.distill_history_for_llm(body.get("messages", []))
        # v3.5: Support skill and terminal prompt injection for planner
        tmp_params = {"__user__": self.metadata.get("__user__", {})}
        # Resolve skills from pipe metadata for the planner
        skill_ids, skill_prompt = await self.registry._resolve_model_skills(
            self.registry.pipe_metadata, tmp_params
        )

        terminal_sys = ""
        enable_planner_terminal = user_valves.ENABLE_PLANNER_TERMINAL_ACCESS
        is_terminal_agent_present = (
            valves.ENABLE_TERMINAL_AGENT or valves.WORKSPACE_TERMINAL_MODELS
        )
        terminal_id = self.metadata.get("__metadata__", {}).get("terminal_id")

        if terminal_id and (enable_planner_terminal or not is_terminal_agent_present):
            try:
                raw_term = await get_terminal_tools(
                    self.context.request,
                    terminal_id,
                    self.context.user,
                    tmp_params,
                )
                _, terminal_sys = unpack_terminal_tools_result(raw_term)
            except Exception:
                pass

        exec_sys = PromptBuilder.build_system_prompt(
            valves,
            user_valves,
            self.registry.get_tools_spec(
                list(self.state.tasks.keys())
            ),
            metadata=self.metadata,
            mode="execute",
            messages=distilled_history,
            skill_prompt=skill_prompt,
            terminal_sys=terminal_sys,
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
            yield ""  # Heartbeat

        external_tools_dict = await self.registry.get_planner_tools_dict()

        while True:
            planner_iteration += 1
            await self.ui.emit_status("Working...")

            # (A) Iteration Limit Check
            can_continue, new_max = await self._handle_iteration_limit(
                planner_iteration, max_planner_iters, valves, user_valves
            )
            max_planner_iters = new_max
            if not can_continue:
                # If we break due to limit, ensure final macros are resolved
                total_emitted = Utils.resolve_references(
                    total_emitted, self.state.results
                )
                self.total_emitted = total_emitted
                await self._emit_replace(total_emitted)
                yield total_emitted
                break

            # (B) Execute Turn
            turn_result = await self._execute_planner_turn(
                exec_history,
                total_emitted,
                valves,
                user_obj,
                body,
                user_valves,
                external_tools_dict,
            )

            content, tc_dict, turn_emitted = (
                turn_result["content"],
                turn_result["tool_calls"],
                turn_result["total_emitted"],
            )
            turn_start_base = turn_result["turn_start_base"]
            yield ""  # Heartbeat

            # Reasoning Promotion (v3 parity): Only if NO content AND NO tool calls (prevent double content)
            if not content and not tc_dict and turn_result.get("reasoning"):
                # v3.6: Clean promoted reasoning to avoid leaking tags or prefixes into the UI content area
                reasoning_text = Utils.clean_thinking(turn_result["reasoning"])
                content = reasoning_text
                # Revert to turn_start_base to delete the thinking block from UI (v3 parity)
                total_emitted = turn_start_base + reasoning_text
                self.total_emitted = total_emitted
            elif content or tc_dict or turn_result.get("reasoning"):
                # v3.6: Ensure total_emitted is always synced from the turn's final state
                total_emitted = turn_emitted
                self.total_emitted = total_emitted
                yield ""  # Heartbeat

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
                    yield ""  # Heartbeat
                    if should_continue:
                        # Ensure UI reflects the judge's decision to continue before the next planner turn
                        self.total_emitted = total_emitted
                        await self._emit_replace(total_emitted)
                        judge_retries += 1
                        continue

                # Final completion: resolve references in the accumulated UI text
                # v3.6: Resolve macros directly on total_emitted to avoid rfind failures due to chunk-based cleaning.
                total_emitted = Utils.resolve_references(
                    total_emitted, self.state.results
                )

                # Ensure UI continues spinning until the entire PlannerEngine.run finishes
                await self.ui.emit_status("Working...")

                self.total_emitted = total_emitted
                await self._emit_replace(total_emitted)

                # We yield the final string as the last item
                yield total_emitted
                return  # End generator

            # (D) Handle Tool Calls
            tool_calls_list = list(tc_dict.values())
            exec_history.append(
                {
                    "role": "assistant",
                    "content": content
                    or "",  # Some providers prefer "" over None with tools
                    "tool_calls": tool_calls_list,
                }
            )

            # Use async generator for tool calls too
            async for delta in self._handle_tool_calls(
                tool_calls_list,
                exec_history,
                total_emitted,
                external_tools_dict,
                chat_id,
                valves,
                user_valves,
                user_obj,
                body,
            ):
                if isinstance(delta, str):
                    total_emitted = delta
                    yield ""  # Heartbeat

    async def _handle_iteration_limit(
        self, iteration: int, max_iters: int, valves: Any, user_valves: Any
    ) -> tuple[bool, int]:
        """Prompts user if iteration limit reached."""
        if user_valves.YOLO_MODE or max_iters <= 0 or iteration <= max_iters:
            return True, max_iters
        if self.context.event_call:
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
                    # Extend by original valve amount
                    extension = valves.MAX_PLANNER_ITERATIONS or 20
                    return True, max_iters + extension
                await self.ui.emit_status("Planner stopped by user.", True)
            except Exception as e:
                logger.error(f"Iteration limit modal error: {e}")
                await self.ui.emit_status("Planner stopped by iteration limit.", True)
                return False, max_iters

        await self.ui.emit_status("Planner reached iteration limit.", True)
        return False, max_iters

    async def _execute_planner_turn(
        self,
        exec_history: list,
        total_emitted: str,
        valves: Any,
        user_obj: Any,
        body: dict,
        user_valves: Any,
        external_tools_dict: dict = None,
    ) -> dict:
        """Performs a single LLM turn for the planner with live reasoning and clean content streaming."""
        tc_dict, content_chunks, reasoning_chunks = {}, [], []
        tools = self.registry.get_tools_spec(
            list(self.state.tasks.keys())
        )
        if external_tools_dict:
            tools.extend(
                [
                    {"type": "function", "function": t["spec"]}
                    for t in external_tools_dict.values()
                ]
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

        request = self.context.request
        app_models = getattr(request.app.state, "MODELS", {}) if request else {}
        workspace_model = merge_workspace_model_dict(
            app_models, body.get("model", "") or ""
        )
        has_bi = bool(
            external_tools_dict
            and any(v.get("type") == "builtin" for v in external_tools_dict.values())
        )
        await apply_native_completion_file_prep(
            request,
            planner_body,
            user_obj,
            workspace_model,
            self.metadata.get("__metadata__", {}),
            self.context.event_emitter,
            has_bi,
        )

        # Reasoning state for live emission (v3 parity)
        reasoning_buffer = ""
        reasoning_start_time = None
        total_emitted_base = total_emitted
        self.total_emitted = total_emitted
        error_occurred = False

        # Debounce control
        last_emit_time = 0
        EMIT_INTERVAL_S = 0.5  # Emit reasoning every 500ms
        token_count = 0
        TOKENS_THRESHOLD = 50

        async for event in Utils.get_streaming_completion(
            self.context.request, planner_body, user_obj
        ):
            etype = event["type"]

            if etype == "reasoning":
                piece = event.get("text", "")
                if piece:
                    if reasoning_start_time is None:
                        reasoning_start_time = time.monotonic()
                    reasoning_chunks.append(piece)
                    reasoning_buffer += piece
                    token_count += 1

                    # Debounce Reasoning: Only emit every 500ms or 50 tokens
                    now = time.monotonic()
                    if (now - last_emit_time > EMIT_INTERVAL_S) or (
                        token_count >= TOKENS_THRESHOLD
                    ):
                        last_emit_time = now
                        token_count = 0
                        # Clean tool calls and stray tags from reasoning display
                        display = Utils.hide_tool_calls(reasoning_buffer)
                        display = Utils.THINKING_TAG_CLEANER_PATTERN.sub("", display)
                        display_quoted = "\n".join(
                            f"> {l}" if not l.startswith(">") else l
                            for l in display.splitlines()
                        )
                        self.total_emitted = (
                            total_emitted_base
                            + f'<details type="reasoning" done="false">\n<summary>Thinking\u2026</summary>\n{display_quoted}\n</details>\n'
                        )
                        await self._emit_replace(self.total_emitted)

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
                    display_quoted = "\n".join(
                        f"> {l}" if not l.startswith(">") else l
                        for l in display.splitlines()
                    )
                    sealed_reasoning = f'<details type="reasoning" done="true" duration="{dur}">\n<summary>Thought for {dur} seconds</summary>\n{display_quoted}\n</details>\n'
                    total_emitted_base += sealed_reasoning
                    reasoning_buffer = ""
                    self.total_emitted = total_emitted_base + "".join(content_chunks)

                    # v3.15: Reasoning Seal. We MUST issue a structural 'replace' exactly once to seal the block.
                    # If we're entering content, we skip this and bundle the seal into the first chunk below.
                    if etype != "content":
                        await self._emit_replace(self.total_emitted)

                if etype == "content":
                    text = event["text"]
                    content_chunks.append(text)

                    full_content = "".join(content_chunks)
                    display_content = Utils.clean_thinking(full_content)
                    new_total = total_emitted_base + display_content

                    if new_total != self.total_emitted:
                        # v3.6/v3.15 Optimized Streaming: Determine if we need a full replace (structural change) 
                        # or can use a silent append delta (performance).
                        delta = new_total[len(self.total_emitted):]
                        
                        # Use replace for FIRST chunk after reasoning seal OR if we detect complex thinking tags
                        is_transition = total_emitted_base in new_total and total_emitted_base not in self.total_emitted
                        is_complex = Utils.THINKING_TAG_CLEANER_PATTERN.search(text)
                        
                        if is_transition or is_complex:
                            self.total_emitted = new_total
                            await self._emit_replace(new_total)
                        else:
                            # Efficient append-only delta
                            self.total_emitted = new_total
                            await self._emit_message(delta)
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
                self.total_emitted = total_emitted + f"\n\n> [!CAUTION]\n> {error_msg}"
                await self._emit_replace(self.total_emitted)
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
            display_quoted = "\n".join(
                f"> {l}" if not l.startswith(">") else l for l in display.splitlines()
            )
            sealed_reasoning = f'<details type="reasoning" done="true" duration="{dur}">\n<summary>Thought for {dur} seconds</summary>\n{display_quoted}\n</details>\n'
            total_emitted_base += sealed_reasoning

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
            "total_emitted": self.total_emitted,
            "turn_start_base": total_emitted_base,
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
    ) -> AsyncGenerator[str, None]:
        """Executes tool calls while providing immediate UI feedback via Details tags."""

        # Count tools for native grouping logic (handled by OWUI)
        tc_tag_map = {}
        history_lock = asyncio.Lock()
        sorted_tcs = sorted(tool_calls, key=lambda x: str(x.get("id", "")))

        # Ensure spacing before the tool call batch
        if total_emitted and not total_emitted.endswith("\n\n"):
            total_emitted += "\n\n"

        for tc in sorted_tcs:
            func_name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            call_id = tc.get("id", str(uuid4()))

            # Build initial "Executing..." tag
            tc_tag = self.ui.build_tool_call_details(
                call_id, func_name, args_str, done=False
            )
            tc_tag_map[call_id] = tc_tag
            total_emitted += tc_tag

        # Ensure spacing after the tool call batch
        total_emitted += "\n\n"

        # Update UI with all initial tags
        self.total_emitted = total_emitted
        await self._emit_replace(total_emitted)
        yield ""  # Heartbeat

        # 2. Sequential or Parallel execution
        subagent_calls = [tc for tc in tool_calls if tc["function"]["name"] == "call_subagent"]
        self.metadata["__consolidated__"] = valves.PARALLEL_TOOL_EXECUTION and len(subagent_calls) > 1
        if self.metadata["__consolidated__"]:
            await self.ui.emit_status(f"Consulting {len(subagent_calls)} agents...")

        if valves.PARALLEL_TOOL_EXECUTION:
            # Parallel execution with real-time UI updates via Pydantic model
            ui_state = UIState(total_emitted=total_emitted)

            tasks = [
                self._execute_single_tool_call(
                    tc,
                    exec_history,
                    external_tools_dict,
                    chat_id,
                    valves,
                    user_valves,
                    user_obj,
                    body,
                    tc_tag_map,
                    ui_state,
                    history_lock,
                    append_to_history=False,
                )
                for tc in sorted_tcs
            ]

            # We wait for all tasks, but we want to yield heartbeats while they run.
            # asyncio.as_completed is better here.
            # NOTE: as_completed returns in completion order, not submission order.
            # The subsequent sort on history_tail_start below ensures the messages
            # are appended to the execution history in the correct tool_call order.
            results = []
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                yield ""  # Heartbeat whenever a tool finishes

            # v3.15: Safe History Append for Parallel Mode
            # We append all results in the correct order to match tool_calls
            # to avoid the race condition where history_tail_start is stale.
            async with history_lock:
                for done_tag, tool_res, msg_dict in results:
                    exec_history.append(msg_dict)

            # Re-sort is still useful if results came back out of order via as_completed
            call_id_order = {tc.get("id"): i for i, tc in enumerate(tool_calls)}
            history_tail_start = len(exec_history) - len(tool_calls)
            tool_tail = exec_history[history_tail_start:]
            tool_tail.sort(
                key=lambda m: call_id_order.get(m.get("tool_call_id", ""), 999)
            )
            exec_history[history_tail_start:] = tool_tail

            total_emitted = ui_state.total_emitted
        else:
            # Sequential execution (original behavior)
            for tc in sorted_tcs:
                call_id = tc.get("id")
                done_tag, _, _ = await self._execute_single_tool_call(
                    tc,
                    exec_history,
                    external_tools_dict,
                    chat_id,
                    valves,
                    user_valves,
                    user_obj,
                    body,
                    tc_tag_map,
                    None,  # No shared state for sequential
                    history_lock,
                )

                # Sibling update in sequential mode
                total_emitted = total_emitted.replace(tc_tag_map[call_id], done_tag)
                tc_tag_map[call_id] = done_tag
                self.total_emitted = total_emitted
                await self._emit_replace(total_emitted)
                yield ""  # Heartbeat

        # Clear the consolidation flag after the tool batch completes
        self.metadata.pop("__consolidated__", None)
        await self.ui.emit_status("Working...")
        yield total_emitted
        return

    async def _execute_single_tool_call(
        self,
        tc: dict,
        exec_history: list,
        external_tools_dict: dict,
        chat_id: str,
        valves: Any,
        user_valves: Any,
        user_obj: Any,
        body: dict,
        tc_tag_map: dict,
        ui_state: Optional[UIState] = None,
        history_lock: Optional[asyncio.Lock] = None,
        append_to_history: bool = True,
    ) -> tuple[str, str, dict]:
        """Helper to execute a single tool call and return the UI tag and tool result."""
        func_name, args_str, call_id = (
            tc["function"]["name"],
            tc["function"]["arguments"],
            tc.get("id", str(uuid4())),
        )

        args = Utils.parse_tool_arguments(args_str)
        # Skip macro expansion (@task_id) for prompts (W6) - we use related_tasks instead
        resolved_args = Utils.resolve_dict_references(
            args,
            self.state.results,
            skip_keys=["task_id", "task_ids", "related_tasks", "prompt"],
        )

        tool_res = ""
        tc_files = []
        tc_embeds = [] # v3.15: Initialise to avoid locals() check failure on exception
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

                    # v3.15: Main Planner Tool Context Injection (Parity with subagents)
                    # We inject special context variables if requested by the tool signature,
                    # bypassing the JSON schema (allowed_keys) which often excludes them.
                    context_vars = {
                        "__request__": self.context.request,
                        "__user__": self.metadata.get("__user__"),
                        "__event_emitter__": self.ui.emitter,
                        "__event_call__": self.context.event_call,
                        "__chat_id__": self.context.chat_id
                        or self.metadata.get("chat_id"),
                        "__message_id__": self.context.message_id
                        or self.metadata.get("message_id"),
                        "__files__": self.metadata.get("__files__"),
                        "__metadata__": self.metadata.get("__metadata__"),
                    }

                    # v3.15: Main Planner Tool Context Injection (Parity with subagents)
                    # We inject special context variables only if they are in the tool's signature.
                    # We use self.ui.emitter (the structural QueueEmitter) to ensure
                    # events are properly queued and synchronized with the UI.
                    try:
                        import inspect
                        sig = inspect.signature(tool_data["callable"])
                        for k, v in context_vars.items():
                            if v is not None and k in sig.parameters and k not in filtered_args:
                                filtered_args[k] = v
                    except Exception as e:
                        logger.warning(f"Signature inspection failed for main planner tool {name}: {e}")

                    res = await tool_data["callable"](**filtered_args)
                    tc_return = process_tool_result(
                        self.context.request,
                        name,
                        res,
                        tool_data.get("type", ""),
                        False,
                        self.metadata.get("__metadata__"),
                        user_obj,
                    )

                    # Handle multiple return values (tuple: str/dict, files, embeds) (v3.15)
                    res_str = ""
                    # tc_embeds already initialized at top of try block
                    r_val = tc_return
                    if isinstance(tc_return, tuple):
                        r_val, tc_files, tc_embeds = tc_return

                    # Extract clean history content for the LLM (v3.15 parity with subagents)
                    if isinstance(r_val, dict):
                        res_str = (
                            r_val.get("message")
                            or r_val.get("description")
                            or json.dumps(r_val, ensure_ascii=False)
                        )
                    else:
                        res_str = str(r_val) if r_val is not None else ""

                    # v3.15: Embed emission is handled via the HTML tool_calls tag (done_tag) below.
                    # This ensures consistency with the "Execution Plan" UI pattern.

                    # Persistence & Visibility (v3.5)
                    if tc_files:
                        target_chat_id = self.context.chat_id or chat_id
                        target_msg_id = self.context.message_id
                        if target_chat_id and target_msg_id:
                            try:
                                # Synchronize with DB for multi-turn persistence
                                Chats.add_message_files_by_id_and_message_id(
                                    target_chat_id, target_msg_id, tc_files
                                )
                                # Update internal metadata list for this turn
                                current_files = self.metadata.get("__files__")
                                if isinstance(current_files, list):
                                    current_files.extend(tc_files)
                            except Exception as e:
                                logger.error(f"Failed to persist tool files to DB: {e}")

                    tool_res = res_str
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

        # Maintain global history - ordering is critical for KV caching and model consistency.
        msg_dict = {
            "role": "tool",
            "content": tool_res,
            "tool_call_id": call_id,
            "name": func_name,
        }
        if append_to_history:
            async with history_lock or asyncio.Lock():
                exec_history.append(msg_dict)

        # Update UI to "Done" status for THIS tool call
        # v3.15: Include embeds derived from process_tool_result to match Plan pattern.
        done_tag = self.ui.build_tool_call_details(
            call_id,
            func_name,
            args_str,
            done=True,
            result=tool_res,
            files=tc_files,
            embeds=tc_embeds,
        )

        # Real-time UI updates for parallel mode
        if ui_state:
            async with ui_state.lock:
                # Direct sibling replacement in total_emitted
                ui_state.total_emitted = ui_state.total_emitted.replace(
                    tc_tag_map[call_id], done_tag
                )
                tc_tag_map[call_id] = done_tag
                current_total = ui_state.total_emitted
                self.total_emitted = current_total

            # Release lock before slow streaming emission
            await self._emit_replace(current_total)

        return done_tag, tool_res, msg_dict

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
            # Revert: Don't prepend here; let PlannerEngine.run handle it once.
            return False, total_emitted

        await self.ui.emit_status("Verifying task states...")
        # Build detailed task context for the judge (W4)
        task_context_lines = []
        for tid in unresolved:
            task_obj = self.state.tasks.get(tid)
            desc = task_obj.description if task_obj else "(no description)"
            status = task_obj.status if task_obj else "unknown"
            task_context_lines.append(f"  - {tid} [{status}]: {desc}")
        task_context_block = "\n".join(task_context_lines)

        # Append compact results summary for inter-task dependency validation
        results_summary = []
        for tid, result in self.state.results.items():
            preview = result[:300] + "..." if len(result) > 300 else result
            results_summary.append(f"  @{tid}: {preview}")

        appended_summary = False
        results_block = ""
        if results_summary:
            results_block = (
                "SYSTEM: Completed task results summary:\n"
                + "\n".join(results_summary)
                + "\n\n"
            )
            exec_history.append({"role": "system", "content": results_block})
            appended_summary = True

        judge_msg = (
            f"{results_block}"
            f"SYSTEM: Review the conversation history above.\n"
            f"The following tasks have not been marked as completed:\n"
            f"{task_context_block}\n\n"
            f"For each incomplete task, determine whether the conversation history already contains "
            f"a completed result that satisfies the task description. If yes, mark it 'completed'. "
            f"If the task was genuinely not attempted or failed, mark it 'failed' and provide a "
            f"specific follow-up instruction that will resolve it.\n"
            f"Respond with a JSON object following the schema exactly."
        )

        # v3.6.8: Use a copy to avoid mutating the core execution history in place
        # and prevent 'assistant' -> 'assistant' invariant violations if retrying.
        msgs = copy.deepcopy(exec_history)
        if not msgs or msgs[-1].get("role") != "assistant":
            msgs.append({"role": "assistant", "content": content})
        else:
            # If the last message is already assistant, update the copy's content
            if msgs[-1].get("content") != content:
                msgs[-1]["content"] = content

        # v3.6.8: Use msgs for the judge call
        judge_body = {
            **body,
            "model": valves.REVIEW_MODEL or valves.PLANNER_MODEL,
            "messages": msgs + [{"role": "user", "content": judge_msg}],
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
                self.context.request, judge_body, user_obj
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
                        if tid in self.state.tasks and status in [
                            "completed",
                            "failed",
                        ]:
                            self.state.update_task(tid, status, upd.get("description"))
                            updated = True

                    if updated:
                        self.current_plan_html = self.ui.get_html_status_block(
                            self.state.tasks
                        )
                        await self._emit_replace(total_emitted)

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
                            display_quoted = "\n".join(
                                f"> {l}" if not l.startswith(">") else l
                                for l in reasoning.splitlines()
                            )
                            total_emitted += f'<details type="reasoning" done="true" duration="{dur}">\n<summary>Judge Verification Feedback</summary>\n{display_quoted}\n</details>\n'

                        await self.ui.emit_status(
                            "Continuing based on judge feedback..."
                        )
                        if appended_summary and exec_history:
                            exec_history.pop()
                        return True, total_emitted
                    break
                else:
                    raise ValueError(f"No JSON brace found in judge response: {raw}")
            except Exception as e:
                if current_judge_retry < max_judge_retries:
                    logger.warning(
                        f"Judge verification parsing failed: {e}. Retrying..."
                    )
                    judge_body["messages"].append(
                        {"role": "assistant", "content": "".join(judge_chunks)}
                    )
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

        if appended_summary and exec_history:
            exec_history.pop()

        # Revert: Don't prepend here; let PlannerEngine.run handle it once.
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
        WORKSPACE_TERMINAL_MODELS: str = Field(
            default="",
            description="Comma-separated list of model IDs available to be queried as subagents with terminal access. These will override the default virtual terminal agent check.",
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
        PARALLEL_TOOL_EXECUTION: bool = Field(
            default=False,
            description="Enable parallel execution of tool calls using asyncio.gather. WARNING: Use with caution if tools have stateful dependencies within the same turn.",
        )
        PARALLEL_SUBAGENT_EXECUTION: bool = Field(
            default=False,
            description="Enable parallel execution of subagent tool calls using asyncio.gather. WARNING: Use with caution if tools have stateful dependencies within the same turn.",
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
            default=0.3,
            description="Temperature for the code interpreter subagent. Low values (0.0-0.2) produce more deterministic, accurate code.",
        )
        SUBAGENT_TIMEOUT: int = Field(
            default=1200,
            description="Timeout in seconds for subagent turns, tool calls, and resolution. Increase for extremely long reasoning models (DeepSeek-R1, Kimi k2.5).",
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
        ENABLE_SUBAGENT_CHECK: bool = Field(
            default=False,
            description="Enable a judge model to verify subagent responses for task completion and correct asset referencing BEFORE returning to the planner.",
        )
        ENABLE_PLANNER_TERMINAL_ACCESS: bool = Field(
            default=False,
            description="Explicitly grant the planner agent terminal access, even if a dedicated terminal agent is present or defined in WORKSPACE_TERMINAL_MODELS.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def _ui_worker(self, queue: asyncio.Queue, emitter: Callable):
        """Processes UI events and emits them sequentially to the real emitter."""
        while True:
            event = await queue.get()
            try:
                # v3.15: Increased watchdog timeout to 5.0s to accommodate large HTML plan blocks/replacements.
                async with asyncio.timeout(5.0):
                    await emitter(event)
            except (asyncio.TimeoutError, Exception) as e:
                pass  # Drop failed/timed-out emissions (client disconnected or pipe saturated)
            finally:
                queue.task_done()

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
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
        Requires a live Request; Open WebUI always provides __request__ for tool and model resolution.
        """
        if __request__ is None:
            raise TypeError(
                "Planner pipe requires __request__ (FastAPI/Starlette Request). "
                "It must not be None."
            )

        # Ensure metadata is present even if not passed
        __metadata__ = __metadata__ or body.get("metadata", {})

        self.user_valves = (
            __user__.pop("valves", None)
            if isinstance(__user__, dict)
            else getattr(__user__, "valves", None)
        ) or self.UserValves()
        # v3.6.5 Structural Overhaul: Initialize Producer-Consumer Queues
        ui_queue = asyncio.Queue()

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

        # 1. Resolve full user object and pre-fetch accessible skills (v3.15 optimization)
        # Fetch skills once here to avoid redundant synchronous DB hits in the loop.
        user_id = __user__.get("id")
        user_obj, user_skills = await asyncio.gather(
            asyncio.to_thread(Users.get_user_by_id, user_id),
            asyncio.to_thread(Skills.get_skills_by_user_id, user_id, "read"),
        )
        accessible_skills = {s.id: s for s in user_skills if s.is_active}

        # Comprehensive metadata for engine components (v3 parity + internal objects)
        metadata = {
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": pipe_metadata,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__user_obj__": user_obj,
            "__user_skills__": accessible_skills,
            "__files__": __files__,
            "__chat_id__": chat_id,
            "__message_id__": message_id,
        }

        # Resolve base URL (Valve -> Env -> Request)
        base_url = self.valves.OPEN_WEBUI_URL
        if not base_url:
            base_url = os.environ.get("WEBUI_URL", "")
        if not base_url:
            base_url = str(__request__.base_url).rstrip("/")

        # Extract model knowledge and features for tool management
        model_knowledge = pipe_metadata.get("knowledge") or pipe_metadata.get(
            "model_knowledge"
        )
        app_models = getattr(__request__.app.state, "MODELS", {})
        # Use the current pipe model ID for metadata/skill resolution (v3 parity)
        planner_info = app_models.get(body.get("model", ""), {})

        # Subagent Token Handling
        # Extraction and substitution are handled internally during subagent context preparation.

        # Life cycle: Step 0: Create Planner Context
        context = PlannerContext(
            request=__request__,
            user=user_obj,
            metadata=metadata,
            event_emitter=__event_emitter__,
            event_call=__event_call__,
            valves=self.valves,
            user_valves=self.user_valves,
            body=body,
            planner_info=planner_info,
            model_knowledge=model_knowledge,
            chat_id=chat_id,
            message_id=message_id,
        )

        # Life cycle: Step 1: Create state and UI
        state = PlannerState()
        ui = UIRenderer(__event_emitter__, __event_call__, ui_queue=ui_queue)

        # Life cycle: Step 2: Create registry (requires engine for MCPHub)
        registry = ToolRegistry(context)

        # Life cycle: Step 3: Create subagent manager
        subagents = SubagentManager(
            context=context,
            ui=ui,
            state=state,
            registry=registry,
        )

        # Life cycle: Step 4: Create engine and link all components
        engine = PlannerEngine(
            context=context,
            ui=ui,
            state=state,
            subagents=subagents,
            registry=registry,
        )

        # Link components together
        registry.engine = engine
        subagents.engine = engine

        # Wire engine to emitter for file tracking parity
        if hasattr(ui.emitter, "engine"):
            ui.emitter.engine = engine

        # Life cycle: Step 3.1: Start Background Consumption Workers
        ui_worker = asyncio.create_task(self._ui_worker(ui_queue, __event_emitter__))


        # Use a generator to keep the SSE stream active and the "Exploring" status valid
        try:
            # v3.6: Add a background "Keep-Alive" task to ensure the generator yields periodically
            # even if complex tools are running in the background.
            async for delta in engine.run(
                chat_id, self.valves, self.user_valves, body, __files__
            ):
                yield delta
        finally:
            # Lifecycle cleanup: Ensure all subagents and MCP connections are terminated
            # when the pipe generator is stopped/cancelled.
            try:
                # 1. Stop workers
                ui_worker.cancel()

                await engine.stop()
            except Exception as e:
                logger.debug(f"Error during engine.stop() in pipe finally: {e}")
