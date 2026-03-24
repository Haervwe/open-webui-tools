"""
title: Planner v3
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 3.2.0
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
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
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

THINKING_TAG_CLEANER_PATTERN = re.compile(
    r'</?(?:think|thinking|reason|reasoning|thought|Thought)>|\|begin_of_thought\||\|end_of_thought\|', 
    re.IGNORECASE
)



def clean_thinking_tags(message: str) -> str:
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL | re.IGNORECASE,
    )
    return re.sub(pattern, "", message).strip()


# ---------------------------------------------------------------------------
# JS modal builders (theme-aware via Open WebUI CSS custom properties)
# ---------------------------------------------------------------------------

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


def _build_ask_user_js(prompt_text: str, placeholder: str = "Type your response...", timeout_s: int = 120) -> str:
    import json as _json
    p = _json.dumps(prompt_text)
    ph = _json.dumps(placeholder)
    return f"""
return (function() {{
  return new Promise((resolve) => {{
{_base_theme_js()}
    let _timer;
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
    const panel = document.createElement('div');
    panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:520px;padding:28px;display:flex;flex-direction:column;gap:20px;`;
    const titleEl = document.createElement('div');
    titleEl.textContent = {p};
    titleEl.style.cssText = `font-size:18px;font-weight:700;color:${{col.text}};line-height:1.3;`;
    panel.appendChild(titleEl);
    const input = document.createElement('textarea');
    input.placeholder = {ph};
    input.style.cssText = `background:${{col.input}};border:1px solid ${{col.inputBorder}};border-radius:12px;padding:14px;color:${{col.text}};font-size:14px;min-height:90px;resize:none;outline:none;font-family:inherit;width:100%;box-sizing:border-box;`;
    input.onfocus = () => input.style.borderColor = col.btnPrimary;
    input.onblur = () => input.style.borderColor = col.inputBorder;
    panel.appendChild(input);
    const footer = document.createElement('div');
    footer.style.cssText = 'display:flex;gap:10px;justify-content:stretch;';
    const makeBtn = (label, primary) => {{
      const b = document.createElement('button');
      b.textContent = label;
      b.style.cssText = `flex:1;padding:11px 18px;border-radius:9999px;font-size:14px;font-weight:600;cursor:pointer;border:1px solid ${{primary ? 'transparent' : col.btnBorder}};background:${{primary ? col.btnPrimary : col.btn}};color:${{primary ? col.btnPrimaryText : col.btnText}};transition:opacity 0.15s;`;
      b.onmouseenter = () => b.style.opacity = '0.85';
      b.onmouseleave = () => b.style.opacity = '1';
      return b;
    }};
    const acceptBtn = makeBtn('Accept', true);
    const skipBtn = makeBtn('Skip', false);
    const skipAllBtn = makeBtn('Skip All', false);
    acceptBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'accept', value: input.value.trim()}})); }};
    skipBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }};
    skipAllBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'skip_all', value: ''}})); }};
    footer.appendChild(acceptBtn); footer.appendChild(skipBtn); footer.appendChild(skipAllBtn);
    panel.appendChild(footer);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    input.focus();
    input.onkeydown = (e) => {{ if((e.ctrlKey||e.metaKey)&&e.key==='Enter') acceptBtn.click(); if(e.key==='Escape') skipBtn.click(); }};
    overlay.onclick = (e) => {{ if(e.target===overlay) {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }} }};
    function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
    _timer = setTimeout(() => {{ cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }}, {timeout_s * 1000});
  }});
}})()"""


def _build_give_options_js(prompt_text: str, choices: list, context: str = "", timeout_s: int = 120) -> str:
    import json as _json
    p = _json.dumps(prompt_text)
    cx = _json.dumps(context)
    ch = _json.dumps(choices)
    return f"""
return (function() {{
  return new Promise((resolve) => {{
{_base_theme_js()}
    let _timer;
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;background:${{col.overlay}};display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);`;
    const panel = document.createElement('div');
    panel.style.cssText = `background:${{col.panel}};border:1px solid ${{col.border}};border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);color:${{col.text}};font-family:ui-sans-serif,system-ui,sans-serif;width:100%;max-width:480px;padding:28px;display:flex;flex-direction:column;gap:18px;`;
    const titleEl = document.createElement('div');
    titleEl.textContent = {p};
    titleEl.style.cssText = `font-size:18px;font-weight:700;color:${{col.text}};`;
    panel.appendChild(titleEl);
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
      b.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'accept', value: c}})); }};
      grid.appendChild(b);
    }});
    panel.appendChild(grid);
    const footer = document.createElement('div'); footer.style.cssText = 'display:flex;gap:10px;';
    const makeBtn = (label) => {{ const b = document.createElement('button'); b.textContent = label; b.style.cssText = `flex:1;padding:10px 16px;border-radius:9999px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid ${{col.btnBorder}};background:${{col.btn}};color:${{col.btnText}};transition:opacity 0.15s;`; b.onmouseenter = () => b.style.opacity='0.8'; b.onmouseleave = () => b.style.opacity='1'; return b; }};
    const skipBtn = makeBtn('Skip'); const skipAllBtn = makeBtn('Skip All');
    skipBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }};
    skipAllBtn.onclick = () => {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'skip_all', value: ''}})); }};
    footer.appendChild(skipBtn); footer.appendChild(skipAllBtn);
    panel.appendChild(footer);
    overlay.appendChild(panel); document.body.appendChild(overlay);
    overlay.onclick = (e) => {{ if(e.target===overlay) {{ clearTimeout(_timer); cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }} }};
    function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
    _timer = setTimeout(() => {{ cleanup(); resolve(JSON.stringify({{action:'skip', value: ''}})); }}, {timeout_s * 1000});
  }});
}})()"""


def _build_continue_cancel_js(context_msg: str, timeout_s: int = 300) -> str:
    import json as _json
    msg = _json.dumps(context_msg)
    return f"""
return (function() {{
  return new Promise((resolve) => {{
{_base_theme_js()}
    let _timer;
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
    const _cd = setInterval(() => {{ remaining--; countdown.textContent = `Auto-cancels in ${{remaining}}s`; if(remaining <= 0) {{ clearInterval(_cd); }} }}, 1000);
    _timer = setTimeout(() => {{ clearInterval(_cd); cleanup(); resolve(JSON.stringify({{action:'cancel'}})); }}, {timeout_s * 1000});
    function cleanup() {{ if(overlay.parentNode) overlay.parentNode.removeChild(overlay); }}
  }});
}})()"""


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: Any
    __model__: str
    __request__: Any

    class Valves(BaseModel):
        PLANNER_MODEL: str = Field(
            default="", 
            description="Mandatoy. The main model driving the planner, works Best with a Base Model (not workspace presets) | (must support Tool Calling and Structured Outputs and only native tool calling is supported) "
        )
        SUBAGENT_MODELS: str = Field(
            default="", 
            description="Comma-separated list of model IDs available to be queried as subagents works best with Workspace Model presets | only native tool calling is supported"
        )
        TEMPERATURE: float = Field(
            default=0.7, 
            description="Temperature for the planner agent"
        )
        REVIEW_MODEL: str = Field(
            default="",
            description="Model used for review_tasks , works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)"
        )
        ENABLE_TERMINAL_AGENT: bool = Field(
            default=True,
            description="Enable terminal subagent (only active when a terminal is attached to the request)"
        )
        TERMINAL_AGENT_MODEL: str = Field(
            default="",
            description="Model for the terminal agent, works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)"
        )
        ENABLE_IMAGE_GENERATION_AGENT: bool = Field(
            default=True,
            description="Enable built-in image generation subagent"
        )
        IMAGE_GENERATION_AGENT_MODEL: str = Field(
            default="",
            description="Model for the image generation agent , works Best with a Base Model (not workspace presets) |(leave blank to use the planner model)"
        )
        ENABLE_WEB_SEARCH_AGENT: bool = Field(
            default=True,
            description="Enable built-in web search subagent"
        )
        WEB_SEARCH_AGENT_MODEL: str = Field(
            default="",
            description="Model for the web search agent , works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)"
        )
        ENABLE_KNOWLEDGE_AGENT: bool = Field(
            default=True,
            description="Enable built-in knowledge, notes, and chat retrieval subagent"
        )
        KNOWLEDGE_AGENT_MODEL: str = Field(
            default="",
            description="Model for the knowledge agent , works Best with a Base Model (not workspace presets) | (leave blank to use the planner model)"
        )
        ENABLE_CODE_INTERPRETER_AGENT: bool = Field(
            default=True,
            description="Enable built-in code interpreter subagent. Executes Python code and returns results. The code_interpreter tool is moved here exclusively."
        )
        CODE_INTERPRETER_AGENT_MODEL: str = Field(
            default="",
            description="Model for the code interpreter agent, works best with a Base Model (not workspace presets) | (leave blank to use the planner model)"
        )
        CODE_INTERPRETER_TEMPERATURE: float = Field(
            default=0.1,
            description="Temperature for the code interpreter subagent. Low values (0.0-0.2) produce more deterministic, accurate code."
        )
        SYSTEM_PROMPT: str = Field(
            default="""You are an advanced agentic Planner. You have the ability to formulate a plan, act on it by delegating tasks to specialized subagents or using tools, and track your progress.
Your goal is to fulfill the user's request.

You have access to the following built-in special tools:
1. `update_state(task_id: str, status: str, description: str)`: Use this ONLY to track the tasks you are working on. Call this tool when you finish a logical step to mark it 'completed' (or 'failed'). The 'in_progress' status is handled automatically when you call a subagent. Do not change the original 'description'.
2. `call_subagent(model_id: str, prompt: str, task_id: str, related_tasks: list[str])`: Use this to delegate a subtask to a specialized model. 
   - **Threading & Context**: The `task_id` identifies the conversation thread with the subagent within this execution. To **continue or follow up** on a previous interaction, you MUST use the **same** `task_id`. To start a **fresh** conversation, use a **new** `task_id`.
   - **@task_id Text Replacement**: When you write `@task_id` (e.g., `@task_1`) in a **prompt** or your **final response**, it will be **automatically replaced** with the LAST complete subagent response text for that task_id. This is a literal text substitution — the entire `@task_1` token gets swapped for the full output of that task. Example: If task_1's subagent returned an analysis, writing "Here is the analysis: @task_1" in your final response will embed the full analysis text at that location.
   - **Raw Task ID (no @)**: Use the plain ID (`task_1`) in tool parameters like `task_id`, `task_ids`, and `related_tasks`. NEVER prefix with @ in these parameter fields.
   - **CRITICAL — `related_tasks` for cross-task data passing**: Subagents are ISOLATED — they CANNOT see any other task's results unless you explicitly pass them. When a subagent needs data produced by a DIFFERENT task, you MUST list that task's raw ID in the `related_tasks` array. This injects the full result text into the subagent's system context so it can read and use it.
     - Example: If `task_research` produced research data and you now call a `terminal_agent` via `task_write_report` to save that research to a file, you MUST include `related_tasks: ["task_research"]` — otherwise the terminal agent has NO access to the research content.
     - Do NOT confuse `related_tasks` with thread persistence: `related_tasks` injects the final result of OTHER completed tasks; using the same `task_id` continues the conversation history of THAT SAME thread.

You must:
- BE STRICT WITH STATE STRUCTURE. Follow the plan provided exactly.
- Dont call subagents for unrelated tasks.
- **CODING RULE**: For ANY coding, scripting, calculation, or data-processing task, ALWAYS delegate to a `code agent` if not present do it yourself. NEVER use web_search_agent or knowledge_agent to fetch or produce code.
- Methodically execute the steps, using `call_subagent` for complex analysis, generation, or reasoning steps.
- As you finish each small step, call `update_state` to mark that specific task as 'completed'.
- **ALWAYS pass `related_tasks`** when a subagent needs results from previous tasks. Subagents are isolated and cannot see other tasks' outputs without this.
- **RESULT TRUNCATION**: The `result` field in a subagent tool response may be truncated for context efficiency. The FULL, untruncated output is ALWAYS available via `@task_id` (literal replacement in prompts/final response) or `read_task_result(task_id)`. Use these when you need the complete content.
- Once the objective is complete, compile the final result. Use `@task_id` references in your final response to include large previous outputs — remember these are **literal text replacements** with the LAST subagent message for that task.
- Relative API addresses like `/api/v1/...` are fully valid and should be used exactly as is.
- Final Output is YOUR responsability. Make it look good dont output placeholder text from previous task results instead use the @task_id to refeplace it with the full output of that task.
""",
            description="System Prompt for the planner agent (used when Plan Mode is ON)"
        )
        NO_PLAN_SYSTEM_PROMPT: str = Field(
            default="""You are an advanced agentic assistant with the ability to delegate tasks to specialized subagents and use tools.
Your goal is to fulfill the user's request by leveraging the available subagents and tools.

You have access to the following built-in special tools:
1. `call_subagent(model_id: str, prompt: str, task_id: str, related_tasks: list[str])`: Use this to delegate a subtask to a specialized model.
   - **Threading & Context**: The `task_id` identifies the conversation thread with the subagent. To **continue or follow up** on a previous interaction, use the **same** `task_id`. To start a **fresh** conversation, use a **new** `task_id`.
   - **@task_id Text Replacement**: When you write `@task_id` (e.g., `@task_1`) in a **prompt** or your **final response**, it will be **automatically replaced** with the LAST complete subagent response text for that task_id. This is literal text substitution — `@task_1` gets swapped for the full output. Example: writing "Here is the result: @task_1" embeds the complete task_1 output at that location.
   - **Raw Task ID (no @)**: Use the plain ID (`task_1`) in tool parameters like `task_id`, `task_ids`, and `related_tasks`. NEVER prefix with @ in parameter fields.
   - **CRITICAL — `related_tasks` for cross-task data passing**: Subagents are ISOLATED — they CANNOT see any other task's results unless you explicitly pass them. When a subagent needs data produced by a DIFFERENT task, you MUST list that task's raw ID in the `related_tasks` array. This injects the full result text into the subagent's system context.
     - Example: If `task_research` produced research data and you now call a `terminal_agent` via `task_save` to save it, you MUST include `related_tasks: ["task_research"]` — otherwise the agent has NO access to the content.
     - Do NOT confuse `related_tasks` with thread persistence: `related_tasks` injects results of OTHER tasks; same `task_id` continues the SAME thread's history.
2. `read_task_result(task_id: str)`: Read the full text result of a previously completed task.
3. `review_tasks(task_ids: list, prompt: str)`: Spawn an LLM cross-review over task results using a custom prompt.

You must:
- Delegate complex work to subagents using `call_subagent`.
- Dont call subagents for unrelated tasks.
- **CODING RULE**: For ANY coding, scripting, calculation, or data-processing task, ALWAYS delegate to a `code_agent` if not present do it yourself. NEVER use web_search_agent or knowledge_agent to fetch or produce code.
- **ALWAYS pass `related_tasks`** when a subagent needs results from previous tasks. Subagents are isolated and cannot see other tasks' outputs without this.
- **RESULT TRUNCATION**: The `result` field in a subagent tool response may be truncated for context efficiency. The FULL, untruncated output is ALWAYS available via `@task_id` (literal replacement in prompts/final response) or `read_task_result(task_id)`. Use these when you need the complete content.
- Use `@task_id` references in your final response to include large previous outputs — these are **literal text replacements** with the LAST subagent message for that task.
- Compose a clear final response for the user once all work is done.
- Relative API addresses like `/api/v1/...` are fully valid and should be used exactly as is.
- Final Output is YOUR responsability. Make it look good dont output placeholder text from previous task results instead use the @task_id to refeplace it with the full output of that task.
""",
            description="System Prompt for the agent when Plan Mode is OFF (no state tracking)"
        )
        USER_INPUT_TIMEOUT: int = Field(
            default=120,
            description="Timeout in seconds for user-input modal responses (ask_user / give_options). After this time the input is auto-skipped."
        )
        MAX_PLANNER_ITERATIONS: int = Field(
            default=25,
            description="Maximum planner loop iterations before asking the user to continue or cancel. Set to 0 to disable."
        )
        MAX_SUBAGENT_ITERATIONS: int = Field(
            default=25,
            description="Maximum tool-call iterations per subagent thread before asking the user to continue or cancel. Set to 0 to disable."
        )

    class UserValves(BaseModel):
        PLAN_MODE: bool = Field(
            default=True,
            description="Enable Plan Mode with visual task state tracking (HTML plan embed, state updates, completion verification). When disabled, the agent delegates to subagents directly without structured planning overhead."
        )
        ENABLE_USER_INPUT_TOOLS: bool = Field(
            default=True,
            description="Allow the planner to call ask_user and give_options to request clarification or choices from you during execution. Disable to let the planner run fully autonomously."
        )
        YOLO_MODE: bool = Field(
            default=False,
            description="YOLO: disable all iteration limits for both the planner and subagents. The planner will run until it naturally finishes with no Continue/Cancel interruptions."
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

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
        import hashlib, time as _time
        embed_id = "pe-" + hashlib.md5(str(_time.monotonic()).encode()).hexdigest()[:8]

        status_colors = {
            "pending":     "#9ca3af",
            "in_progress": "#60a5fa",
            "completed":   "#10b981",
            "failed":      "#ef4444",
        }
        check_icon = {
            "pending":     '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle></svg>',
            "in_progress": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
            "completed":   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
            "failed":      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
        }

        tasks_html = ""
        for task_id, task_info in planner_state.items():
            status = task_info.get("status", "pending")
            desc   = task_info.get("description", "")
            color  = status_colors.get(status, status_colors["pending"])
            icon   = check_icon.get(status, "")
            tasks_html += f'''
            <div class="pe-card" style="margin-bottom:12px;padding:16px;border-left:4px solid {color};border-radius:12px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                    <div style="display:flex;align-items:center;gap:8px;color:{color};">
                        {icon}
                        <strong class="pe-title" style="font-size:14px;font-weight:600;letter-spacing:0.3px;">{task_id}</strong>
                    </div>
                    <span style="font-size:10px;font-weight:700;padding:4px 10px;border-radius:99px;text-transform:uppercase;letter-spacing:1px;color:{color};background:rgba(128,128,128,0.15);">
                        {status.replace("_", " ")}
                    </span>
                </div>
                <div class="pe-desc" style="font-size:13px;line-height:1.5;font-weight:400;padding-left:26px;">
                    {desc}
                </div>
            </div>'''

        if not tasks_html:
            tasks_html = '<div class="pe-empty" style="padding:16px;text-align:center;font-size:13px;font-style:italic;border-radius:12px;">Planning...</div>'

        # The embed runs inside a sandboxed srcdoc iframe.
        # window.parent.document is accessible (allow-same-origin + allow-scripts).
        # Strategy: read the ACTUAL computed body backgroundColor from the parent page,
        # parse its RGB values, and derive all embed colors mathematically.
        # This auto-adapts to ALL OWUI themes (light, dark, oled-dark, her, custom).
        theme_script = f'''<script>
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
    // Read the actual body background of the parent page (handles all 4 OWUI themes)
    var bodyBg=window.parent.getComputedStyle(r.body).backgroundColor;
    var base=parseRgb(bodyBg);
    // If body is transparent (rgba(0,0,0,0)), walk to documentElement
    if(!base||luma(base)===0&&bodyBg.indexOf('rgba')>-1){{
      bodyBg=window.parent.getComputedStyle(r.documentElement).backgroundColor;
      base=parseRgb(bodyBg);
    }}
    if(!base)base=[17,24,39]; // safe dark fallback
    var dark=luma(base)<0.5;
    var step=dark?18:-18;
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
</script>'''

        html = (
            '<style>html,body{margin:0;padding:0;background:transparent!important}</style>\n'
            f'<div id="{embed_id}" style="background:#1e293b;border:1px solid #334155;'
            'border-radius:20px;padding:28px;margin:6px;'
            'font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;'
            'box-shadow:0 4px 20px rgba(0,0,0,0.5);">\n'
            '  <div style="display:flex;flex-direction:column;align-items:center;'
            'text-align:center;gap:12px;margin-bottom:24px;">\n'
            '    <div style="font-size:32px;">🧠</div>\n'
            '    <div>\n'
            '      <h3 style="margin:0;color:#f1f5f9;font-size:18px;font-weight:800;'
            'letter-spacing:-0.2px;">Planner Subagents</h3>\n'
            '      <p class="pe-subtitle" style="margin:4px 0 0 0;font-size:12px;'
            'color:#94a3b8;font-weight:500;">Live Execution State</p>\n'
            '    </div>\n'
            '  </div>\n'
            f'  <div style="display:flex;flex-direction:column;gap:4px;">\n    {tasks_html}\n  </div>\n'
            f'</div>\n{theme_script}'
        )
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
            # Skip resolution for task ID fields to preserve raw IDs
            if k in ["task_id", "task_ids", "related_tasks"]:
                resolved[k] = v
                continue
                
            if isinstance(v, str):
                resolved[k] = self.resolve_action_references(v, action_results)
            elif isinstance(v, dict):
                resolved[k] = self.resolve_dict_references(v, action_results)
            elif isinstance(v, list):
                resolved_list = []
                for item in v:
                    if isinstance(item, str):
                        # Only resolve if NOT an ID field
                        if k not in ["task_ids", "related_tasks"]:
                            resolved_list.append(self.resolve_action_references(item, action_results))
                        else:
                            resolved_list.append(item)
                    elif isinstance(item, dict):
                        resolved_list.append(self.resolve_dict_references(item, action_results))
                    else:
                        resolved_list.append(item)
                resolved[k] = resolved_list
            else:
                resolved[k] = v
        return resolved

    def _extract_json_array(self, text: str) -> list:
        """Extract the first valid JSON array from text, handled redundantly for robustness."""
        # 1. Clean thinking tags
        text = clean_thinking_tags(text)
        
        # 2. Extract from markdown if present
        markdown_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if markdown_match:
            text = markdown_match.group(1)
            
        # 3. Basic cleanup
        text = text.strip()
        
        # 4. Try finding the first [
        start = text.find("[")
        if start == -1:
            return []
            
        # 5. Use raw_decode to find the first valid JSON array
        decoder = json.JSONDecoder()
        remaining_text = text[start:]
        
        try:
            obj, _ = decoder.raw_decode(remaining_text)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict) and "tasks" in obj:
                # Handle potential wrap in an object if structured output used a field name
                return obj["tasks"]
        except json.JSONDecodeError:
            # If it failed, try cleaning trailing commas
            try:
                cleaned = re.sub(r",\s*([\]}])", r"\1", remaining_text)
                obj, _ = decoder.raw_decode(cleaned)
                if isinstance(obj, list):
                    return obj
                elif isinstance(obj, dict) and "tasks" in obj:
                    return obj["tasks"]
            except:
                pass
                
        return []

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

    async def get_streaming_completion(self, messages, model_id: str, body: dict, tools: list = None, temperature_override: float = None):
        form_data = {**body}
        form_data["model"] = model_id
        form_data["messages"] = messages
        form_data["temperature"] = temperature_override if temperature_override is not None else self.valves.TEMPERATURE
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
        __metadata__: dict = None,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __event_call__: Callable[[Any], Awaitable[None]] = None,
        __request__=None
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__event_call__ = __event_call__
        self.__user_valves__ = __user__.pop("valves", None) or self.UserValves()
        self.__user__ = Users.get_user_by_id(__user__.get("id"))
        self.__request__ = __request__
        self.__model__ = body.get("model", "")
        plan_mode = self.__user_valves__.PLAN_MODE
        self.__skip_all_inputs__ = False  # Shared flag: Skip All mutes all user-input requests for this pipe() call

        # OWUI pops metadata from the body before passing to pipe functions,
        # but injects it as __metadata__. Use that as primary source.
        pipe_metadata = __metadata__ or body.get("metadata", {}) or {}
        
        # Identify chat for context persistence
        chat_id = pipe_metadata.get("chat_id") or body.get("chat_id") or body.get("id") or "default"
        
        # Extract terminal_id from metadata (middleware moves it there)
        terminal_id = pipe_metadata.get("terminal_id") or body.get("terminal_id")
        logger.debug(f"[Planner] terminal_id={terminal_id}, metadata keys={list(pipe_metadata.keys())}")

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
        
        # ── Inject virtual native-feature subagents into the list ──
        # Map of virtual_id -> config dict including restrictive builtinTools to prevent tool leakage
        virtual_agents = {}
        
        # Base dict that disables ALL builtin tool categories
        _all_builtins_off = {
            "time": False, "knowledge": False, "chats": False, "memory": False,
            "web_search": False, "image_generation": False, "code_interpreter": False,
            "notes": False, "channels": False,
        }
        
        if self.valves.ENABLE_IMAGE_GENERATION_AGENT:
            virtual_agents["image_gen_agent"] = {
                "model": self.valves.IMAGE_GENERATION_AGENT_MODEL or model_id,
                "description": "Built-in image generation and editing subagent. Can generate and edit images from text prompts.",
                "system_message": (
                    "You are a specialized image generation subagent. Your role is to generate or edit images based on the user's prompt. "
                    "Use the generate_image tool for creating new images and edit_image for modifying existing ones. "
                    "Always return the image URLs or file paths in your final response so the planner can use them."
                ),
                "features": {"image_generation": True},
                "type": "builtin",
                "builtin_model_override": {"info": {"meta": {"builtinTools": {**_all_builtins_off, "image_generation": True}}}},
            }
            subagents_list.append("image_gen_agent")
            subagent_descriptions.append(
                "- ID: image_gen_agent (Name: Image Generation Agent)\n  Description: Built-in image generation and editing subagent. Can generate and edit images from text prompts."
            )
            
        if self.valves.ENABLE_WEB_SEARCH_AGENT:
            virtual_agents["web_search_agent"] = {
                "model": self.valves.WEB_SEARCH_AGENT_MODEL or model_id,
                "description": "Built-in web search subagent. Can search the web and fetch URL content.",
                "system_message": (
                    "You are a specialized web search and research subagent. Your role is to search the web for information and fetch content from URLs. "
                    "Use search_web to find relevant results and fetch_url to retrieve full page content. "
                    "Synthesize and return the relevant information clearly in your response."
                ),
                "features": {"web_search": True},
                "type": "builtin",
                "builtin_model_override": {"info": {"meta": {"builtinTools": {**_all_builtins_off, "web_search": True, "time": True}}}},
            }
            subagents_list.append("web_search_agent")
            subagent_descriptions.append(
                "- ID: web_search_agent (Name: Web Search Agent)\n  Description: Built-in web search subagent. Can search the web and fetch URL content."
            )
            
        if self.valves.ENABLE_KNOWLEDGE_AGENT:
            virtual_agents["knowledge_agent"] = {
                "model": self.valves.KNOWLEDGE_AGENT_MODEL or model_id,
                "description": "Built-in knowledge, notes, and chat history retrieval subagent. Can search and read notes, knowledge bases, and past conversations.",
                "system_message": (
                    "You are a specialized knowledge retrieval subagent. Your role is to search through notes, knowledge bases, and chat history to find relevant information. "
                    "Use the available search and retrieval tools to find the information requested. "
                    "Return the relevant findings clearly and completely in your response."
                ),
                "features": {},
                "type": "builtin",
                "builtin_model_override": {"info": {"meta": {"builtinTools": {**_all_builtins_off, "knowledge": True, "chats": True, "notes": True, "channels": True}}}},
            }
            subagents_list.append("knowledge_agent")
            subagent_descriptions.append(
                "- ID: knowledge_agent (Name: Knowledge Agent)\n  Description: Built-in knowledge, notes, and chat history retrieval subagent. Can search and read notes, knowledge bases, and past conversations."
            )

        if self.valves.ENABLE_CODE_INTERPRETER_AGENT:
            virtual_agents["code_interpreter_agent"] = {
                "model": self.valves.CODE_INTERPRETER_AGENT_MODEL or model_id,
                "description": "Built-in code interpreter subagent. Can generate content in ANY language (HTML, CSS, JS, Python, shell scripts, JSON, etc.) and execute Python in a sandboxed Jupyter environment. Use this for ALL coding, scripting, content generation, and computation tasks.",
                "system_message": (
                    "You are a specialized code and content generation subagent. "
                    "You can generate content in ANY language — HTML, CSS, JavaScript, Python, shell scripts, JSON, YAML, and more. "
                    "You also have access to a code_interpreter tool that runs Python in a sandboxed Jupyter environment "
                    "(maintains state across calls) — use it when you need to EXECUTE code, not just generate it.\n"
                    "Rules:\n"
                    "- For HTML/CSS/JS or any web content: output the FULL, complete, self-contained content DIRECTLY — "
                    "do NOT wrap it in Python code that generates it. Return it as-is so the planner can use it immediately.\n"
                    "- For computation, data processing, or tasks that need execution: use the code_interpreter tool.\n"
                    "- Output ONLY the final, complete, working content unless the user explicitly asks for explanations.\n"
                    "- Do NOT add prose, commentary, or markdown outside of code blocks unless asked.\n"
                    "- Return generated file paths, URLs, or raw content in your response so the planner can use them.\n"
                    "- If the task requires a file to be created, return its absolute path in your response.\n"
                    "- For visualizations or plots, save them to a file and return the path."
                ),
                "features": {"code_interpreter": True},
                "type": "builtin",
                "temperature": self.valves.CODE_INTERPRETER_TEMPERATURE,
                "builtin_model_override": {"info": {"meta": {"builtinTools": {**_all_builtins_off, "code_interpreter": True}}}},
            }
            subagents_list.append("code_interpreter_agent")
            subagent_descriptions.append(
                "- ID: code_interpreter_agent (Name: Code Interpreter Agent)\n  Description: Built-in code interpreter subagent. Can generate content in ANY language (HTML, CSS, JS, Python, etc.) and execute Python in a sandbox. Use for ALL coding, content generation, scripting, and computation tasks."
            )
            
        if self.valves.ENABLE_TERMINAL_AGENT and terminal_id:
            virtual_agents["terminal_agent"] = {
                "model": self.valves.TERMINAL_AGENT_MODEL or model_id,
                "description": "Built-in terminal subagent. Can execute commands, read/write files, and interact with the system terminal.",
                "system_message": (
                    "You are a specialized terminal subagent. Your role is to execute terminal commands, read and write files, and perform system operations. "
                    "Use the available terminal tools (run_command, write_file, etc.) to fulfill the request. "
                    "Always return the command outputs and results clearly in your response."
                ),
                "features": {},
                "type": "terminal",
                "terminal_id": terminal_id,
                "builtin_model_override": {"info": {"meta": {"builtinTools": {**_all_builtins_off}}}},
            }
            subagents_list.append("terminal_agent")
            subagent_descriptions.append(
                "- ID: terminal_agent (Name: Terminal Agent)\n  Description: Built-in terminal subagent. Can execute commands, read/write files, and interact with the system terminal."
            )
        
        base_prompt = self.valves.SYSTEM_PROMPT if plan_mode else self.valves.NO_PLAN_SYSTEM_PROMPT
        subagent_models_text = "\n".join(subagent_descriptions) if subagent_descriptions else "None specified in config"
        system_prompt = base_prompt + f"\nAvailable Subagent Models:\n{subagent_models_text}\n"
        if self.__user_valves__.ENABLE_USER_INPUT_TOOLS and self.__event_call__:
            system_prompt += (
                "\nUser Interaction Tools (ask_user, give_options) are ACTIVE. "
                "Use them whenever you need user input or a choice from the user. "
                "NEVER ask the user a question in plain text — ALWAYS use the appropriate tool instead.\n"
            )

        sys_message = {"role": "system", "content": system_prompt}
        planner_messages = [sys_message] + messages

        metadata = pipe_metadata
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
        
        # ── Planner tools: chat payload + planner model OWUI settings (with dedup) ──
        planner_settings_tool_ids = []
        planner_db_info = Models.get_model_by_id(model_id)
        if planner_db_info:
            p_meta = planner_db_info.meta.model_dump() if hasattr(planner_db_info.meta, "model_dump") else planner_db_info.meta
            p_params = planner_db_info.params.model_dump() if hasattr(planner_db_info.params, "model_dump") else planner_db_info.params
            if isinstance(p_meta, dict):
                planner_settings_tool_ids.extend(p_meta.get("toolIds", []))
            if isinstance(p_params, dict):
                planner_settings_tool_ids.extend(p_params.get("toolIds", []) or p_params.get("tools", []))
        planner_model_info = app_models.get(model_id, {})
        if planner_model_info:
            pi_meta = planner_model_info.get("info", {}).get("meta", {})
            pi_params = planner_model_info.get("info", {}).get("params", {})
            planner_settings_tool_ids.extend(pi_meta.get("toolIds", []))
            planner_settings_tool_ids.extend(pi_params.get("toolIds", []) or pi_params.get("tools", []))
        
        # Merge chat payload tools with planner settings tools (dedup)
        all_planner_tool_ids = list(set([*(tool_ids or []), *planner_settings_tool_ids]))
            
        tools_dict = {}
        if all_planner_tool_ids:
            tools_dict = await get_tools(self.__request__, all_planner_tool_ids, self.__user__, extra_params)

        # ── Preload and cache subagent tools ──
        subagent_tools_cache = {}
        for sub_model in subagents_list:
            # Check if this is a virtual native-feature agent
            if sub_model in virtual_agents:
                va = virtual_agents[sub_model]
                va_model = va["model"]
                va_model_info = app_models.get(va_model, {})
                s_tools_dict = {}
                
                if va["type"] == "terminal":
                    # Terminal agent: load terminal tools
                    try:
                        s_tools_dict = await get_terminal_tools(
                            self.__request__, va["terminal_id"], self.__user__, extra_params
                        )
                    except Exception as e:
                        logger.error(f"Failed to load terminal tools: {e}")
                elif va["type"] == "builtin":
                    # Native feature agent: load builtin tools with restrictive model override
                    # to prevent tool leakage (e.g., web_search_agent getting chat/knowledge tools)
                    builtin_model = va.get("builtin_model_override", va_model_info)
                    try:
                        s_tools_dict = get_builtin_tools(
                            self.__request__, extra_params, features=va["features"], model=builtin_model
                        )
                    except Exception as e:
                        logger.error(f"Failed to load builtin tools for {sub_model}: {e}")
                
                s_tools = [{"type": "function", "function": t["spec"]} for t in s_tools_dict.values()] if s_tools_dict else None
                subagent_tools_cache[sub_model] = {
                    "dict": s_tools_dict,
                    "specs": s_tools,
                    "system_message": va["system_message"],
                    "actual_model": va_model,
                    "temperature_override": va.get("temperature"),
                }
                continue
            
            # Regular subagent: use ONLY preconfigured tools
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
                
            p_features = {}
            if model_db_info:
                meta = model_db_info.meta.model_dump() if hasattr(model_db_info.meta, "model_dump") else model_db_info.meta
                params = model_db_info.params.model_dump() if hasattr(model_db_info.params, "model_dump") else model_db_info.params

                for obj in [meta, params, model_info.get("info", {}).get("meta", {}), model_info.get("info", {}).get("params", {})]:
                    if isinstance(obj, dict):
                        if isinstance(obj.get("features"), dict):
                            p_features.update(obj["features"])
                        for f_id in obj.get("defaultFeatureIds", []):
                            p_features[f_id] = True
                            
            # Subagents use ONLY their own preconfigured tools - NOT chat payload tools
            combined_tool_ids = list(set(subagent_tool_ids))
            s_tools_dict = {}
            if combined_tool_ids:
                s_tools_dict = await get_tools(self.__request__, combined_tool_ids, self.__user__, extra_params)
                
            try:
                builtin_tools = get_builtin_tools(self.__request__, extra_params, features=p_features, model=model_info)
                if builtin_tools:
                    s_tools_dict.update(builtin_tools)
            except Exception as e:
                logger.error(f"Failed to load built-in tools for subagent {sub_model}: {e}")
                
            s_tools = [{"type": "function", "function": t["spec"]} for t in s_tools_dict.values()] if s_tools_dict else None
            subagent_tools_cache[sub_model] = {
                "dict": s_tools_dict,
                "specs": s_tools,
                "system_message": model_system_message
            }

        planner_state = {}
        action_results = {}
        subagent_contexts = {} # Local storage for subagent follow-ups within this session
        total_emitted = ""

        if plan_mode:
            # Emit initial html embed state
            await self.emit_html_embed(planner_state)
            
            # --- Phase 1: Plan Creation ---
            plan_system_prompt = (
                f"{system_prompt}\n\n"
                "You are currently in the PLANNING PHASE. Your ONLY job is to create a detailed, step-by-step plan to fulfill the user's request. "
                "You MUST keep in mind the capabilities of the subagents and tools available to you when forming the plan. "
                "Output your plan strictly as a JSON array of objects, where each object has a 'task_id' (string, e.g., 'task_1') and a 'description' (string, short summary of the step). "
                "Do not output any options, greetings, or other text. Only the JSON array."
            )
            plan_messages = [{"role": "system", "content": plan_system_prompt}] + messages
            
            # Add structured output if possible
            plan_body = {**body}
            plan_body["response_format"] = {
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
                                        "description": {"type": "string"}
                                    },
                                    "required": ["task_id", "description"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["tasks"],
                        "additionalProperties": False
                    }
                }
            }

            await self.emit_status("Planning...", False)
            plan_content_chunks = []
            try:
                async for event in self.get_streaming_completion(plan_messages, model_id, body=plan_body, tools=None):
                    if event["type"] == "content":
                        plan_content_chunks.append(event["text"])
            except Exception as e:
                logger.error(f"Error during plan formation: {e}")

            full_plan_text = "".join(plan_content_chunks)

            # Try to parse the json plan using robust extraction

            try:
                plan_json = self._extract_json_array(full_plan_text)
                if plan_json:
                    for task in plan_json:
                        tid = task.get("task_id")
                        desc = task.get("description")
                        if tid and desc:
                            planner_state[tid] = {"status": "pending", "description": desc}
                
                if not planner_state:
                    raise ValueError("No tasks parsed from plan.")
                    
                # Emit the newly populated state
                await self.emit_html_embed(planner_state)
            except Exception as e:
                logger.warning(f"Could not parse planner JSON effectively: {e}. Raw: {full_plan_text}")
                # Fallback if failed: add a single task
                planner_state["task_1"] = {"status": "pending", "description": "Process user request"}
                await self.emit_html_embed(planner_state)

            # Inform the main agent of the approved plan
            planner_messages.append({"role": "system", "content": f"Here is the established plan. Do not deviate from it. Execute the steps logically:\n{json.dumps(planner_state)}"})
            # ------------------------------

        final_pass_done = False
        
        # Build dynamic tool schemas ONCE after planning to avoid cache busting
        available_tasks = list(planner_state.keys()) if planner_state else ["task_1"]
        
        tools_spec = []
        
        # update_state only available in plan mode
        if plan_mode:
            tools_spec.append({
                "type": "function",
                "function": {
                    "name": "update_state",
                    "description": "Track task progress by providing a task id, status, and description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "Unique identifier for the task", "enum": available_tasks},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"]},
                            "description": {"type": "string", "description": "Short description of the task"}
                        },
                        "required": ["task_id", "status", "description"]
                    }
                }
            })
        
        # call_subagent: in plan mode, task_id/related_tasks are constrained to available_tasks enum; in no-plan mode, free-form
        call_subagent_task_id_schema = {"type": "string", "description": "ID identifying the conversation thread. Use the raw ID (e.g. 'task_1'). Using the same task_id follows up or continues a previous conversation."}
        related_tasks_items_schema = {"type": "string"}
        if plan_mode:
            call_subagent_task_id_schema["enum"] = available_tasks
            related_tasks_items_schema["enum"] = available_tasks
        
        tools_spec.append({
            "type": "function",
            "function": {
                "name": "call_subagent",
                "description": "Call a specialized subagent model to perform a task. Returns the output from the model. Using the same task_id continues the same conversation thread (thread persistence).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "The ID of the model to use", "enum": subagents_list if subagents_list else ["__none__"]},
                        "prompt": {"type": "string", "description": "Detailed instructions for the subagent. Use '@task_id' (e.g. '@task_1') as a TEXT REPLACEMENT MACRO — it will be automatically replaced with the LAST complete subagent response text for that task_id."},
                        "task_id": call_subagent_task_id_schema,
                        "related_tasks": {
                            "type": "array",
                            "items": related_tasks_items_schema,
                            "description": "Optional list of previously completed Task IDs (e.g. 'task_1') whose results you need contextually available to the subagent."
                        }
                    },
                    "required": ["model_id", "prompt", "task_id"]
                }
            }
        })
        
        # read_task_result and review_tasks: constrained enum in plan mode, free-form in no-plan mode
        read_task_id_schema = {"type": "string", "description": "The Task ID (e.g. 'task_1')"}
        review_items_schema = {"type": "string"}
        if plan_mode:
            read_task_id_schema["enum"] = available_tasks
            review_items_schema["enum"] = available_tasks
        
        tools_spec.append({
            "type": "function",
            "function": {
                "name": "read_task_result",
                "description": "Read the FULL, untruncated raw text result of a completed task verbatim. Use this when the result shown in the call_subagent response was truncated and you need the complete content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": read_task_id_schema
                    },
                    "required": ["task_id"]
                }
            }
        })
        tools_spec.append({
            "type": "function",
            "function": {
                "name": "review_tasks",
                "description": "Spawn an invisible LLM cross-review over massive task results using a custom prompt, saving your own context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_ids": {
                            "type": "array",
                            "items": review_items_schema
                        },
                        "prompt": {"type": "string", "description": "Instructions on what to review or extract from these given task IDs"}
                    },
                    "required": ["task_ids", "prompt"]
                }
            }
        })
        for tool_name, tool_data in tools_dict.items():
            if "spec" in tool_data:
                tools_spec.append({"type": "function", "function": tool_data["spec"]})

        # ask_user and give_options: only added when the user valve is enabled
        if self.__user_valves__.ENABLE_USER_INPUT_TOOLS and self.__event_call__:
            tools_spec.append({
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Ask the user for free-form text input. Use when you need clarification or additional information from the user.  never send HTML or code here, use it only if necesary, Returns the text, or a skip/skip-all sentinel if the user declines.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt_text": {"type": "string", "description": "The question or request to present to the user"},
                            "placeholder": {"type": "string", "description": "Optional hint text for the input field"}
                        },
                        "required": ["prompt_text"]
                    }
                }
            })
            tools_spec.append({
                "type": "function",
                "function": {
                    "name": "give_options",
                    "description": "Present the user with a list of choices and wait for their selection. Returns the chosen option, never send HTML or code here, use it only if necesary, or a skip/skip-all sentinel if the user declines.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt_text": {"type": "string", "description": "The question or prompt to display"},
                            "choices": {"type": "array", "items": {"type": "string"}, "description": "List of options to present to the user"},
                            "context": {"type": "string", "description": "Optional background context to show beneath the title"}
                        },
                        "required": ["prompt_text", "choices"]
                    }
                }
            })

        planner_iteration = 0

        while True:
            # --- Planner iteration limit check ---
            planner_iteration += 1
            yolo = self.__user_valves__.YOLO_MODE
            if not yolo and self.valves.MAX_PLANNER_ITERATIONS > 0 and planner_iteration > self.valves.MAX_PLANNER_ITERATIONS:
                if self.__event_call__:
                    try:
                        js = _build_continue_cancel_js(
                            f"The planner has completed {planner_iteration - 1} iterations. Would you like it to continue working?",
                            timeout_s=300
                        )
                        raw = await self.__event_call__({"type": "execute", "data": {"code": js}})
                        raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                        decision = json.loads(raw_str or '{"action":"cancel"}')
                        if decision.get("action") == "continue":
                            planner_iteration = 1  # Reset counter
                        else:
                            await self.emit_status("Planner stopped by user after iteration limit.", True)
                            break
                    except Exception as e:
                        logger.error(f"Planner iteration limit modal error: {e}")
                        break
                else:
                    logger.warning("Planner iteration limit reached. No event_call available. Stopping.")
                    break

            try:
                import time
                content_chunks = []
                tool_calls_dict = {}
                reasoning_buffer = ""
                full_reasoning = ""
                reasoning_start_time = None
                total_emitted_base = total_emitted
                turn_start_base = total_emitted # Save original to revert during promotion if needed
                
                async for event in self.get_streaming_completion(planner_messages, model_id, body=body, tools=tools_spec):
                    event_type = event["type"]
                    
                    if event_type == "reasoning":
                        reasoning_piece = event.get("text", "")
                        if reasoning_piece:
                            if reasoning_start_time is None:
                                reasoning_start_time = time.monotonic()
                            reasoning_buffer += reasoning_piece
                            full_reasoning += reasoning_piece
                            display = "\n".join(
                                f"> {line}" if not line.startswith(">") else line
                                for line in reasoning_buffer.splitlines()
                            )
                            await self.emit_replace(
                                total_emitted
                                + '\n\n<details type="reasoning" done="false">\n'
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
                            total_emitted_base += (
                                f'\n\n<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                + display
                                + "\n</details>\n\n"
                            )
                            reasoning_buffer = ""
                            total_emitted = total_emitted_base + clean_thinking_tags("".join(content_chunks))
                            await self.emit_replace(total_emitted)

                        if event_type == "content":
                            text = event["text"]
                            content_chunks.append(text)
                            current_content = "".join(content_chunks)
                            display_content = clean_thinking_tags(current_content)
                            display_content = re.sub(r'<(?:think|thinking|reason|reasoning|thought|Thought)>.*', '', display_content, flags=re.DOTALL | re.IGNORECASE)
                            total_emitted = total_emitted_base + display_content
                            await self.emit_replace(total_emitted)
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
                            total_emitted_base += (
                                f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                + display
                                + "\n</details>\n\n"
                            )
                            reasoning_buffer = ""
                            total_emitted = total_emitted_base + clean_thinking_tags("".join(content_chunks))
                            await self.emit_replace(total_emitted)

                raw_content = "".join(content_chunks)
                # Only clean if not promoting reasoning later
                final_content = clean_thinking_tags(raw_content)
                final_content = THINKING_TAG_CLEANER_PATTERN.sub('', final_content).strip()

                # Intercept hallucinated `<tool_call>` XML before promotion check
                if not tool_calls_dict:
                    extracted_any = False
                    # Check both content and reasoning for hallucinated tool calls
                    xml_count = 0
                    for source_text in [final_content, full_reasoning]:
                        if "<tool_call>" in source_text:
                            xml_matches = re.finditer(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|$)', source_text, re.DOTALL)
                            for match in xml_matches:
                                tc_data = match.group(1)
                                extracted_any = True
                                func_match = re.search(r'<function\s*=\s*"?([^>"]+)"?>', tc_data)
                                if func_match:
                                    func_name = func_match.group(1).strip()
                                    kwargs = {}
                                    param_matches = re.findall(r'<parameter\s*=\s*"?([^>"]+)"?>(.*?)(?=</parameter>|<parameter\s*=|</function>|$)', tc_data, re.DOTALL)
                                    for p_name, p_val in param_matches:
                                        kwargs[p_name.strip()] = p_val.strip()
                                        
                                    tool_calls_dict[f"xml_{xml_count}"] = {
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
                                            tool_calls_dict[f"xml_{xml_count}"] = {
                                                "id": str(uuid4()),
                                                "function": {
                                                    "name": data["name"],
                                                    "arguments": json.dumps(data.get("arguments", data.get("parameters", {})))
                                                }
                                            }
                                    except:
                                        pass
                                xml_count += 1
                    
                    if extracted_any:
                        # Clean the XML tool call tags from final_content
                        final_content = re.sub(r'<tool_call>.*?</tool_call>', '', final_content, flags=re.DOTALL).strip()
                        
                        # Also clean XML tool calls from the reasoning that was already flushed into
                        # total_emitted_base, distinguishing two cases:
                        #
                        # Case 1: Reasoning had ONLY XML tool calls (no other real thinking text)
                        #   → Remove the entire reasoning <details> block (nothing useful to show).
                        # Case 2: Reasoning had real thinking + XML tool calls mixed in
                        #   → Keep the <details> block but strip just the <tool_call>…</tool_call> from it.
                        if not final_content and full_reasoning.strip():
                            cleaned_reasoning = re.sub(r'<tool_call>.*?</tool_call>', '', full_reasoning, flags=re.DOTALL).strip()
                            if not cleaned_reasoning:
                                # Case 1: pure XML in reasoning — drop the whole block
                                total_emitted_base = re.sub(
                                    r'\n*<details type="reasoning"[^>]*>.*?</details>\n*',
                                    '',
                                    total_emitted_base,
                                    flags=re.DOTALL
                                ).rstrip()
                            else:
                                # Case 2: real thinking content exists — update the block in-place,
                                # replacing any <tool_call>…</tool_call> inside it with nothing.
                                total_emitted_base = re.sub(
                                    r'(<details type="reasoning"[^>]*>.*?</details>)',
                                    lambda m: re.sub(r'<tool_call>.*?</tool_call>', '', m.group(1), flags=re.DOTALL),
                                    total_emitted_base,
                                    flags=re.DOTALL
                                )
                
                # Promotion logic: only promote if NO content chunks were ever added AND we have reasoning
                if not content_chunks and not tool_calls_dict and full_reasoning.strip():
                    # Promoting reasoning to final content (clean tags)
                    final_content = THINKING_TAG_CLEANER_PATTERN.sub('', full_reasoning).strip()
                    
                    # Revert total_emitted_base to turn start to avoid duplicating or deleting previous history
                    total_emitted_base = turn_start_base.strip()
                    if total_emitted_base:
                        total_emitted_base += "\n\n"
                
                # Double-check cleanup: if we extracted XML tool calls, make sure they are NOT in the final user-facing text
                if tool_calls_dict:
                    final_content = re.sub(r'<tool_call>.*?</tool_call>', '', final_content, flags=re.DOTALL).strip()
                    
                total_emitted = total_emitted_base + final_content
                await self.emit_replace(total_emitted)

                # Append message to history
                assistant_message = {"role": "assistant", "content": final_content}
                if tool_calls_dict:
                    tool_calls_list = list(tool_calls_dict.values())
                    assistant_message["tool_calls"] = tool_calls_list
                planner_messages.append(assistant_message)
                
                # Resolve aliases in the newly streamed response content for the UI
                resolved_content = self.resolve_action_references(final_content, action_results)
                if resolved_content != final_content:
                    total_emitted = total_emitted.replace(final_content, resolved_content)
                    final_content = resolved_content
                    await self.emit_replace(total_emitted)

                if not tool_calls_dict:
                    # Final verification pass (plan mode only)
                    if plan_mode:
                        unresolved_tasks = [tid for tid, info in planner_state.items() if info["status"] not in ["completed"]]
                        if unresolved_tasks and not final_pass_done:
                            final_pass_done = True
                            await self.emit_status("Verifying task states...", False)
                            
                            judge_messages = planner_messages.copy()
                            judge_messages.append({
                                "role": "user",
                                "content": (
                                    f"SYSTEM: Review the conversation. Tasks {', '.join(unresolved_tasks)} are not marked as completed. "
                                    "Respond with a JSON object (no extra text) with two fields:\n"
                                    "1. \"updates\": array of {\"task_id\": string, \"status\": \"completed\"|\"failed\", \"description\": string} "
                                    "for any tasks that were actually completed or clearly failed in the conversation.\n"
                                    "2. \"follow_up_prompt\": string — if tasks are genuinely incomplete and need more work, "
                                    "write a short instruction for the planner to continue. If all tasks are resolved or no further work "
                                    "is possible, set this to an empty string \"\"."
                                )
                            })
                            
                            # Structured output schema for the judge
                            judge_body = {**body}
                            judge_body["response_format"] = {
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
                                                        "status": {"type": "string", "enum": ["completed", "failed"]},
                                                        "description": {"type": "string"}
                                                    },
                                                    "required": ["task_id", "status", "description"],
                                                    "additionalProperties": False
                                                }
                                            },
                                            "follow_up_prompt": {"type": "string"}
                                        },
                                        "required": ["updates", "follow_up_prompt"],
                                        "additionalProperties": False
                                    }
                                }
                            }
                            
                            judge_chunks = []
                            try:
                                async for sub_event in self.get_streaming_completion(judge_messages, model_id, body=judge_body, tools=None):
                                    if sub_event["type"] == "content":
                                        judge_chunks.append(sub_event["text"])
                            except Exception as e:
                                logger.error(f"Error in judge verification: {e}")
                            
                            # Parse judge response — strip thinking tags first
                            judge_raw = clean_thinking_tags("".join(judge_chunks))
                            judge_result = {}
                            try:
                                # Find the first { and raw_decode from there
                                brace_start = judge_raw.find("{")
                                if brace_start != -1:
                                    decoder = json.JSONDecoder()
                                    judge_result, _ = decoder.raw_decode(judge_raw[brace_start:])
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse judge JSON. Raw: {judge_raw[:300]}")
                            
                            # Apply state updates from judge
                            updated_any = False
                            for upd in judge_result.get("updates", []):
                                tid = upd.get("task_id", "")
                                status = upd.get("status", "")
                                if tid in planner_state and status in ["completed", "failed"]:
                                    planner_state[tid]["status"] = status
                                    if upd.get("description"):
                                        planner_state[tid]["description"] = upd["description"]
                                    updated_any = True
                                    
                            if updated_any:
                                await self.emit_html_embed(planner_state)
                                await asyncio.sleep(0.1)
                                await self.emit_replace(total_emitted)
                            
                            # Check what remains incomplete after judge updates
                            still_incomplete = [tid for tid, info in planner_state.items() if info["status"] not in ["completed"]]
                            follow_up = judge_result.get("follow_up_prompt", "").strip()
                            
                            if not still_incomplete:
                                pass  # All resolved — fall through to break
                            elif follow_up:
                                # Judge identified remaining work — inject follow-up and let planner retry
                                planner_messages.append({
                                    "role": "user",
                                    "content": f"SYSTEM: The following tasks are still incomplete: {', '.join(still_incomplete)}. {follow_up}"
                                })
                                await self.emit_status("Continuing incomplete tasks...", False)
                                continue
                            else:
                                # Judge found nothing to update and no follow-up — warn user
                                incomplete_details = ", ".join(
                                    f"{tid} ({planner_state[tid].get('description', 'no description')})"
                                    for tid in still_incomplete
                                )
                                warning = f"\n\n⚠️ **Some tasks could not be completed:** {incomplete_details}\n"
                                await self.emit_message(warning)
                                total_emitted += warning

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

                    # Build a compact version of arguments for the <details> tag attributes
                    # to prevent long choices arrays / prompts from breaking HTML rendering
                    try:
                        display_args = json.loads(arguments_str)
                        for k, v in display_args.items():
                            if isinstance(v, str) and len(v) > 60:
                                display_args[k] = v[:57] + "..."
                            elif isinstance(v, list) and len(v) > 3:
                                display_args[k] = v[:3] + [f"... +{len(v) - 3} more"]
                        display_arguments_str = json.dumps(display_args, ensure_ascii=False)
                    except Exception:
                        display_arguments_str = arguments_str

                    # Emit "Executing..." tag
                    executing_tag = self._build_tool_call_details(call_id, function_name, display_arguments_str, done=False)
                    if not total_emitted.endswith("\n"):
                        total_emitted += "\n"
                    total_emitted += executing_tag
                    await self.emit_replace(total_emitted)

                    tool_result_str = ""
                    
                    if function_name == "update_state" and plan_mode:
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
                        
                        # Auto mark in_progress (plan mode only)
                        if plan_mode and sub_task_id and sub_task_id in planner_state:
                            planner_state[sub_task_id]["status"] = "in_progress"
                            await self.emit_html_embed(planner_state)
                            await asyncio.sleep(0.1)
                            await self.emit_replace(total_emitted)
                            
                        await self.emit_status(f"[Subagent: {sub_model}] Executing {sub_task_id}...", False)
                        
                        cached = subagent_tools_cache.get(sub_model, {})
                        sub_sys = cached.get("system_message", "")
                        if not sub_sys:
                            sub_sys = f"You are a specialized subagent acting as {sub_model}. Follow the prompt directly and accurately using your tools."
                            
                        # Ensure subagents understand they aren't interacting with the UI directly
                        sub_sys += "\n\nCRITICAL CONTEXT: You are running as a headless subagent entirely in the background. DO NOT return markdown elements that rely on Open WebUI UI embeds for tool generation output. Any tools you use (like generate_image, search, etc.) will return URLs, base64 data, or raw paths. You MUST return these raw HTML references, URLs, files or images relative or absolute paths unconditionally in your final reply so the main planner can use them."
                        if "related_tasks" in args and isinstance(args["related_tasks"], list):
                            for rt in args["related_tasks"]:
                                # Robustness: strip '@' if hallucinated
                                rt_clean = rt.lstrip("@")
                                if rt_clean in action_results:
                                    sub_sys += f"\n\n--- RESULTS FROM PREVIOUS TASK {rt} ---\n{action_results[rt_clean]}\n--- END OF {rt} ---\n"
                            
                        context_key = (chat_id, sub_task_id)
                        if context_key in subagent_contexts:
                            sub_messages = subagent_contexts[context_key]
                            # Update system message with potentially new sub_sys (including new related_tasks results)
                            if sub_messages and sub_messages[0]["role"] == "system":
                                sub_messages[0]["content"] = sub_sys
                            sub_messages.append({"role": "user", "content": sub_prompt})
                        else:
                            sub_messages = [
                                {"role": "system", "content": sub_sys},
                                {"role": "user", "content": sub_prompt}
                            ]
                            subagent_contexts[context_key] = sub_messages
                        
                        try:
                            sub_tools_dict = cached.get("dict", {})
                            sub_tools = cached.get("specs", None)
                            
                            sub_final_answer_chunks = []
                            sub_reasoning_chunks = []
                            sub_called_tools = []
                            sub_iteration = 0
                            
                            while True:
                                sub_tc_dict = {}
                                sub_iteration += 1
                                if not self.__user_valves__.YOLO_MODE and self.valves.MAX_SUBAGENT_ITERATIONS > 0 and sub_iteration > self.valves.MAX_SUBAGENT_ITERATIONS:
                                    if self.__event_call__:
                                        try:
                                            js = _build_continue_cancel_js(
                                                f"Subagent '{sub_model}' ({sub_task_id}) has completed {sub_iteration - 1} tool-call iterations. Continue?",
                                                timeout_s=300
                                            )
                                            raw = await self.__event_call__({"type": "execute", "data": {"code": js}})
                                            raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                                            decision = json.loads(raw_str or '{"action":"cancel"}')
                                            if decision.get("action") == "continue":
                                                sub_iteration = 1
                                            else:
                                                sub_final_answer_chunks.append("[Subagent stopped at iteration limit by user.]")
                                                break
                                        except Exception as e:
                                            logger.error(f"Subagent iteration limit modal error: {e}")
                                            break
                                    else:
                                        sub_final_answer_chunks.append("[Subagent stopped at iteration limit.]")
                                        break


                                # Resolve actual model for virtual agents (virtual IDs -> real model IDs)
                                actual_sub_model = cached.get("actual_model", sub_model)
                                sub_temp_override = cached.get("temperature_override")
                                async for sub_event in self.get_streaming_completion(sub_messages, actual_sub_model, body=body, tools=sub_tools, temperature_override=sub_temp_override):
                                    event_type = sub_event["type"]
                                    if event_type == "content":
                                        sub_final_answer_chunks.append(sub_event["text"])
                                    elif event_type == "reasoning":
                                        sub_reasoning_chunks.append(sub_event["text"])
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
                                                    
                                raw_sub_content = "".join(sub_final_answer_chunks)
                                sub_content = clean_thinking_tags(raw_sub_content)
                                sub_content = THINKING_TAG_CLEANER_PATTERN.sub('', sub_content).strip()
                                
                                if not sub_content and not sub_tc_dict:
                                    if raw_sub_content.strip():
                                        extracted = THINKING_TAG_CLEANER_PATTERN.sub('', raw_sub_content).strip()
                                        if extracted:
                                            sub_content = extracted
                                    elif "".join(sub_reasoning_chunks).strip():
                                        sub_content = "".join(sub_reasoning_chunks).strip()
                                        
                                if not sub_content:
                                    sub_content = raw_sub_content
                                
                                sub_final_answer_chunks = [] # Reset for next cycle if tool calls happen
                                sub_reasoning_chunks = []
                                
                                # Intercept hallucinated `<tool_call>` XML for subagents
                                if not sub_tc_dict and "<tool_call>" in sub_content:
                                    xml_matches = re.finditer(r'<tool_call>\s*(.*?)\s*(?:</tool_call>|$)', sub_content, re.DOTALL)
                                    for idx, match in enumerate(xml_matches):
                                        tc_data = match.group(1)
                                        func_match = re.search(r'<function\s*=\s*"?([^>"]+)"?>', tc_data)
                                        if func_match:
                                            func_name = func_match.group(1).strip()
                                            kwargs = {}
                                            param_matches = re.findall(r'<parameter\s*=\s*"?([^>"]+)"?>(.*?)(?=</parameter>|<parameter\s*=|</function>|$)', tc_data, re.DOTALL)
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
                                    sub_messages.append({"role": "assistant", "content": sub_content})
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
                                        tc_return = process_tool_result(self.__request__, stc_name, tc_res, target_tool.get("type", ""), False, pipe_metadata, self.__user__)
                                        res_str = tc_return[0] if len(tc_return) > 0 else str(tc_res)
                                        
                                        # Track tool call: name, truncated params, success
                                        truncated_args = {k: (str(v)[:80] + "..." if len(str(v)) > 80 else str(v)) for k, v in stc_args_obj.items()}
                                        sub_called_tools.append({"tool": stc_name, "arguments": truncated_args, "success": True})
                                        
                                        sub_messages.append({
                                            "role": "tool",
                                            "tool_call_id": call_id,
                                            "name": stc_name,
                                            "content": str(res_str)
                                        })
                                    else:
                                        err_res = f"Error: Tool {stc_name} not found"
                                        sub_called_tools.append({"tool": stc_name, "arguments": {}, "success": False})
                                        sub_messages.append({
                                            "role": "tool",
                                            "tool_call_id": call_id,
                                            "name": stc_name,
                                            "content": err_res
                                        })

                            raw_result = "".join(sub_final_answer_chunks)
                            
                            # Store ONLY the raw result for @task_id replacement and related_tasks injection
                            if sub_task_id:
                                action_results[sub_task_id] = raw_result
                            
                            # Build structured response for planner context
                            # NOTE: result may be truncated here to preserve planner context window.
                            # The FULL result is always available via @task_id or read_task_result().
                            was_truncated = len(raw_result) > 2000
                            result_preview = raw_result if not was_truncated else raw_result[:2000] + f"\n... [TRUNCATED — {len(raw_result) - 2000} chars omitted. Use @{sub_task_id} or read_task_result('{sub_task_id}') to get the FULL untruncated output.]"
                            structured_response = {
                                "task_id": sub_task_id,
                                "called_tools": sub_called_tools,
                                "result": result_preview,
                            }
                            if was_truncated:
                                structured_response["note"] = f"IMPORTANT: Result was truncated. Use @{sub_task_id} in prompts/final response for FULL literal text replacement, or call read_task_result('{sub_task_id}') to read the complete output."
                            tool_result_str = json.dumps(structured_response, ensure_ascii=False)
                                
                            await self.emit_status(f"Planner evaluating...", False)
                                
                        except Exception as e:
                            logger.error(f"Subagent error: {e}")
                            tool_result_str = f"Error calling subagent: {e}"

                    elif function_name == "ask_user":
                        if self.__skip_all_inputs__:
                            tool_result_str = "User has opted out of all further input requests for this session."
                        else:
                            try:
                                js = _build_ask_user_js(
                                    args.get("prompt_text", ""),
                                    args.get("placeholder", "Type your response..."),
                                    self.valves.USER_INPUT_TIMEOUT,
                                )
                                raw = await self.__event_call__({"type": "execute", "data": {"code": js}})
                                raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                                parsed = json.loads(raw_str or "{}")
                                action = parsed.get("action", "skip")
                                if action == "skip_all":
                                    self.__skip_all_inputs__ = True
                                    tool_result_str = "User has opted out of all further input requests for this session."
                                elif action == "accept" and parsed.get("value"):
                                    tool_result_str = f"User provided: {parsed['value']}"
                                else:
                                    tool_result_str = "User skipped this input (no details provided)."
                            except Exception as e:
                                tool_result_str = f"ask_user error: {e}"

                    elif function_name == "give_options":
                        if self.__skip_all_inputs__:
                            tool_result_str = "User has opted out of all further input requests for this session."
                        else:
                            try:
                                js = _build_give_options_js(
                                    args.get("prompt_text", ""),
                                    args.get("choices", []),
                                    args.get("context", ""),
                                    self.valves.USER_INPUT_TIMEOUT,
                                )
                                raw = await self.__event_call__({"type": "execute", "data": {"code": js}})
                                raw_str = raw if isinstance(raw, str) else (raw.get("result") or raw.get("value") or "{}") if raw else "{}"
                                parsed = json.loads(raw_str or "{}")
                                action = parsed.get("action", "skip")
                                if action == "skip_all":
                                    self.__skip_all_inputs__ = True
                                    tool_result_str = "User has opted out of all further input requests for this session."
                                elif action == "accept" and parsed.get("value"):
                                    tool_result_str = f"User selected: {parsed['value']}"
                                else:
                                    tool_result_str = "User skipped this selection (no option chosen)."
                            except Exception as e:
                                tool_result_str = f"give_options error: {e}"

                    elif function_name == "read_task_result":

                        rt_id = args.get("task_id", "")
                        if rt_id in action_results:
                            tool_result_str = action_results[rt_id]
                        else:
                            tool_result_str = f"Task {rt_id} not found."
                            
                    elif function_name == "review_tasks":
                        rt_ids = args.get("task_ids", [])
                        rt_prompt = args.get("prompt", "")
                        if not rt_ids or not rt_prompt:
                            tool_result_str = "Error: must specify task_ids and prompt."
                        else:
                            await self.emit_status("Reviewing tasks cross-reference...", False)
                            review_sys = "You are a specialized review subagent. Read the provided task results carefully and respond strictly to the user's prompt by synthesizing or evaluating the information."
                            for rx in rt_ids:
                                if rx in action_results:
                                    review_sys += f"\n\n--- RESULTS FROM TASK {rx} ---\n{action_results[rx]}\n--- END OF {rx} ---\n"
                            
                            review_messages = [{"role": "system", "content": review_sys}, {"role": "user", "content": rt_prompt}]
                            
                            try:
                                review_model = self.valves.REVIEW_MODEL or model_id
                                review_chunks = []
                                async for rx_event in self.get_streaming_completion(review_messages, review_model, body=body, tools=None):
                                    if rx_event["type"] == "content":
                                        review_chunks.append(rx_event["text"])
                                tool_result_str = "".join(review_chunks)
                            except Exception as e:
                                tool_result_str = f"Review failed: {str(e)}"

                    elif function_name in tools_dict:
                        tool_data = tools_dict[function_name]
                        try:
                            allowed_keys = tool_data.get("spec", {}).get("parameters", {}).get("properties", {}).keys()
                            filtered_args = {k: v for k, v in args.items() if k in allowed_keys}
                            
                            res = await tool_data["callable"](**filtered_args)
                            tool_return = process_tool_result(self.__request__, function_name, res, tool_data.get("type", ""), False, pipe_metadata, self.__user__)
                            tool_result_str = tool_return[0] if len(tool_return) > 0 else str(res)
                        except Exception as e:
                            tool_result_str = f"Error executing tool {function_name}: {e}"
                    else:
                        tool_result_str = f"Error: Function {function_name} is not recognized or available."

                    context_result_str = tool_result_str
                    if len(context_result_str) > 4000 and function_name == "call_subagent":
                        ref_id = args.get("task_id", function_name)
                        context_result_str = context_result_str[:3800] + f"\n\n...[TRUNCATED. Full output saved. Use 'review_tasks' tool passing 'task_ids': ['{ref_id}'] to review it, or use 'related_tasks' parameter in 'call_subagent' to inject its full text context]..."

                    # Complete executing tag in the UI stream
                    total_emitted = total_emitted.replace(executing_tag, "")
                    done_tag = self._build_tool_call_details(
                        call_id, function_name, display_arguments_str, done=True, result=tool_result_str
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
