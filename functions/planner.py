"""
title: Planner
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 2.0.2
"""

import logging
import json
import asyncio
from fastapi import Request
from typing import List, Dict, Optional, Callable, Awaitable, Any
from pydantic import BaseModel, Field
from datetime import datetime
from open_webui.constants import TASKS

from open_webui.utils.chat import generate_chat_completion  # type: ignore
from open_webui.utils.tools import get_tools  # type: ignore

from open_webui.models.users import Users, User
from open_webui.models.tools import Tools

import re


name = "Planner_2"


def clean_thinking_tags(message: str) -> str:
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL,
    )

    return re.sub(pattern, "", message).strip()


def clean_json_response(response_text: str) -> str:

    start = response_text.find("{")
    end = response_text.rfind("}") + 1

    if start == -1 or end == -1:

        return "{}"

    return response_text[start:end]


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


class Action(BaseModel):
    """Model for a single action in the plan"""

    id: str
    type: str
    description: str
    params: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    tool_ids: Optional[list[str]] = None
    output: Optional[Dict[str, str]] = None
    status: str = "pending"  # pending, in_progress, completed, failed, warning
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model: Optional[str] = None  # NEW: model to use for this action


class Plan(BaseModel):
    """Model for the complete execution plan"""

    goal: str
    actions: List[Action]
    metadata: dict[str, Any] = Field(default_factory=dict[str, Any])
    final_output: Optional[str] = None
    execution_summary: Optional[dict[str, str | int | dict[str, str | int | None]]] = (
        None
    )


class ReflectionResult(BaseModel):
    """A simplified model for storing reflection analysis results."""

    is_successful: bool
    quality_score: float = Field(
        ..., description="A score from 0.0 to 1.0 indicating output quality."
    )
    issues: List[str] = Field(
        default_factory=list, description="Specific issues found in the output."
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions for how to fix the issues.",
    )


class Pipe:
    __current_event_emitter__: Callable[[dict[str, Any]], Awaitable[None]]
    __user__: User
    __model__: str

    class Valves(BaseModel):
        MODEL: str = Field(
            default="", description="Model to use (model id from ollama)"
        )
        ACTION_MODEL: str = Field(
            default="", description="Model to use (model id from ollama)"
        )
        WRITER_MODEL: str = Field(
            default="",
            description="Model to use for text/documentation actions (e.g., RP/Writer model)",
        )
        ACTION_PROMPT_REQUIREMENTS_TEMPLATE: str = Field(
            default="""Requirements:
1. Use the specified tool(s) for this action if provided. Do NOT default to code/scripts unless the user explicitly requests it.
2. The output must directly achieve the step goal and be actionable.
3. For tool-based actions:
   - Use the tool(s) as intended and provide clear, relevant output.
4. For code actions (only if requested):
   - Code must be complete, runnable, and all variables defined.
   - No placeholder functions or TODO comments.
5. For text/documentation actions:
   - Be specific, actionable, and include all relevant details.
6. For actions with dependencies, use ONLY the outputs of the listed dependencies.
7. Do not ask for clarifications; just perform the action as described.
            """,
            description="General requirements for task completions, used in ALL action steps, change it to make the outputs of the task align to your general goal",
        )
        AUTOMATIC_TAKS_REQUIREMENT_ENHANCEMENT: bool = Field(
            default=False,
            description="Use an LLM call to refine the requirements of each ACTION based on the whole PLAN and GOAL before executing an ACTION (uses the ACTION_PROMPT_REQUIREMENTS_TEMPLATE as an example of requirements)",
        )
        MAX_RETRIES: int = Field(
            default=3, description="Maximum number of retry attempts"
        )
        CONCURRENT_ACTIONS: int = Field(
            default=1,
            description="Maximum concurrent actions (experimental try on your own risk)",
        )
        ACTION_TIMEOUT: int = Field(
            default=300, description="Action timeout in seconds"
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.current_output = ""

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def get_completion(
        self,
        prompt: str | list[dict[str, Any]],
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        model: str | dict[str, Any] = "",
        tools: dict[str, dict[Any, Any]] = {},
    ) -> str:
        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a Helpful agent that does exactly as told and dont ask clarifications",
                },
                {"role": "user", "content": prompt},
            ]
            if isinstance(prompt, str)
            else prompt
        )
        __model = model if model and not tools else self.valves.ACTION_MODEL
        try:
            form_data: dict[str, Any] = {
                "model": __model,
                "messages": messages,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "tools": (
                    [
                        {"type": "function", "function": tool.get("spec", {})}
                        for tool in tools.values()
                    ]
                    if tools
                    else None
                ),
            }
            response: dict[str, Any] = await generate_chat_completion(
                self.__request__,
                form_data,
                user=self.__user__,
            )
            response_content = str(response["choices"][0]["message"]["content"])
            tool_calls: list[dict[str, Any]] | None = None
            try:
                tool_calls = response["choices"][0]["message"].get("tool_calls")
            except Exception:
                tool_calls = None
            if not tool_calls or not isinstance(tool_calls, list):
                if response_content == "\n":
                    logger.debug(f"No tool calls: {response}")
                return clean_thinking_tags(response_content)
            for tool_call in tool_calls:
                tool_function_name = tool_call["function"].get("name", None)
                if tool_function_name not in tools:
                    tool_result = f"{tool_function_name} not in {tools}"
                else:
                    tool = tools[tool_function_name]
                    spec = tool.get("spec", {})
                    allowed_params = (
                        spec.get("parameters", {}).get("properties", {}).keys()
                    )
                    tool_function_params = json.loads(
                        tool_call["function"].get("arguments", {})
                    )
                    tool_function_params = {
                        k: v
                        for k, v in tool_function_params.items()
                        if k in allowed_params
                    }

                    tool_function = tool["callable"]
                    tool_result = await tool_function(**tool_function_params)

                messages: list[dict[str, Any]] = messages + [
                    {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                    {
                        "role": "assistant",
                        "tool_call_id": tool_call["id"],
                        "name": tool_function_name,
                        "content": str(tool_result),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The tool '{tool_function_name}' has been executed and returned the output above. "
                            "Now, based on this output and the original task, provide the final, comprehensive answer for this step. "
                            "Do not simply repeat the tool's output. Synthesize it into a complete response. Your response is the final output for this action."
                        ),
                    },
                ]
            if model == self.valves.WRITER_MODEL:
                tools = {}
            tool_response = await self.get_completion(
                prompt=messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                model=model,
                tools=tools,
            )

            return tool_response
        except Exception as e:
            logger.error(f"LLM Call Error: {e}")
            raise e

    async def generate_mermaid(self, plan: Plan) -> str:
        """Generate Mermaid diagram representing the current plan state"""
        mermaid = ["graph TD", f'    Start["Goal: {plan.goal[:30]}..."]']

        status_emoji = {
            "pending": "⭕",
            "in_progress": "⚙️",
            "completed": "✅",
            "failed": "❌",
            "warning": "⚠️",
        }

        def sanitize_action_id(id_str: str) -> str:
            """Create a safe node ID by replacing invalid characters"""
            return f"action_{re.sub(r'[^a-zA-Z0-9]', '_', id_str)}"

        styles: list[str] = []
        for action in plan.actions:
            action_id = sanitize_action_id(action.id)
            mermaid.append(
                f'    {action_id}["{status_emoji[action.status]} {action.description[:40]}..."]'
            )

            if action.status == "in_progress":
                styles.append(f"style {action_id} fill:#fff4cc")
            elif action.status == "completed":
                styles.append(f"style {action_id} fill:#e6ffe6")
            elif action.status == "warning":
                styles.append(f"style {action_id} fill:#fffbe6")
            elif action.status == "failed":
                styles.append(f"style {action_id} fill:#ffe6e6")

        entry_actions = [action for action in plan.actions if not action.dependencies]
        for action in entry_actions:
            action_id = sanitize_action_id(action.id)
            mermaid.append(f"    Start --> {action_id}")

        for action in plan.actions:
            action_id = sanitize_action_id(action.id)
            for dep in action.dependencies:
                mermaid.append(f"    {sanitize_action_id(dep)} --> {action_id}")

        mermaid.extend(styles)

        return "\n".join(mermaid)

    async def create_plan(self, goal: str) -> Plan:
        tools: list[dict[str, Any]] = [
            {
                "tool_id": tool.id,
                "tool_name": tool.name,
            }
            for tool in Tools.get_tools()
        ]
        """Create an execution plan for the given goal"""
        system_prompt = f"""
You are an expert planning agent. Your job is to break down a complex goal into a precise, step-by-step plan. The plan must be a Directed Acyclic Graph (DAG).

AVAILABLE TOOLS:
{json.dumps(tools, indent=2)}

PLANNING PRINCIPLES (Follow these strictly!):
1.  **Dependency Graph (DAG):** Create a logical flow of actions. "Write Chapter 2" MUST depend on "Write Chapter 1". "Generate Illustration for Chapter 1" MUST depend on "Write Chapter 1".
2.  **Action Design:**
    - Each action must be a single, clear, and independently executable task.
    - For each action, specify the model to use in the 'model' field:
        - For tool-based actions (e.g., type: 'tool', 'research'), use the TASK/ACTION model: '{self.valves.ACTION_MODEL}'
        - For generative text actions (e.g., type: 'text', 'documentation'), use the WRITER model: '{self.valves.WRITER_MODEL}'

3.  **Final Synthesis Action (CRITICAL - READ CAREFULLY):**
    - The very last action in the plan MUST have the `id` set to `"final_synthesis"`.
    - The `type` for this action should be `"synthesis"`.
    - **This is a TEMPLATE action, not a generative one.**
    - The `description` for this action MUST be the final document structure, using placeholders in the format `{{action_id}}` where the output of a dependency should go.
    - DO NOT write instructions like "combine the outputs." Write the actual template.
    - **NEVER generate code (HTML, Python, etc.) directly in the template.** If code is needed, create a separate action to generate it and reference it with `{{action_id}}`.
    - **Only use simple placeholders like `{{action_id}}`, NOT nested ones like `{{action_id.field}}` or `{{action_id.output.field}}`.**
    - If a generative step is required to create the final output, create it as a regular action and link it to the final_synthesis.
    - When outputs can be used AS IS (for example text generation), use them directly in the template with simple placeholders.
    - Its `dependencies` list must contain the IDs of the final content actions.

    - **Correct Example for a 2-chapter story with illustrations:**
      ```json
      {{
        "id": "final_synthesis",
        "type": "synthesis",
        "description": "## Chapter 1\\n\\n{{write_chapter_1}}\\n\\n*Illustration: {{generate_illustration_1}}*\\n\\n---\\n\\n## Chapter 2\\n\\n{{write_chapter_2}}\\n\\n*Illustration: {{generate_illustration_2}}*",
        "dependencies": ["write_chapter_1", "generate_illustration_1", "write_chapter_2", "generate_illustration_2"],
        "model": "" // Model is not needed for a template action
      }}
      ```

    - **WRONG Example (DO NOT DO THIS):**
      ```json
      {{
        "id": "final_synthesis",
        "type": "synthesis",
        "description": "<html><head><title>{{title}}</title></head><body>{{content}}</body></html>",
        // This is WRONG because it generates HTML directly in template
      }}
      ```

    - **CORRECT Example for HTML output:**
      ```json
      {{
        "id": "create_html_page",
        "type": "text",
        "description": "Generate an HTML page that displays the research results, song, and image with proper HTML structure",
        "dependencies": ["research_action", "song_action", "image_action"],
        "model": "writer_model"
      }},
      {{
        "id": "final_synthesis",
        "type": "synthesis",
        "description": "{{create_html_page}}",
        "dependencies": ["create_html_page"],
        "model": ""
      }}
      ```

OUTPUT FORMAT:
Return ONLY a JSON object with the exact structure below. Do not add any other text, explanations, or markdown.

{{
    "goal": "<original goal / user_prompt>",
    "actions": [
        {{
            "id": "<unique_id>",
            "type": "<tool|text|research|code|script>",
            "description": "<SPECIFIC task description, or for final_synthesis, the template>",
            "tool_ids": ["<tool IDs if using tools>"],
            "params": {{}},
            "dependencies": ["<IDs of parent actions>"],
            "model": "<model to use for this action>"
        }},
        {{
        "id": "final_synthesis",
        "type": "synthesis",
        "description": "The `description` for this action MUST be the final document structure, using placeholders in the format `{{action_id}}` where the output of a dependency should go, in markdown format for titles.",
        "dependencies": ["<IDs of actions whos outputs will be used in the template above>"],
        "model": "" // Model is not needed for a template action
        }},
    ]
}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]
        for attempt in range(self.valves.MAX_RETRIES):
            try:
                result = await self.get_completion(
                    prompt=messages, temperature=0.8, top_k=60, top_p=0.95
                )
                clean_result = clean_json_response(result)
                plan_dict = json.loads(clean_result)

                actions = plan_dict.get("actions", [])

                for action in actions:
                    if "model" not in action:
                        if action.get("type") in ["text", "documentation", "synthesis"]:
                            action["model"] = self.valves.WRITER_MODEL
                        else:
                            action["model"] = self.valves.ACTION_MODEL

                plan = Plan(
                    goal=plan_dict.get("goal", goal),
                    actions=[Action(**a) for a in actions],
                )

                if not any(a.id == "final_synthesis" for a in plan.actions):
                    msg = "The generated plan is missing the required 'final_synthesis' action."
                    messages += [
                        {
                            "role": "assistant",
                            "content": f"previous attempt: {clean_result}",
                        },
                        {"role": "user", "content": f"error:: {msg}"},
                    ]
                    raise ValueError(msg)

                final_synthesis = next(
                    (a for a in plan.actions if a.id == "final_synthesis"), None
                )
                if final_synthesis:
                    template = final_synthesis.description

                    placeholder_pattern = r"\{([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}"
                    all_placeholders = re.findall(placeholder_pattern, template)

                    invalid_placeholders = [p for p in all_placeholders if "." in p]
                    if invalid_placeholders:
                        msg = (
                            f"Template contains invalid nested placeholders: {invalid_placeholders}. "
                            f"Use simple {{action_id}} format only, not {{action_id.field}} or {{action_id.output.field}}."
                        )
                        messages += [
                            {
                                "role": "assistant",
                                "content": f"previous attempt: {clean_result}",
                            },
                            {"role": "user", "content": f"error:: {msg}"},
                        ]
                        raise ValueError(msg)

                    code_patterns = [
                        (r"<[a-zA-Z][^>]*>", "HTML tags"),
                        (r"def\s+\w+\s*\(", "Python function definitions"),
                        (r"class\s+\w+\s*[:\(]", "Python class definitions"),
                        (r"import\s+\w+", "Python imports"),
                        (r"function\s+\w+\s*\(", "JavaScript functions"),
                        (r"<!DOCTYPE", "HTML DOCTYPE declarations"),
                        (r"<\?xml", "XML declarations"),
                    ]

                    for pattern, description in code_patterns:
                        if re.search(pattern, template):
                            msg = (
                                f"Template contains {description}. Templates should not contain code. "
                                f"Create a separate action to generate code and reference it with {{action_id}}."
                            )
                            messages += [
                                {
                                    "role": "assistant",
                                    "content": f"previous attempt: {clean_result}",
                                },
                                {"role": "user", "content": f"error:: {msg}"},
                            ]
                            raise ValueError(msg)

                    simple_placeholders = [p for p in all_placeholders if "." not in p]
                    action_ids = {a.id for a in plan.actions}
                    missing_actions = [
                        p for p in simple_placeholders if p not in action_ids
                    ]
                    if missing_actions:
                        msg = (
                            f"Template references non-existent actions: {missing_actions}. "
                            f"All placeholders must reference valid action IDs."
                        )
                        messages += [
                            {
                                "role": "assistant",
                                "content": f"previous attempt: {clean_result}",
                            },
                            {"role": "user", "content": f"error:: {msg}"},
                        ]
                        raise ValueError(msg)

                return plan
            except Exception as e:
                logger.error(
                    f"Error creating plan (attempt {attempt + 1}/{self.valves.MAX_RETRIES}): {e}"
                )
                if attempt < self.valves.MAX_RETRIES - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise
        raise RuntimeError(
            f"Failed to create plan after {self.valves.MAX_RETRIES} attempts"
        )

    async def enhance_requirements(self, plan: Plan, action: Action):
        dependencies_str = (
            json.dumps(action.dependencies) if action.dependencies else "None"
        )
        has_dependencies = bool(action.dependencies)

        requirements_prompt = f"""
You are an expert requirements generator for a generalist agent that can use a variety of tools, not just code. Focus on the following action:
Action Description: {action.description}
Parameters: {json.dumps(action.params)}
Tool(s) to use: {action.tool_ids if action.tool_ids else "None"}
Dependencies: {dependencies_str if has_dependencies else "None"}

Instructions:
- Generate a concise, numbered list of requirements to ensure this action is performed correctly.
- If a tool is specified, requirements should focus on correct and effective tool usage.
- Only require code/scripts if the user explicitly requested it; otherwise, prefer tool or text/documentation outputs.
- For actions with dependencies, clearly state how outputs from dependencies should be used.
- For text/documentation actions, be specific and actionable.
- For code actions (if requested), ensure code is complete, runnable, and all variables are defined.

Return ONLY a numbered list of requirements. Do not include explanations or extra text.
"""
        enhanced_requirements = await self.get_completion(
            prompt=requirements_prompt,
            temperature=0.7,
            top_k=40,
            top_p=0.8,
        )
        return enhanced_requirements

    async def execute_action(
        self, plan: Plan, action: Action, results: dict[str, Any], step_number: int
    ) -> dict[str, Any]:

        action.start_time = datetime.now().strftime("%H:%M:%S")
        action.status = "in_progress"

        def gather_all_parent_results(
            action_id: str,
            results: dict[str, Any],
            plan: Plan,
            visited: set[Any] | None = None,
        ) -> dict[Any, Any]:
            if visited is None:
                visited = set()
            if action_id in visited:
                return {}
            visited.add(action_id)
            action_to_check = next((a for a in plan.actions if a.id == action_id), None)
            if not action_to_check or not action_to_check.dependencies:
                return {}
            parent_results: Dict[str, Any] = {}
            for dep in action_to_check.dependencies:
                parent_results[dep] = results.get(dep, {})
                parent_results.update(
                    gather_all_parent_results(dep, results, plan, visited)
                )
            return parent_results

        context = gather_all_parent_results(action.id, results, plan)
        requirements = (
            await self.enhance_requirements(plan, action)
            if self.valves.AUTOMATIC_TAKS_REQUIREMENT_ENHANCEMENT
            else self.valves.ACTION_PROMPT_REQUIREMENTS_TEMPLATE
        )
        base_prompt = f"""
            Execute step {step_number}: {action.description}
            Overall Goal: {plan.goal}
        
            Context from dependent steps:
            - Parameters: {json.dumps(action.params)}
            - Previous Results: {json.dumps(context)}
        
            {requirements}
            
            Focus ONLY on this specific step's output.
            """

        attempts_remaining = self.valves.MAX_RETRIES
        best_output = None
        best_reflection = None
        best_quality_score = -1
        while attempts_remaining >= 0:
            try:
                current_attempt = self.valves.MAX_RETRIES - attempts_remaining
                if current_attempt == 0:
                    await self.emit_status(
                        "info",
                        f"Attempt {current_attempt + 1}/{self.valves.MAX_RETRIES + 1} for action {action.id}",
                        False,
                    )
                action.status = "in_progress"
                await self.emit_replace("")
                await self.emit_replace_mermaid(plan)

                if current_attempt > 0 and best_reflection:
                    base_prompt += f"""
                        
                        Previous attempt had these issues:
                        {json.dumps(best_reflection.issues, indent=2)}
                        
                        Required corrections based on suggestions:
                        {json.dumps(best_reflection.suggestions, indent=2)}
                        
                        Please address ALL issues above in this new attempt.
                        """

                try:
                    extra_params: dict[str, Any] = {
                        "__event_emitter__": self.__current_event_emitter__,
                        "__user__": self.user,
                        "__request__": self.__request__,
                    }

                    tools: dict[str, dict[Any, Any]] = get_tools(  # type: ignore
                        self.__request__,
                        action.tool_ids or [],
                        self.__user__,
                        extra_params,
                    )
                    system_prompt = f"""SYSTEM: You are the Action Agent, an expert at executing specific tasks within a larger plan. Your role is to focus solely on executing the current step, using ONLY the available tools and context provided.

    TASK CONTEXT:
    - Step {step_number} Description: {action.description}
    - Available Tools: {action.tool_ids if action.tool_ids else "None"}
    
    DEPENDENCIES AND INPUTS:
    - Parameters: {json.dumps(action.params)}
    - Input from Previous Steps: {json.dumps(context)}

    EXECUTION REQUIREMENTS:
    {requirements}

    CRITICAL GUIDELINES:
    1. Focus EXCLUSIVELY on this step's task - do not try to solve the overall goal
    2. Use ONLY the outputs from listed dependencies - do not reference other steps
    3. When using tools:
       - Use EXACTLY as specified in the tool documentation
       - Process and format the tool output appropriately for this step
    4. Produce a complete, self-contained output that can be used by dependent steps
    5. Never ask for clarification - work with what is provided
    6. Never output an empty message
    7. Remember that tool outputs are only visible to you - include relevant results in your response
    """
                    response = await self.get_completion(
                        prompt=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": base_prompt},
                        ],
                        temperature=0.9,
                        top_k=70,
                        top_p=0.95,
                        model=(
                            action.model
                            if action.model
                            else (
                                self.valves.ACTION_MODEL
                                if (self.valves.ACTION_MODEL != "")
                                else self.valves.MODEL
                            )
                        ),
                        tools=tools,
                    )

                    await self.emit_message(response)
                    logger.info(f"response complete  : {response}")

                    if not response or not response.strip():
                        await self.emit_status(
                            "warning",
                            "Action produced an empty output. Retrying...",
                            False,
                        )

                        best_reflection = ReflectionResult(
                            is_successful=False,
                            quality_score=0.0,
                            issues=["The action produced no output."],
                            suggestions=[
                                "The action must generate a non-empty response that directly addresses the task."
                            ],
                        )
                        attempts_remaining -= 1
                        continue

                    current_output = {"result": response}

                except Exception as api_error:
                    if attempts_remaining > 0:
                        attempts_remaining -= 1
                        await self.emit_status(
                            "warning",
                            f"API error, retrying... ({attempts_remaining + 1} attempts remaining)",
                            False,
                        )
                        continue
                    else:
                        action.status = "failed"
                        action.end_time = datetime.now().strftime("%H:%M:%S")
                        await self.emit_status(
                            "error",
                            f"API error in action {action.id} after all attempts",
                            True,
                        )
                        raise api_error

                await self.emit_status(
                    "info",
                    "Analyzing output ...",
                    False,
                )

                current_reflection = await self.analyze_output(
                    plan=plan,
                    action=action,
                    output=response,
                )

                await self.emit_status(
                    "info",
                    f"Analyzed output (Quality Score: {current_reflection.quality_score:.2f})",
                    False,
                )

                if current_reflection.quality_score >= best_quality_score:
                    best_output = current_output
                    best_reflection = current_reflection
                    best_quality_score = current_reflection.quality_score

                if current_reflection.is_successful:
                    break

                if attempts_remaining > 0:
                    attempts_remaining -= 1
                    await self.emit_status(
                        "warning",
                        f"Output needs improvement. Retrying... ({attempts_remaining + 1} attempts remaining) (Quality Score: {current_reflection.quality_score:.2f})",
                        False,
                    )
                    continue
                break

            except Exception as e:
                if attempts_remaining > 0:
                    attempts_remaining -= 1
                    await self.emit_status(
                        "warning",
                        f"Execution error, retrying... ({attempts_remaining + 1} attempts remaining)",
                        False,
                    )
                    continue
                else:
                    action.status = "failed"
                    action.end_time = datetime.now().strftime("%H:%M:%S")
                    await self.emit_status(
                        "error", f"Action failed after all attempts: {str(e)}", True
                    )
                    raise

        if best_output is None or best_reflection is None:
            action.status = "failed"
            action.end_time = datetime.now().strftime("%H:%M:%S")
            await self.emit_status(
                "error",
                "Action failed to produce any valid output after all attempts",
                True,
            )
            raise RuntimeError("No valid output produced after all attempts")

        if not best_reflection.is_successful:
            action.status = "warning"
            action.end_time = datetime.now().strftime("%H:%M:%S")
            action.output = best_output

        else:
            action.status = "completed"
            action.end_time = datetime.now().strftime("%H:%M:%S")
            action.output = best_output
        await self.emit_status(
            "success",
            f"Action completed with best output (Quality: {best_reflection.quality_score:.2f})",
            True,
        )
        return best_output

    async def analyze_output(
        self,
        plan: Plan,
        action: Action,
        output: str,
    ) -> ReflectionResult:
        """Simplified output analysis using an LLM to reflect on the action's result."""
        analysis_prompt = f"""
You are an expert evaluator for a generalist agent that can use a variety of tools, not just code. Analyze the output of an action based on the project goal, the action's description, and the tools used.

Overall Goal: {plan.goal}
Action Description: {action.description}
Tool(s) used: {action.tool_ids if action.tool_ids else "None"}
Action Output to Analyze:
---
{output}
---

Instructions:
Critically evaluate the output based on the following criteria:
1. **Completeness**: Does the output fully address the action's description and requirements?
2. **Correctness**: Is the information, tool usage, or code (if present) accurate and functional?
3. **Relevance**: Does the output directly contribute to the overall goal?
4. **Appropriateness**: Was the correct tool or method used for the task (not defaulting to code unless requested)?

Your response MUST be a single, valid JSON object with the following structure. Do not add any text before or after the JSON object.
{{
    "is_successful": <boolean>,
    "quality_score": <float, 0.0-1.0>,
    "issues": ["<A list of specific, concise issues found in the output>"],
    "suggestions": ["<A list of actionable suggestions to fix the issues>"]
}}

Scoring Guide:
- 0.9-1.0: Perfect, no issues.
- 0.7-0.89: Minor issues, but mostly correct and usable.
- 0.5-0.69: Significant issues that prevent the output from being used as-is.
- 0.0-0.49: Severely flawed, incorrect, or incomplete.

Be brutally honest. A high `quality_score` should only be given to high-quality outputs.
"""
        analysis_response = ""
        try:
            analysis_response = await self.get_completion(
                prompt=analysis_prompt,
                temperature=0.4,
                top_k=40,
                top_p=0.9,
            )

            clean_response = clean_json_response(analysis_response)
            analysis_data = json.loads(clean_response)

            return ReflectionResult(**analysis_data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(
                f"Failed to parse reflection analysis: {e}. Raw response: {analysis_response}"
            )

            return ReflectionResult(
                is_successful=False,
                quality_score=0.0,
                issues=[
                    "Failed to analyze the output due to a formatting error from the analysis model."
                ],
                suggestions=[
                    "The action should be retried, focusing on generating a simpler, clearer output."
                ],
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during output analysis: {e}. Raw response: {analysis_response}"
            )
            return ReflectionResult(
                is_successful=False,
                quality_score=0.0,
                issues=[f"An unexpected error occurred during analysis: {e}"],
                suggestions=["Retry the action."],
            )

    async def execute_plan(self, plan: Plan) -> None:
        """
        Execute the complete plan based on dependencies.
        Handles a special 'final_synthesis' action for templating.
        """
        completed_results: dict[str, dict[str, str]] = {}
        in_progress: set[str] = set()
        completed: set[str] = set()
        step_counter = 1
        all_outputs: list[dict[str, int | str]] = []

        async def can_execute(action: Action) -> bool:
            return all(dep in completed for dep in action.dependencies)

        while len(completed) < len(plan.actions):
            await self.emit_replace_mermaid(plan)

            available = [
                action
                for action in plan.actions
                if action.id not in completed
                and action.id not in in_progress
                and await can_execute(action)
            ]

            if not available:
                if not in_progress:
                    failed_actions = [a for a in plan.actions if a.status == "failed"]
                    if failed_actions or len(completed) < len(plan.actions):
                        logger.error(
                            "Execution stalled. Not all actions could be completed."
                        )
                    break
                await asyncio.sleep(0.1)
                continue

            synthesis_action = next(
                (a for a in available if a.id == "final_synthesis"), None
            )
            if synthesis_action:
                action = synthesis_action
                available.remove(action)
            elif available and len(in_progress) < self.valves.CONCURRENT_ACTIONS:
                action = available.pop(0)
            else:
                await asyncio.sleep(0.1)
                continue

            if action.id == "final_synthesis":
                await self.emit_status(
                    "info", "Assembling final output from template...", False
                )
                action.status = "in_progress"
                action.start_time = datetime.now().strftime("%H:%M:%S")
                await self.emit_replace_mermaid(plan)

                final_output_template = action.description

                placeholder_ids = re.findall(
                    r"\{([a-zA-Z0-9_]+)\}", final_output_template
                )

                final_output = final_output_template
                for action_id in placeholder_ids:
                    placeholder = f"{{{action_id}}}"
                    if action_id in completed_results:
                        dependency_output = completed_results[action_id].get(
                            "result", ""
                        )
                        final_output = final_output.replace(
                            placeholder, dependency_output
                        )
                    else:
                        logger.warning(
                            f"Could not find output for placeholder '{placeholder}'. It may have failed or was not executed. It will be left in the final output."
                        )

                action.output = {"result": final_output}
                action.status = "completed"
                action.end_time = datetime.now().strftime("%H:%M:%S")
                completed.add(action.id)
                completed_results[action.id] = action.output
                await self.emit_status("success", "Final output assembled.", True)
                await self.emit_replace_mermaid(plan)
                continue  #

            in_progress.add(action.id)

            try:
                context: dict[Any, Any] = {
                    dep: completed_results.get(dep, {}) for dep in action.dependencies
                }

                result = await self.execute_action(plan, action, context, step_counter)

                completed_results[action.id] = result
                completed.add(action.id)

                all_outputs.append(
                    {
                        "step": step_counter,
                        "id": action.id,
                        "output": result.get("result", ""),
                        "status": action.status,
                    }
                )
                step_counter += 1

            except Exception as e:
                step_counter += 1
                logger.error(f"Action {action.id} failed: {e}")
                action.status = "failed"
                completed.add(action.id)
            finally:
                if action.id in in_progress:
                    in_progress.remove(action.id)

            await self.emit_replace_mermaid(plan)

        plan.execution_summary = {
            "total_steps": len(plan.actions),
            "completed_steps": len(
                [a for a in plan.actions if a.status == "completed"]
            ),
            "failed_steps": len([a for a in plan.actions if a.status == "failed"]),
            "execution_time": {
                "start": plan.actions[0].start_time if plan.actions else None,
                "end": datetime.now().strftime("%H:%M:%S"),
            },
        }

        plan.metadata["execution_outputs"] = all_outputs
        return

    async def emit_replace_mermaid(self, plan: Plan):
        """Emit current state as Mermaid diagram, replacing the old one"""
        mermaid = await self.generate_mermaid(plan)
        await self.emit_replace(f"\n\n```mermaid\n{mermaid}\n```\n")

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_replace(self, message: str):
        await self.__current_event_emitter__(
            {"type": "replace", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
        )

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | User,
        __request__: Request,
        __event_emitter__: Callable[..., Awaitable[None]],
        __task__: TASKS | None = None,
        __model__: str | dict[str, Any] | None = None,
        user: dict[str, Any] | None = None,
    ) -> None | str:
        model = self.valves.MODEL
        self.__user__ =  Users.get_user_by_id(__user__["id"])
        self.__request__ = __request__
        self.user = __user__
        if __task__ and __task__ != TASKS.DEFAULT:
            response: dict[str, Any] = await generate_chat_completion(  # type: ignore
                self.__request__,
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        self.__current_event_emitter__ = __event_emitter__  # type: ignore
        self.__model__ = model

        goal = body.get("messages", [])[-1].get("content", "").strip()

        await self.emit_status("info", "Creating execution plan...", False)
        try:
            plan = await self.create_plan(goal)
        except Exception as e:
            await self.emit_status("error", f"Failed to create a valid plan: {e}", True)
            return

        await self.emit_replace_mermaid(plan)

        await self.emit_status("info", "Executing plan...", False)
        await self.execute_plan(plan)

        final_synthesis_action = next(
            (a for a in plan.actions if a.id == "final_synthesis"), None
        )
        if final_synthesis_action and final_synthesis_action.output:
            final_result = final_synthesis_action.output.get("result", "")
            await self.emit_status("success", "Final result ready.", True)
            await self.emit_replace("")
            await self.emit_replace_mermaid(plan)
            await self.emit_message(final_result)

        return
