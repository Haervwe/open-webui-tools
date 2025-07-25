"""
title: Planner
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 2.0.5
"""

import re
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


def parse_structured_output(response: str) -> dict[str, str]:
    """
    Parse agent output into structured format {"primary_output": str, "supporting_details": str}.
    If the response is not in the expected JSON format, treat the entire response as 'primary_output'.
    """
    try:
        clean_response = clean_json_response(response)
        parsed = json.loads(clean_response)

        if isinstance(parsed, dict) and "primary_output" in parsed:
            return {
                "primary_output": str(parsed.get("primary_output", "")),
                "supporting_details": str(parsed.get("supporting_details", "")),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {"primary_output": response, "supporting_details": ""}


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
    params: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    tool_ids: Optional[list[str]] = None
    output: Optional[Dict[str, str]] = None
    status: str = "pending"  # pending, in_progress, completed, failed, warning
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model: Optional[str] = None
    use_lightweight_context: bool = Field(
        default=False,
        description="If True, only action IDs and supporting_details are passed as context instead of full primary_output content to reduce context size",
    )
    tool_calls: List[str] = Field(
        default_factory=list,
        description="Names of tools that were actually called during execution",
    )
    tool_results: Dict[str, str] = Field(
        default_factory=dict, description="Results from tool calls, keyed by tool name"
    )


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
            description="Model to use for text/documentation actions (e.g., RP/Writer model). This model should focus ONLY on content generation and should NOT handle tool calls for post-processing like saving files. Create separate tool-based actions for any file operations.",
        )
        CODER_MODEL: str = Field(
            default="",
            description="Model to use for code/script generation actions (e.g., Coding specialized model). This model should focus ONLY on code generation and should NOT handle tool calls for post-processing like saving files. Create separate tool-based actions for any file operations.",
        )
        WRITER_SYSTEM_PROMPT: str = Field(
            default="""You are a Creative Writing Agent, specialized in generating high-quality narrative content, dialogue, and creative text. Your role is to focus on producing engaging, well-written content that matches the requested style and tone.

CREATIVE WRITING GUIDELINES:
1. Focus on creating compelling, well-structured narrative content
2. Maintain consistent character voices and narrative style
3. Use vivid descriptions and engaging dialogue when appropriate
4. Follow the specified genre, tone, and style requirements
5. Create content that flows naturally and maintains reader engagement
6. Pay attention to pacing, character development, and plot progression
7. Adapt your writing style to match the context (formal, casual, creative, etc.)
8. Never break character or mention that you are an AI
9. Produce complete, polished content ready for use

FIELD-SPECIFIC REQUIREMENTS:
- "primary_output": The complete written content (full articles, stories, chapters, documentation, etc.) ready for immediate use
- "supporting_details": Writing process notes, style considerations, or additional context about the content""",
            description="System prompt template for the Writer Model",
        )
        CODER_SYSTEM_PROMPT: str = Field(
            default="""You are a Coding Specialist Agent, expert in software development, scripting, and technical implementation. Your role is to generate clean, efficient, and well-documented code solutions.

CODING GUIDELINES:
1. Write clean, readable, and well-commented code
2. Follow best practices and conventions for the target language
3. Include proper error handling and validation where appropriate
4. Make code modular and reusable when possible
5. Provide complete, runnable code with no placeholders or TODOs
6. Include necessary imports, dependencies, and setup instructions
7. Add inline comments to explain complex logic
8. Consider security, performance, and maintainability
9. Test your code logic mentally before providing the solution
10. Structure code clearly with proper indentation and organization

FIELD-SPECIFIC REQUIREMENTS:
- "primary_output": The complete functional code (full scripts, functions, classes, etc.) ready to run
- "supporting_details": Code explanations, setup instructions, dependency notes, or implementation details""",
            description="System prompt template for the Coder Model",
        )
        ACTION_SYSTEM_PROMPT: str = Field(
            default="""You are the Action Agent, an expert at executing specific tasks within a larger plan. Your role is to focus solely on executing the current step, using ONLY the available tools and context provided.

CRITICAL GUIDELINES:
1. Focus EXCLUSIVELY on this step's task - do not try to solve the overall goal
2. Use ONLY the outputs from listed dependencies - do not reference other steps
3. When using tools:
   - Use EXACTLY as specified in the tool documentation
   - Process and format the tool output appropriately for this step
   - You can reference previous action outputs directly in tool parameters using the format: "@action_id" (e.g., "@search_results" to use the complete output from the search_results action)
   - When using "@action_id" references, the complete output will be automatically substituted - you don't need to copy/paste content manually
   - If the referenced output contains extra text that isn't needed for the tool call, you can either handle it manually by extracting what you need, or use additional tools to process it first
4. Produce a complete, self-contained output that can be used by dependent steps
5. Never ask for clarification - work with what is provided
6. Never output an empty message
7. Remember that tool outputs are only visible to you - include relevant results in your response
8. Always attach images in final responses as a markdown embedded images or other markdown embedable content with the ![caption](<image uri>) or [title](<hyperlink>)""",
            description="System prompt template for the Action Model",
        )
        ACTION_PROMPT_REQUIREMENTS_TEMPLATE: str = Field(
            default="""Requirements:
1. Focus EXCLUSIVELY on this specific action - do not attempt to solve the entire goal
2. Use ONLY the provided context and dependencies - do not reference other steps
3. Produce output that directly achieves this step's objective
4. Do not ask for clarifications; work with the information provided
5. Never output an empty response""",
            description="General requirements template applied to ALL actions",
        )
        WRITER_REQUIREMENTS_SUFFIX: str = Field(
            default="""
WRITER-SPECIFIC REQUIREMENTS:
- Focus ONLY on this specific action - do not attempt to complete the entire goal
- Create engaging, well-structured content that matches the requested style
- Use vivid descriptions and maintain character voices
- Incorporate sensory details to enhance immersion
- Include internal thoughts and feelings of characters
- Be very descriptive and creative
- Use metaphors and similes to create vivid imagery
- Focus on primary_output field and dont shy away on length
- Maintain consistent voice and tone throughout
- Focus on narrative flow and reader engagement
- Produce polished, publication-ready content for this action step only
- Do not break character or reference being an AI
- Your response must be a JSON object with "primary_output" and "supporting_details" fields
- The "primary_output" field must contain the COMPLETE written content, not just a title or description
- The "supporting_details" fieldwill never be shown to the user is meant for internal agents comunication.
- DO NOT add placeholder links 
- DO NOT use the "supporting_details" for more than 150 characters is meant to be just an epigraph to explain
- DO NOT attempt to save, write to files, or perform any tool operations - those are handled in separate actions""",
            description="Additional requirements specifically for Writer Model actions",
        )
        CODER_REQUIREMENTS_SUFFIX: str = Field(
            default="""
CODER-SPECIFIC REQUIREMENTS:
- Focus ONLY on this specific action - do not attempt to solve the entire goal
- Write clean, readable, and well-commented code for this action step only
- Include all necessary imports and dependencies
- Provide complete, runnable code with no placeholders or TODOs
- Follow best practices and conventions for the target language
- Include error handling where appropriate
- Add inline comments for complex logic
- Your response must be a JSON object with "primary_output" and "supporting_details" fields
- The "primary_output" field must contain the COMPLETE functional code, not just snippets or descriptions
- The "supporting_details" field will never be shown to the user is meant for internal agents comunication.
- DO NOT use the "supporting_details" for more than 150 characters is meant to be just an epigraph to explain
- DO NOT attempt to save, write to files, or perform any tool operations - those are handled in separate actions""",
            description="Additional requirements specifically for Coder Model actions",
        )
        ACTION_REQUIREMENTS_SUFFIX: str = Field(
            default="""
ACTION-SPECIFIC REQUIREMENTS:
- Use the specified tool(s) exactly as documented
- Process and synthesize tool outputs appropriately
- Provide clear, actionable results that can be used by dependent steps
- Include relevant details from tool outputs in your response
- Do not simply repeat tool outputs - synthesize them meaningfully
- You can use @action_id references in tool parameters to reference complete outputs from previous actions (e.g., "@search_results" to use the full output from the search_results action)
- When using @action_id references, the complete output will be automatically substituted - handle any extra text appropriately for your tool's needs
- Focus on executing the task with the available tools, not on formatting your response
- After tool execution, provide a natural, comprehensive response that incorporates the tool results""",
            description="Additional requirements specifically for Action Model (tool-using) actions",
        )
        LIGHTWEIGHT_CONTEXT_REQUIREMENTS_SUFFIX: str = Field(
            default="""
LIGHTWEIGHT CONTEXT REQUIREMENTS:
- You are working with lightweight context mode - only action IDs and supporting details are provided
- When referencing dependencies, use the action ID name as a parameter in your tool calls (e.g., use "research_results" as a file name or identifier)
- The actual content from previous steps is NOT included in context to save space
- Focus on using tools with the provided action IDs as identifiers/parameters
- Work with action IDs as if they were file names, content identifiers, or operation references
- Trust that the action IDs represent valid content that exists in the system
- Use @action_id references in tool parameters when you need to reference previous action outputs
- The supporting_details field contains hints about what each action ID represents
- Do not ask for the full content - work with the IDs as provided
- Your tools should be able to operate using these IDs as references""",
            description="Additional requirements for actions using lightweight context mode",
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

    def get_system_prompt_for_model(
        self,
        action: Action,
        step_number: int,
        context: dict[str, Any],
        requirements: str,
        model: str,
    ) -> str:
        """Generate model-specific system prompts based on the model type."""
        enhanced_requirements = requirements
        
        # Handle lightweight context mode
        if action.use_lightweight_context:
            enhanced_requirements += self.valves.LIGHTWEIGHT_CONTEXT_REQUIREMENTS_SUFFIX
        else:
            match model:
                case self.valves.WRITER_MODEL:
                    enhanced_requirements += self.valves.WRITER_REQUIREMENTS_SUFFIX
                case self.valves.CODER_MODEL:
                    enhanced_requirements += self.valves.CODER_REQUIREMENTS_SUFFIX
                case _:  # ACTION_MODEL (default)
                    enhanced_requirements += self.valves.ACTION_REQUIREMENTS_SUFFIX

        # Format context based on lightweight mode
        if action.use_lightweight_context:
            # Only include action IDs and supporting details
            lightweight_context = {}
            for dep_id, dep_data in context.items():
                if isinstance(dep_data, dict):
                    lightweight_context[dep_id] = {
                        "action_id": dep_id,
                        "supporting_details": dep_data.get("supporting_details", "")
                    }
                else:
                    lightweight_context[dep_id] = {
                        "action_id": dep_id,
                        "supporting_details": ""
                    }
            
            base_context = f"""
    TASK CONTEXT:
    - Step {step_number} Description: {action.description}
    - Available Tools: {action.tool_ids if action.tool_ids else "None"}
    - Context Mode: LIGHTWEIGHT (only action IDs and hints provided)
    
    DEPENDENCIES AND INPUTS:
    - Parameters: {json.dumps(action.params)}
    - Lightweight Context (IDs and hints only): {json.dumps(lightweight_context)}
    
    NOTE: You are working in lightweight context mode. Previous step results contain only action IDs and supporting_details hints.
    Use the action IDs as identifiers/parameters in your tool calls. The actual content is not provided to save context space.

    EXECUTION REQUIREMENTS:
    {enhanced_requirements}
"""
        else:
            # Full context mode (existing behavior)
            base_context = f"""
    TASK CONTEXT:
    - Step {step_number} Description: {action.description}
    - Available Tools: {action.tool_ids if action.tool_ids else "None"}
    
    DEPENDENCIES AND INPUTS:
    - Parameters: {json.dumps(action.params)}
    - Input from Previous Steps: {json.dumps(context)}
    
    NOTE: Previous step results are structured as {{"primary_output": "main_deliverable_content", "supporting_details": "additional_context"}}. 
    You have access to both fields for context, but focus on using the "primary_output" field which contains the actual deliverable content from previous steps.

    EXECUTION REQUIREMENTS:
    {enhanced_requirements}
"""

        match model:
            case m if m == self.valves.WRITER_MODEL:
                return f"SYSTEM: {self.valves.WRITER_SYSTEM_PROMPT}\n{base_context}"

            case m if m == self.valves.CODER_MODEL:
                return f"SYSTEM: {self.valves.CODER_SYSTEM_PROMPT}\n{base_context}"

            case _:  # ACTION_MODEL (default)
                return f"SYSTEM: {self.valves.ACTION_SYSTEM_PROMPT}\n{base_context}"

    async def get_completion(
        self,
        prompt: str | list[dict[str, Any]],
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        model: str | dict[str, Any] = "",
        tools: dict[str, dict[Any, Any]] = {},
        format: dict[str, Any] | None = None,
        action_results: dict[str, dict[str, str]] = {},
        action: Optional[Action] = None,
    ) -> str:
        system_content = "You are a Helpful agent that does exactly as told and dont ask clarifications"
        if format is not None:
            system_content += ". When responding with structured data, ensure your response is valid JSON format without any additional text, markdown formatting, or explanations."

        messages = (
            [
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ]
            if isinstance(prompt, str)
            else prompt
        )

        if model in [self.valves.WRITER_MODEL, self.valves.CODER_MODEL] and tools:
            __model = (
                self.valves.ACTION_MODEL
                if self.valves.ACTION_MODEL
                else self.valves.MODEL
            )
        else:
            __model = model if model else self.valves.ACTION_MODEL
        _tools = (
            [
                {"type": "function", "function": tool.get("spec", {})}
                for tool in tools.values()
            ]
            if tools
            else None
        )

        try:
            form_data: dict[str, Any] = {
                "model": __model,
                "messages": messages,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "tools": _tools,
            }
            logger.debug(f"{_tools}")
            if format and not tools:
                form_data["response_format"] = format
            response: dict[str, Any] = await generate_chat_completion(
                self.__request__,
                form_data,
                user=self.__user__,
            )
            response_content = str(response["choices"][0]["message"]["content"])
            tool_calls: list[dict[str, Any]] | None = None
            logger.debug(f"{tool_calls}")
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

                if action and tool_function_name:
                    if tool_function_name not in action.tool_calls:
                        action.tool_calls.append(tool_function_name)

                if tool_function_name not in tools:
                    tool_result = f"{tool_function_name} not in {tools}"
                    if action:
                        action.tool_results[tool_function_name] = (
                            f"ERROR: {tool_result}"
                        )
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

                    def resolve_action_references(
                        params: dict[str, Any | dict[str, Any]] | list[Any],
                    ) -> dict[str, Any]:
                        """Recursively resolve @action_id references in tool parameters"""
                        resolved_params: dict[str, Any] = {}
                        for key, value in params.items():
                            if isinstance(value, str) and value.startswith("@"):
                                action_id = value[1:]
                                if action_id in action_results:
                                    resolved_params[key] = action_results[
                                        action_id
                                    ].get("primary_output", "")
                                    logger.info(
                                        f"Resolved @{action_id} reference in parameter '{key}'"
                                    )
                                else:
                                    resolved_params[key] = value
                                    logger.warning(
                                        f"Action ID '{action_id}' not found for reference in parameter '{key}'"
                                    )
                            elif isinstance(value, dict):
                                resolved_params[key] = resolve_action_references(value)
                            elif isinstance(value, list):
                                resolved_list: list[str] = []
                                for item in value:
                                    if isinstance(item, str) and item.startswith("@"):
                                        action_id = item[1:]
                                        if action_id in action_results:
                                            resolved_list.append(
                                                action_results[action_id].get(
                                                    "result", ""
                                                )
                                            )
                                            logger.info(
                                                f"Resolved @{action_id} reference in list parameter '{key}'"
                                            )
                                        else:
                                            resolved_list.append(item)
                                            logger.warning(
                                                f"Action ID '{action_id}' not found for reference in list parameter '{key}'"
                                            )
                                    elif isinstance(item, dict):
                                        resolved_list.append(
                                            str(resolve_action_references(item))
                                        )
                                    else:
                                        resolved_list.append(item)
                                resolved_params[key] = resolved_list
                            else:
                                resolved_params[key] = value
                        return resolved_params

                    tool_function_params = resolve_action_references(
                        tool_function_params
                    )

                    tool_function = tool["callable"]
                    logger.debug(f"{tool_call} , {tool_function_params}")
                    tool_result = await tool_function(**tool_function_params)

                    # Store successful tool result
                    if action:
                        action.tool_results[tool_function_name] = str(tool_result)

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
                            "Do not simply repeat the tool's output. Synthesize it into a complete response that accomplishes the step's objective. "
                            "Focus on the actual results and deliverables from the tool execution."
                        ),
                    },
                ]

            if model in [self.valves.WRITER_MODEL, self.valves.CODER_MODEL]:
                specialist_response = await self.get_completion(
                    prompt=messages,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    model=model,
                    action_results=action_results,
                    format=format,
                )
                return specialist_response
            else:
                messages[-1][
                    "content"
                ] += """                       
                            OUTPUT FORMAT REQUIREMENT:
                            Your response MUST be formatted as a JSON object with this exact structure:
                            {
                                "primary_output": "The main deliverable content that directly addresses this step's objective and will be used in the final output and by dependent steps. For image generation tasks, this should be the actual image URL or file path. For text content, this should be the actual written content. For code tasks, this should be the complete functional code.",
                                "supporting_details": "Additional context, process information, technical details, or supplementary information that may help subsequent steps understand how this output was created, but should not appear in the final result."
                            }
                            
                            CRITICAL: The "primary_output" field must contain the ACTUAL deliverable (URLs for images, complete text for writing tasks, functional code for coding tasks, etc.), not just descriptions or titles. This content will be directly used by other steps and in the final synthesis.
                            OUTPUT STRUCTURE:
                            - "primary_output": The main deliverable content that will be used in the final output and by dependent steps (actual URLs for images, complete text for writing, functional code for coding, etc.)
                            - "supporting_details": Additional context, process information, or details useful for subsequent steps but not for final output"""
                tool_response = await self.get_completion(
                    prompt=messages,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    model=model,
                    tools=tools,
                    action_results=action_results,
                    action=action,
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
You are a planning agent. Break down the goal into a logical sequence of actions that build upon each other.

OUTPUT FORMAT CONSIDERATIONS:
- ALL action outputs are TEXT or HYPERLINKS (URLs/URIs)
- Image generation tools return URLs/file paths as text, NOT actual image files
- Web content, files, and media are represented as hyperlinks/URLs in text format
- Every action produces text-based output that can be directly included in the final synthesis
- Do NOT assume any special formatting - treat all outputs as plain text or clickable links
- Action IDs are indepedent of tools and tool calls, Asign an action_id that correspond with the step specific Goal

FINAL SYNTHESIS REQUIREMENT:
- The final_synthesis action MUST include ALL RELEVANT content outputs in its template
- Include actions that produce deliverable content: text, images, code, research results, etc.
- Do NOT miss important outputs like images, documents, or other key deliverables
- Exclude only auxiliary/setup actions that don't produce end-user content (like configuration or intermediate processing steps)
- When in doubt, include the action output rather than exclude it

AVAILABLE TOOLS (use exact tool_id):
{json.dumps(tools, indent=2)}

CRITICAL UNDERSTANDING - DEPENDENCIES:
When action B depends on action A, action B will receive A's complete output as context. This means:
- "Write Chapter 2" depending on "Write Chapter 1" gets the full Chapter 1 text as context
- "Generate Illustration for Chapter 1" depending on "Write Chapter 1" gets the chapter content to create relevant imagery
- "Research AI trends" → "Write summary based on research" → "Create presentation from summary" forms a logical chain

**EXPLICIT DEPENDENCY RULE**: If an action needs content from previous actions, it MUST explicitly list ALL required actions in its dependencies array. Transitive dependencies are NOT automatically included.

TOOL TYPES EXPLAINED:
- Search tools: For web research, finding information
- Image generation tools: For creating visual content  
- File/save tools: For saving content to files or specific formats, break saving in to multiple intermediate steps instead of one aggreated one.
- API tools: For specific integrations or data processing
- Always use the exact "tool_id" from the available tools list when an action needs external capabilities in the correct tool_ids field

ACTION TYPES:
- type="tool": Uses external tools, MUST specify tool_ids
- type="text": Pure content creation (writing, documentation)
- type="code": Code generation
- type="synthesis": Final template action only

MODEL ASSIGNMENT:
- Tool actions: 'ACTION_MODEL'
- Text/writing actions: 'WRITER_MODEL'
- Code actions: 'CODER_MODEL'

LIGHTWEIGHT CONTEXT MODE:
- use_lightweight_context: Set to true for actions that work with large file operations, bulk processing, or when dependencies might produce very large content that would overflow context
- When true, the action receives only action IDs and supporting_details from dependencies instead of full primary_output content
- Use this for actions like file operations, data processing, bulk operations where the tool can work with identifiers/names rather than full content
- Actions in lightweight mode should use @action_id references in tool parameters to reference previous results
- Best for: file saving, data compilation, operations on multiple large documents, complex transformations where content size might be prohibitive
- Default: false (full context mode)

DEPENDENCY EXAMPLES - EXPLICIT LINKING REQUIRED:
❌ WRONG - No context flow:
research_ai → write_ch1, write_ch2, write_ch3 (chapters get no context from each other)

❌ WRONG - Implicit/transitive dependencies:
research_ai → write_ch1 → write_ch2 → write_ch3 
(ch3 only gets ch2, missing ch1 and research_ai context)

❌ WRONG - Assuming transitive dependencies work:
chapter_1 → chapter_2 → chapter_3 
(chapter_3 WON'T automatically get chapter_1 content, only chapter_2)

✅ CORRECT - Explicit dependencies for all needed context:
research_ai → write_ch1 
write_ch2 depends on [research_ai, write_ch1]
write_ch3 depends on [research_ai, write_ch1, write_ch2] 
(each chapter EXPLICITLY lists ALL previous actions it needs)

✅ CORRECT - Story development with explicit dependencies:
research_ai → story_outline → character_sheet → write_ch1
write_ch2 depends on [story_outline, character_sheet, write_ch1]
write_ch3 depends on [story_outline, character_sheet, write_ch1, write_ch2]

✅ CORRECT - Book compilation example:
research → outline → ch1 → ch2 → ch3 → compile_book
compile_book depends on [outline, ch1, ch2, ch3] (explicitly lists all chapters needed)

REMEMBER: Dependencies are NOT transitive. If action C needs content from action A and action B, 
it must explicitly list BOTH A and B in its dependencies, even if B already depends on A.

FINAL SYNTHESIS - COMPREHENSIVE TEMPLATING GUIDE:
The final_synthesis action is a SPECIAL TEMPLATING ACTION that assembles the final output by combining results from previous actions.

CRITICAL TEMPLATING RULES:
1. **Placeholder Format**: Use {{action_id}} to reference any action's primary_output
   - Example: {{research_results}} will be replaced with the full primary_output from the "research_results" action
   - Example: {{chapter_1}} will be replaced with the complete text content from the "chapter_1" action
   - Example: {{generated_image}} will be replaced with the image URL/path from the "generated_image" action

2. **Template Structure**: The description field contains the FINAL OUTPUT TEMPLATE with placeholders
   - Use Markdown formatting for proper presentation
   - Include all relevant content using placeholders
   - Structure the template as you want the final output to appear

3. **Content Replacement**: During execution, each {{action_id}} placeholder is replaced with:
   - The complete "primary_output" field from that action's result
   - This is literal text substitution - what you see is what you get
   - Images come already in embedded markdown: ![caption](URL)
   - Code becomes code blocks if the original action formatted it that way
   - Text content is inserted as-is

4. **Dependencies**: MUST list ALL actions whose outputs you reference in the template
   - If template uses {{research}} and {{summary}}, dependencies must include ["research", "summary"]
   - Missing dependencies will result in placeholder not being replaced

TEMPLATING EXAMPLES:

Example 1 - Simple Report:
```
{{
  "id": "final_synthesis",
  "type": "synthesis",
  "description": "# Research Report\n\n## Background\n{{background_research}}\n\n## Analysis\n{{data_analysis}}\n\n## Conclusion\n{{conclusions}}",
  "dependencies": ["background_research", "data_analysis", "conclusions"]
}}
```

Example 2 - Blog Post with Images:
```
{{
  "id": "final_synthesis", 
  "type": "synthesis",
  "description": "# {{blog_title}}\n\n{{blog_intro}}\n\n![Featured Image]({{hero_image}})\n\n## Main Content\n{{main_content}}\n\n![Supporting Image]({{supporting_image}})\n\n## Conclusion\n{{conclusion}}",
  "dependencies": ["blog_title", "blog_intro", "hero_image", "main_content", "supporting_image", "conclusion"]
}}
```

Example 3 - Code Documentation:
```
{{
  "id": "final_synthesis",
  "type": "synthesis", 
  "description": "# {{project_name}} Documentation\n\n## Overview\n{{overview}}\n\n## Installation\n```bash\n{{installation_commands}}\n```\n\n## Code\n```python\n{{main_code}}\n```\n\n## Usage Examples\n{{usage_examples}}",
  "dependencies": ["project_name", "overview", "installation_commands", "main_code", "usage_examples"]
}}
```

Example 4 - Multi-Chapter Story:
```
{{
  "id": "final_synthesis",
  "type": "synthesis",
  "description": "# {{story_title}}\n\n{{story_intro}}\n\n## Chapter 1\n{{chapter_1}}\n\n## Chapter 2\n{{chapter_2}}\n\n## Chapter 3\n{{chapter_3}}\n\n## Epilogue\n{{epilogue}}",
  "dependencies": ["story_title", "story_intro", "chapter_1", "chapter_2", "chapter_3", "epilogue"]
}}
```

TEMPLATE FORMATTING TIPS:
- Use `\n\n` for paragraph breaks in the description string
- Use `\n` for single line breaks
- Include proper Markdown headers (#, ##, ###) for structure  
- For code blocks, use: `\n```language\n{{code_action}}\n```\n`
- For images, use: `![Description]({{image_action}})`
- For links, use: `[Link Text]({{url_action}})`

WHAT GETS REPLACED:
- {{action_id}} → Complete primary_output content from that action
- If action_id produced an image URL: {{action_id}} → "https://example.com/image.jpg"
- If action_id produced text content: {{action_id}} → "The complete text content here..."
- If action_id produced code: {{action_id}} → "def function():\n    return 'code'"

FINAL SYNTHESIS REQUIREMENTS:
- Must be the LAST action in the plan
- Must have id="final_synthesis" 
- Must have type="synthesis"
- Must include ALL content-producing actions in dependencies
- Template must reference all important deliverables using {{action_id}} placeholders
- Model field should be empty "" (no LLM needed for templating)

REMEMBER: The final_synthesis action is a PURE TEMPLATING step - it takes the "primary_output" from each referenced action and substitutes it into the template. No AI generation happens here, just text replacement. Design your template to be the exact final output you want users to see.

JSON OUTPUT:
{{
    "goal": "<original goal>",
    "actions": [
        {{
            "id": "unique_id",
            "type": "tool|text|code",
            "description": "Specific task description",
            "tool_ids": ["exact_tool_id_from_list"], 
            "params": {{}},
            "dependencies": ["action_ids_this_needs"],
            "model": "model_name",
            "use_lightweight_context": false
        }},
        {{
            "id": "final_synthesis",
            "type": "synthesis",
            "description": "# Title\\n\\n{{action1}}\\n\\n{{action2}}",
            "dependencies": ["action1", "action2"],
            "model": "",
            "use_lightweight_context": false
        }}
    ]
}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]
        for attempt in range(self.valves.MAX_RETRIES):
            try:
                plan_format: dict[str, Any] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "execution_plan",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string"},
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "type": {"type": "string"},
                                            "description": {"type": "string"},
                                            "tool_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "params": {},
                                            "dependencies": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "model": {"type": "string"},
                                            "use_lightweight_context": {
                                                "type": "boolean",
                                                "default": False
                                            },
                                        },
                                        "required": [
                                            "id",
                                            "type",
                                            "description",
                                            "params",
                                            "dependencies",
                                            "model",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["goal", "actions"],
                            "additionalProperties": False,
                        },
                    },
                }

                result = await self.get_completion(
                    prompt=messages,
                    temperature=0.8,
                    top_k=60,
                    top_p=0.95,
                    format=plan_format,
                    action_results={},
                    action=None,
                )
                clean_result = clean_json_response(result)
                plan_dict = json.loads(clean_result)

                actions = plan_dict.get("actions", [])

                model_mapping = {
                    "ACTION_MODEL": self.valves.ACTION_MODEL,
                    "WRITER_MODEL": self.valves.WRITER_MODEL,
                    "CODER_MODEL": self.valves.CODER_MODEL,
                }

                for action in actions:

                    if action.get("model") in model_mapping:
                        action["model"] = model_mapping[action["model"]]
                    elif "model" not in action or not action["model"]:

                        if action.get("type") in ["text", "documentation", "synthesis"]:
                            action["model"] = self.valves.WRITER_MODEL
                        elif action.get("type") in ["code", "script"]:
                            action["model"] = self.valves.CODER_MODEL
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

                final_synthesis_index = next(
                    (
                        i
                        for i, a in enumerate(plan.actions)
                        if a.id == "final_synthesis"
                    ),
                    None,
                )
                if final_synthesis_index is not None:
                    if final_synthesis_index != len(plan.actions) - 1:
                        msg = (
                            f"The 'final_synthesis' action must be the last action in the plan. "
                            f"Currently it's at position {final_synthesis_index + 1} out of {len(plan.actions)} actions."
                        )
                        messages += [
                            {
                                "role": "assistant",
                                "content": f"previous attempt: {clean_result}",
                            },
                            {"role": "user", "content": f"error:: {msg}"},
                        ]
                        raise ValueError(msg)

                    actions_depending_on_final = [
                        a.id
                        for a in plan.actions
                        if "final_synthesis" in a.dependencies
                    ]
                    if actions_depending_on_final:
                        msg = (
                            f"No actions can depend on 'final_synthesis'. "
                            f"Found actions depending on it: {actions_depending_on_final}. "
                            f"The 'final_synthesis' action must be the absolute final step."
                        )
                        messages += [
                            {
                                "role": "assistant",
                                "content": f"previous attempt: {clean_result}",
                            },
                            {"role": "user", "content": f"error:: {msg}"},
                        ]
                        raise ValueError(msg)
                logger.debug(f"Plan: {plan}")
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
            action_results={},
            action=None,
        )
        return enhanced_requirements

    async def execute_action(
        self, plan: Plan, action: Action, context: dict[str, Any], step_number: int
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

        # Gather additional context for the base prompt based on lightweight context setting
        if action.use_lightweight_context:
            # For lightweight context, only use direct dependencies with IDs and supporting details
            context_for_prompt = {}
            for dep in action.dependencies:
                if dep in context:
                    dep_result = context.get(dep, {})
                    context_for_prompt[dep] = {
                        "action_id": dep,
                        "supporting_details": dep_result.get("supporting_details", "")
                    }
                else:
                    context_for_prompt[dep] = {
                        "action_id": dep,
                        "supporting_details": ""
                    }
        else:
            # Full context mode - use the complete context as provided
            context_for_prompt = context
        
        requirements = (
            await self.enhance_requirements(plan, action)
            if self.valves.AUTOMATIC_TAKS_REQUIREMENT_ENHANCEMENT
            else self.valves.ACTION_PROMPT_REQUIREMENTS_TEMPLATE
        )
        
        if action.use_lightweight_context:
            base_prompt = f"""
            Execute step {step_number}: {action.description}
            Overall Goal: {plan.goal}
        
            Context from dependent steps (LIGHTWEIGHT MODE - IDs and hints only):
            - Parameters: {json.dumps(action.params)}
            - Action IDs and hints: {json.dumps(context_for_prompt)}
        
            {requirements}
            
            Focus ONLY on this specific step's output.
            Use the action IDs as identifiers/parameters in your tool calls when referencing previous results.
            """
        else:
            base_prompt = f"""
            Execute step {step_number}: {action.description}
            Overall Goal: {plan.goal}
        
            Context from dependent steps:
            - Parameters: {json.dumps(action.params)}
            - Previous Results: {json.dumps(context_for_prompt)}
        
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

                if current_attempt > 0:
                    action.tool_calls.clear()
                    action.tool_results.clear()

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
                    retry_guidance = ""
                    if action.tool_ids and not action.tool_calls:
                        retry_guidance += f"""
                        
                        IMPORTANT: You have access to these tools: {action.tool_ids}
                        Your previous attempt did not use any tools, which may be why it failed.
                        Consider using the appropriate tools to complete this task effectively.
                        """
                    elif action.tool_ids and action.tool_calls:
                        retry_guidance += f"""
                        
                        Your previous attempt used tools: {action.tool_calls}
                        But the output was still inadequate. Try different approaches or parameters.
                        """

                    base_prompt += f"""
                        
                        Previous attempt had these issues:
                        {json.dumps(best_reflection.issues, indent=2)}
                        
                        Required corrections based on suggestions:
                        {json.dumps(best_reflection.suggestions, indent=2)}
                        
                        {retry_guidance}
                        
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

                    execution_model = (
                        action.model
                        if action.model
                        else (
                            self.valves.ACTION_MODEL
                            if (self.valves.ACTION_MODEL != "")
                            else self.valves.MODEL
                        )
                    )

                    system_prompt = self.get_system_prompt_for_model(
                        action, step_number, context, requirements, execution_model
                    )

                    action_format: dict[str, Any] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "action_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "primary_output": {"type": "string"},
                                    "supporting_details": {"type": "string"},
                                },
                                "required": ["primary_output", "supporting_details"],
                                "additionalProperties": False,
                            },
                        },
                    }

                    response = await self.get_completion(
                        prompt=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": base_prompt},
                        ],
                        temperature=0.9,
                        top_k=70,
                        top_p=0.95,
                        model=execution_model,
                        tools=tools,
                        format=action_format,
                        action_results=context,
                        action=action,
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

                    structured_output = parse_structured_output(response)
                    current_output = structured_output

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

        # Prepare tool call verification information
        expected_tools = action.tool_ids if action.tool_ids else []
        actual_tool_calls = action.tool_calls
        tool_results_summary = {
            tool: result[:200] + "..." if len(result) > 200 else result
            for tool, result in action.tool_results.items()
        }

        analysis_prompt = f"""
You are an expert evaluator for a generalist agent that can use a variety of tools, not just code. Analyze the output of an action based on the project goal, the action's description, and the tools used.

Overall Goal: {plan.goal}
Action Description: {action.description}
Expected Tool(s): {expected_tools}
Actually Called Tool(s): {actual_tool_calls}
Tool Results Summary: {json.dumps(tool_results_summary, indent=2)}

Action Output to Analyze:
---
{output}
---

CRITICAL TOOL VERIFICATION:
- If the action was expected to use tools ({expected_tools}) but no tools were called ({actual_tool_calls}), this is a MAJOR failure
- If tools were called, verify that the output actually incorporates their results meaningfully
- If the output claims tools were used but no actual tool calls occurred, this is FALSE and should be heavily penalized
- Tool results should be properly processed and integrated into the final output

Instructions:
Critically evaluate the output based on the following criteria:
1. **Tool Usage Verification**: STRICTLY verify that claimed tool usage matches actual tool calls. False claims about tool usage should result in quality_score <= 0.3
2. **Output Format**: The output should be a valid JSON object with "primary_output" and "supporting_details" fields
3. **Completeness**: Does the output fully address the action's description and requirements?
4. **Correctness**: Is the information, tool usage, or code (if present) accurate and functional?
5. **Relevance**: Does the output directly contribute to the overall goal?
6. **Tool Integration**: If tools were used, are their results properly integrated and processed in the output?
7. **Content Quality**: Is the primary_output field clean, complete, and ready for use by subsequent steps?
8. **Markdown integration**: Markdown format for deliverables is preferable using the embeding formats for example ![caption](<image uri>) to show image or embedable content.
9. **Missing Tool Calls**: if tool calls werent done Ask the model to call them but do not mention Format at this step.
Your response MUST be a single, valid JSON object with the following structure. Do not add any text before or after the JSON object.
{{
    "is_successful": <boolean>,
    "quality_score": <float, 0.0-1.0>,
    "issues": ["<A list of specific, concise issues found in the output>"],
    "suggestions": ["<A list of actionable suggestions to fix the issues>"]
}}

Scoring Guide:
- 0.9-1.0: Perfect, properly structured, tools used correctly, no issues
- 0.7-0.89: Minor issues, but mostly correct and usable
- 0.5-0.69: Significant issues that prevent the output from being used as-is
- 0.3-0.49: Major problems, incorrect tool usage claims, or incomplete execution
- 0.0-0.29: Severely flawed, false tool claims, or completely incorrect

Be brutally honest. A high `quality_score` should only be given to high-quality outputs that properly use tools when expected and follow the correct format.
"""
        analysis_response = ""
        try:
            reflection_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "reflection_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_successful": {"type": "boolean"},
                            "quality_score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "issues": {"type": "array", "items": {"type": "string"}},
                            "suggestions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "is_successful",
                            "quality_score",
                            "issues",
                            "suggestions",
                        ],
                        "additionalProperties": False,
                    },
                },
            }

            analysis_response = await self.get_completion(
                prompt=analysis_prompt,
                temperature=0.4,
                top_k=40,
                top_p=0.9,
                format=reflection_format,
                action_results={},
                action=None,
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
                            "primary_output", ""
                        )
                        final_output = final_output.replace(
                            placeholder, dependency_output
                        )
                    else:
                        logger.warning(
                            f"Could not find output for placeholder '{placeholder}'. It may have failed or was not executed. It will be left in the final output."
                        )

                action.output = {
                    "primary_output": final_output,
                    "supporting_details": "Final synthesis completed",
                }
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
                        "output": result.get("primary_output", ""),
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
        self.__user__ = Users.get_user_by_id(__user__["id"])
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
            final_result = final_synthesis_action.output.get("primary_output", "")
            await self.emit_status("success", "Final result ready.", True)
            await self.emit_replace("")
            await self.emit_replace_mermaid(plan)
            await self.emit_message(final_result)

        return
