"""
title: Planner
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 2.0.6
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


name = "Planner"


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

CRITICAL OUTPUT STRUCTURE - MANDATORY:
Your response MUST be a JSON object with exactly these fields:
{
    "primary_output": "THE COMPLETE WRITTEN CONTENT GOES HERE",
    "supporting_details": "Brief process notes or context (max 150 chars)"
}

FIELD USAGE RULES - AUTOMATIC FAILURE IF VIOLATED:
- "primary_output": MUST contain the COMPLETE written content (full articles, stories, chapters, documentation, etc.) ready for immediate use by users or subsequent steps
- "supporting_details": MUST only contain brief explanatory notes, writing process context, or metadata - NEVER the main content
- WRONG: putting main content in supporting_details while primary_output has just a title/summary
- WRONG: putting "See supporting details" in primary_output
- WRONG: primary_output contains only brief descriptions while actual content is elsewhere

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
10. ALWAYS put the complete written work in "primary_output" - this is what users will see""",
            description="System prompt template for the Writer Model",
        )
        CODER_SYSTEM_PROMPT: str = Field(
            default="""You are a Coding Specialist Agent, expert in software development, scripting, and technical implementation. Your role is to generate clean, efficient, and well-documented code solutions.

CRITICAL OUTPUT STRUCTURE - MANDATORY:
Your response MUST be a JSON object with exactly these fields:
{
    "primary_output": "THE COMPLETE FUNCTIONAL CODE GOES HERE",
    "supporting_details": "Brief setup notes or context (max 150 chars)"
}

FIELD USAGE RULES - AUTOMATIC FAILURE IF VIOLATED:
- "primary_output": MUST contain the COMPLETE functional code (full scripts, functions, classes, etc.) ready to run immediately
- "supporting_details": MUST only contain brief setup instructions, dependency notes, or implementation context - NEVER the main code
- WRONG: putting main code in supporting_details while primary_output has just a description/title
- WRONG: putting "See supporting details" in primary_output
- WRONG: primary_output contains only code snippets while full implementation is elsewhere

CODING GUIDELINES:
1. Write clean, readable, and well-commented code
2. Follow best practices and conventions for the target language
3. Include proper error handling and validation where appropriate
4. Make code modular and reusable when possible
5. Provide complete, runnable code with no placeholders or TODOs
6. Include necessary imports, dependencies, and setup instructions within the code
7. Add inline comments to explain complex logic
8. Consider security, performance, and maintainability
9. Test your code logic mentally before providing the solution
10. Structure code clearly with proper indentation and organization
11. ALWAYS put the complete working code in "primary_output" - this is what will be used""",
            description="System prompt template for the Coder Model",
        )
        ACTION_SYSTEM_PROMPT: str = Field(
            default="""You are the Action Agent, an expert at executing specific tasks within a larger plan. Your role is to focus solely on executing the current step, using ONLY the available tools and context provided.

CRITICAL OUTPUT STRUCTURE - MANDATORY:
Your response MUST be a JSON object with exactly these fields:
{
    "primary_output": "THE MAIN DELIVERABLE/RESULT GOES HERE",
    "supporting_details": "Brief context or metadata (max 150 chars)"
}

FIELD USAGE RULES - AUTOMATIC FAILURE IF VIOLATED:
- "primary_output": MUST contain the MAIN DELIVERABLE content that users need or subsequent steps will use
- "supporting_details": MUST only contain brief explanatory notes, source info, or metadata - NEVER the main deliverable
- WRONG: putting the main result/content in supporting_details while primary_output has just a title/summary
- WRONG: putting "See supporting details" in primary_output
- WRONG: primary_output contains only brief descriptions while actual deliverable content is elsewhere

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
8. Always attach images in final responses as a markdown embedded images or other markdown embedable content with the ![caption](<image uri>) or [title](<hyperlink>)
9. ALWAYS put the main deliverable/result in "primary_output" - this is what users and subsequent steps will see and use""",
            description="System prompt template for the Action Model",
        )
        ACTION_PROMPT_REQUIREMENTS_TEMPLATE: str = Field(
            default="""Requirements:
1. Focus EXCLUSIVELY on this specific action - do not attempt to solve the entire goal
2. Use ONLY the provided context and dependencies - do not reference other steps
3. Produce output that directly achieves this step's objective
4. Do not ask for clarifications; work with the information provided
5. Never output an empty response
6. CRITICAL: Your response MUST be a JSON object with "primary_output" and "supporting_details" fields
7. CRITICAL: Put the MAIN DELIVERABLE/RESULT in "primary_output" - this is what users and subsequent steps will see
8. CRITICAL: Use "supporting_details" ONLY for brief metadata/context (max 150 chars) - NEVER for main content
9. FAILURE to follow field usage rules will result in automatic action failure""",
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
- Maintain consistent voice and tone throughout
- Focus on narrative flow and reader engagement
- Produce polished, publication-ready content for this action step only
- Do not break character or reference being an AI
- Your response must be a JSON object with "primary_output" and "supporting_details" fields

CRITICAL FIELD REQUIREMENTS - AUTOMATIC FAILURE IF VIOLATED:
- The "primary_output" field MUST contain the COMPLETE written content, not just a title or description
- The "supporting_details" field is for internal communication only (max 150 chars) - brief writing notes
- NEVER put the main written content in "supporting_details"
- NEVER put "See supporting details" in "primary_output"
- WRONG EXAMPLE: {"primary_output": "Story Title", "supporting_details": "Once upon a time..."}
- CORRECT EXAMPLE: {"primary_output": "Once upon a time...", "supporting_details": "Fantasy genre, 1200 words"}

- DO NOT add placeholder links 
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

CRITICAL FIELD REQUIREMENTS - AUTOMATIC FAILURE IF VIOLATED:
- The "primary_output" field MUST contain the COMPLETE functional code, not just snippets or descriptions
- The "supporting_details" field is for internal communication only (max 150 chars) - brief setup notes
- NEVER put the main code in "supporting_details"
- NEVER put "See supporting details" in "primary_output"
- WRONG EXAMPLE: {"primary_output": "Python script", "supporting_details": "def main(): print('hello')"}
- CORRECT EXAMPLE: {"primary_output": "def main(): print('hello')", "supporting_details": "Python 3.8+, no deps"}

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
- After tool execution, provide a natural, comprehensive response that incorporates the tool results
- Tool outputs should be processed and the important information included in your final response
- If tools produce files, images, or URLs, include them properly formatted in your response""",
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
        SHOW_ACTION_SUMMARIES: bool = Field(
            default=True,
            description="Show detailed summaries for completed actions in dropdown format",
        )
        ACTION_TEMPERATURE: float = Field(
            default=0.7,
            description="Temperature setting for the ACTION_MODEL (tool-using actions)",
        )
        WRITER_TEMPERATURE: float = Field(
            default=0.9,
            description="Temperature setting for the WRITER_MODEL (creative text generation)",
        )
        CODER_TEMPERATURE: float = Field(
            default=0.3,
            description="Temperature setting for the CODER_MODEL (code generation)",
        )
        PLANNING_TEMPERATURE: float = Field(
            default=0.8,
            description="Temperature setting for planning phase",
        )
        ANALYSIS_TEMPERATURE: float = Field(
            default=0.4,
            description="Temperature setting for output analysis and reflection",
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

        if action.use_lightweight_context:
            enhanced_requirements += self.valves.LIGHTWEIGHT_CONTEXT_REQUIREMENTS_SUFFIX
        else:
            match model:
                case self.valves.WRITER_MODEL:
                    enhanced_requirements += self.valves.WRITER_REQUIREMENTS_SUFFIX
                case self.valves.CODER_MODEL:
                    enhanced_requirements += self.valves.CODER_REQUIREMENTS_SUFFIX
                case _:
                    enhanced_requirements += self.valves.ACTION_REQUIREMENTS_SUFFIX

        if action.use_lightweight_context:
            lightweight_context = {}
            for dep_id, dep_data in context.items():
                if isinstance(dep_data, dict):
                    lightweight_context[dep_id] = {
                        "action_id": dep_id,
                        "supporting_details": dep_data.get("supporting_details", ""),
                    }
                else:
                    lightweight_context[dep_id] = {
                        "action_id": dep_id,
                        "supporting_details": "",
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
            response_content = response["choices"][0]["message"].get("content","")
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
                            
                            CRITICAL FIELD REQUIREMENTS - AUTOMATIC FAILURE IF VIOLATED:
                            - The "primary_output" field MUST contain the MAIN DELIVERABLE/RESULT from this action
                            - The "supporting_details" field is for internal communication only (max 150 chars) - brief metadata/context
                            - NEVER put the main deliverable/result in "supporting_details"
                            - NEVER put "See supporting details" in "primary_output"
                            - WRONG EXAMPLE: {"primary_output": "Task completed", "supporting_details": "Here is the actual important result..."}
                            - CORRECT EXAMPLE: {"primary_output": "Here is the actual important result...", "supporting_details": "Tool: search, 3 results"}
                            
                            CRITICAL: The "primary_output" field must contain the ACTUAL deliverable (URLs for images, complete text for writing tasks, functional code for coding tasks, etc.), not just descriptions or titles. This content will be directly used by other steps and in the final synthesis.
                            Tool outputs should be processed and the important information included in "primary_output"
                            If tools produce files, images, or URLs, include them properly formatted in "primary_output" """
                tool_response = await self.get_completion(
                    prompt=messages,
                    temperature=temperature,
                    model=model,
                    tools=tools,
                    action_results=action_results,
                    action=action,
                    format=format,
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
- Always use the exact "tool_id" in the "tool_ids" field from the available tools list when an action needs external capabilities in the correct tool_ids field
- Return a JSON array of tool_ids, for example: ["tool_id_1", "tool_id_2"] in the "tool_ids" field if the "action" type is "tool". MANDATORY for tool actions.

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
- When false, the action receives full context from dependencies (default behavior)
- Best for file operations that require saving or processing content verbatim, like saving chapters, saving generated images, or saving code, 
that dont need the full context of the previous steps but only a reference to the previous action's output
- Use lightweight context mode when the action can operate with just the action IDs and supporting details,
  such as when the action can process files or data based on identifiers rather than needing the full
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
                                                "default": False,
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
                    temperature=self.valves.PLANNING_TEMPERATURE,
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

                try:
                    await self.validate_and_fix_tool_actions(plan)
                except Exception as validation_error:
                    await self.emit_status(
                        "warning",
                        f"Tool validation failed but continuing with plan: {str(validation_error)}",
                        False,
                    )
                    logger.warning(f"Tool validation error: {validation_error}")

                try:
                    await self.validate_and_enhance_template(plan)
                except Exception as template_error:
                    await self.emit_status(
                        "warning",
                        f"Template validation failed but continuing with plan: {str(template_error)}",
                        False,
                    )
                    logger.warning(f"Template validation error: {template_error}")

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
            temperature=self.valves.ACTION_TEMPERATURE,
            action_results={},
            action=None,
        )
        return enhanced_requirements

    async def validate_and_fix_tool_actions(self, plan: Plan):
        """Check for tool actions missing tool_ids and automatically populate them."""
        await self.emit_status(
            "info", "Starting tool validation for plan actions...", False
        )

        tools: list[dict[str, Any]] = [
            {
                "tool_id": tool.id,
                "tool_name": tool.name,
            }
            for tool in Tools.get_tools()
        ]

        actions_needing_tools = [
            action
            for action in plan.actions
            if action.type == "tool" and (not action.tool_ids)
        ]

        if not actions_needing_tools:
            await self.emit_status(
                "success", "All tool actions have proper tool_ids specified.", False
            )
            return

        await self.emit_status(
            "info",
            f"Found {len(actions_needing_tools)} tool action(s) missing tool_ids. Auto-fixing...",
            False,
        )

        for action in actions_needing_tools:
            await self.emit_status(
                "info", f"Identifying tools for action: {action.id}", False
            )

            tool_selection_prompt = f"""
You are a tool selection expert. Given an action description, select the most appropriate tool(s) from the available list.

ACTION TO ANALYZE:
- Description: {action.description}
- Parameters: {json.dumps(action.params)}
- Dependencies: {action.dependencies}

AVAILABLE TOOLS:
{json.dumps(tools, indent=2)}

INSTRUCTIONS:
1. Analyze the action description to understand what needs to be accomplished
2. Select the most appropriate tool(s) from the available list
3. Return ONLY the tool_id(s) that best match the action requirements
4. If multiple tools are needed, list all relevant tool_ids
5. Focus on tools that directly accomplish the action's objective

OUTPUT FORMAT:
Return a JSON array of tool_ids, for example: ["tool_id_1", "tool_id_2"]
If no suitable tools are found, return an empty array: []
"""

            try:
                tool_format: dict[str, Any] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tool_selection",
                        "strict": True,
                        "schema": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                }

                result = await self.get_completion(
                    prompt=tool_selection_prompt,
                    temperature=self.valves.ACTION_TEMPERATURE,
                    model="",
                    format=tool_format,
                    action_results={},
                    action=None,
                )

                clean_result = clean_json_response(result)
                selected_tools = json.loads(clean_result)

                available_tool_ids = {tool["tool_id"] for tool in tools}
                valid_tools = [
                    tool_id
                    for tool_id in selected_tools
                    if tool_id in available_tool_ids
                ]

                if valid_tools:
                    action.tool_ids = valid_tools
                    await self.emit_status(
                        "success",
                        f"Added tools to {action.id}: {', '.join(valid_tools)}",
                        False,
                    )
                else:
                    await self.emit_status(
                        "warning",
                        f"No suitable tools found for action {action.id}. Action may need manual review.",
                        False,
                    )

            except Exception as e:
                await self.emit_status(
                    "warning",
                    f"Failed to auto-select tools for {action.id}: {str(e)}",
                    False,
                )

        await self.emit_status(
            "success", "Tool validation and auto-fixing completed.", False
        )

        still_missing_tools = [
            action.id
            for action in plan.actions
            if action.type == "tool" and (not action.tool_ids)
        ]

        if still_missing_tools:
            await self.emit_status(
                "warning",
                f"Actions still missing tools (may need manual review): {', '.join(still_missing_tools)}",
                False,
            )

    async def validate_and_enhance_template(self, plan: Plan):
        """Check if the final_synthesis template has all required action IDs and enhance it if needed."""
        await self.emit_status(
            "info", "Validating final_synthesis template...", False
        )

        final_synthesis = next(
            (a for a in plan.actions if a.id == "final_synthesis"), None
        )
        
        if not final_synthesis:
            return  # No final_synthesis action to validate

        template = final_synthesis.description
        
        # Extract placeholders from template
        placeholder_pattern = r"\{([a-zA-Z0-9_]+)\}"
        template_placeholders = set(re.findall(placeholder_pattern, template))
        
        # Get action IDs from dependencies (these are what should be in template)
        dependency_ids = set(final_synthesis.dependencies)
        
        # Find missing action IDs in template
        missing_in_template = dependency_ids - template_placeholders
        
        if not missing_in_template:
            await self.emit_status(
                "success", "Template validation passed - all dependencies are referenced in template.", False
            )
            return

        await self.emit_status(
            "info", f"Template missing references to: {', '.join(missing_in_template)}. Enhancing template...", False
        )

        # Get all actions for context
        actions_info = []
        for action in plan.actions:
            if action.id != "final_synthesis":
                actions_info.append({
                    "id": action.id,
                    "type": action.type,
                    "description": action.description
                })

        template_enhancement_prompt = f"""
You are a template enhancement expert for the final_synthesis action. Your job is to enhance an incomplete template by adding missing action references in logical positions.

UNDERSTANDING TEMPLATES:
The final_synthesis action uses a TEMPLATING SYSTEM where:
- {{action_id}} placeholders get replaced with the complete primary_output from that action
- Each placeholder becomes the actual content (text, URLs, code, etc.) from the referenced action
- The template structure becomes the final user-facing output after placeholder substitution
- This is PURE TEXT REPLACEMENT - no AI processing happens during synthesis

CURRENT SITUATION:
GOAL: {plan.goal}

EXISTING TEMPLATE:
{template}

TEMPLATE DEPENDENCIES: {list(dependency_ids)}
MISSING REFERENCES: {list(missing_in_template)}

ALL PLAN ACTIONS CONTEXT:
{json.dumps(actions_info, indent=2)}

ENHANCEMENT REQUIREMENTS:

1. **PRESERVE EXISTING STRUCTURE**: Keep all current content, formatting, and existing {{action_id}} references exactly as they are

2. **ADD MISSING REFERENCES**: Include ALL missing action references ({list(missing_in_template)}) using {{action_id}} format

3. **LOGICAL PLACEMENT**: Position missing references where they make sense contextually:
   - Research actions → Background/Introduction sections
   - Content creation → Main body sections  
   - Images/media → Visual elements with proper markdown
   - Code → Code blocks with language specification
   - Analysis → Analysis/Results sections
   - Conclusions → Summary/Conclusion sections

4. **CONTENT TYPE AWARENESS**: Consider what each action produces:
   - Text actions: Insert as paragraphs or sections
   - Image actions: Use ![Description]({{action_id}}) format
   - Code actions: Use ```language\\n{{action_id}}\\n``` format
   - Search/Research: Use in background or reference sections
   - URL/Link actions: Use [Text]({{action_id}}) format

5. **MARKDOWN STRUCTURE**: Maintain proper hierarchy with headers (#, ##, ###)

6. **COMPREHENSIVE COVERAGE**: The enhanced template should create a complete deliverable that includes ALL valuable outputs from the plan

7. **USER-FACING QUALITY**: The final output should be professional, well-organized, and immediately useful to users

EXAMPLE ENHANCEMENT PATTERNS:

If missing "research_background":
Add: "## Background\\n{{research_background}}\\n\\n"

If missing "generated_image": 
Add: "![Generated Visualization]({{generated_image}})\\n\\n"

If missing "code_solution":
Add: "```python\\n{{code_solution}}\\n```\\n\\n"

If missing "data_analysis":
Add: "## Analysis\\n{{data_analysis}}\\n\\n"

CRITICAL RULES:
- Use ONLY {{action_id}} format - NO nested properties like {{action.output}}
- Every missing action MUST be included somewhere logical
- Maintain existing template flow and structure
- Ensure the template will create a cohesive final document
- Use proper Markdown syntax for all formatting

Return ONLY the enhanced template description. Do not include explanations, comments, or extra text.
"""

        try:
            enhanced_template = await self.get_completion(
                prompt=template_enhancement_prompt,
                temperature=self.valves.PLANNING_TEMPERATURE,
                action_results={},
                action=None,
            )
            
            # Update the final_synthesis action with enhanced template
            final_synthesis.description = enhanced_template.strip()
            
            await self.emit_status(
                "success", f"Template enhanced successfully with missing references: {', '.join(missing_in_template)}", False
            )
            
        except Exception as e:
            await self.emit_status(
                "warning", f"Failed to enhance template: {str(e)}", False
            )

    async def execute_action(
        self, plan: Plan, action: Action, context: dict[str, Any], step_number: int
    ) -> dict[str, Any]:

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

        if action.use_lightweight_context:
            context_for_prompt = {}
            for dep in action.dependencies:
                if dep in context:
                    dep_result = context.get(dep, {})
                    context_for_prompt[dep] = {
                        "action_id": dep,
                        "supporting_details": dep_result.get("supporting_details", ""),
                    }
                else:
                    context_for_prompt[dep] = {
                        "action_id": dep,
                        "supporting_details": "",
                    }
        else:
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
                # Note: action.status is already set to "in_progress" in execute_plan

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

                    # Determine appropriate temperature based on model type
                    if execution_model == self.valves.WRITER_MODEL:
                        model_temperature = self.valves.WRITER_TEMPERATURE
                    elif execution_model == self.valves.CODER_MODEL:
                        model_temperature = self.valves.CODER_TEMPERATURE
                    else:
                        model_temperature = self.valves.ACTION_TEMPERATURE

                    response = await self.get_completion(
                        prompt=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": base_prompt},
                        ],
                        temperature=model_temperature,
                        model=execution_model,
                        tools=tools,
                        format=action_format,
                        action_results=context,
                        action=action,
                    )

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

                    # Show the current attempt immediately to user (before analysis)
                    formatted_output = self.format_action_output(action, current_output)
                    await self.emit_message(formatted_output)

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
                    # Analysis passed - action output will become summary in main loop
                    break
                else:
                    # Analysis failed - if we're retrying, prepare for next attempt
                    if attempts_remaining > 0:
                        # Show retry status instead of just clearing
                        await self.emit_status(
                            "warning",
                            f"Output needs improvement. Retrying... ({attempts_remaining} attempts remaining) (Quality Score: {current_reflection.quality_score:.2f})",
                            False,
                        )
                        # Note: unsuccessful output will be replaced by retry attempt

                if attempts_remaining > 0:
                    attempts_remaining -= 1
                    continue
                else:
                    # Final failed attempt - output will be handled in summary
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
            # No regular message for failed actions - only summary

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

CRITICAL FIELD USAGE VERIFICATION - AUTOMATIC FAILURE CONDITIONS:
- PRIMARY_OUTPUT must contain the MAIN DELIVERABLE content (the actual result users need)
- SUPPORTING_DETAILS must contain only ADDITIONAL CONTEXT, metadata, or explanatory information
- If the main content/deliverable is in supporting_details instead of primary_output: AUTOMATIC quality_score = 0.1
- If primary_output contains only brief summaries while actual content is in supporting_details: AUTOMATIC quality_score = 0.1
- If primary_output is empty or just says "See supporting details": AUTOMATIC quality_score = 0.1

CRITICAL TOOL VERIFICATION:
- If the action was expected to use tools ({expected_tools}) but no tools were called ({actual_tool_calls}), this is a MAJOR failure
- If tools were called, verify that the output actually incorporates their results meaningfully
- If the output claims tools were used but no actual tool calls occurred, this is FALSE and should be heavily penalized
- Tool results should be properly processed and integrated into the final output

Instructions:
Critically evaluate the output based on the following criteria:
1. **FIELD CORRECTNESS (CRITICAL)**: The primary_output field MUST contain the main deliverable content. Supporting_details MUST only contain auxiliary information. Content in wrong fields = AUTOMATIC FAILURE
2. **Tool Usage Verification**: STRICTLY verify that claimed tool usage matches actual tool calls. False claims about tool usage should result in quality_score <= 0.3
3. **Output Format**: The output should be a valid JSON object with "primary_output" and "supporting_details" fields
4. **Completeness**: Does the output fully address the action's description and requirements?
5. **Correctness**: Is the information, tool usage, or code (if present) accurate and functional?
6. **Relevance**: Does the output directly contribute to the overall goal?
7. **Tool Integration**: If tools were used, are their results properly integrated and processed in the output?
8. **Content Quality**: Is the primary_output field clean, complete, and ready for use by subsequent steps?
9. **Markdown integration**: Markdown format for deliverables is preferable using the embeding formats for example ![caption](<image uri>) to show image or embedable content.
10.**Missing Tool Calls**: if tool calls werent done Ask the model to call them but do not mention Format at this step.

EXAMPLES OF CORRECT vs INCORRECT FIELD USAGE:

✅ CORRECT:
{{"primary_output": "# AI News Report\\n\\nGoogle's Gemini Advancements...", "supporting_details": "Source: TechCrunch. Search performed at 10:30 AM."}}

❌ INCORRECT (AUTOMATIC FAIL):
{{"primary_output": "AI News Report", "supporting_details": "# AI News Report\\n\\nGoogle's Gemini Advancements..."}}

❌ INCORRECT (AUTOMATIC FAIL):
{{"primary_output": "See supporting details", "supporting_details": "# AI News Report\\n\\nGoogle's Gemini Advancements..."}}

❌ INCORRECT (AUTOMATIC FAIL):
{{"primary_output": "Summary: AI news compiled", "supporting_details": "# AI News Report\\n\\nGoogle's Gemini Advancements..."}}

Remember: PRIMARY_OUTPUT = Main content that users need. SUPPORTING_DETAILS = Extra context only.
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

        # Retry loop for analysis
        attempts_remaining = self.valves.MAX_RETRIES
        while attempts_remaining >= 0:
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
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
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
                    temperature=self.valves.ANALYSIS_TEMPERATURE,
                    format=reflection_format,
                    action_results={},
                    action=None,
                )

                clean_response = clean_json_response(analysis_response)
                analysis_data = json.loads(clean_response)

                return ReflectionResult(**analysis_data)

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.error(
                    f"Failed to parse reflection analysis (attempt {self.valves.MAX_RETRIES - attempts_remaining + 1}/{self.valves.MAX_RETRIES + 1}): {e}. Raw response: {analysis_response}"
                )

                if attempts_remaining > 0:
                    attempts_remaining -= 1
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    # All retries exhausted - return default failure result
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
                    f"An unexpected error occurred during output analysis (attempt {self.valves.MAX_RETRIES - attempts_remaining + 1}/{self.valves.MAX_RETRIES + 1}): {e}. Raw response: {analysis_response}"
                )

                if attempts_remaining > 0:
                    attempts_remaining -= 1
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    # All retries exhausted - return default failure result
                    return ReflectionResult(
                        is_successful=False,
                        quality_score=0.0,
                        issues=[f"An unexpected error occurred during analysis: {e}"],
                        suggestions=["Retry the action."],
                    )

        # This should never be reached, but added for completeness
        return ReflectionResult(
            is_successful=False,
            quality_score=0.0,
            issues=["Analysis failed after all retry attempts."],
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
        completed_summaries: list[str] = []  # Track completed action summaries

        async def can_execute(action: Action) -> bool:
            return all(dep in completed for dep in action.dependencies)

        while len(completed) < len(plan.actions):
            await self.emit_full_state(plan, completed_summaries)

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
                await self.emit_full_state(plan, completed_summaries)

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

                # Check if this is truly the final action (no more actions to execute)
                remaining_actions = [a for a in plan.actions if a.id not in completed]
                if not remaining_actions:
                    # This is the final result - show as normal message
                    formatted_output = self.format_action_output(
                        action, action.output, is_final_result=True
                    )
                    await self.emit_message(formatted_output)
                else:
                    # There are still actions to execute - add to summaries for now
                    summary = self.generate_action_summary(action, plan)
                    if summary:
                        completed_summaries.append(summary)

                continue  #

            in_progress.add(action.id)

            # Set status to in_progress and update mermaid immediately
            action.status = "in_progress"
            action.start_time = datetime.now().strftime("%H:%M:%S")
            await self.emit_full_state(plan, completed_summaries)

            try:
                context: dict[Any, Any] = {
                    dep: completed_results.get(dep, {}) for dep in action.dependencies
                }

                result = await self.execute_action(plan, action, context, step_counter)

                completed_results[action.id] = result
                completed.add(action.id)

                # Add action summary to collection
                summary = self.generate_action_summary(action, plan)
                if summary:
                    completed_summaries.append(summary)

                # Update mermaid to show completion
                await self.emit_full_state(plan, completed_summaries)

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
                # Update mermaid to show failure
                await self.emit_full_state(plan, completed_summaries)
            finally:
                if action.id in in_progress:
                    in_progress.remove(action.id)

        # Final state update
        await self.emit_full_state(plan, completed_summaries)

        # Check if final synthesis was completed and show it as final message
        final_synthesis_action = next(
            (
                a
                for a in plan.actions
                if a.id == "final_synthesis" and a.status == "completed"
            ),
            None,
        )
        if final_synthesis_action and final_synthesis_action.output:
            # Remove final synthesis from summaries if it exists
            completed_summaries = [
                s for s in completed_summaries if "🎯 Final Synthesis Complete" not in s
            ]

            # Show final synthesis as the ultimate result
            formatted_output = self.format_action_output(
                final_synthesis_action,
                final_synthesis_action.output,
                is_final_result=True,
            )
            await self.emit_message(formatted_output)

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

    def clean_nested_markdown(self, text: str) -> str:
        """Clean nested markdown image syntax like ![alt](![alt2](url)) to just ![alt2](url)"""
        import re

        nested_pattern = r"!\[([^\]]*)\]\(!\[([^\]]*)\]\(([^)]+)\)\)"

        cleaned_text = re.sub(nested_pattern, r"![\2](\3)", text)

        return cleaned_text

    def format_action_output(
        self, action: Action, output: dict[str, str], is_final_result: bool = False
    ) -> str:
        """Format action output for user display (non-JSON format)"""
        primary_output = output.get("primary_output", "")
        supporting_details = output.get("supporting_details", "")

        primary_output = self.clean_nested_markdown(primary_output)
        supporting_details = self.clean_nested_markdown(supporting_details)

        if action.id == "final_synthesis":
            if is_final_result:
                # Clean output without header for final result
                formatted_content = f"{primary_output}\n\n"
            else:
                formatted_content = (
                    f"## 🎯 Final Synthesis Complete\n\n{primary_output}\n\n---\n"
                )
            return formatted_content

        formatted_content = f"## 🔄 Action: {action.description}\n\n"

        if primary_output:
            formatted_content += f"{primary_output}\n\n"

        if supporting_details and supporting_details.strip():
            formatted_content += f"<details>\n<summary>📋 Supporting Details</summary>\n\n{supporting_details}\n\n</details>\n\n"

        formatted_content += "---\n"
        return formatted_content

    async def emit_full_state(self, plan: Plan, completed_summaries: list[str]):
        """Emit the full state including mermaid diagram and all summaries"""
        mermaid = await self.generate_mermaid(plan)

        content_parts = [f"```mermaid\n{mermaid}\n```"]

        if completed_summaries:
            content_parts.append("---")
            content_parts.extend(completed_summaries)

        final_synthesis_action = next(
            (a for a in plan.actions if a.id == "final_synthesis"), None
        )

        if final_synthesis_action and self.valves.SHOW_ACTION_SUMMARIES:
            template = final_synthesis_action.description
            preview_template = template

            template_placeholders = re.findall(r"\{([a-zA-Z0-9_]+)\}", template)
            total_placeholders = len(template_placeholders)

            completed_actions = [
                a
                for a in plan.actions
                if a.status in ["completed", "warning"] and a.output
            ]
            pending_actions = [
                a
                for a in plan.actions
                if a.status == "pending" and a.id != "final_synthesis"
            ]

            completed_placeholders = 0

            for action in completed_actions:
                placeholder = f"{{{action.id}}}"
                if placeholder in template and action.output:
                    completed_placeholders += 1
                    preview_content = action.output.get("primary_output", "")
                    if len(preview_content) > 200:
                        preview_content = preview_content[:200] + "..."
                    preview_template = preview_template.replace(
                        placeholder, f"✅ [{action.id}]: {preview_content}"
                    )

            for action in pending_actions:
                placeholder = f"{{{action.id}}}"
                if placeholder in template:
                    preview_template = preview_template.replace(
                        placeholder, f"⏳ [{action.id}]: Pending..."
                    )

            final_synthesis_content = f"""<details>
<summary>📋 Final Synthesis Template ({completed_placeholders}/{total_placeholders} outputs ready)</summary>

**Template Preview**:
{preview_template}

**Status**: {completed_placeholders} of {total_placeholders} template placeholders completed

---

</details>"""
            content_parts.append(final_synthesis_content)

        full_content = "\n\n".join(content_parts)
        await self.emit_replace(full_content)

    def generate_action_summary(self, action: Action, plan: Plan) -> str:
        """Generate a detailed summary of a completed action in dropdown format"""
        if not self.valves.SHOW_ACTION_SUMMARIES:
            return ""

        if action.id == "final_synthesis":
            summary_title = "🎯 Final Synthesis Complete"
        else:
            status_emoji = "✅" if action.status == "completed" else "⚠️"
            summary_title = (
                f"{status_emoji} {action.status.title()}: {action.description}"
            )

        tool_calls_str = ", ".join(action.tool_calls) if action.tool_calls else "None"

        tool_results_summary = ""
        if action.tool_results:
            tool_results_summary = "\n".join(
                [
                    f"- **{tool}**: {result[:100]}{'...' if len(result) > 100 else ''}"
                    for tool, result in action.tool_results.items()
                ]
            )
        else:
            tool_results_summary = "None"

        execution_time = ""
        if action.start_time and action.end_time:
            execution_time = (
                f"**Execution Time**: {action.start_time} - {action.end_time}\n"
            )

        summary_content = f"""**Action ID**: {action.id}
**Type**: {action.type}
**Status**: {action.status}
**Model**: {action.model or "Default"}
{execution_time}**Tool Calls**: {tool_calls_str}

**Tool Results**:
{tool_results_summary}

**Supporting Details**: {action.output.get('supporting_details', 'None') if action.output else 'None'}

**Primary Output**:
{action.output.get('primary_output', 'No output available') if action.output else 'No output available'}"""

        return f"<details>\n<summary>{summary_title}</summary>\n\n{summary_content}\n\n---\n\n</details>"

    async def emit_action_summary(self, action: Action, plan: Plan):
        """Emit a detailed summary of a completed action in dropdown format"""
        summary = self.generate_action_summary(action, plan)
        if summary:
            await self.__current_event_emitter__(
                {
                    "type": "message",
                    "data": {"content": summary},
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

        await self.emit_full_state(plan, [])

        await self.emit_status("info", "Executing plan...", False)
        await self.execute_plan(plan)

        await self.emit_status("success", "Plan execution completed.", True)

        return
