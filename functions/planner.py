"""
title: Planner
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.8.3
"""

import logging
import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Callable, Awaitable, Union
from pydantic import BaseModel, Field
from datetime import datetime
from open_webui.constants import TASKS
from open_webui.utils.chat import generate_chat_completion
from dataclasses import dataclass
from fastapi import Request
import re
import difflib

# Constants and Setup
name = "Planner"


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str


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
    params: Dict = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    output: Optional[Dict] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class Plan(BaseModel):
    """Model for the complete execution plan"""

    goal: str
    actions: List[Action]
    metadata: Dict = Field(default_factory=dict)
    final_output: Optional[str] = None
    execution_summary: Optional[Dict] = None


class ReflectionResult(BaseModel):
    """Model for storing reflection analysis results"""

    is_successful: bool
    confidence_score: float
    issues: Union[str, List[str]] = Field(default_factory=list)
    suggestions: Union[str, List[str]] = Field(default_factory=list)
    required_corrections: Dict[str, Union[str, List[str]]] = Field(default_factory=dict)
    output_quality_score: float


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __model__: str

    class Valves(BaseModel):
        MODEL: str = Field(
            default="", description="Model to use (model id from ollama)"
        )
        ACTION_MODEL: str = Field(
            default="", description="Model to use (model id from ollama)"
        )
        ACTION_PROMPT_REQUIREMENTS_TEMPLATE: str = Field(
            default="""Requirements:
            1. Provide implementation that DIRECTLY achieves the step goal
            2. Include ALL necessary components
            3. For code:
               - Must be complete and runnable
               - All variables must be properly defined
               - No placeholder functions or TODO comments
            4. For text/documentation:
               - Must be specific and actionable
               - Include all relevant details
        
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

    async def get_streaming_completion(
        self,
        messages,
        temperature: float = 0.7,  # Default fallback temperature
        top_k: int = 50,  # Default fallback top_k
        top_p: float = 0.9,  # Default fallback top_p
        model: str = "",
    ):
        model = model if model else self.valves.MODEL
        try:
            form_data = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
            response = await generate_chat_completion(
                self.__request__,
                form_data,
                user=self.__user__,
            )

            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")

            async for chunk in response.body_iterator:
                for part in self.get_chunk_content(chunk):
                    yield part

        except Exception as e:
            raise RuntimeError(f"Streaming completion failed: {e}")

    def get_chunk_content(self, chunk):
        # Convert bytes to string if needed
        if isinstance(chunk, bytes):
            chunk_str = chunk.decode("utf-8")
        else:
            chunk_str = chunk

        # Remove the data: prefix if present
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:]

        chunk_str = chunk_str.strip()

        if chunk_str == "[DONE]" or not chunk_str:
            return

        try:
            chunk_data = json.loads(chunk_str)
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except json.JSONDecodeError:
            logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')

    async def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,  # Default fallback temperature
        top_k: int = 50,  # Default fallback top_k
        top_p: float = 0.9,  # Default fallback top_p
    ) -> str:
        response = await generate_chat_completion(
            self.__request__,
            {
                "model": self.__model__,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def generate_mermaid(self, plan: Plan) -> str:
        """Generate Mermaid diagram representing the current plan state"""
        mermaid = ["graph TD", f'    Start["Goal: {plan.goal[:30]}..."]']

        status_emoji = {
            "pending": "⭕",
            "in_progress": "⚙️",
            "completed": "✅",
            "failed": "❌",
        }

        styles = []
        for action in plan.actions:
            node_id = f"action_{action.id}"
            mermaid.append(
                f'    {node_id}["{status_emoji[action.status]} {action.description[:40]}..."]'
            )

            if action.status == "in_progress":
                styles.append(f"style {node_id} fill:#fff4cc")
            elif action.status == "completed":
                styles.append(f"style {node_id} fill:#e6ffe6")
            elif action.status == "failed":
                styles.append(f"style {node_id} fill:#ffe6e6")

        # Add connections
        mermaid.append(f"    Start --> action_{plan.actions[0].id}")
        for action in plan.actions:
            for dep in action.dependencies:
                mermaid.append(f"    action_{dep} --> action_{action.id}")

        # Add styles at the end
        mermaid.extend(styles)

        return "\n".join(mermaid)

    async def create_plan(self, goal: str) -> Plan:
        """Create an execution plan for the given goal"""
        prompt = f"""
Given the goal: {goal}

Generate a detailed execution plan by breaking it down into atomic, sequential steps. Each step should be clear, actionable, and have explicit dependencies.

Return a JSON object with exactly this structure:
{{
    "goal": "<original goal>",
    "actions": [
        {{
            "id": "<unique_id>",
            "type": "<category_of_action>",
            "description": "<specific_actionable_task>",
            "params": {{"param_name": "param_value"}},
            "dependencies": ["<id_of_prerequisite_step>"]
        }}
    ],
    "metadata": {{
        "estimated_time": "<minutes>",
        "complexity": "<low|medium|high>"
    }}
}}

Requirements:
1. Each step must be independently executable
2. Dependencies must form a valid sequence
3. Steps must be concrete and specific
4. For code-related tasks:
   - Focus on implementation only
   - Exclude testing and deployment
   - Each step should produce complete, functional code or complete well writen textual content.

Return ONLY the JSON object. Do not include explanations or additional text.
"""

        for attempt in range(self.valves.MAX_RETRIES):
            try:
                result = await self.get_completion(
                    prompt, temperature=0.8, top_k=60, top_p=0.95
                )
                plan_dict = json.loads(result)
                return Plan(**plan_dict)
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
        Focus on this specific action: {action.description}

        Parameters: {json.dumps(action.params)}

        """
        if has_dependencies:
            requirements_prompt += f"""
            Dependencies: {dependencies_str}
            Consider how dependencies should be used in this action.
            """

        requirements_prompt += f"""
        Generate concise and clear requirements to ensure this action is performed correctly.

        For code actions:
        - Ensure complete and runnable code.
        - Define all variables.
        - Handle errors gracefully.

        For text/documentation actions:
        - Be specific and actionable.
        - Include all relevant details.


        Return a numbered list of requirements.  Be direct and avoid unnecessary explanations.
        """
        enhanced_requirements = await self.get_completion(
            requirements_prompt,
            temperature=0.7,
            top_k=40,
            top_p=0.8,
        )
        logger.debug(f"RAW Enahcned Requirements: {enhanced_requirements}")
        return enhanced_requirements

    async def execute_action(
        self, plan: Plan, action: Action, results: Dict, step_number: int
    ) -> Dict:
        """Execute action with enhanced reflection, always returning best output"""
        action.start_time = datetime.now().strftime("%H:%M:%S")
        action.status = "in_progress"

        context = {dep: results.get(dep, {}) for dep in action.dependencies}
        requirements = (
            await self.enhance_requirements(plan, action)
            if self.valves.AUTOMATIC_TAKS_REQUIREMENT_ENHANCEMENT
            else self.valves.ACTION_PROMPT_REQUIREMENTS_TEMPLATE
        )
        base_prompt = f"""
            Execute step {step_number}: {action.description}
            Overall Goal: {plan.goal}
        
            Context:
            - Parameters: {json.dumps(action.params)}
            - Previous Results: {json.dumps(context)}
        
            {requirements}
            
            Focus ONLY on this specific step's output.
            """

        attempts_remaining = self.valves.MAX_RETRIES
        best_output = None
        best_reflection = None
        best_quality_score = -1

        while attempts_remaining >= 0:  # Changed to include the initial attempt
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
                        
                        Required corrections:
                        {json.dumps(best_reflection.suggestions, indent=2)}
                        
                        Please address ALL issues above in this new attempt.
                        """

                try:
                    complete_response = ""
                    async for chunk in self.get_streaming_completion(
                        [
                            {"role": "system", "content": f"Goal: {plan.goal}"},
                            {"role": "user", "content": base_prompt},
                        ],
                        temperature=0.9,
                        top_k=70,
                        top_p=0.95,
                        model=(
                            self.valves.ACTION_MODEL
                            if (self.valves.ACTION_MODEL != "")
                            else self.valves.MODEL
                        ),
                    ):
                        complete_response += chunk
                        await self.emit_message(chunk)

                    current_output = {"result": complete_response}

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
                        raise

                await self.emit_status(
                    "info",
                    f"Analyzing output ...",
                    False,
                )

                current_reflection = await self.analyze_output(
                    plan=plan,
                    action=action,
                    output=complete_response,
                    attempt=current_attempt,
                )

                await self.emit_status(
                    "info",
                    f"Analyzed output (Quality Score: {current_reflection.output_quality_score:.2f})",
                    False,
                )

                # Update best output if current quality is higher
                if current_reflection.output_quality_score > best_quality_score:
                    best_output = current_output
                    best_reflection = current_reflection
                    best_quality_score = current_reflection.output_quality_score

                # If execution was successful, we can stop retrying
                if current_reflection.is_successful:
                    break

                # If we have attempts remaining and execution failed, continue
                if attempts_remaining > 0:
                    attempts_remaining -= 1
                    await self.emit_status(
                        "warning",
                        f"Output needs improvement. Retrying... ({attempts_remaining + 1} attempts remaining) (Quality Score: {current_reflection.output_quality_score:.2f})",
                        False,
                    )
                    continue

                # If we're here, we're out of attempts but have a best output
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

        # After all attempts, use the best output we got
        if best_output is None:
            action.status = "failed"
            action.end_time = datetime.now().strftime("%H:%M:%S")
            await self.emit_status(
                "error",
                f"Action failed to produce any valid output after all attempts",
                True,
            )
            raise RuntimeError("No valid output produced after all attempts")

        action.status = "completed"
        action.end_time = datetime.now().strftime("%H:%M:%S")
        action.output = best_output
        await self.emit_status(
            "success",
            f"Action completed with best output (Quality: {best_quality_score:.2f} Consolidating Outputs...)",
            True,
        )
        return best_output

    async def analyze_output(
        self,
        plan: Plan,
        action: Action,
        output: str,
        attempt: int,
    ) -> ReflectionResult:
        """Sophisticated output analysis"""

        analysis_prompt = f"""
    Analyze this action output for step {action.id}:

    Goal: {plan.goal}
    Action: {action.description}
    Output: {output}
    Attempt: {attempt + 1}

    Provide a detailed analysis in JSON format with these EXACT keys:
    {{
        "is_successful": boolean,
        "confidence_score": float between 0-1,
        "issues": [
            "specific issue 1",
            "specific issue 2",
            ...
        ],
        "suggestions": [
            "specific correction 1",
            "specific correction 2",
            ...
        ],
        "required_corrections": {{
            "component": "what needs to change",
            "changes": ["a concise string describing the required change 1", "another change", ...]  
        }},
        "output_quality_score": float between 0-1,
        "analysis": {{
            "completeness": float between 0-1,
            "correctness": float between 0-1,
            "clarity": float between 0-1,
            "specific_problems": [
                "detailed problem description 1",
                ...
            ]
        }}
    }}

    Evaluate based on:
    1. Completeness: Are all required components present?
    2. Correctness: Is everything technically accurate?
    3. Clarity: Is the output clear and well-structured?
    4. Specific Issues: What exactly needs improvement?

    For code output, also check:
    - All variables are properly defined
    - No missing functions or imports
    - Proper error handling
    - Code is runnable as-is

    For text output, also check:
    - All required information is included
    - Clear organization and structure
    - No ambiguous statements
    - Actionable content

    Be extremely specific in the analysis. Each issue must have a corresponding suggestion for correction.


    Your response MUST be a single, valid JSON object, and absolutely NOTHING else. 
    No introductory text, no explanations, no comments – ONLY the JSON data structure as described.

    # ======================  New additions here ======================

    Be brutally honest in your assessment. Assume a highly critical perspective and focus on finding flaws and areas for improvement, even minor ones.

    The `output_quality_score` should be a true reflection of the output's overall quality, considering both major and minor flaws. Avoid inflating the score. It should be a conservative estimate of the output's merit.

    Use this general guideline for scoring:
        * 0.9-1.0:  Flawless output with no significant issues.
        * 0.7-0.89: Minor issues or areas for improvement.
        * 0.5-0.69:  Significant issues that need addressing.
        * 0.3-0.49: Major flaws and incomplete implementation.
        * 0.0-0.29:  Severely flawed or unusable output.

    The `output_quality_score` should be inversely proportional to the number and severity of issues found. More issues should generally lead to a lower score.

    Provide a brief justification for the assigned `output_quality_score` within the "analysis" object. 
    "confidence_score": refears to the confidence on your specific assesment output not the input being analyzed.
    # =================================================================

    Return your analysis as a valid JSON object enclosed within these markers:
    ```json
    <JSON_START>
    {{ 
      // ... your JSON structure ...
    }}
    <JSON_END>
    ```
"""
        try:
            analysis_response = await self.get_completion(
                analysis_prompt,
                temperature=0.5,
                top_k=40,
                top_p=0.9,
            )
            logger.debug(f"RAW analysis response: {analysis_response}")

            # Improved regex for JSON extraction
            json_match = re.search(
                r"<JSON_START>\s*({.*?})\s*<JSON_END>", analysis_response, re.DOTALL
            )

            if json_match:
                try:
                    # Extract and parse the JSON string
                    analysis_json = json_match.group(1)
                    analysis_data = json.loads(analysis_json)

                    # Ensure list fields are indeed lists
                    for field in ["issues", "suggestions"]:
                        if isinstance(analysis_data[field], str):
                            analysis_data[field] = [analysis_data[field]]

                    # Ensure 'changes' within 'required_corrections' is a list
                    if "changes" in analysis_data["required_corrections"]:
                        if isinstance(
                            analysis_data["required_corrections"]["changes"], str
                        ):
                            analysis_data["required_corrections"]["changes"] = [
                                analysis_data["required_corrections"]["changes"]
                            ]

                    return ReflectionResult(**analysis_data)

                except json.JSONDecodeError as e:
                    logger.error(
                        f"JSON decode error: {e}. Raw response: {analysis_response}"
                    )
                    try:
                        # Sanitization logic: Remove non-JSON characters
                        analysis_json = json_match.group(1)
                        analysis_json = re.sub(
                            r"[^{}\s\[\],:\"'\d\.-]", "", analysis_json
                        )
                        analysis_data = json.loads(analysis_json)

                        for field in ["issues", "suggestions"]:
                            if isinstance(analysis_data[field], str):
                                analysis_data[field] = [analysis_data[field]]

                        # Ensure 'changes' within 'required_corrections' is a list
                        if "changes" in analysis_data["required_corrections"]:
                            if isinstance(
                                analysis_data["required_corrections"]["changes"], str
                            ):
                                analysis_data["required_corrections"]["changes"] = [
                                    analysis_data["required_corrections"]["changes"]
                                ]
                        return ReflectionResult(**analysis_data)

                    except json.JSONDecodeError as e2:
                        logger.error(
                            f"JSON decode error after sanitization: {e2}. Sanitized response: {analysis_json}"
                        )
                        return ReflectionResult(  # Fallback result
                            is_successful=False,
                            confidence_score=0.0,
                            issues=["Analysis failed"],
                            suggestions=["Retry with simplified output"],
                            required_corrections={},
                            output_quality_score=0.0,
                        )

            else:
                try:
                    analysis_data = json.loads(analysis_response)

                    # Ensure list fields are indeed lists (same as above)
                    for field in ["issues", "suggestions"]:
                        if isinstance(analysis_data[field], str):
                            analysis_data[field] = [analysis_data[field]]

                    # Ensure 'changes' within 'required_corrections' is a list (same as above)
                    if "changes" in analysis_data["required_corrections"]:
                        if isinstance(
                            analysis_data["required_corrections"]["changes"], str
                        ):
                            analysis_data["required_corrections"]["changes"] = [
                                analysis_data["required_corrections"]["changes"]
                            ]

                    return ReflectionResult(**analysis_data)

                except json.JSONDecodeError as e:
                    logger.error(
                        f"JSON decode error (direct parsing): {e}. Raw response: {analysis_response}"
                    )

        except Exception as e:
            logger.error(f"Error in output analysis: {e}")
            return ReflectionResult(  # Fallback result
                is_successful=False,
                confidence_score=0.0,
                issues=["Analysis failed"],
                suggestions=["Retry with simplified output"],
                required_corrections={},
                output_quality_score=0.0,
            )

    async def consolidate_branch_with_llm(
        self, plan: Plan, action_id: str, completed_results: Dict
    ) -> Dict:
        """
        Consolidate a branch using the LLM, ensuring code is merged and narrative is preserved.
        Uses consolidated outputs from previous steps when available.
        """

        await self.emit_status("info", f"Consolidating branch {action_id}...", False)

        # Gather branch outputs with step information
        branch_outputs = []
        action = next(a for a in plan.actions if a.id == action_id)

        # Include the current action's raw output in the consolidation
        branch_outputs.append(
            {
                "id": action.id,
                "description": action.description,
                "result": completed_results[action_id][
                    "result"
                ],  # Raw output for current action
            }
        )

        for dep_id in action.dependencies:
            dep_action = next((a for a in plan.actions if a.id == dep_id), None)
            if dep_action:
                # Use consolidated output if available, otherwise fallback to raw output
                result = (
                    completed_results.get(dep_id, {})
                    .get("consolidated", {})
                    .get("result", "")
                )
                if not result:
                    result = completed_results.get(dep_id, {}).get("result", "")
                branch_outputs.append(
                    {
                        "id": dep_id,
                        "description": dep_action.description,
                        "result": result,
                    }
                )

        for dep_id in action.dependencies:
            dep_action = next((a for a in plan.actions if a.id == dep_id), None)
            if dep_action:
                branch_outputs.append(
                    {
                        "id": dep_id,
                        "description": dep_action.description,  # Include description
                        "result": completed_results.get(dep_id, {}).get("result", ""),
                    }
                )

        # Construct consolidation prompt with focus on merging and clarity
        consolidation_prompt = f"""
Consolidate these outputs into a single coherent unit for the goal: {plan.goal}

Branch Outputs:
{json.dumps(branch_outputs, indent=2)}

Consolidate these outputs into a single coherent unit for the goal: {plan.goal}

Branch Outputs:
{json.dumps(branch_outputs, indent=2)}

!!!EXTREMELY IMPORTANT!!!

*   **DO NOT** add any explanations, comments, or any content not present in the branch outputs.
*   **DO NOT** generate any new code, text, or any other type of content.
*   **DO NOT** mention steps, their origins, or any other meta-information not relevant to the task output.
*   Focus exclusively on creating a single, cohesive version of the given outputs, **WITHOUT ANY ADDITIONS OR MODIFICATIONS bEYOND THE PROVIDED IN CONTEXT.**

Requirements:

1.  **Strict Consolidation:**
    *   **ONLY** merge and integrate the provided outputs.
    *   Adhere to the **EXTREMELY IMPORTANT** instructions above.

2.  **Content Integration:**
    *   Intelligently merge all types of content, including code, text, diagrams, lists, etc., while adhering to Strict Consolidation rules.
    *   Preserve the functionality of code blocks while ensuring clear integration with other content types.
    *   For text, combine narrative elements seamlessly, maintaining the original intent and flow.
    *   For diagrams, ensure they are correctly placed and referenced within the consolidated output.

3.  **Conflict Resolution:**
    *   If outputs contradict each other, prioritize information from later steps.
    *   If there is ambiguity, resolve it in a way that aligns with the overall goal and the specific step, without adding new information.

4.  **Structure and Clarity:**
    *   Use appropriate formatting (headings, lists, code fences, etc.) to organize the consolidated output, based on the structure of the provided outputs.
    *   Ensure a clear and logical flow of information, derived from the organization of the original outputs.

5.  **Completeness and Accuracy:**
    *   Include ALL relevant content from the branch outputs without modification. Do not summarize or omit any details.
    *   Ensure the consolidated output accurately reflects the combined information from all steps, without any additions or alterations.

        """

        # Stream the consolidation process
        consolidated_output = ""
        await self.emit_replace("")
        await self.emit_replace_mermaid(plan)
        try:
            async for chunk in self.get_streaming_completion(
                [
                    {"role": "system", "content": f"Goal: {plan.goal}"},
                    {"role": "user", "content": consolidation_prompt},
                ],
                temperature=0.6,
                top_k=40,
                top_p=0.9,
            ):
                consolidated_output += chunk
                await self.emit_message(chunk)

            await self.emit_status("success", f"Consolidated branch {action_id}", True)

        except Exception as e:
            await self.emit_status(
                "error", f"Error consolidating branch {action_id}: {e}", True
            )
            raise  # Re-raise the exception to be handled elsewhere

        return {"result": consolidated_output}

    async def execute_plan(self, plan: Plan) -> Dict:
        """
        Execute the complete plan with dependency-based context and LLM-driven consolidation.
        """

        completed_results = {}
        in_progress = set()
        completed = set()
        step_counter = 1
        all_outputs = []  # Initialize all_outputs here
        all_dependencies = {}  # Store all dependencies for pruning

        # Build the all_dependencies dictionary
        for action in plan.actions:
            all_dependencies[action.id] = action.dependencies

        def prune_redundant_dependencies(dependencies, action_id, all_dependencies):
            """
            Recursively prunes redundant dependencies from a list of dependencies.

            Args:
              dependencies: A list of dependencies for the current action.
              action_id: The ID of the current action.
              all_dependencies: A dictionary mapping action IDs to their dependencies.

            Returns:
              A list of pruned dependencies.
            """
            pruned_dependencies = []
            for dependency in dependencies:
                if dependency not in all_dependencies:
                    pruned_dependencies.append(dependency)
                else:
                    # Recursively check if the dependency has any dependencies that are also
                    # dependencies of the current action.
                    redundant_dependencies = set(all_dependencies[dependency]) & set(
                        dependencies
                    )
                    if not redundant_dependencies:
                        pruned_dependencies.append(dependency)
                    else:
                        # If a redundant dependency is found, prune it and add its pruned
                        # dependencies to the list.
                        pruned_dependencies.extend(
                            prune_redundant_dependencies(
                                all_dependencies[dependency],
                                dependency,
                                all_dependencies,
                            )
                        )
            return pruned_dependencies

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
                    break
                await asyncio.sleep(0.1)
                continue

            while available and len(in_progress) < self.valves.CONCURRENT_ACTIONS:
                action = available.pop(0)
                in_progress.add(action.id)

                try:
                    # Prune redundant dependencies
                    pruned_dependencies = prune_redundant_dependencies(
                        action.dependencies, action.id, all_dependencies
                    )

                    # Gather ONLY direct dependency outputs, using pruned dependencies
                    context = {
                        dep: completed_results.get(dep, {})
                        .get("consolidated", {})
                        .get("result", "")
                        for dep in pruned_dependencies  # Use pruned dependencies here
                    }

                    # Execute action with dependency context and best result logic
                    result = await self.execute_action(
                        plan, action, context, step_counter
                    )

                    completed_results[action.id] = {
                        "result": result["result"]
                    }  # Store original result
                    completed.add(action.id)

                    # Consolidate the branch with LLM AFTER execution and marking as completed
                    if len(pruned_dependencies) > 0:
                        consolidated_output = await self.consolidate_branch_with_llm(
                            plan, action.id, completed_results
                        )
                    else:
                        consolidated_output = result
                    completed_results[action.id]["consolidated"] = consolidated_output

                    # Append BOTH original and consolidated results to all_outputs
                    all_outputs.append(
                        {
                            "step": step_counter,
                            "id": action.id,
                            "original_output": result["result"],
                            "consolidated_output": consolidated_output["result"],
                            "status": action.status,
                        }
                    )
                    step_counter += 1

                except Exception as e:
                    logger.error(f"Action {action.id} failed: {e}")
                    action.status = "failed"
                finally:
                    in_progress.remove(action.id)

            await self.emit_replace_mermaid(plan)

        plan.execution_summary = {
            "total_steps": len(plan.actions),
            "completed_steps": len(completed),
            "failed_steps": len(plan.actions) - len(completed),
            "execution_time": {
                "start": plan.actions[0].start_time if plan.actions else None,
                "end": plan.actions[-1].end_time if plan.actions else None,
            },
        }

        plan.metadata["execution_outputs"] = (
            all_outputs  # Assign all_outputs to metadata
        )

        return completed_results  # Return completed_results

    async def synthesize_results(self, plan: Plan, results: Dict) -> str:
        """
        Synthesize final results with comprehensive context gathering and merging.
        Uses consolidated outputs and handles dependencies effectively.
        """
        logger.debug("Starting enhanced result synthesis")

        # Collect all nodes without dependents (final nodes)
        final_nodes = []
        for action in plan.actions:
            # Check if this action has NO dependents
            is_final_node = not any(
                action.id in a.dependencies for a in plan.actions if a.id != action.id
            )

            if is_final_node:
                # Gather comprehensive context with consolidated outputs
                node_context = {
                    "id": action.id,
                    "description": action.description,
                    "consolidated_output": results.get(action.id, {})
                    .get("consolidated", {})
                    .get("result", ""),
                    "dependencies": [],
                }

                # Collect dependency details with consolidated outputs
                for dep_id in action.dependencies:
                    dep_action = next((a for a in plan.actions if a.id == dep_id), None)
                    if dep_action:
                        dep_context = {
                            "id": dep_id,
                            "description": dep_action.description,
                            "consolidated_output": results.get(dep_id, {})
                            .get("consolidated", {})
                            .get("result", ""),
                        }
                        node_context["dependencies"].append(dep_context)

                final_nodes.append(node_context)

        logger.debug(f"Found {len(final_nodes)} final nodes")

        # Handle cases with no final nodes
        if not final_nodes:
            error_msg = "No final outputs found in execution"
            await self.emit_status("error", error_msg, True)
            await self.emit_replace(error_msg)
            return error_msg

        # Handle single final node with completeness check
        if len(final_nodes) == 1:
            final_node = final_nodes[0]
            completeness_check = await self.check_output_completeness(
                {"output": {"result": final_node["consolidated_output"]}},
                plan.goal,
                final_node["dependencies"],
            )

            if completeness_check.get("is_complete", True):
                plan.final_output = final_node["consolidated_output"]
                await self.emit_status(
                    "success", "Final output verified complete", True
                )
                await self.emit_replace("")
                await self.emit_message(plan.final_output)
                await self.emit_replace_mermaid(plan)
                return plan.final_output

        # If multiple final nodes or completeness check failed, merge them
        merge_prompt = f"""
        Merge these final outputs into a single coherent result for the goal: {plan.goal}
        
        Final Outputs to merge:
        {json.dumps([
            {
                'id': node['id'],
                'description': node['description'],
                'output': node['consolidated_output'],
                'dependencies': [
                    {
                        'id': dep['id'], 
                        'description': dep['description'], 
                        'output': dep['consolidated_output']
                    } for dep in node['dependencies']
                ]
            } for node in final_nodes
        ], indent=2)}
        
        Requirements:
        1. Combine ALL outputs maintaining their complete functionality
        2. Preserve ALL implementation details from consolidated outputs
        3. Ensure proper integration between parts
        4. Use clear organization and structure
        5. Include ALL content - NO summarization
        6. Verify ALL dependencies are properly incorporated
        7. CHECK FOR MISSING COMPONENTS:
           - Ensure all CSS is included
           - Verify all required imports and dependencies
           - Check for any missing features or placeholders
        """

        try:
            complete_response = ""
            async for chunk in self.get_streaming_completion(
                [
                    {"role": "system", "content": f"Goal: {plan.goal}"},
                    {"role": "user", "content": merge_prompt},
                ],
                temperature=0.2,
                top_k=20,
                top_p=0.7,
            ):
                complete_response += chunk
                await self.emit_message(chunk)

            plan.final_output = complete_response
            await self.emit_status("success", "Successfully merged final outputs", True)
            await self.emit_replace("")
            await self.emit_replace_mermaid(plan)
            await self.emit_message(plan.final_output)
            return plan.final_output

        except Exception as e:
            error_message = (
                f"Failed to merge outputs: {str(e)}\n\nIndividual outputs:\n"
            )
            for node in final_nodes:
                error_message += (
                    f"\n--- Output for {node['id']} ---\n"
                    f"Description: {node['description']}\n"
                    f"Consolidated Output:\n{node['consolidated_output']}\n"
                )

            plan.final_output = error_message
            await self.emit_status(
                "error",
                "Failed to merge outputs - showing individual results",
                True,
            )
            await self.emit_message(error_message)
            await self.emit_replace(error_message)
            await self.emit_replace_mermaid(plan)
            return error_message

    async def check_output_completeness(
        self, final_output: Dict, goal: str, dependencies_info: Dict
    ) -> Dict:
        """Check if the final output is complete and incorporates all necessary dependencies"""

        check_prompt = f"""
        Analyze this final output for completeness:
    
        Goal: {goal}
        Output: {final_output['output'].get('result', '')}
        Dependencies: {json.dumps(dependencies_info, indent=2)}
    
        Verify:
        1. Output fully achieves the stated goal
        2. All dependencies are properly incorporated
        3. No missing components or references
        4. Implementation is complete and functional
        5. NO Missing features of any kind
        6. NO SUMARIZATION
        7. ALL CODE OR TEXT COMPLETE AND CORRECT
    
        Return a JSON object with:
        {{
            "is_complete": boolean,
            "confidence": float,
            "issues": [list of specific issues],
            "missing_dependencies": [list of missing dependency references]
        }}
        """

        try:
            response = await self.get_completion(
                check_prompt,
                temperature=0.4,  # Slightly lower for more focused analysis
                top_k=35,
                top_p=0.85,
            )
            result = json.loads(response)
            logger.debug(f"Completeness check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in completeness check: {e}")
            return {
                "is_complete": False,
                "confidence": 0.0,
                "issues": [f"Failed to verify completeness: {str(e)}"],
                "missing_dependencies": [],
            }

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
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None
    ) -> str:
        model = self.valves.MODEL
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        if __task__ and __task__ != TASKS.DEFAULT:
            response = await generate_chat_completion(
                self.__request__,
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        logger.debug(f"Pipe {name} received: {body}")
        self.__current_event_emitter__ = __event_emitter__
        self.__model__ = model

        goal = body.get("messages", [])[-1].get("content", "").strip()

        await self.emit_status("info", "Creating execution plan...", False)
        plan = await self.create_plan(goal)
        await self.emit_replace_mermaid(plan)  # Initial Mermaid graph

        await self.emit_status("info", "Executing plan...", False)
        results = await self.execute_plan(plan)

        await self.emit_status("info", "Creating final result...", False)
        final_result = await self.synthesize_results(plan, results)
        await self.emit_replace("")
        await self.emit_replace_mermaid(plan)
        await self.emit_message(final_result)
        # await self.emit_status("success", "Execution complete", True)
        return ""
