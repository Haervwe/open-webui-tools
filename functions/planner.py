"""
title: Planner
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/open-webui
version: 0.2
"""

import logging
import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Callable, Awaitable, Union
from pydantic import BaseModel, Field
from datetime import datetime
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from dataclasses import dataclass
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
            default=None, description="Model to use (model id from ollama)"
        )
        MAX_RETRIES: int = Field(
            default=3, description="Maximum number of retry attempts"
        )
        CONCURRENT_ACTIONS: int = Field(
            default=2, description="Maximum concurrent actions"
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
    ) -> AsyncGenerator[str, None]:
        try:
            form_data = {
                "model": self.__model__,
                "messages": messages,
                "stream": True,
            }
            response = await generate_chat_completions(
                form_data,
                user=self.__user__,
                bypass_filter=False,
            )

            # Ensure the response has body_iterator
            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")

            async for chunk in response.body_iterator:
                # Use the updated chunk content method
                for part in self.get_chunk_content(chunk):
                    yield part

        except Exception as e:
            raise RuntimeError(f"Streaming completion failed: {e}")

    def get_chunk_content(self, chunk):
        # Directly process the chunk since it's already a string
        chunk_str = chunk
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

    async def get_completion(self, prompt: str) -> str:
        response = await generate_chat_completions(
            {
                "model": self.__model__,
                "messages": [{"role": "user", "content": prompt}],
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
   - Each step should produce complete, functional code

Return ONLY the JSON object. Do not include explanations or additional text.
"""

        for attempt in range(self.valves.MAX_RETRIES):
            try:
                result = await self.get_completion(prompt)
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

    async def execute_action(
        self, plan: Plan, action: Action, results: Dict, step_number: int
    ) -> Dict:
        """Execute action with enhanced reflection, always returning best output"""
        action.start_time = datetime.now().strftime("%H:%M:%S")
        action.status = "in_progress"

        context = {dep: results.get(dep, {}) for dep in action.dependencies}

        base_prompt = f"""
            Execute step {step_number}: {action.description}
            Overall Goal: {plan.goal}
        
            Context:
            - Parameters: {json.dumps(action.params)}
            - Previous Results: {json.dumps(context)}
        
            Requirements:
            1. Provide implementation that DIRECTLY achieves the step goal
            2. Include ALL necessary components
            3. For code:
               - Must be complete and runnable
               - All variables must be properly defined
               - No placeholder functions or TODO comments
            4. For text/documentation:
               - Must be specific and actionable
               - Include all relevant details
        
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
                        ]
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
        self, plan: Plan, action: Action, output: str, attempt: int
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
            analysis_response = await self.get_completion(analysis_prompt)
            logger.debug(f"RAW analysis response: {analysis_response}")

            # Extract JSON using regex, considering the tags
            json_match = re.search(
                r"<JSON_START>\s*(.*?)\s*<JSON_END>", analysis_response, re.DOTALL
            )
            if json_match:
                analysis_data = json.loads(
                    json_match.group(1)
                )  # Parse the captured group

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
            else:
                # Fallback to basic analysis if JSON extraction fails
                logger.error(
                    f"Error extracting JSON from analysis response: {analysis_response}"
                )
                return ReflectionResult(
                    is_successful=False,
                    confidence_score=0.0,
                    issues=["Analysis failed"],
                    suggestions=["Retry with simplified output"],
                    required_corrections={},
                    output_quality_score=0.0,
                )

        except Exception as e:
            logger.error(f"Error in output analysis: {e}")
            return ReflectionResult(
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
        """

        # Gather branch outputs
        branch_outputs = []
        action = next(a for a in plan.actions if a.id == action_id)
        for dep_id in action.dependencies:
            branch_outputs.append(
                {
                    "id": dep_id,
                    "result": completed_results.get(dep_id, {}).get("result", ""),
                }
            )
        branch_outputs.append(
            {"id": action_id, "result": completed_results[action_id]["result"]}
        )

        # Construct consolidation prompt
        consolidation_prompt = f"""
        Consolidate these outputs into a single coherent unit for the goal: {plan.goal}

        Branch Outputs:
        {json.dumps(branch_outputs, indent=2)}

        Requirements:
        1. Merge code blocks by language, preserving functionality.
        2. Preserve narrative text (non-code content) in sequence.
        3. Ensure proper integration between code and narrative.
        4. Use clear organization and structure.
        5. Include ALL content - no summarization.
        6. code modifications must be outpute in a SINGLE CODE BLOCK never output variations or previus versions, the latest complete modified version of each code file or idea is the correct.
        7. Do not mention steps or make comentaries. your task is consolidation on an iterative process. so focus on making a sole version with the latest or more relevant/good  code/information.
        """

        # Stream the consolidation process
        consolidated_output = ""
        await self.emit_replace("")
        await self.emit_replace_mermaid(plan)
        async for chunk in self.get_streaming_completion(
            [
                {"role": "system", "content": f"Goal: {plan.goal}"},
                {"role": "user", "content": consolidation_prompt},
            ]
        ):
            consolidated_output += chunk
            await self.emit_message(chunk)

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
                    # Gather ONLY direct dependency outputs
                    context = {
                        dep: completed_results.get(dep, {})
                        .get("consolidated", {})
                        .get("result", "")
                        for dep in action.dependencies
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
                    consolidated_output = await self.consolidate_branch_with_llm(
                        plan, action.id, completed_results
                    )
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
        """Synthesize final results focusing on final outputs or merging dangling parts"""
        logger.debug("Starting result synthesis")

        # Find final outputs or dangling parts
        final_outputs = []
        dependencies_info = {}

        for action in reversed(plan.actions):
            # If this action has no dependents, it's a final output
            is_final = not any(
                action.id in dep_action.dependencies for dep_action in plan.actions
            )

            # Log dependencies for debugging
            if action.dependencies:
                dependencies_info[action.id] = {
                    "deps": action.dependencies,
                    "content": {
                        dep: (
                            results.get(dep, {}).get("result", "")[:100] + "..."
                            if results.get(dep, {}).get("result")
                            else "No content"
                        )
                        for dep in action.dependencies
                    },
                }
                logger.debug(
                    f"Dependencies for {action.id}: {dependencies_info[action.id]}"
                )

            if is_final and action.output:
                final_outputs.append(
                    {
                        "id": action.id,
                        "output": action.output,
                        "dependencies": dependencies_info.get(action.id, {}),
                    }
                )

        logger.debug(f"Found {len(final_outputs)} final outputs")

        if not final_outputs:
            error_msg = "No final outputs found in execution"
            await self.emit_status(
                "error",
                error_msg,
                True,
            )
            await self.emit_replace(error_msg)
            return error_msg

        # If there's only one final output, verify its completeness
        if len(final_outputs) == 1:
            final_output = final_outputs[0]
            completeness_check = await self.check_output_completeness(
                final_output, plan.goal, dependencies_info
            )

            if completeness_check["is_complete"]:
                plan.final_output = final_output["output"].get("result", "")
                await self.emit_status(
                    "success",
                    "Final output verified complete",
                    True,
                )
                # Ensure the output is visible by using both emit methods
                await self.emit_message(plan.final_output)
                await self.emit_replace(plan.final_output)
                await self.emit_replace_mermaid(plan)
                return plan.final_output
            else:
                logger.warning(
                    f"Output completeness check failed: {completeness_check['issues']}"
                )

        # If multiple outputs or completeness check failed, create merge prompt
        merge_prompt = f"""
        Merge these final outputs into a single coherent result for the goal: {plan.goal}
        
        Final Outputs to merge:
        {json.dumps([{
            'output': output['output'].get('result', ''),
            'dependencies': output['dependencies']
        } for output in final_outputs], indent=2)}
        
        Requirements:
        1. Combine all outputs maintaining their complete functionality
        2. Preserve all implementation details
        3. Ensure proper integration between parts
        4. Use clear organization and structure
        5. Include ALL content - no summarization
        6. Verify all dependencies are properly incorporated
        """

        try:
            complete_response = ""
            async for chunk in self.get_streaming_completion(
                [
                    {"role": "system", "content": f"Goal: {plan.goal}"},
                    {"role": "user", "content": merge_prompt},
                ]
            ):
                complete_response += chunk
                await self.emit_message(chunk)

            plan.final_output = complete_response
            # Ensure visibility of final output
            await self.emit_status(
                "success",
                "Successfully merged final outputs",
                True,
            )
            await self.emit_message(plan.final_output)
            await self.emit_replace(plan.final_output)
            await self.emit_replace_mermaid(plan)
            return plan.final_output

        except Exception as e:
            error_message = (
                f"Failed to merge outputs: {str(e)}\n\nIndividual outputs:\n"
            )
            for i, output in enumerate(final_outputs, 1):
                error_message += (
                    f"\n--- Output {i} ---\n{output['output'].get('result', '')}\n"
                )

            plan.final_output = error_message
            await self.emit_status(
                "error",
                "Failed to merge outputs - showing individual results",
                True,
            )
            # Ensure error message is visible
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
    
        Return a JSON object with:
        {{
            "is_complete": boolean,
            "confidence": float,
            "issues": [list of specific issues],
            "missing_dependencies": [list of missing dependency references]
        }}
        """

        try:
            response = await self.get_completion(check_prompt)
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
    ) -> str:
        model = self.valves.MODEL
        self.__user__ = User(**__user__)
        if __task__ == TASKS.TITLE_GENERATION:
            response = await generate_chat_completions(
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
        final_result = await self.synthesize_results(
            plan, results
        )  # No need to return here
        await self.emit_replace("")
        await self.emit_replace_mermaid(plan)
        await self.emit_message(final_result)
        # await self.emit_status("success", "Execution complete", True)
        return ""
