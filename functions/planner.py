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

        reflection_max_retries = self.valves.MAX_RETRIES
        reflection_attempts = 0
        best_output = None
        best_reflection = None
        best_quality_score = -1

        while reflection_attempts <= reflection_max_retries:
            try:
                await self.emit_status(
                    "info",
                    f"Starting Attempt {reflection_attempts + 1}/{reflection_max_retries + 1} for action {action.id}",
                    False,
                )
                action.status = "in_progress"
                await self.emit_replace("")
                await self.emit_replace_mermaid(plan)

                if reflection_attempts > 0 and best_reflection:
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
                except Exception as api_error:
                    action.status = "failed"
                    action.end_time = datetime.now().strftime("%H:%M:%S")
                    await self.emit_status(
                        "error",
                        f"API error in action {action.id}",
                        True,
                    )
                    raise

                current_output = {"result": complete_response}

                await self.emit_status(
                    "info",
                    f"Analyzing output ...",
                    False,
                )

                current_reflection = await self.analyze_output(
                    plan=plan,
                    action=action,
                    output=complete_response,
                    attempt=reflection_attempts,
                )

                await self.emit_status(
                    "info",
                    f"Analyzing output (Quality Score: {current_reflection.output_quality_score:.2f})",
                    False,
                )

                # Update best output if current quality is higher
                if current_reflection.output_quality_score > best_quality_score:
                    best_output = current_output
                    best_reflection = current_reflection
                    best_quality_score = current_reflection.output_quality_score

                # Only continue if execution actually failed
                if not current_reflection.is_successful:
                    if reflection_attempts < reflection_max_retries:
                        await self.emit_status(
                            "warning",
                            f"Output needs improvement. Retrying...",
                            False,
                        )
                        reflection_attempts += 1
                        continue

                # If we reach here, either execution was successful or we're out of retries
                action.status = "completed"
                action.end_time = datetime.now().strftime("%H:%M:%S")
                action.output = best_output
                await self.emit_status(
                    "success",
                    f"Action completed with best output (Quality: {best_quality_score:.2f})",
                    True,
                )
                return best_output

            except Exception as e:
                action.status = "failed"
                action.end_time = datetime.now().strftime("%H:%M:%S")
                await self.emit_status("error", f"Action failed: {str(e)}", True)
                raise

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

    async def execute_plan(self, plan: Plan) -> Dict:
        """Execute the complete plan with visualization"""
        results = {}
        in_progress = set()
        completed = set()
        step_counter = 1

        all_outputs = []

        async def can_execute(action: Action) -> bool:
            return all(dep in completed for dep in action.dependencies)

        while len(completed) < len(plan.actions):
            await self.emit_replace_mermaid(
                plan
            )  # Always replace the old mermaid graph

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
                    result = await self.execute_action(
                        plan, action, results, step_counter
                    )
                    results[action.id] = result
                    completed.add(action.id)
                    step_counter += 1
                except Exception as e:
                    logger.error(f"Action {action.id} failed: {e}")

                in_progress.remove(action.id)

        await self.emit_replace_mermaid(plan)  # Update the final state of the graph

        plan.execution_summary = {
            "total_steps": len(plan.actions),
            "completed_steps": len(completed),
            "failed_steps": len(plan.actions) - len(completed),
            "execution_time": {
                "start": plan.actions[0].start_time,
                "end": plan.actions[-1].end_time,
            },
        }
        if action.output:
            all_outputs.append(
                {
                    "step": step_counter,
                    "id": action.id,
                    "output": action.output,
                    "status": action.status,
                }
            )

        plan.metadata["execution_outputs"] = all_outputs

        return results

    async def synthesize_results(self, plan: Plan, results: Dict) -> str:
        """Synthesize final results focusing on final outputs or merging dangling parts"""

        # Find final outputs or dangling parts
        final_outputs = []
        for action in reversed(plan.actions):
            # If this action has no dependents, it's a final output
            is_final = not any(
                action.id in dep_action.dependencies for dep_action in plan.actions
            )
            if is_final and action.output:
                final_outputs.append(action.output)

        if not final_outputs:
            await self.emit_status(
                "error",
                "No final outputs found in execution",
                True,
            )
            return "No final outputs were generated during execution"

        # If there's only one final output, use it directly
        if len(final_outputs) == 1:
            plan.final_output = final_outputs[0].get("result", "")
            await self.emit_status(
                "success",
                "Using single final output",
                True,
            )
            await self.emit_replace(plan.final_output)
            await self.emit_replace_mermaid(plan)
            return plan.final_output

        # If multiple final outputs, create merge prompt
        merge_prompt = f"""
        Merge these final outputs into a single coherent result for the goal: {plan.goal}
        
        Final Outputs to merge:
        {json.dumps([output.get("result", "") for output in final_outputs], indent=2)}
        
        Requirements:
        1. Combine all outputs maintaining their complete functionality
        2. Preserve all implementation details
        3. Ensure proper integration between parts
        4. Use clear organization and structure
        5. Include ALL content - no summarization
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
            await self.emit_status(
                "success",
                "Successfully merged final outputs",
                True,
            )
            await self.emit_replace(plan.final_output)
            await self.emit_replace_mermaid(plan)
            return plan.final_output

        except Exception as e:
            error_message = (
                f"Failed to merge outputs: {str(e)}\n\nIndividual outputs:\n"
            )
            for i, output in enumerate(final_outputs, 1):
                error_message += f"\n--- Output {i} ---\n{output.get('result', '')}\n"

            plan.final_output = error_message
            await self.emit_status(
                "error",
                "Failed to merge outputs - showing individual results",
                True,
            )
            await self.emit_replace(plan.final_output)
            await self.emit_replace_mermaid(plan)
            return plan.final_output

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
        await self.synthesize_results(plan, results)  # No need to return here

        # await self.emit_status("success", "Execution complete", True)
        # return ""