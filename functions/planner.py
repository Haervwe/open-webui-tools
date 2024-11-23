"""
title: Planner
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/open-webui
version: 0.1
"""

import logging
import json
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
from datetime import datetime
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from dataclasses import dataclass

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
        return [
            {"id": f"{name}-{self.valves.MODEL}", "name": f"{name} {self.valves.MODEL}"}
        ]

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

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
        You are a helpful and harmless AI assistant. Create a step-by-step plan for: {goal}

        Return ONLY a JSON object with this structure:
        {{
            "goal": "goal description",
            "actions": [
                {{
                    "id": "unique_id",
                    "type": "step_type",
                    "description": "detailed step description",
                    "params": {{"key": "value"}},
                    "dependencies": ["dependent_step_id"]
                }}
            ],
            "metadata": {{
                "estimated_time": "minutes",
                "complexity": "low|medium|high"
            }}
        }}

        Break down the goal into logical, sequential steps with clear dependencies.
        Do not include any additional text or explanations outside the JSON object.
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
        """Execute a single action with streaming output"""
        action.start_time = datetime.now().strftime("%H:%M:%S")
        action.status = "in_progress"

        context = {dep: results.get(dep, {}) for dep in action.dependencies}

        prompt = f"""
        Execute step {step_number}: {action.description}
        Goal: {plan.goal}
        Parameters: {json.dumps(action.params)}
        Previous context: {json.dumps(context)}
        
        Provide ONLY the output of the execution.
        Do not include any additional text or explanations.
        If the output is code, enclose it in a code block.
        Ensure that the code does not reference variables not defined within the scope of this step.
        """

        reflection_max_retries = (
            self.valves.MAX_RETRIES
        )  # Maximum retries for reflection steps
        reflection_attempts = 0

        while reflection_attempts <= reflection_max_retries:
            try:
                await self.emit_status(
                    "info",
                    f"Starting action {action.id}: {action.description} (Attempt {reflection_attempts + 1}/{reflection_max_retries})",
                    False,
                )
                action.status = "in_progress"
                await self.emit_replace("")
                await self.emit_replace_mermaid(plan)

                complete_response = ""
                async for chunk in self.get_streaming_completion(
                    [
                        {"role": "system", "content": f"Goal: {plan.goal}"},
                        {"role": "user", "content": prompt},
                    ]
                ):
                    complete_response += chunk
                    await self.emit_message(chunk)  # Emit each chunk as a message

                action.status = "completed"
                action.end_time = datetime.now().strftime("%H:%M:%S")
                action.output = {
                    "result": complete_response
                }  # Store only the relevant output

                # Reflection step after completing the action
                await self.emit_status(
                    "info", "Analyzing intermediate result...", False
                )
                await asyncio.sleep(1)  # Simulate some thinking time

                reflection_prompt = f"""
                Was the following action completed **successfully**? 
                
                **Goal:** {plan.goal}
                **Action:** {action.description}
                **Expected Output (based on instructions):** [Describe the expected output clearly based on the action's purpose] 
                **Actual Output:** {complete_response}
                
                Answer with a simple 'yes' or 'no'.
                """
                reflection_result = await self.get_completion(reflection_prompt)

                if reflection_result.strip().lower() == "no":
                    if reflection_attempts < reflection_max_retries:
                        await self.emit_status(
                            "warning",
                            f"Action {action.id} was not completed correctly. Retrying reflection... (Attempt {reflection_attempts + 1}/{reflection_max_retries})",
                            False,
                        )
                        # Generate a re-prompt with corrections
                        correction_prompt = f"""
                        The previous action was not completed correctly. Please provide a corrected output for:
                        Goal: {plan.goal}
                        Action: {action.description}
                        Parameters: {json.dumps(action.params)}
                        Previous context: {json.dumps(context)}
        
                        Provide ONLY the corrected output.
                        Do not include any additional text or explanations.
                        If the output is code, enclose it in a code block.
                        Ensure that the code does not reference variables not defined within the scope of this step.
                        """
                        prompt = correction_prompt
                        reflection_attempts += 1
                        continue  # Retry with the corrected prompt
                    else:
                        await self.emit_status(
                            "warning",
                            f"Action {action.id} failed after {reflection_max_retries} reflection attempts.",
                            True,
                        )
                        action.status = "failed"
                        action.end_time = datetime.now().strftime("%H:%M:%S")
                        raise Exception("Action failed after reflection retries")
                else:
                    await self.emit_status(
                        "success", f"Action {action.id} completed successfully.", True
                    )
                    return action.output

            except Exception as e:
                action.status = "failed"
                action.end_time = datetime.now().strftime("%H:%M:%S")
                logger.error(f"Error executing action {action.id}: {e}")
                await self.emit_status(
                    "error",
                    f"Action {action.id} failed.",
                    True,
                )
                raise

        # If maximum reflection retries are reached, return the last message
        action.status = "failed"
        action.end_time = datetime.now().strftime("%H:%M:%S")
        await self.emit_status(
            "error",
            f"Action {action.id} failed after {reflection_max_retries} reflection attempts.",
            True,
        )
        return action.output

    async def execute_plan(self, plan: Plan) -> Dict:
        """Execute the complete plan with visualization"""
        results = {}
        in_progress = set()
        completed = set()
        step_counter = 1

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

        return results

    async def synthesize_results(self, plan: Plan, results: Dict) -> str:
        # ...
        prompt = f"""
        Create a final result for goal: {plan.goal}
    
        Plan execution summary:
        - Total steps: {plan.execution_summary['total_steps']}
        - Completed: {plan.execution_summary['completed_steps']}
        - Failed: {plan.execution_summary['failed_steps']}
        - Time: {plan.execution_summary['execution_time']['start']} to {plan.execution_summary['execution_time']['end']}
    
        Results by step:
        {json.dumps({action.id: action.output for action in plan.actions}, indent=2)}
    
        Provide a comprehensive result by AGGREGATING the outputs of all steps.
        If code was generated, provide the COMPLETE and CORRECT code.
        Do not summarize.
        Format in clear markdown with sections and bullet points.
        """

        for attempt in range(self.valves.MAX_RETRIES):
            try:
                await self.emit_status("info", "Creating final result...", False)
                result = ""
                async for chunk in self.get_streaming_completion(
                    [
                        {"role": "system", "content": f"Goal: {plan.goal}"},
                        {"role": "user", "content": prompt},
                    ]
                ):
                    result += chunk
                    await self.emit_message(
                        chunk
                    )  # Replace the previous chunk with the new one

                plan.final_output = result

                # Reflection step after getting the final result
                await self.emit_status("info", "Analyzing final result...", False)
                await asyncio.sleep(1)  # Simulate some thinking time

                # Check if the final result was correct
                reflection_prompt = f"""
                Was the final result for the goal '{plan.goal}' correct?
                Final result: {result}
    
                Answer with a simple 'yes' or 'no'.
                """

                reflection_result = await self.get_completion(reflection_prompt)

                if reflection_result.strip().lower() == "no":
                    await self.emit_status(
                        "warning",
                        "Final result was not correct. Retrying... (Attempt {attempt + 1}/{self.valves.MAX_RETRIES})",
                        False,
                    )
                    # Generate a re-prompt with corrections
                    correction_prompt = f"""
                    The final result was not correct. Please provide a corrected final result for the goal:
                    Goal: {plan.goal}
                    Plan execution summary:
                    - Total steps: {plan.execution_summary['total_steps']}
                    - Completed: {plan.execution_summary['completed_steps']}
                    - Failed: {plan.execution_summary['failed_steps']}
                    - Time: {plan.execution_summary['execution_time']['start']} to {plan.execution_summary['execution_time']['end']}
    
                    Results by step:
                    {json.dumps({action.id: action.output for action in plan.actions}, indent=2)}
    
                    Provide a comprehensive result by AGGREGATING the outputs of all steps.
                    If code was generated, provide the COMPLETE and CORRECT code.
                    Do not summarize.
                    Format in clear markdown with sections and bullet points.
                    """

                    prompt = correction_prompt
                    continue  # Retry with the corrected prompt

                await self.emit_status(
                    "success", "Final result created successfully.", True
                )
                return plan.final_output

            except Exception as e:
                logger.error(
                    f"Error synthesizing results (attempt {attempt + 1}/{self.valves.MAX_RETRIES}): {e}"
                )
                await self.emit_status(
                    "error",
                    "Error creating final result. Retrying... (Attempt {attempt + 1}/{self.valves.MAX_RETRIES})",
                    False,
                )
                if attempt < self.valves.MAX_RETRIES - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    await self.emit_status(
                        "error",
                        f"Failed to create final result after {self.valves.MAX_RETRIES} attempts.",
                        True,
                    )
                    raise

        raise RuntimeError(
            f"Failed to synthesize results after {self.valves.MAX_RETRIES} attempts"
        )

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
        model = self.resolve_model(body)
        self.__user__ = User(**__user__)
        if __task__ == TASKS.TITLE_GENERATION:
            response = await generate_chat_completions(
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=__user__,
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

        await self.emit_status("success", "Execution complete", True)
