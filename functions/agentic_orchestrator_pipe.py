"""
title: Agentic Orchestrator Pipe
author: Haervwe
version: 0.9.0
"""
import json
import re
import logging
from typing import List, Dict, Optional, AsyncGenerator, Callable, Awaitable, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from open_webui.constants import TASKS
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import User, Users
from open_webui.models.tools import Tools
from dataclasses import dataclass
from fastapi import Request
from open_webui.constants import TASKS

# Utility to clean filter tags (from semantic_router)
def clean_thinking_tags(message: str) -> str:
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\\1>"
        r"|"
        r"\\|begin_of_thought\\|.*?\\|end_of_thought\\|",
        re.DOTALL,
    )
    return re.sub(pattern, "", message).strip()

# Logger setup
def setup_logger():
    logger = logging.getLogger("AgenticOrchestratorPipe")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name("AgenticOrchestratorPipe")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

class Action(BaseModel):
    id: str
    type: str
    description: str
    params: Dict = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    output: Optional[Dict] = None
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    agent_profile: Optional[Dict] = None

class Plan(BaseModel):
    goal: str
    actions: List[Action]
    metadata: Dict = Field(default_factory=dict)
    final_output: Optional[str] = None
    execution_summary: Optional[Dict] = None

class ReflectionResult(BaseModel):
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
        ORCHESTRATOR_MODEL: str = Field(
            default="", description="Model to use for orchestration (planning, reasoning)"
        )
        ACTION_MODEL: str = Field(
            default="", description="Model to use for tool calling and step execution"
        )
        WRITER_MODEL: str = Field(
            default="", description="Model to use for writing/creative tasks"
        )
        CODING_MODEL: str = Field(
            default="", description="Model to use for code generation (optional)"
        )
        ACTION_PROMPT_REQUIREMENTS_TEMPLATE: str = Field(
            default="""Requirements:\n1. Provide implementation that DIRECTLY achieves the step goal\n2. Include ALL necessary components\n3. For code:\n   - Must be complete and runnable\n   - All variables must be properly defined\n   - No placeholder functions or TODO comments\n4. For text/documentation:\n   - Must be specific and actionable\n   - Include all relevant details\n""",
            description="General requirements for task completions, used in ALL action steps."
        )
        MAX_RETRIES: int = Field(default=3, description="Maximum number of retry attempts")
        CONCURRENT_ACTIONS: int = Field(default=1, description="Maximum concurrent actions")
        ACTION_TIMEOUT: int = Field(default=300, description="Action timeout in seconds")
        TOOL_SELECTION_STATUS: bool = Field(default=True, description="Show tool selection status updates")

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.current_output = ""

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "agentic-orchestrator-pipe", "name": "Agentic Orchestrator Pipe"}]

    async def get_completion(self, prompt: str, model: str = None, tools: list = None, user: User = None, request: Request = None) -> str:
        payload = {
            "model": model or self.valves.ORCHESTRATOR_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        response = await generate_chat_completion(request, payload, user=user)
        return response

    async def create_plan(self, goal: str, user: User, request: Request) -> Plan:
        prompt = f"""
Given the goal: {goal}
Generate a detailed execution plan by breaking it down into atomic, sequential steps. Each step should be clear, actionable, and have explicit dependencies.\nReturn a JSON object with exactly this structure:\n{{\n    "goal": "<original goal>",\n    "actions": [\n        {{\n            "id": "<unique_id>",\n            "type": "<category_of_action>",\n            "description": "<specific_actionable_task>",\n            "params": {{"param_name": "param_value"}},\n            "dependencies": ["<id_of_prerequisite_step>"]\n        }}\n    ],\n    "metadata": {{\n        "estimated_time": "<minutes>",\n        "complexity": "<low|medium|high>"\n    }}\n}}\nReturn ONLY the JSON object. Do not include explanations or additional text.\n"""
        for attempt in range(self.valves.MAX_RETRIES):
            try:
                plan_json = await self.get_completion(prompt, model=self.valves.ORCHESTRATOR_MODEL, user=user, request=request)
                plan_data = json.loads(plan_json[plan_json.find("{"):plan_json.rfind("}")+1])
                plan = Plan(**plan_data)
                return plan
            except Exception as e:
                logger.error(f"Plan creation attempt {attempt+1} failed: {e}")
        raise RuntimeError(f"Failed to create plan after {self.valves.MAX_RETRIES} attempts")

    def get_tools_for_action(self, action: Action, model_meta: dict) -> list:
        # Fetch available tools for the model (from semantic_router/AutoTool logic)
        available_tool_ids = model_meta.get("info", {}).get("meta", {}).get("toolIds", [])
        # Return tool IDs (not dicts)
        return [tid for tid in available_tool_ids]

    def get_agent_profile(self, action: Action, model_meta: dict) -> dict:
        # Select model for the agent profile based on action type
        if action.type == "writing":
            model = self.valves.WRITER_MODEL or self.valves.ACTION_MODEL or self.valves.ORCHESTRATOR_MODEL
        elif action.type == "code":
            model = self.valves.CODING_MODEL or self.valves.ACTION_MODEL or self.valves.ORCHESTRATOR_MODEL
        else:
            model = self.valves.ACTION_MODEL or self.valves.ORCHESTRATOR_MODEL
        tools = self.get_tools_for_action(action, model_meta)
        return {"model": model, "tools": tools}
    async def execute_action(self, plan: Plan, action: Action, results: Dict, step_number: int, user: User, request: Request, model_meta: dict) -> Dict:
        action.start_time = datetime.now().strftime("%H:%M:%S")
        action.status = "in_progress"
        context = {dep: results.get(dep, {}) for dep in action.dependencies}
        requirements = self.valves.ACTION_PROMPT_REQUIREMENTS_TEMPLATE
        agent_profile = self.get_agent_profile(action, model_meta)
        action.agent_profile = agent_profile
        tools = agent_profile["tools"]
        base_prompt = (
            f"Execute step {step_number}: {action.description}\n"
            f"Overall Goal: {plan.goal}\n"
            f"Context:\n- Parameters: {json.dumps(action.params)}\n- Previous Results: {json.dumps(context)}\n"
            f"{requirements}\n"
            f"Focus ONLY on this specific step's output.\n"
        )
        messages = [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": base_prompt},
        ]
        max_retries = self.valves.MAX_RETRIES
        best_output = None
        best_reflection = None
        best_quality_score = -1
        for attempt in range(max_retries):
            payload = {
                "model": agent_profile["model"],
                "messages": messages,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
            response = await generate_chat_completion(request, payload, user=user)
            content = response["choices"][0]["message"]["content"]
            content = clean_thinking_tags(content)
            # The backend handles tool-calling and returns the final output
            # Reflection/analysis
            reflection = await self.analyze_output(plan, action, content, attempt, user, request)
            if reflection.output_quality_score > best_quality_score:
                best_output = content
                best_reflection = reflection
                best_quality_score = reflection.output_quality_score
            if reflection.is_successful and reflection.output_quality_score >= 0.7:
                break  # Acceptable output
        action.status = "completed"
        action.end_time = datetime.now().strftime("%H:%M:%S")
        action.output = {"result": best_output, "reflection": best_reflection.dict() if best_reflection else {}}
        await self.emit_status("success", f"Action {action.id} completed", True)
        return action.output
    
    async def analyze_output(self, plan: Plan, action: Action, output: str, attempt: int, user: User, request: Request) -> ReflectionResult:
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
    "issues": ["specific issue 1", ...],
    "suggestions": ["specific correction 1", ...],
    "required_corrections": {{"component": "what needs to change", "changes": ["a concise string describing the required change 1", ...]}},
    "output_quality_score": float between 0-1,
    "analysis": {{
        "completeness": float between 0-1,
        "correctness": float between 0-1,
        "clarity": float between 0-1,
        "specific_problems": ["detailed problem description 1", ...]
    }}
}}
Return ONLY the JSON object. Do not include explanations or additional text.
"""
        response = await self.get_completion(analysis_prompt, model=self.valves.ORCHESTRATOR_MODEL, user=user, request=request)
        try:
            # Extract JSON from response
            text = response["choices"][0]["message"]["content"]
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            data = json.loads(json_str)
            return ReflectionResult(**data)
        except Exception as e:
            logger.error(f"Reflection parse error: {e}")
            # Fallback: mark as failed
            return ReflectionResult(is_successful=False, confidence_score=0.0, issues=["Reflection parse error"], suggestions=[], required_corrections={}, output_quality_score=0.0)

    async def generate_mermaid(self, plan: Plan) -> str:
        mermaid = ["graph TD", f'    Start[\"Goal: {plan.goal[:30]}...\"]']
        status_emoji = {"pending": "⭕", "in_progress": "⚙️", "completed": "✅", "failed": "❌"}
        styles = []
        for action in plan.actions:
            label = f"{action.id}: {action.description[:20]} {status_emoji.get(action.status, '')}"
            mermaid.append(f'    action_{action.id}[\"{label}\"]')
            styles.append(f"    style action_{action.id} fill:#f9f,stroke:#333,stroke-width:2px;")
        mermaid.append(f"    Start --> action_{plan.actions[0].id}")
        for action in plan.actions:
            for dep in action.dependencies:
                mermaid.append(f"    action_{dep} --> action_{action.id}")
        mermaid.extend(styles)
        return "\n".join(mermaid)

    async def execute_plan(self, plan: Plan, user: User, request: Request, model_meta: dict) -> Dict:
        completed_results = {}
        step_counter = 1
        for action in plan.actions:
            result = await self.execute_action(plan, action, completed_results, step_counter, user, request, model_meta)
            completed_results[action.id] = result
            step_counter += 1
        return completed_results

    async def emit_status(self, level: str, message: str, done: bool):
        if hasattr(self, "__current_event_emitter__") and self.__current_event_emitter__:
            await self.__current_event_emitter__({
                "type": "status",
                "data": {"description": message, "done": done, "level": level},
            })

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        
        if __task__ and __task__ != TASKS.DEFAULT:
            response = await generate_chat_completion(
                self.__request__,
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"
        
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__) if isinstance(__user__, dict) else __user__
        self.__model__ = __model__
        user = self.__user__
        request = __request__
        model_meta = __model__ or {}
        goal = body.get("goal") or ""
        await self.emit_status("info", "Creating plan...", False)
        plan = await self.create_plan(goal, user, request)
        await self.emit_status("info", "Executing plan...", False)
        results = await self.execute_plan(plan, user, request, model_meta)
        plan.final_output = results.get(plan.actions[-1].id, {}).get("result", "")
        plan.execution_summary = results
        await self.emit_status("success", "Plan execution complete.", True)
        return {
            "plan": plan.dict(),
            "mermaid": await self.generate_mermaid(plan),
            "final_output": plan.final_output,
        }
