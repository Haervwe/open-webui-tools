"""
title: Letta_Agent_Connector
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.4
description: A pipe to connect with Letta agents, enabling seamless integration of autonomous agents into Open WebUI conversations. Supports task-specific processing and maintains conversation context while communicating with the agent API.
"""

import logging
from typing import Dict, List, Callable, Awaitable
from pydantic import BaseModel, Field
from dataclasses import dataclass
import aiohttp
import json
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import Users
import asyncio
import time

name = "Letta Agent"


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


class Pipe:
    class Valves(BaseModel):
        Agent_ID: str = Field(
            default="Demether",
            description="The ID of the Letta agent to communicate with",
        )
        API_URL: str = Field(
            default="http://localhost:8283",
            description="Base URL for the Letta agent API",
        )
        API_Token: str = Field(
            default="", description="Bearer token for API authentication"
        )
        Task_Model: str = Field(
            default="",
            description="Model to use for title/tags generation tasks. If empty, uses the default model.",
        )
        Custom_Name: str = Field(
            default="",
            description="Name of the agent",
        )
        Timeout: int = Field(
            default=400,
            description="Timeout to wait for Letta agent response in seconds",
        )

    def __init__(self):
        self.type = "manifold"
        self.conversation_history = []
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        pipe_name = self.valves.Custom_Name if self.valves.Custom_Name != "" else name
        return [
            {
                "id": f"{name}-pipe",
                "name": f"{pipe_name} Pipe",
            }
        ]

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
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

    async def format_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Format messages according to the Letta API specification."""
        formatted_messages = []
        for msg in messages:
            # Only include supported roles
            if msg.get("role") not in ["user", "system"]:
                continue

            formatted_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            formatted_messages.append(formatted_msg)

        # Ensure we have at least one message
        if not formatted_messages:
            formatted_messages.append({"role": "user", "content": "Hello"})

        logger.debug(f"Formatted messages: {json.dumps(formatted_messages, indent=2)}")
        return formatted_messages

    async def get_letta_response(self, message: Dict[str, str]) -> str:
        """
        Send the user message and wait for the full response.
        Aggregate reasoning messages and the final response into one combined output.
        This version uses aiohttp for asynchronous HTTP calls.
        """
        start_time = time.monotonic()
        headers = {
            "Authorization": f"Bearer {self.valves.API_Token}",
            "Content-Type": "application/json",
        }
        data = {"messages": [message]}
        url = f"{self.valves.API_URL}/v1/agents/{self.valves.Agent_ID}/messages"
        timeout = aiohttp.ClientTimeout(total=self.valves.Timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Make the initial POST request
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 422:
                    text = await response.text()
                    logger.error(f"API Validation Error. Response: {text}")
                    raise ValueError(f"API Validation Error: {text}")
                response.raise_for_status()
                result = await response.json()
                logger.debug(f"Initial API response: {result}")

            # URL for checking message status
            status_url = (
                f"{self.valves.API_URL}/v1/agents/{self.valves.Agent_ID}/messages"
            )

            def find_last_user_message_index(messages):
                """Find the index of the last user message in the list."""
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("message_type") == "user_message":
                        return i
                return -1

            def process_messages(messages_data, elapsed):
                final_response = ""
                reasoning_details = []
                # Handle both list and dictionary responses
                messages = (
                    messages_data
                    if isinstance(messages_data, list)
                    else messages_data.get("messages", [])
                )

                # Find last user message and filter subsequent messages
                last_user_idx = find_last_user_message_index(messages)
                if last_user_idx >= 0:
                    messages = messages[last_user_idx + 1 :]

                for msg in messages:
                    msg_type = msg.get("message_type")
                    if msg_type == "reasoning_message":
                        reasoning = msg.get("reasoning", "").strip()
                        if reasoning:
                            header = f"Thought for {elapsed} seconds"
                            details = (
                                f'<details type="reasoning" done="true" duration="{elapsed}">\n'
                                f"<summary>{header}</summary>\n"
                                f"> {reasoning}\n"
                                "</details>"
                            )
                            # Only keep the latest reasoning
                            reasoning_details = [details]
                    elif msg_type == "assistant_message":
                        content = msg.get("content", "").strip()
                        if content:
                            final_response = content
                    elif msg_type == "tool_return_message":
                        tool_name = msg.get("tool_call_id", "").split("-")[0]
                        content = msg.get("tool_return", "").strip()
                        if content:
                            try:
                                # Try to parse JSON content
                                content_json = json.loads(content)
                                if isinstance(content_json, dict):
                                    content = content_json.get("message", content)
                            except json.JSONDecodeError:
                                pass
                            final_response = f"> {tool_name}: {content}"

                return final_response, reasoning_details

            elapsed = int(time.monotonic() - start_time)
            final_response, reasoning_details = process_messages(
                result if isinstance(result, list) else result.get("messages", []),
                elapsed,
            )

            # Poll for updates until we have a complete response
            while not final_response:
                logger.debug("Waiting for non-empty response from Letta agent...")
                await asyncio.sleep(2)
                async with session.get(status_url, headers=headers) as response:
                    result = await response.json()
                    logger.debug(f"Polling API response: {result}")
                    final_response, new_reasoning = process_messages(
                        (
                            result
                            if isinstance(result, list)
                            else result.get("messages", [])
                        ),
                        elapsed,
                    )
                    reasoning_details.extend(new_reasoning)

            # Combine reasoning details and the final response
            combined = ""
            if reasoning_details:
                combined += "\n".join(reasoning_details) + "\n"
            if final_response:
                combined += final_response

            return combined

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        """Process messages through the Letta agent pipe."""
        # Store event_emitter in instance variable for future use
        if __event_emitter__:
            self.__current_event_emitter__ = __event_emitter__
        elif (
            not hasattr(self, "__current_event_emitter__")
            or not self.__current_event_emitter__
        ):
            logger.error("Event emitter not provided")
            return ""
        print(__user__)
        self.__user__ = Users.get_user_by_id(__user__["id"])
        self.__model__ = __model__
        self.__request__ = __request__

        # Handle task-specific processing
        if __task__ and __task__ != TASKS.DEFAULT:
            try:
                task_model = (
                    self.valves.Task_Model if self.valves.Task_Model else self.__model__
                )
                response = await generate_chat_completions(
                    self.__request__,
                    {
                        "model": task_model,
                        "messages": body.get("messages"),
                        "stream": False,
                    },
                    user=self.__user__,
                )
                return f"{name}: {response['choices'][0]['message']['content']}"
            except Exception as e:
                logger.error(f"Error processing task {__task__}: {e}")
                return f"{name}: Error processing {__task__}"

        # Regular message processing
        messages = body.get("messages", [])
        if not messages:
            await self.emit_status("error", "No messages provided", True)
            return ""

        # Only send the last user message
        user_message = messages[-1]
        if isinstance(user_message, str):
            user_message = {"role": "user", "content": user_message}

        await self.emit_status("info", "Demether is thinking...", False)

        try:
            response = await self.get_letta_response(user_message)
            logger.debug(f"Letta agent response: {response}")
            if response:
                await self.emit_message(str(response))
                await self.emit_status("success", "", True)
                return response
            else:
                await self.emit_status("error", "Empty response from Letta agent", True)
                return ""
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("error", error_msg, True)
            return ""
