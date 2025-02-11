"""
title: Letta_Agent_Connector
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.2
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
import requests
import asyncio


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

@dataclass
class User:
    id: str
    email: str
    name: str
    role: str

class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __model__: str
    __request__: None

    class Valves(BaseModel):
        Agent_ID: str = Field(
            default="Demether",
            description="The ID of the Letta agent to communicate with"
        )
        API_URL: str = Field(
            default="http://localhost:8283",
            description="Base URL for the Letta agent API"
        )
        API_Token: str = Field(
            default="",
            description="Bearer token for API authentication"
        )
        Task_Model: str = Field(
            default="",
            description="Model to use for title/tags generation tasks. If empty, uses the default model."
        )

    def __init__(self):
        self.type = "manifold"
        self.conversation_history = []
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

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
                }
            }
        )

    async def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages according to the Letta API specification."""
        formatted_messages = []
        for msg in messages:
            # Only include supported roles
            if msg.get("role") not in ["user", "system"]:
                continue
                
            formatted_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }
            formatted_messages.append(formatted_msg)
        
        # Ensure we have at least one message
        if not formatted_messages:
            formatted_messages.append({
                "role": "user",
                "content": "Hello"
            })
            
        logger.debug(f"Formatted messages: {json.dumps(formatted_messages, indent=2)}")
        return formatted_messages

    def get_letta_response(self, message: Dict[str, str]) -> str:
        """Send the last user message to the Letta agent and get its response using requests."""
        headers = {
            "Authorization": f"Bearer {self.valves.API_Token}",
            "Content-Type": "application/json"
        }
        data = {
            "messages": [message]
        }
        url = f"{self.valves.API_URL}/v1/agents/{self.valves.Agent_ID}/messages"
        
        logger.debug(f"Sending request to {url}")
        logger.debug(f"Request data: {data}")
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 422:
                logger.error(f"API Validation Error. Response: {response.text}")
                raise ValueError(f"API Validation Error: {response.text}")
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Raw API response: {result}")
            
            if "messages" in result and result["messages"]:
                for msg in reversed(result["messages"]):
                    msg_type = msg.get("message_type")
                    if msg_type == "assistant_message":
                        content = msg.get("content", "")
                        if content:
                            return content
                    elif msg_type == "tool_return_message":
                        content = msg.get("return_value", "")
                        if content:
                            return content
            
            if "usage" in result:
                logger.debug(f"Usage statistics: {result['usage']}")
            
            raise ValueError("No valid response content found")
        except requests.RequestException as e:
            logger.error(f"Error communicating with Letta agent: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

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
        elif not hasattr(self, '__current_event_emitter__') or not self.__current_event_emitter__:
            logger.error("Event emitter not provided")
            return ""
            
        self.__user__ = User(**__user__)
        self.__model__ = __model__
        self.__request__ = __request__

        # Handle task-specific processing
        if __task__ and __task__ != TASKS.DEFAULT:
            try:
                task_model = self.valves.Task_Model or self.__model__
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

        await self.emit_status("info", "Sending request to Letta agent...", False)

        try:
            response = await asyncio.to_thread(self.get_letta_response, user_message)
            logger.debug(f"Letta agent response: {response}")
            if response:
                await self.emit_message(str(response))
                await self.emit_status("success", "Letta agent response received", True)
                return response
            else:
                await self.emit_status("error", "Empty response from Letta agent", True)
                return ""
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("error", error_msg, True)
            return ""
