"""
title: Multi Model Conversations
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools/
version: 0.2
"""

import logging
import json
import asyncio
from typing import Dict, List, Callable, Awaitable, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions

name = "Conversation"

logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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

    class Valves(BaseModel):
        NUM_PARTICIPANTS: int = Field(
            default=1,
            description="Number of participants in the conversation (1-5)",
            ge=1,
            le=5,
        )
        ROUNDS_PER_USER_MESSAGE: int = Field(
            default=1,
            description="Number of rounds of replies before user can send a new message",
            ge=1,
        )
        Participant1Model: str = Field(
            default="", description="Model tag for Participant 1"
        )
        Participant1Alias: str = Field(
            default="", description="Alias tag for Participant 1"
        )
        Participant1SystemMessage: str = Field(
            default="", description="Character sheet for Participant 1"
        )
        Participant2Model: str = Field(
            default="", description="Model tag for Participant 2"
        )
        Participant2Alias: str = Field(
            default="", description="Alias tag for Participant 2"
        )
        Participant2SystemMessage: str = Field(
            default="", description="Character sheet for Participant 2"
        )
        Participant3Model: str = Field(
            default="", description="Model tag for Participant 3"
        )
        Participant3Alias: str = Field(
            default="", description="Alias tag for Participant 3"
        )
        Participant3SystemMessage: str = Field(
            default="", description="Character sheet for Participant 3"
        )
        Participant4Model: str = Field(
            default="", description="Model tag for Participant 4"
        )
        Participant4Alias: str = Field(
            default="", description="Alias tag for Participant 4"
        )
        Participant4SystemMessage: str = Field(
            default="", description="Character sheet for Participant 4"
        )
        Participant5Model: str = Field(
            default="", description="Model tag for Participant 5"
        )
        Participant5Alias: str = Field(
            default="", description="Alias tag for Participant 5"
        )
        Participant5SystemMessage: str = Field(
            default="", description="Character sheet for Participant 5"
        )
        AllParticipantsApendedMessage: str = Field(
            default="Respond only as your specified character and never use your name as title, just output the response as if you really were talking(no one says his name before a phrase), do not go off character in any situation, Your acted response as",
            description="Appended message to all participants internally to prime them propperly to not go off character",
        )
        Temperature: float = Field(default=1, description="Models temperature")
        Top_k: int = Field(default=50, description="Models top_k")
        Top_p: float = Field(default=0.8, description="Models top_p")

    def __init__(self):
        self.type = "manifold"
        self.conversation_history = []
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def get_streaming_completion(
        self,
        messages,
        model: str,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        try:
            form_data = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": self.valves.Temperature,
                "top_k": self.valves.Top_k,
                "top_p": self.valves.Top_p,
            }
            response = await generate_chat_completions(
                form_data,
                user=self.__user__,
                bypass_filter=False,
            )

            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")

            async for chunk in response.body_iterator:
                for part in self.get_chunk_content(chunk):
                    yield part

        except Exception as e:
            raise RuntimeError(f"Streaming completion failed: {e}")

    def get_chunk_content(self, chunk):
        chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
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

    async def emit_model_title(self, model_name: str):
        """Helper function to emit the model title with a separator."""
        await self.emit_message(f"\n\n---\n\n**{model_name}:**\n\n")

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__)
        self.__model__ = __model__  # Store the default model

        if __task__ == TASKS.TITLE_GENERATION:
            model = (
                self.valves.Participant1Model or self.__model__
            )  # Use Participant 1 or default
            response = await generate_chat_completions(
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        user_message = body.get("messages", [])[-1].get("content", "").strip()

        # Fetch valve configurations
        num_participants = self.valves.NUM_PARTICIPANTS
        rounds_per_user_message = self.valves.ROUNDS_PER_USER_MESSAGE
        participants = []

        for i in range(1, num_participants + 1):
            model_field = f"Participant{i}Model"
            system_field = f"Participant{i}SystemMessage"
            alias_field = f"Participant{i}Alias"
            model = getattr(self.valves, model_field, "")
            alias = getattr(self.valves, alias_field, "")
            system_message = getattr(self.valves, system_field, "")
            if not model:
                logger.warning(f"No model set for Participant {i}")
                continue
            participants.append(
                {
                    "model": model,
                    "alias": alias if alias else model,
                    "system_message": system_message,
                }
            )

        if not participants:
            await self.emit_status("error", "No participants configured", True)
            return "No participants configured."

        self.conversation_history.append({"role": "user", "content": user_message})

        for _ in range(rounds_per_user_message):
            for participant in participants:
                model = participant["model"]
                alias = participant["alias"]
                system_message = participant["system_message"]

                messages = [
                    {"role": "system", "content": system_message},
                    *self.conversation_history,
                    {
                        "role": "user",
                        "content": f"{self.valves.AllParticipantsApendedMessage} {alias}",
                    },
                ]

                await self.emit_status(
                    "info",
                    f"Getting response from: {f'{alias}/{model}' if alias else model}...",
                    False,
                )

                try:
                    await self.emit_model_title(alias)  # Emit title *before* streaming
                    full_response = ""
                    async for chunk in self.get_streaming_completion(
                        messages, model=model
                    ):
                        full_response += chunk
                        await self.emit_message(
                            chunk
                        )  # Stream the response without modification

                    cleaned_response = full_response.strip()  # Clean after streaming
                    self.conversation_history.append(
                        {"role": "assistant", "content": cleaned_response}
                    )
                    logger.debug(f"History:{self.conversation_history}")
                except Exception as e:
                    await self.emit_status(
                        "error",
                        f"Error getting response from {model}: {e}",
                        True,
                    )
                    logger.error(f"Error with {model}: {e}")

        await self.emit_status("success", "Conversation completed", True)
        return ""