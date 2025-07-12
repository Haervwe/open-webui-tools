"""
title: Multi Model Conversations
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.7.0
"""

import logging
import json
import asyncio
from typing import Dict, List, Callable, Awaitable, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import User

name = "Conversation"


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
        ROUNDS_PER_CONVERSATION: int = Field(
            default=3,
            description="Number of rounds in the entire conversation",
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
            description="Appended message to all participants internally to prime them properly to not go off character",
        )
        UseGroupChatManager: bool = Field(
            default=False, description="Use Group Chat Manager"
        )
        ManagerModel: str = Field(default="", description="Model for the Manager")
        ManagerSystemMessage: str = Field(
            default="You are a group chat manager. Your role is to decide who should speak next in a multi-participant conversation. You will be given the conversation history and a list of participant aliases. Choose the alias of the participant who is most likely to provide a relevant and engaging response to the latest message. Consider the context of the conversation, the personalities of the participants, and avoid repeatedly selecting the same participant.",
            description="System message for the Manager",
        )
        ManagerSelectionPrompt: str = Field(
            default="Conversation History:\n{history}\n\nThe last speaker was '{last_speaker}'. Based on the flow of the conversation, who should speak next? Choose exactly one from the following list of participants: {participant_list}\n\nRespond with ONLY the alias of your choice, and nothing else.",
            description="Template for the Manager's selection prompt. Use {history}, {last_speaker}, and {participant_list}.",
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
        __request__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__)
        self.__model__ = __model__
        self.__request__ = __request__
        if __task__ and __task__ != TASKS.DEFAULT:
            model = self.valves.Participant1Model or self.__model__
            response = await generate_chat_completions(
                self.__request__,
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        user_message = body.get("messages", [])[-1].get("content", "").strip()
        num_participants = self.valves.NUM_PARTICIPANTS
        participants = []
        for i in range(1, num_participants + 1):
            model = getattr(self.valves, f"Participant{i}Model", "")
            alias = getattr(self.valves, f"Participant{i}Alias", "") or model
            system_message = getattr(self.valves, f"Participant{i}SystemMessage", "")
            if not model:
                logger.warning(f"No model set for Participant {i}, skipping.")
                continue
            participants.append(
                {"model": model, "alias": alias, "system_message": system_message}
            )

        if not participants:
            await self.emit_status("error", "No valid participants configured.", True)
            return "Error: No participants configured. Please set at least one participant's model in the valves."

        self.conversation_history.append({"role": "user", "content": user_message})
        last_speaker = None

        for round_num in range(self.valves.ROUNDS_PER_CONVERSATION):
            if self.valves.UseGroupChatManager:
                participant_aliases = [p["alias"] for p in participants]
                history_str = "\n".join(
                    f"{msg['role']}: {msg['content']}"
                    for msg in self.conversation_history
                )

                manager_prompt = self.valves.ManagerSelectionPrompt.format(
                    history=history_str,
                    last_speaker=last_speaker or "None",
                    participant_list=", ".join(participant_aliases),
                )

                manager_messages = [
                    {"role": "system", "content": self.valves.ManagerSystemMessage},
                    {"role": "user", "content": manager_prompt},
                ]

                manager_response = await generate_chat_completions(
                    self.__request__,
                    {
                        "model": self.valves.ManagerModel,
                        "messages": manager_messages,
                        "stream": False,
                    },
                    user=self.__user__,
                )
                selected_alias = manager_response["choices"][0]["message"][
                    "content"
                ].strip()

                selected_participant = next(
                    (p for p in participants if p["alias"] == selected_alias), None
                )

                if not selected_participant or selected_alias == last_speaker:
                    await self.emit_status(
                        "info",
                        f"Manager selection '{selected_alias}' was invalid or repeated. Using fallback.",
                        False,
                    )
                    
                    last_speaker_index = next(
                        (
                            i
                            for i, p in enumerate(participants)
                            if p["alias"] == last_speaker
                        ),
                        -1,
                    )
                    # Pick the next participant in order, wrapping around
                    fallback_index = (last_speaker_index + 1) % len(participants)
                    selected_participant = participants[fallback_index]
                    logger.warning(
                        f"Manager failed. Fallback selected: {selected_participant['alias']}"
                    )

                # A single participant turn
                participants_to_run = [selected_participant]
                last_speaker = selected_participant["alias"]
            else:
                participants_to_run = participants

            for participant in participants_to_run:
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
                    "info", f"Getting response from: {alias} ({model})...", False
                )
                try:
                    await self.emit_model_title(alias)
                    full_response = ""
                    async for chunk in self.get_streaming_completion(
                        messages, model=model
                    ):
                        full_response += chunk
                        await self.emit_message(chunk)

                    cleaned_response = full_response.strip()
                    self.conversation_history.append(
                        {"role": "assistant", "content": cleaned_response}
                    )
                    logger.debug(f"History updated with response from {alias}")

                except Exception as e:
                    error_message = f"Error getting response from {model}: {e}"
                    await self.emit_status("error", error_message, True)
                    logger.error(f"Error with {model}: {e}", exc_info=True)

        await self.emit_status("success", "Conversation round completed.", True)
        return ""
