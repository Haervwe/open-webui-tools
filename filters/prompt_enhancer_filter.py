"""
title: Prompt Enhancer
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.5.2
"""

import logging
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional
import json
from dataclasses import dataclass
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message
from open_webui.models.models import Models
from open_webui.models.users import User

name = "enhancer"


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


class Filter:
    class Valves(BaseModel):
        user_customizable_template: str = Field(
            default="""\
You are an expert prompt engineer. Your task is to enhance the given prompt by making it more detailed, specific, and effective. Consider the context and the user's intent.

Response Format:
- Provide only the enhanced prompt.
- No additional text, markdown, or titles.
- The enhanced prompt should start immediately without any introductory phrases.

Example:
Given Prompt: Write a poem about flowers.
Enhanced Prompt: Craft a vivid and imaginative poem that explores the beauty and diversity of flowers, using rich imagery and metaphors to bring each bloom to life.

Now, enhance the following prompt:
""",
            description="Prompt to use in the Prompt enhancer System Message",
        )
        show_status: bool = Field(
            default=False,
            description="Show status indicators",
        )
        show_enhanced_prompt: bool = Field(
            default=False,
            description="Show Enahcend Prompt in chat",
        )
        model_id: Optional[str] = Field(
            default=None,
            description="Model to use for the prompt enhancement, leave empty to use the same as selected for the main response.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.__current_event_emitter__ = None
        self.__user__ = None
        self.__model__ = None
        self.__request__ = None

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        self.__current_event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__model__ = __model__
        self.__user__ = User(**__user__) if isinstance(__user__, dict) else __user__

        # Fetch available models and log their relevant details
        available_models = await get_models(self.__request__, self.__user__)
        logger.debug("Available Models (truncated image data):")
        for model in available_models:
            truncated_model = model.model_dump()  # Convert to dict for modification
            if "meta" in truncated_model:
                if isinstance(truncated_model["meta"], dict):
                    if "profile_image_url" in truncated_model["meta"]:
                        truncated_model["meta"]["profile_image_url"] = (
                            truncated_model["meta"]["profile_image_url"][:50] + "..."
                            if isinstance(
                                truncated_model["meta"]["profile_image_url"], str
                            )
                            else None
                        )
                    if "profile_image_url" in truncated_model["user"]:
                        truncated_model["user"]["profile_image_url"] = (
                            truncated_model["user"]["profile_image_url"][:50] + "..."
                            if isinstance(
                                truncated_model["user"]["profile_image_url"], str
                            )
                            else None
                        )
                else:
                    logger.warning(
                        f"Unexpected type for model.meta: {type(truncated_model['meta'])}"
                    )
            else:
                logger.warning("Model missing 'meta' key: %s", model)

            # Truncate files information
            if "knowledge" in truncated_model and isinstance(
                truncated_model["knowledge"], list
            ):
                for knowledge_item in truncated_model["knowledge"]:
                    if isinstance(knowledge_item, dict) and "files" in knowledge_item:
                        knowledge_item["files"] = "List of files (truncated)"

            logger.debug(json.dumps(truncated_model, indent=2))

        messages = body["messages"]
        user_message = get_last_user_message(messages)

        if self.valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Enhancing the prompt...",
                        "done": False,
                    },
                }
            )

        # Prepare context from chat history, excluding the last user message
        context_messages = [
            msg
            for msg in messages
            if msg["role"] != "user" or msg["content"] != user_message
        ]
        context = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in context_messages]
        )

        # Build context block
        context_str = f'\n\nContext:\n"""{context}"""\n\n' if context else ""

        # Construct the system prompt with clear delimiters
        system_prompt = self.valves.user_customizable_template
        user_prompt = (
            f"Context: {context_str}" f'Prompt to enhance:\n"""{user_message}"""\n\n'
        )

        # Log the system prompt before sending to LLM

        logger.debug("System Prompt: %s", system_prompt)  # Fixed string formatting

        # Determine the model to use
        # model_to_use = self.valves.model_id if self.valves.model_id else (body["model"])
        model_to_use = None
        if self.valves.model_id:
            model_to_use = self.valves.model_id
        else:
            model_to_use = body["model"]

        # Check if the selected model has "-pipe" or "pipe" in its name.
        is_pipeline_model = False
        if "-pipe" in model_to_use.lower() or "pipe" in model_to_use.lower():
            is_pipeline_model = True
            logger.warning(
                f"Selected model '{model_to_use}' appears to be a pipeline model.  Consider using the base model."
            )

        # If a pipeline model is *explicitly* chosen, use it. Otherwise, fall back to the main model.
        if not self.valves.model_id and is_pipeline_model:
            logger.warning(
                f"Pipeline model '{model_to_use}' selected without explicit model_id.  Using main model instead."
            )
            model_to_use = body["model"]  # Fallback to main model
            is_pipeline_model = False

        # Construct payload for LLM request
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Enhance the given user prompt based on context: {user_prompt}",
                },
            ],
            "stream": False,
        }

        try:
            # Use the User object directly, as done in other scripts
            logger.debug(
                "API CALL:\n Request: %s\n Form_data: %s\n User: %s",
                str(self.__request__),
                json.dumps(payload),
                self.__user__,
            )

            response = await generate_chat_completion(
                self.__request__, payload, user=self.__user__, bypass_filter=True
            )

            enhanced_prompt = response["choices"][0]["message"]["content"]
            logger.debug("Enhanced prompt: %s", enhanced_prompt)

            # Update the messages with the enhanced prompt
            messages[-1]["content"] = enhanced_prompt
            body["messages"] = messages

            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Prompt successfully enhanced.",
                            "done": True,
                        },
                    }
                )
            if self.valves.show_enhanced_prompt:
                enhanced_prompt_message = f"<details>\n<summary>Enhanced Prompt</summary>\n{enhanced_prompt}\n\n---\n\n</details>"
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": enhanced_prompt_message,
                        },
                    }
                )

        except ValueError as ve:
            logger.error("Value Error: %s", str(ve))
            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error: {str(ve)}",
                            "done": True,
                        },
                    }
                )
        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "An unexpected error occurred.",
                            "done": True,
                        },
                    }
                )

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        self.__current_event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__model__ = __model__
        self.__user__ = User(**__user__) if isinstance(__user__, dict) else __user__
        print(body)
        return body
