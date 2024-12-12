"""
title: Prompt Enhancer
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/open-webui
version: 0.1
"""

import logging
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional
import json

from open_webui.apps.webui.models.users import Users
from open_webui.main import generate_chat_completions
from open_webui.utils.misc import get_last_user_message

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

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
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

        logger.debug("System Prompt:\n%s", system_prompt)

        # Determine the model to use
        model_to_use = self.valves.model_id if self.valves.model_id else body["model"]

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
            user = Users.get_user_by_id(__user__["id"])
            response = await generate_chat_completions(form_data=payload, user=user)
            enhanced_prompt = response["choices"][0]["message"]["content"]

            # Update the messages with the enhanced prompt
            messages[-1]["content"] = enhanced_prompt
            logger.debug(f"""After:{body["messages"]}""")
            body["messages"] = messages
            logger.debug(f"""Before:{body["messages"]}""")
            logger.debug("Enhanced prompt: %s", enhanced_prompt)
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
                enhanced_prompt_message = (
                    f"### Enhanced Prompt:\n{enhanced_prompt}\n\n---\n\n"
                )
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": enhanced_prompt_message},
                    }
                )

        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error enhancing prompt.",
                            "done": True,
                        },
                    }
                )

        return body
