"""
title: Prompt Enhancer
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.6.5
important note: if you are going to sue this filter with custom pipes, do not use the show enhanced prompt valve setting
"""

import logging
import re
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional
import json
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message
from open_webui.models.users import User, Users
from open_webui.routers.models import get_models
from open_webui.constants import TASKS
from datetime import datetime
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


def remove_tagged_content(text: str) -> str:

    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought>)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL,
    )

    return re.sub(pattern, "", text).strip()


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

IMPORTANT: DO NOT INCLUDE ANY HEADERS SUCH AS "Enhanced Prompt:" in the response. just the final refined enhaced prompt 

Now, enhance the following prompt using the Context and The user prompt, return only the enhanced user prompt.:
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
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9IiMwMDAwMDAiPjxnIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+PHBhdGggZD0ibTEyLjU5NCAyMy4yNThsLS4wMTIuMDAybC0uMDcxLjAzNWwtLjAyLjAwNGwtLjAxNC0uMDA0bC0uMDcxLS4wMzZxLS4wMTYtLjAwNC0uMDI0LjAwNmwtLjAwNC4wMWwtLjAxNy40MjhsLjAwNS4wMmwuMDEuMDEzbC4xMDQuMDc0bC4wMTUuMDA0bC4wMTItLjAwNGwuMTA0LS4wNzRsLjAxMi0uMDE2bC4wMDQtLjAxN2wtLjAxNy0uNDI3cS0uMDA0LS4wMTYtLjAxNi0uMDE4bS4yNjQtLjExM2wtLjAxNC4wMDJsLS4xODQuMDkzbC0uMDEuMDFsLS4wMDMuMDExbC4wMTguNDNsLjAwNS4wMTJsLjAwOC4wMDhsLjIwMS4wOTJxLjAxOS4wMDUuMDI5LS4wMDhsLjAwNC0uMDE0bC0uMDM0LS42MTRxLS4wMDUtLjAxOS0uMDItLjAyMm0tLjcxNS4wMDJhLjAyLjAyIDAgMCAwLS4wMjcuMDA2bC0uMDA2LjAxNGwtLjAzNC42MTRxLjAwMS4wMTguMDE3LjAyNGwuMDE1LS4wMDJsLjIwMS0uMDkzbC4wMS0uMDA4bC4wMDMtLjAxMWwuMDE4LS40M2wtLjAwMy0uMDEybC0uMDEtLjAxeiIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGQ9Ik0xOSAxOWExIDEgMCAwIDEgLjExNyAxLjk5M0wxOSAyMWgtN2ExIDEgMCAwIDEtLjExNy0xLjk5M0wxMiAxOXptLjYzMS0xNC42MzJhMi41IDIuNSAwIDAgMSAwIDMuNTM2TDguNzM1IDE4LjhhMS41IDEuNSAwIDAgMS0uNDQuMzA1bC0zLjgwNCAxLjcyOWMtLjg0Mi4zODMtMS43MDgtLjQ4NC0xLjMyNS0xLjMyNmwxLjczLTMuODA0YTEuNSAxLjUgMCAwIDEgLjMwNC0uNDRMMTYuMDk2IDQuMzY4YTIuNSAyLjUgMCAwIDEgMy41MzUgMG0tMi4xMiAxLjQxNEw2LjY3NyAxNi42MTRsLS41ODkgMS4yOTdsMS4yOTYtLjU5TDE4LjIxNyA2LjQ5YS41LjUgMCAxIDAtLjcwNy0uNzA3TTYgMWExIDEgMCAwIDEgLjk0Ni42NzdsLjEzLjM3OGEzIDMgMCAwIDAgMS44NjkgMS44N2wuMzc4LjEyOWExIDEgMCAwIDEgMCAxLjg5MmwtLjM3OC4xM2EzIDMgMCAwIDAtMS44NyAxLjg2OWwtLjEyOS4zNzhhMSAxIDAgMCAxLTEuODkyIDBsLS4xMy0uMzc4YTMgMyAwIDAgMC0xLjg2OS0xLjg3bC0uMzc4LS4xMjlhMSAxIDAgMCAxIDAtMS44OTJsLjM3OC0uMTNhMyAzIDAgMCAwIDEuODctMS44NjlsLjEyOS0uMzc4QTEgMSAwIDAgMSA2IDFtMCAzLjE5NkE1IDUgMCAwIDEgNS4xOTYgNXEuNDQ4LjM1NS44MDQuODA0cS4zNTUtLjQ0OC44MDQtLjgwNEE1IDUgMCAwIDEgNiA0LjE5NiIvPjwvZz48L3N2Zz4="""

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __task__=None,
        __request__: Optional[Request] = None,
    ) -> dict:
        self.__current_event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__model__ = __model__
        self.__user__ = Users.get_user_by_id(__user__["id"]) if __user__ else None
        if __task__ and __task__ != TASKS.DEFAULT:
            return body

        messages = body["messages"]
        user_message = get_last_user_message(messages)
        if messages[-1]["role"] != "user":
            print(messages[-1]["content"])
            return body
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
        context = "\n".join(f"""Tools: {body.get("tool_ids")}""")
        context += f'\nCurrent Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        print(body.get("metadata"))
        # Build context block
        context_str = f'\n\nContext:\n"""{context}"""\n\n' if context else ""

        # Construct the system prompt with clear delimiters
        system_prompt = self.valves.user_customizable_template
        user_prompt = f"{context_str}" + f'Prompt to enhance:\n"""{user_message}"""\n\n'

        # Log the system prompt before sending to LLM

        logger.debug("System Prompt: %s", system_prompt)  # Fixed string formatting
        logger.debug("User Prompt: %s", user_prompt)
        model_to_use = None
        if self.valves.model_id:
            model_to_use = self.valves.model_id
        else:
            print("""##########################""")
            print(json.dumps(__model__, indent=4))
            base_model = __model__.get("base_model_id", "")
            model_to_use = base_model if base_model else __model__["info"]["id"]

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
            model_to_use = body["model"]["base_model_id"]  # Fallback to main model
            is_pipeline_model = False

        # Construct payload for LLM request
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Enhance: {user_prompt}",
                },
            ],
            "stream": False,
        }

        try:

            response = await generate_chat_completion(
                self.__request__, payload, user=self.__user__, bypass_filter=True
            )

            message = response["choices"][0]["message"]["content"]
            enhanced_prompt = remove_tagged_content(message)
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
                html_content = f"""<!DOCTYPE html>
<html style="margin:0; padding:0; overflow:hidden;">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .container {{
            margin: 0;
            padding: 0;
            border: none;
            line-height: 0;
        }}
        .prompt-bubble {{
            background: rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0 0 0;
            font-size: 13px;
            line-height: 1.5;
            color: #333;
            word-wrap: break-word;
        }}
        .prompt-label {{
            font-weight: 600;
            color: #666;
            margin-bottom: 4px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .prompt-text {{
            color: #444;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="prompt-bubble">
            <div class="prompt-label">Enhanced Prompt:</div>
            <div class="prompt-text">{enhanced_prompt}</div>
        </div>
    </div>
</body>
</html>"""

                await __event_emitter__(
                    {
                        "type": "embeds",
                        "data": {
                            "embeds": [html_content],
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
        self.__user__ = Users.get_user_by_id(__user__["id"]) if __user__ else None
        return body
