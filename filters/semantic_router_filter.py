"""
title: Semantic Router Filter
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.3
description: Filter that acts a model router, using model descriptions
(make sure to set them in the models you want to be presented to the router)
and the prompt, selecting the best model base,
pipe or preset for the task completion
"""

import logging
import json
import re
from typing import Callable, Awaitable, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.utils.chat import (
    generate_chat_completion,
)
from open_webui.utils.misc import get_last_user_message
from open_webui.models.users import User
from open_webui.routers.models import get_models, get_base_models
from open_webui.models.files import Files

name = "semantic_router"

# Setup logger
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


def get_model_attr(model, attr, default=None):
    """Safely get attributes from dict or Pydantic objects"""
    return (
        getattr(model, attr, default)
        if hasattr(model, "model_dump")
        else model.get(attr, default)
    )


def clean_thinking_tags(message: str) -> str:
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL,
    )

    return re.sub(pattern, "", message).strip()


class Filter:
    class Valves(BaseModel):
        vision_model_id: str = Field("", description="Model ID for image queries")
        banned_models: List[str] = Field(
            default_factory=list, description="Models to exclude"
        )
        system_prompt: str = Field(
            default=(
                "You are a model router assistant. Analyze the user's message and select the most appropriate model.\n"
                'Return ONLY a JSON object with: {"selected_model_id": "id of selected model", "reasoning": "explanation"}'
            ),
            description="System prompt for router",
        )
        disable_qwen_thinking: bool = Field(
            default=True, description="toggle to add /no_think to qwen 3 models"
        )
        show_reasoning: bool = Field(False, description="Show reasoning in chat")
        status: bool = Field(True, description="Show status updates")
        debug: bool = Field(False, description="Show debug statements in logs")

    def __init__(self):
        self.valves = self.Valves()
        self.__request__ = None
        self.__user__ = None
        self.__model__ = None

    def _has_images(self, messages):
        user_message = get_last_user_message(messages)
        if not user_message:
            return False
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)
        return bool(messages[-1].get("images"))

    def _get_available_models(self, models_data):
        available = []
        for model in models_data:
            model_dict = model.model_dump() if hasattr(model, "model_dump") else model
            model_id = model_dict.get("id")
            meta = model_dict.get("meta", {})
            if (
                model_id
                and model_id not in self.valves.banned_models
                and meta.get("description")
                and model_dict.get("pipeline", {}).get("type") != "filter"
            ):
                available.append(
                    {
                        "id": model_id,
                        "name": model_dict.get("name", model_id),
                        "description": meta["description"],
                    }
                )
        return available

    async def _get_model_recommendation(self, body, available_models, user_message):
        system_prompt = (
            (self.valves.system_prompt + " /no_think")
            if self.valves.disable_qwen_thinking
            else self.valves.system_prompt
        )
        models_data = available_models.copy() + [
            {
                "id": body["model"],
                "name": "Base Model",
                "description": "General-purpose language model suitable for various tasks",
            }
        ]
        _temp_body = body.copy()
        if body["messages"][0]["role"] == "system":
            messages = (
                [
                    {
                        "role": "system",
                        "content": system_prompt
                        + f"\nAvailable models:\n{json.dumps(models_data, indent=2)}\n",
                    }
                ]
                + body["messages"][1::-1]
                + [
                    {
                        "role": "user",
                        "content": f"Available models:\n{json.dumps(models_data, indent=2)}\n\nUser request: {user_message}",
                    }
                ]
            )
        else:
            messages = (
                [
                    {
                        "role": "system",
                        "content": system_prompt
                        + f"\nAvailable models:\n{json.dumps(models_data, indent=2)}\n",
                    }
                ]
                + body["messages"][0::-1]
                + [
                    {
                        "role": "user",
                        "content": f"Available models:\n{json.dumps(models_data, indent=2)}\n\nUser request: {user_message}",
                    }
                ]
            )
        payload = {
            "model": body["model"],
            "messages": messages,
            "stream": False,
            "metadata": {
                "direct": True,
                "preset": True,
                "user_id": self.__user__.id if self.__user__ else None,
            },
            "files": [],
            "tool_ids": [],
        }
        response = await generate_chat_completion(
            self.__request__, payload, user=self.__user__, bypass_filter=True
        )
        result = clean_thinking_tags(response["choices"][0]["message"]["content"])
        print(result)
        return json.loads(result)

    def _process_files_for_model(self, files_data):
        collections = {}
        for file_data in files_data:
            meta = file_data.get("meta", {})
            cid = meta.get("collection_name")
            if not cid:
                continue
            collections.setdefault(cid, []).append(
                {
                    "id": file_data["id"],
                    "meta": {
                        **(meta or {}),
                        "citations": True,
                        "source": {
                            "name": meta.get("name", file_data["id"]),
                            "id": file_data["id"],
                            "collection": cid,
                        },
                    },
                    "source": {
                        "name": meta.get("name", file_data["id"]),
                        "id": file_data["id"],
                    },
                    "document": [meta.get("content", "")],
                    "metadata": [
                        {
                            "name": meta.get("name", file_data["id"]),
                            "file_id": file_data["id"],
                            "collection_name": cid,
                            "citations": True,
                        }
                    ],
                    "distances": [1.0],
                }
            )
        # Format into collection objects
        return [
            {
                "id": cid,
                "data": {"file_ids": files, "citations": True},
                "type": "collection",
                "meta": {"citations": True, "source": {"name": cid, "id": cid}},
                "source": {"name": cid, "id": cid},
                "document": [f"Collection: {cid}"],
                "metadata": [{"name": cid, "collection_name": cid, "citations": True}],
                "distances": [1.0],
            }
            for cid, files in collections.items()
        ]

    async def _get_files_from_collections(self, knowledge_collections):
        files_data = []
        for collection in knowledge_collections:
            if not isinstance(collection, dict):
                continue
            cid = collection.get("id")
            file_ids = collection.get("data", {}).get("file_ids", [])
            for fid in file_ids:
                try:
                    file = Files.get_file_metadata_by_id(fid)
                    if file and not any(f["id"] == file.id for f in files_data):
                        file_dict = {
                            "id": file.id,
                            "meta": {
                                **(file.meta or {}),
                                "collection_name": cid,
                                "citations": True,
                                "source": {
                                    "name": file.meta.get("name", file.id),
                                    "id": file.id,
                                    "collection": cid,
                                },
                            },
                            "created_at": file.created_at,
                            "updated_at": file.updated_at,
                            "collection_name": cid,
                            "source": {
                                "name": file.meta.get("name", file.id),
                                "id": file.id,
                            },
                            "document": [file.meta.get("content", "")],
                            "metadata": [
                                {
                                    "name": file.meta.get("name", file.id),
                                    "file_id": file.id,
                                    "collection_name": cid,
                                    "citations": True,
                                }
                            ],
                            "distances": [1.0],
                        }
                        files_data.append(file_dict)
                except Exception as e:
                    logger.error(f"Error getting file {fid}: {str(e)}")
        return files_data

    def _preserve_metadata(self, new_body, original_metadata, original_config):
        for key, value in original_metadata.items():
            if key not in ["model", "features", "filterIds", "files", "tool_ids"]:
                new_body["metadata"][key] = value
        for key in ["user_id", "chat_id", "message_id", "session_id"]:
            if key in original_metadata:
                new_body["metadata"][key] = original_metadata[key]
        new_body["metadata"]["direct"] = original_metadata.get(
            "direct", original_config["metadata"].get("direct", False)
        )
        new_body["metadata"]["variables"] = original_metadata.get(
            "variables", original_config.get("variables", {})
        )
        return new_body

    def _update_model_metadata(
        self, selected_model, selected_model_full, original_config
    ):
        meta = {}
        if selected_model_full:
            model_data = (
                selected_model_full.model_dump()
                if hasattr(selected_model_full, "model_dump")
                else selected_model_full
            )
            meta = model_data.get("meta", {})
        updated_model = {
            "id": selected_model["id"],
            "name": selected_model["name"],
            "description": meta.get("description", ""),
        }
        original_model = original_config.get("model_metadata", {})
        for field in ["object", "created", "owned_by", "preset", "actions"]:
            if field in original_model:
                updated_model[field] = original_model[field]
        updated_model["info"] = {
            "id": selected_model["id"],
            "name": selected_model["name"],
            "base_model_id": (
                model_data.get("base_model_id")
                if selected_model_full and model_data.get("base_model_id")
                else model_data.get("info", {}).get("base_model_id", None)
            ),
        }
        # Copy critical info fields
        for field in [
            "user_id",
            "updated_at",
            "created_at",
            "access_control",
            "is_active",
        ]:
            if "info" in original_model and field in original_model["info"]:
                updated_model.setdefault("info", {})[field] = original_model["info"][
                    field
                ]
        if "info" in original_model and "meta" in original_model["info"]:
            updated_model["info"]["meta"] = {
                k: v
                for k, v in original_model["info"]["meta"].items()
                if k != "toolIds"
            }
            if selected_model_full and meta.get("toolIds"):
                updated_model["info"]["meta"]["toolIds"] = meta.get("toolIds", [])
            else:
                updated_model["info"]["meta"]["toolIds"] = []
        if "info" in original_model:
            for k, v in original_model["info"].items():
                if k not in [
                    "id",
                    "name",
                    "user_id",
                    "updated_at",
                    "created_at",
                    "access_control",
                    "is_active",
                    "meta",
                ]:
                    updated_model["info"][k] = v
        if "params" in original_model:
            updated_model["info"]["params"] = original_model["params"]
        return updated_model

    def _merge_files(self, new_body, files_data):
        merged = new_body.get("files", []).copy() if new_body.get("files") else []
        for file in files_data:
            idx = next(
                (i for i, ef in enumerate(merged) if ef.get("id") == file.get("id")),
                None,
            )
            if idx is not None:
                merged[idx].update(
                    {
                        "meta": {
                            **(merged[idx].get("meta", {})),
                            **file.get("meta", {}),
                            "citations": True,
                            "source": file.get("meta", {}).get("source", {}),
                        },
                        "source": file.get("source", {}),
                        "document": file.get("document", []),
                        "metadata": file.get("metadata", []),
                        "distances": file.get("distances", [1.0]),
                    }
                )
            else:
                merged.append(file)
        new_body["files"] = merged
        return new_body

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        self.__request__ = __request__
        self.__model__ = __model__
        self.__user__ = User(**__user__) if isinstance(__user__, dict) else __user__

        original_config = {
            "stream": body.get("stream", False),
            "features": body.get("features", {}),
            "metadata": body.get("metadata", {}),
            "variables": body.get("metadata", {}).get("variables", {}),
            "model_metadata": body.get("metadata", {}).get("model", {}),
            "session_info": {
                k: body.get("metadata", {}).get(k)
                for k in ["user_id", "chat_id", "message_id", "session_id"]
            },
        }

        messages = body.get("messages", [])
        # Route to vision model if images exist
        if self._has_images(messages) and self.valves.vision_model_id:
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Routing image request to vision model",
                            "done": True,
                        },
                    }
                )
            body["model"] = self.valves.vision_model_id
            return body

        if self.valves.status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Analyzing request to select best model...",
                        "done": False,
                    },
                }
            )

        try:
            models = await get_models(
                self.__request__, self.__user__
            ) + await get_base_models(self.__user__)
            logger.debug(models)
            if not models:
                logger.warning("No models returned from get_models()")
                return body
            if isinstance(models, dict):
                models = models.get("data", [])
            available_models = self._get_available_models(models)
            if not available_models:
                logger.warning("No valid models found for routing")
                return body

            user_message = get_last_user_message(messages)
            result = await self._get_model_recommendation(
                body, available_models, user_message
            )

            if self.valves.show_reasoning:
                reasoning_message = f"<details>\n<summary>Model Selection</summary>\nSelected Model: {result['selected_model_id']}\n\nReasoning: {result['reasoning']}\n\n---\n\n</details>"
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {"content": reasoning_message},
                    }
                )

            selected_model = next(
                (m for m in available_models if m["id"] == result["selected_model_id"]),
                None,
            )
            if not selected_model:
                logger.error(f"Selected model {result['selected_model_id']} not found")
                return body

            selected_model_full = next(
                (m for m in models if get_model_attr(m, "id") == selected_model["id"]),
                None,
            )
            new_body = body.copy()
            new_body["model"] = selected_model["id"]
            new_body.setdefault("metadata", {})

            # Update filterIds from model metadata if present
            meta = {}
            if selected_model_full:
                model_data = (
                    selected_model_full.model_dump()
                    if hasattr(selected_model_full, "model_dump")
                    else selected_model_full
                )
                meta = model_data.get("meta", {})
            new_body["metadata"]["filterIds"] = [
                fid
                for fid in meta.get("filterIds", [])
                if fid != "semantic_router_filter"
            ]

            # Remove any tool_ids
            new_body.pop("tool_ids", None)
            new_body.get("metadata", {}).pop("tool_ids", None)

            new_body = self._preserve_metadata(
                new_body, body.get("metadata", {}), original_config
            )
            new_body["metadata"]["model"] = self._update_model_metadata(
                selected_model, selected_model_full, original_config
            )
            new_body["metadata"]["features"] = original_config.get("features", {})

            if selected_model_full and meta.get("toolIds"):
                new_body["tool_ids"] = meta.get("toolIds", []).copy()

            # Process knowledge files if defined
            files_data = []
            if selected_model_full and isinstance(meta.get("knowledge"), list):
                files_data = await self._get_files_from_collections(meta["knowledge"])
            if files_data:
                new_body = self._merge_files(new_body, files_data)
            elif "files" in body:
                new_body["files"] = body["files"]

            # If files exist, process for knowledge collections
            if new_body.get("files"):
                meta_info = (
                    new_body["metadata"]["model"].get("info", {}).setdefault("meta", {})
                )
                meta_info["knowledge"] = (
                    self._process_files_for_model(new_body["files"]) or []
                )

            # Preserve top-level settings and parameters
            new_body["stream"] = body.get("stream", False)
            for field in [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "seed",
            ]:
                if field in body:
                    new_body[field] = body[field]

            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Selected: {selected_model['name']} Model",
                            "done": True,
                        },
                    }
                )

            return new_body

        except Exception as e:
            logger.error("Error in semantic routing: %s", str(e), exc_info=True)
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error during model selection",
                            "done": True,
                        },
                    }
                )
            return body
