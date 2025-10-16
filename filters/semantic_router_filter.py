"""
title: Semantic Router Filter
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.5
description: Filter that acts a model router, using model descriptions and capabilities
(make sure to set them in the models you want to be presented to the router)
and the prompt, selecting the best model base, pipe or preset for the task completion.
Supports vision model filtering, proper type handling, and passes images to the router model
for contextual routing decisions. Automatically switches to vision fallback model if the
router model doesn't support vision. Strictly filters out inactive/deleted models (is_active must be True).
Preserves all original request parameters and relies on Open WebUI's built-in payload
conversion system to handle backend-specific parameter translation.
"""

import logging
import json
import re
from typing import (
    Callable,
    Awaitable,
    Any,
    Optional,
    List,
    Dict,
    Union,
    TypedDict,
)
from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message
from open_webui.utils.payload import convert_payload_openai_to_ollama
from open_webui.models.users import UserModel, Users
from open_webui.models.files import FileMetadataResponse, Files
from open_webui.models.models import ModelUserResponse, ModelModel
from open_webui.routers.models import get_models, get_base_models

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


class ModelInfo(TypedDict):
    """Model information structure for routing decisions"""

    id: str
    name: str
    description: str
    vision_capable: bool
    created_at: Optional[int]
    updated_at: Optional[int]


class RouterResponse(TypedDict):
    """Response structure from the router model"""

    selected_model_id: str
    reasoning: str


def clean_thinking_tags(message: str) -> str:
    """Remove thinking tags from LLM output"""
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL,
    )
    return re.sub(pattern, "", message).strip()


def extract_model_data(
    model: Union[ModelUserResponse, ModelModel, Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract data from model object (Pydantic or dict)"""
    if isinstance(model, dict):
        return model
    return model.model_dump()


def is_vision_capable(model_data: Dict[str, Any]) -> bool:
    """Check if a model has vision capabilities"""
    meta = model_data.get("meta", {})
    capabilities = meta.get("capabilities", {})

    if isinstance(capabilities, dict):
        return bool(capabilities.get("vision", False))

    return False


def clean_ollama_params(payload: dict) -> dict:
    """
    Remove Ollama-specific parameters that cause errors with OpenAI endpoints.
    Called when routing TO an OpenAI endpoint.
    """
    clean = payload.copy()
    
    # Remove Ollama-specific 'options' dict entirely
    if "options" in clean:
        options = clean.pop("options")
        
        # Extract num_predict -> max_tokens
        if "num_predict" in options:
            clean["max_tokens"] = options["num_predict"]
        
        # Move compatible params to root
        for param in ["temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"]:
            if param in options and param not in clean:
                clean[param] = options[param]
    
    # Handle format -> response_format
    if "format" in clean:
        format_val = clean.pop("format")
        if isinstance(format_val, dict):
            clean["response_format"] = {"type": "json_schema", "json_schema": {"schema": format_val}}
        elif format_val == "json":
            clean["response_format"] = {"type": "json_object"}
    
    # Handle system parameter at root (move to messages)
    if "system" in clean:
        system_content = clean.pop("system")
        sys_msg = {"role": "system", "content": system_content}
        clean["messages"] = [sys_msg] + clean.get("messages", [])
    
    return clean


class Filter:
    class Valves(BaseModel):
        vision_fallback_model_id: str = Field(
            "",
            description="Fallback model ID for image queries when no vision models are available in routing",
        )
        banned_models: List[str] = Field(
            default_factory=list, description="Models to exclude from routing"
        )
        allowed_models: List[str] = Field(
            default_factory=list,
            description="Whitelist of models to include (overrides banned_models when set)",
        )
        router_model_id: str = Field(
            "",
            description="Specific model to use for routing decisions (leave empty to use current model)",
        )
        system_prompt: str = Field(
            default=(
                "You are a model router assistant. Analyze the user's message and select the most appropriate model.\n"
                "Consider the task type, complexity, and required capabilities. DO NOT assume capabilties beyond the explicitly stated in the description.\n"
                'Return ONLY a JSON object with: {"selected_model_id": "id of selected model", "reasoning": "brief explanation"}'
            ),
            description="System prompt for router model",
        )
        disable_qwen_thinking: bool = Field(
            default=True,
            description="Append /no_think to router prompt for Qwen models",
        )
        show_reasoning: bool = Field(
            False, description="Display routing reasoning in chat"
        )
        status: bool = Field(True, description="Show status updates in chat")
        debug: bool = Field(False, description="Enable debug logging")

    def __init__(self):
        self.valves = self.Valves()
        self.__request__: Optional[Request] = None
        self.__user__: Optional[UserModel] = None
        self.__model__: Optional[Dict[str, Any]] = None

    def _has_images(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if the message history contains images"""
        if not messages:
            return False

        last_message = messages[-1]
        content = last_message.get("content", "")

        if isinstance(content, list):
            return any(item.get("type") == "image_url" for item in content)

        return bool(last_message.get("images"))

    def _get_available_models(
        self,
        models_data: List[Union[ModelUserResponse, ModelModel, Dict[str, Any]]],
        filter_vision: bool = False,
    ) -> List[ModelInfo]:
        """
        Get available models for routing with proper type handling

        Args:
            models_data: List of model objects (Pydantic models or dicts)
            filter_vision: If True, only return vision-capable models
        """
        available: List[ModelInfo] = []

        if self.valves.debug:
            logger.debug(
                f"Processing {len(models_data)} models for routing (filter_vision={filter_vision})"
            )

        for model in models_data:
            model_dict = extract_model_data(model)
            model_id = model_dict.get("id")
            meta = model_dict.get("meta", {})
            pipeline_type = model_dict.get("pipeline", {}).get("type")

            if not model_id or pipeline_type == "filter":
                continue

            is_active = model_dict.get("is_active", False)
            if self.valves.debug:
                base_model_id = model_dict.get("base_model_id")
                logger.debug(
                    f"Model {model_id}: is_active={is_active} (type: {type(is_active).__name__}), "
                    f"base_model_id={base_model_id}, pipeline={pipeline_type}"
                )

            if not is_active:
                if self.valves.debug:
                    logger.debug(f"Skipping inactive/deleted model: {model_id}")
                continue

            if (
                self.valves.allowed_models
                and model_id not in self.valves.allowed_models
            ):
                if self.valves.debug:
                    logger.debug(f"Skipping model not in allowed_models: {model_id}")
                continue

            if model_id in self.valves.banned_models:
                if self.valves.debug:
                    logger.debug(f"Skipping banned model: {model_id}")
                continue

            description = meta.get("description")
            if not description:
                if self.valves.debug:
                    logger.debug(f"Skipping model without description: {model_id}")
                continue

            is_vision = is_vision_capable(model_dict)
            if filter_vision and not is_vision:
                continue

            model_info: ModelInfo = {
                "id": model_id,
                "name": model_dict.get("name", model_id),
                "description": description,
                "vision_capable": is_vision,
            }
            available.append(model_info)

        if self.valves.debug:
            logger.debug(f"Found {len(available)} available models after filtering")
            if available:
                model_list = ", ".join([f"{m['name']} ({m['id']})" for m in available])
                logger.debug(f"Available models: {model_list}")
            else:
                logger.debug("No models passed filtering criteria")

        return available

    async def _get_model_recommendation(
        self, body: Dict[str, Any], available_models: List[ModelInfo], user_message: str
    ) -> RouterResponse:
        """Get model recommendation from the router LLM"""
        system_prompt = self.valves.system_prompt
        if self.valves.disable_qwen_thinking:
            system_prompt += " /no_think"

        models_json = json.dumps(available_models, indent=2)
        router_messages: List[Dict[str, Any]] = []

        if body.get("messages") and body["messages"][0]["role"] == "system":
            router_messages.append(body["messages"][0])

        router_messages.append(
            {
                "role": "system",
                "content": f"{system_prompt}\n\nAvailable models:\n{models_json}",
            }
        )

        has_images = self._has_images(body.get("messages", []))

        if has_images:
            last_msg = body.get("messages", [])[-1]
            router_user_message = {"role": "user", "content": []}

            if isinstance(last_msg.get("content"), list):
                for item in last_msg["content"]:
                    if item.get("type") == "text":
                        router_user_message["content"].append(
                            {
                                "type": "text",
                                "text": f"User request: {item.get('text', '')}\n\nSelect the most appropriate model based on the text and image(s) provided.",
                            }
                        )
                    elif item.get("type") == "image_url":
                        router_user_message["content"].append(item)
            else:
                router_user_message["content"].append(
                    {
                        "type": "text",
                        "text": f"User request: {user_message}\n\nSelect the most appropriate model based on the text and image(s) provided.",
                    }
                )
                if "images" in last_msg:
                    router_user_message["images"] = last_msg["images"]

            router_messages.append(router_user_message)
        else:
            router_messages.append(
                {
                    "role": "user",
                    "content": f"User request: {user_message}\n\nSelect the most appropriate model.",
                }
            )

        router_model = self.valves.router_model_id
        if not router_model:
            metadata_model = body.get("metadata", {}).get("model", {})
            base_model_id = metadata_model.get("info", {}).get("base_model_id")
            router_model = base_model_id or body.get("model")

        if has_images:
            from open_webui.routers.models import get_models, get_base_models

            all_models = await get_models(
                self.__request__, self.__user__
            ) + await get_base_models(self.__user__)

            router_model_obj = next(
                (
                    m
                    for m in all_models
                    if extract_model_data(m).get("id") == router_model
                ),
                None,
            )

            if router_model_obj:
                router_model_data = extract_model_data(router_model_obj)
                if not is_vision_capable(router_model_data):
                    if self.valves.vision_fallback_model_id:
                        logger.info(
                            f"Router model '{router_model}' is not vision-capable. "
                            f"Using fallback '{self.valves.vision_fallback_model_id}' for routing."
                        )
                        router_model = self.valves.vision_fallback_model_id
                    else:
                        logger.warning(
                            f"Router model '{router_model}' is not vision-capable and no fallback is set. "
                            "Routing decision may not consider image content."
                        )

        payload = {
            "model": router_model,
            "messages": router_messages,
            "stream": False,
            "metadata": {
                "direct": True,
                "preset": True,
                "user_id": self.__user__.id if self.__user__ else None,
            },
        }

        # Get recommendation
        response = await generate_chat_completion(
            self.__request__, payload, user=self.__user__, bypass_filter=True
        )

        # Parse response
        content = self._extract_response_content(response)
        if not content:
            raise Exception("No content found in router response")

        # Clean and parse JSON
        result = clean_thinking_tags(content)
        return self._parse_router_response(result, body.get("model") or "")

    def _extract_response_content(self, response: Any) -> Optional[str]:
        """Extract content from various response formats"""
        # Handle JSONResponse
        if hasattr(response, "body"):
            response_data = json.loads(response.body.decode("utf-8"))
        elif hasattr(response, "json"):
            response_data = (
                response.json if not callable(response.json) else response.json()
            )
        else:
            response_data = response

        # Check for errors
        if isinstance(response_data, dict) and "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            logger.error(f"API error in model recommendation: {error_msg}")
            raise Exception(f"API error: {error_msg}")

        # Extract content from various formats
        if isinstance(response_data, dict):
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"]
            elif "message" in response_data:
                return response_data["message"]["content"]
            elif "content" in response_data:
                return response_data["content"]
            elif "response" in response_data:
                return response_data["response"]

        return str(response_data)

    def _parse_router_response(
        self, content: str, fallback_model: str
    ) -> RouterResponse:
        """Parse JSON response from router with fallbacks"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content}")

            # Try to extract JSON from text
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Final fallback
            logger.warning("Using fallback model selection")
            return {
                "selected_model_id": fallback_model,
                "reasoning": "Fallback to original model due to parsing error",
            }

    def _build_file_data(
        self, file_metadata: FileMetadataResponse, collection_name: str
    ) -> Dict[str, Any]:
        """
        Build file data structure for a single file in the INPUT format
        expected by get_sources_from_items().

        This creates the structure that Open WebUI's RAG system will process,
        NOT the final citation structure.
        """
        file_id = file_metadata.id
        meta = file_metadata.meta or {}

        return {
            "type": "file",
            "id": file_id,
            "name": meta.get("name", file_id),
            "collection_name": collection_name,
            "legacy": False,
            # Optional: Add enriched file data if content is available
            "file": {
                "meta": meta,
                "data": {"content": meta.get("content", "")},
            },
        }

    async def _get_files_from_collections(
        self, knowledge_collections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get files from knowledge collections and format for RAG retrieval.
        Returns INPUT format expected by get_sources_from_items().
        """
        files_data: List[Dict[str, Any]] = []

        for collection in knowledge_collections:
            if not isinstance(collection, dict):
                continue

            collection_id = collection.get("id")
            file_ids = collection.get("data", {}).get("file_ids", [])

            for file_id in file_ids:
                try:
                    file_metadata = Files.get_file_metadata_by_id(file_id)
                    if file_metadata and not any(
                        f["id"] == file_metadata.id for f in files_data
                    ):
                        file_data = self._build_file_data(file_metadata, collection_id)
                        files_data.append(file_data)

                except Exception as e:
                    logger.error(f"Error getting file {file_id}: {str(e)}")

        return files_data

    def _build_collection_data(
        self, collection_id: str, files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build collection data structure"""
        return {
            "id": collection_id,
            "data": {"file_ids": files, "citations": True},
            "type": "collection",
            "meta": {
                "citations": True,
                "source": {"name": collection_id, "id": collection_id},
            },
            "source": {"name": collection_id, "id": collection_id},
            "document": [f"Collection: {collection_id}"],
            "metadata": [
                {
                    "name": collection_id,
                    "collection_name": collection_id,
                    "citations": True,
                }
            ],
            "distances": [1.0],
        }

    def _process_files_for_knowledge(
        self,
        files_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process files and group into collections"""
        collections: Dict[str, List[Dict[str, Any]]] = {}

        for file_data in files_data:
            collection_name = file_data.get("meta", {}).get("collection_name")
            if not collection_name:
                continue

            collections.setdefault(collection_name, []).append(file_data)

        return [
            self._build_collection_data(cid, files)
            for cid, files in collections.items()
        ]

    def _merge_files(
        self, existing_files: List[Dict[str, Any]], new_files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge new files with existing files, avoiding duplicates"""
        merged = existing_files.copy() if existing_files else []
        existing_ids = {f["id"] for f in merged}

        for file_data in new_files:
            file_id = file_data["id"]
            if file_id in existing_ids:
                idx = next(i for i, f in enumerate(merged) if f["id"] == file_id)
                merged[idx].update(dict(file_data))
            else:
                merged.append(dict(file_data))
                existing_ids.add(file_id)

        return merged

    def _build_updated_body(
        self,
        original_body: Dict[str, Any],
        selected_model: ModelInfo,
        selected_model_full: Union[ModelUserResponse, ModelModel, Dict[str, Any]],
        files_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the updated request body with selected model and metadata"""

        new_body = original_body.copy()
        new_body["model"] = selected_model["id"]

        model_data = extract_model_data(selected_model_full)
        meta = model_data.get("meta", {})

        new_body.setdefault("metadata", {})
        original_metadata = original_body.get("metadata", {})

        new_body["metadata"]["filterIds"] = [
            fid for fid in meta.get("filterIds", []) if fid != "semantic_router_filter"
        ]

        new_body.pop("tool_ids", None)
        new_body["metadata"].pop("tool_ids", None)
        if meta.get("toolIds"):
            new_body["tool_ids"] = meta["toolIds"].copy()

        for key in [
            "user_id",
            "chat_id",
            "message_id",
            "session_id",
            "direct",
            "variables",
        ]:
            if key in original_metadata:
                new_body["metadata"][key] = original_metadata[key]

        new_body["metadata"]["model"] = self._build_model_metadata(
            selected_model, model_data, original_metadata.get("model", {})
        )
        new_body["metadata"]["features"] = original_body.get("features", {})

        if files_data:
            new_body["files"] = self._merge_files(new_body.get("files", []), files_data)
        elif "files" in original_body:
            new_body["files"] = original_body["files"]

        if new_body.get("files"):
            new_body["metadata"]["model"].setdefault("info", {}).setdefault("meta", {})[
                "knowledge"
            ] = self._process_files_for_knowledge(new_body["files"])

        return new_body

    def _build_model_metadata(
        self,
        selected_model: ModelInfo,
        model_data: Dict[str, Any],
        original_model: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build model metadata structure"""
        meta = model_data.get("meta", {})

        updated_model = {
            "id": selected_model["id"],
            "name": selected_model["name"],
            "description": meta.get("description", ""),
            "info": {
                "id": selected_model["id"],
                "name": selected_model["name"],
                "base_model_id": model_data.get("base_model_id"),
            },
        }

        for field in ["object", "created", "owned_by", "preset", "actions"]:
            if field in original_model:
                updated_model[field] = original_model[field]

        if "info" in original_model:
            for field in [
                "user_id",
                "updated_at",
                "created_at",
                "access_control",
                "is_active",
            ]:
                if field in original_model["info"]:
                    updated_model["info"][field] = original_model["info"][field]

        if "info" in original_model and "meta" in original_model["info"]:
            updated_model["info"]["meta"] = {
                k: v
                for k, v in original_model["info"]["meta"].items()
                if k != "toolIds"
            }
            if meta.get("toolIds"):
                updated_model["info"]["meta"]["toolIds"] = meta["toolIds"]

        if "params" in original_model:
            updated_model["info"]["params"] = original_model["params"]

        return updated_model

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Request] = None,
    ) -> Dict[str, Any]:
        """Main filter entry point"""
        self.__request__ = __request__
        self.__model__ = __model__
        self.__user__ = Users.get_user_by_id(__user__["id"]) if __user__ else None

        messages = body.get("messages", [])
        has_images = self._has_images(messages)

        try:
            models = await get_models(
                self.__request__, self.__user__
            ) + await get_base_models(self.__user__)

            if not models:
                logger.warning("No models returned from get_models()")
                return body

            if has_images:
                if self.valves.debug:
                    logger.debug("Message contains images, filtering for vision models")

                available_models = self._get_available_models(
                    models, filter_vision=True
                )

                if not available_models and self.valves.vision_fallback_model_id:
                    if self.valves.debug:
                        logger.debug(
                            "No vision-capable models found, using fallback model %s",
                            self.valves.vision_fallback_model_id,
                        )
                    body["model"] = self.valves.vision_fallback_model_id
                    return body
                elif not available_models:
                    logger.warning("No vision-capable models found and no fallback set")
                    return body
            else:
                available_models = self._get_available_models(
                    models, filter_vision=False
                )

            if not available_models:
                logger.warning("No valid models found for routing")
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

            user_message = get_last_user_message(messages)
            result = await self._get_model_recommendation(
                body, available_models, user_message
            )

            if self.valves.show_reasoning:
                reasoning_message = (
                    f"<details>\n<summary>Model Selection</summary>\n"
                    f"Selected Model: {result['selected_model_id']}\n\n"
                    f"Reasoning: {result['reasoning']}\n\n---\n\n</details>"
                )
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
                logger.error(
                    f"Selected model {result['selected_model_id']} not found in available models"
                )
                return body

            selected_model_full = next(
                (
                    m
                    for m in models
                    if extract_model_data(m).get("id") == selected_model["id"]
                ),
                None,
            )

            if not selected_model_full:
                logger.warning(
                    f"Could not find full model data for {selected_model['id']}"
                )
                return body

            files_data: List[Dict[str, Any]] = []
            model_data = extract_model_data(selected_model_full)
            meta = model_data.get("meta", {})
            knowledge = meta.get("knowledge", [])
            if isinstance(knowledge, list) and knowledge:
                files_data = await self._get_files_from_collections(knowledge)

            new_body = self._build_updated_body(
                body, selected_model, selected_model_full, files_data
            )

            # Convert payload based on source/target model backend mismatch
            source_owned_by = body.get("metadata", {}).get("model", {}).get("owned_by")
            target_owned_by = model_data.get("owned_by")
            
            if source_owned_by != target_owned_by:
                if target_owned_by == "ollama":
                    # Target is Ollama, source is OpenAI → convert to Ollama
                    new_body = convert_payload_openai_to_ollama(new_body)
                elif target_owned_by == "openai":
                    # Target is OpenAI, source might be Ollama → clean Ollama params
                    new_body = clean_ollama_params(new_body)

            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Selected: {selected_model['name']}",
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
