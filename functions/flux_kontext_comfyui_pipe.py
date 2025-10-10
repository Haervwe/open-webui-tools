"""
title: ComfyUI Flux Kontext Pipe
authors:
    - Haervwe
    - pupphelper
    - Zed Unknown
author_url: https://github.com/Haervwe/open-webui-tools
description: Edit images using the Flux Kontext workflow API in ComfyUI.
required_open_webui_version: 0.4.0
requirements:
version: 5.0
license: MIT

ComfyUI Required Nodes For Default Workflow:
    - https://github.com/jags111/efficiency-nodes-comfyui
    - https://github.com/glowcone/comfyui-base64-to-image
    - https://github.com/yolain/ComfyUI-Easy-Use
    - https://github.com/SeanScripts/ComfyUI-Unload-Model

Instructions:
- Load the provided workflow in ComfyUI.
- Update the model loader nodes as needed:
        * Use 'Load Diffusion Model' for .safetensors models (enable 'Auto Check Model Loader' in advanced options if unsure).
        * Use 'Unet Loader' for .gguf models.
- Ensure Dual CLIP Loader and VAE Loader nodes are configured with the correct model files.
"""

import json
import os
import uuid
import aiohttp
import asyncio
import random
from typing import Dict, Callable, Optional, Union, Any, cast, Awaitable
from pydantic import BaseModel, Field
from open_webui.utils.misc import get_last_user_message_item  # type: ignore
from open_webui.utils.chat import generate_chat_completion  # type: ignore
from open_webui.models.users import User, Users

from open_webui.constants import TASKS
import logging
import requests

import io
import mimetypes
from fastapi import UploadFile
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW_JSON = json.dumps(
    {
        "6": {
            "inputs": {"text": "THE PROMPT GOES HERE", "clip": ["38", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "35": {
            "inputs": {"guidance": 2.5, "conditioning": ["177", 0]},
            "class_type": "FluxGuidance",
            "_meta": {"title": "FluxGuidance"},
        },
        "38": {
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "flux/t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "flux",
                "device": "cpu",
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "DualCLIPLoader"},
        },
        "39": {
            "inputs": {"vae_name": "Flux/ae.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "42": {
            "inputs": {"image": ["196", 0]},
            "class_type": "FluxKontextImageScale",
            "_meta": {"title": "FluxKontextImageScale"},
        },
        "135": {
            "inputs": {"conditioning": ["6", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        "177": {
            "inputs": {"conditioning": ["6", 0], "latent": ["208", 0]},
            "class_type": "ReferenceLatent",
            "_meta": {"title": "ReferenceLatent"},
        },
        "194": {
            "inputs": {
                "seed": random.randint(0, 2**32 - 1),
                "steps": 20,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "preview_method": "none",
                "vae_decode": "true (tiled)",
                "model": ["197", 0],
                "positive": ["35", 0],
                "negative": ["135", 0],
                "latent_image": ["208", 0],
                "optional_vae": ["39", 0],
            },
            "class_type": "KSampler (Efficient)",
            "_meta": {"title": "KSampler (Efficient)"},
        },
        "196": {
            "inputs": {"data": ""},
            "class_type": "LoadImageFromBase64",
            "_meta": {"title": "Load Image (Base64)"},
        },
        "197": {
            "inputs": {
                "unet_name": "flux-kontext/flux1-dev-kontext_fp8_scaled.safetensors"
            },
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "Unet Loader (GGUF)"},
        },
        "205": {
            "inputs": {"filename_prefix": "ComfyUI_FLUX_KONTEXT", "images": ["209", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "206": {
            "inputs": {"anything": ["194", 5]},
            "class_type": "easy cleanGpuUsed",
            "_meta": {"title": "Clean VRAM Used"},
        },
        "208": {
            "inputs": {"pixels": ["42", 0], "vae": ["39", 0]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"},
        },
        "209": {
            "inputs": {"value": ["206", 0], "model": ["197", 0]},
            "class_type": "UnloadModel",
            "_meta": {"title": "UnloadModel"},
        },
    },
    indent=2,
)


class Pipe:
    class Valves(BaseModel):
        ENABLE_SETUP_FORM: bool = Field(
            title="Enable /setup form",
            default=True,
            description="If disabled, the /setup command is ignored.",
        )
        ENABLE_CONFIG_OVERRIDE: bool = Field(
            title="Enable config override",
            default=True,
            description="Load backend config.json and override valves values when present.",
        )
        CONFIG_BACKEND_PATH: str = Field(
            title="Backend config path",
            default="auto",
            description="Path to persist settings. Use 'auto' to store next to this pipe as flux_kontext_comfyui_config.json",
        )
        COMFYUI_ADDRESS: str = Field(
            title="ComfyUI Address",
            default="http://127.0.0.1:8188",
            description="Address of the running ComfyUI server.",
        )
        COMFYUI_WORKFLOW_JSON: str = Field(
            title="ComfyUI Workflow JSON",
            default=DEFAULT_WORKFLOW_JSON,
            description="The entire ComfyUI workflow in JSON format.",
            json_schema_extra={"type": "textarea"},
        )
        PROMPT_NODE_ID: str = Field(
            title="Prompt Node ID",
            default="6",
            description="The ID of the node that accepts the text prompt.",
        )
        IMAGE_NODE_ID: str = Field(
            title="Image Node ID",
            default="196",
            description="The ID of the node that accepts the Base64 image.",
        )
        KSAMPLER_NODE_ID: str = Field(
            title="KSampler Node ID",
            default="194",
            description="The ID of the sampler node to apply a inline parameters.",
        )
        ENHANCE_PROMPT: bool = Field(
            title="Enhance Prompt",
            default=False,
            description="Use vision model to enhance prompt",
        )
        VISION_MODEL_ID: str = Field(
            title="Vision Model ID",
            default="",
            description="Vision model to be used as prompt enhancer",
        )
        ENHANCER_SYSTEM_PROMPT: str = Field(
            title="Enhancer System Prompt",
            default="""
            You are a visual prompt engineering assistant. 
            For each request, you will receive a user-provided prompt and an image to be edited. 
            Carefully analyze the image’s content (objects, colors, environment, style, mood, etc.) along with the user’s intent. 
            Then generate a single, improved editing prompt for the FLUX Kontext model using best practices. 
            Be specific and descriptive: use exact color names and detailed adjectives, and use clear action verbs like “change,” “add,” or “remove.” 
            Name each subject explicitly (for example, “the woman with short black hair,” “the red sports car”), avoiding pronouns like “her” or “it.” 
            Include relevant details from the image. 
            Preserve any elements the user did not want changed by stating them explicitly (for example, “keep the same composition and lighting”). 
            If the user wants to add or change any text, put the exact words in quotes (for example, replace “joy” with “BFL”).
            Focus only on editing instructions. 
            Finally, output only the final enhanced prompt (the refined instruction) with no additional explanation or commentary.
            """,
            description="System prompt to be used on the prompt enhancement process",
        )
        UNLOAD_OLLAMA_MODELS: bool = Field(
            title="Unload Ollama Models",
            default=False,
            description="Unload all Ollama models from VRAM before running.",
        )
        OLLAMA_URL: str = Field(
            title="Ollama API URL",
            default="http://host.docker.internal:11434",
            description="Ollama API URL for unloading models.",
        )
        MAX_WAIT_TIME: int = Field(
            title="Max Wait Time",
            default=1200,
            description="Max wait time for generation (seconds).",
        )

        AUTO_CHECK_MODEL_LOADER: bool = Field(
            title="Auto Check Model Loader",
            default=True,
            description="Automatically check model loader. Enable if you are not sure about the model loader.",
        )

        CLIP_NAME_1: str = Field(
            title="CLIP 1 Filename",
            default="",
            description="Filename of the first CLIP model for DualCLIPLoader (e.g., clip_l.safetensors).",
        )
        CLIP_NAME_2: str = Field(
            title="CLIP 2 Filename",
            default="",
            description="Filename of the second CLIP/T5 model for DualCLIPLoader (e.g., t5xxl_fp8.safetensors).",
        )
        UNET_MODEL_NAME: str = Field(
            title="UNet / Diffusion Model Filename",
            default="",
            description="Filename of the diffusion model to load (e.g., .safetensors or .gguf).",
        )
        VAE_NAME: str = Field(
            title="VAE Filename",
            default="",
            description="Filename of the VAE model for VAELoader (e.g., flux_vae.safetensors).",
        )
        KSAMPLER_SEED: Optional[int] = Field(
            title="KSampler Seed (optional)",
            default=None,
            description="Seed for the sampler. Leave empty or use -1 to randomize each run.",
        )
        KSAMPLER_STEPS: Optional[int] = Field(
            title="KSampler Steps (optional)",
            default=None,
            description="Number of steps for the sampler. Leave empty to use workflow default.",
        )
        KSAMPLER_CFG: Optional[float] = Field(
            title="KSampler CFG (optional)",
            default=None,
            description="CFG value for the sampler. Leave empty to use workflow default.",
        )
        KSAMPLER_SAMPLER_NAME: Optional[str] = Field(
            title="KSampler Name (optional)",
            default=None,
            description="Sampler name (e.g., euler, heun). Leave empty to use workflow default.",
        )
        KSAMPLER_SCHEDULER: Optional[str] = Field(
            title="KSampler Scheduler (optional)",
            default=None,
            description="Scheduler type (e.g., simple, karras). Leave empty to use workflow default.",
        )
        KSAMPLER_DENOISE: Optional[float] = Field(
            title="KSampler Denoise (optional)",
            default=None,
            description="Denoise strength. Leave empty to use workflow default.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.client_id = str(uuid.uuid4())
        # Attempt to pre-load config on construction (best-effort)
        try:
            self._load_config_and_apply()
        except Exception:
            pass

    # ---------------------------
    # Backend config persistence
    # ---------------------------
    def _get_config_path(self) -> str:
        if getattr(self.valves, "CONFIG_BACKEND_PATH", "auto") and self.valves.CONFIG_BACKEND_PATH != "auto":
            return self.valves.CONFIG_BACKEND_PATH
        # Default to a file next to this module
        return os.path.join(os.path.dirname(__file__), "flux_kontext_comfyui_config.json")

    def _load_config_and_apply(self) -> None:
        """Load config.json and apply overrides to valves (takes precedence)."""
        if not getattr(self.valves, "ENABLE_CONFIG_OVERRIDE", True):
            return
        cfg_path = self._get_config_path()
        if not os.path.exists(cfg_path):
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = cast(Dict[str, Any], json.load(f))
            # Only apply known fields
            for k, v in data.items():
                if hasattr(self.valves, k):
                    try:
                        setattr(self.valves, k, v)
                    except Exception:
                        logger.warning(f"Failed to apply config key '{k}' with value '{v}'")
            logger.info(f"Applied settings from backend config at {cfg_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {cfg_path}: {e}")

    def _save_config(self, values: Dict[str, Any]) -> str:
        """Persist provided values to backend config.json (only known valves). Returns path."""
        cfg_path = self._get_config_path()
        try:
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            # Load existing to merge
            existing: Dict[str, Any] = {}
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        existing = json.load(f) or {}
                except Exception:
                    existing = {}
            # Keep only known fields
            for k in list(values.keys()):
                if not hasattr(self.valves, k):
                    values.pop(k, None)
            merged = {**existing, **values}
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            logger.info(f"Saved settings to backend config at {cfg_path}")
            return cfg_path
        except Exception as e:
            logger.error(f"Failed to save config at {cfg_path}: {e}")
            raise

    def setup_inputs(
        self, messages: list[Dict[str, str]]
    ) -> tuple[
        Optional[str], Optional[str], Dict[str, Optional[Union[int, float, str]]]
    ]:
        prompt: str = ""
        base64_image: Optional[str] = None

        ksampler_options: Dict[str, Optional[Union[int, float, str]]] = {
            "seed": random.randint(0, 2**32 - 1),
            "steps": None,
            "cfg": None,
            "sampler_name": None,
            "scheduler": None,
            "denoise": None,
        }

        if self.valves.KSAMPLER_SEED is not None and self.valves.KSAMPLER_SEED != -1:
            ksampler_options["seed"] = int(self.valves.KSAMPLER_SEED)
        if self.valves.KSAMPLER_STEPS is not None:
            ksampler_options["steps"] = int(self.valves.KSAMPLER_STEPS)
        if self.valves.KSAMPLER_CFG is not None:
            ksampler_options["cfg"] = float(self.valves.KSAMPLER_CFG)
        if self.valves.KSAMPLER_SAMPLER_NAME is not None:
            ksampler_options["sampler_name"] = str(self.valves.KSAMPLER_SAMPLER_NAME)
        if self.valves.KSAMPLER_SCHEDULER is not None:
            ksampler_options["scheduler"] = str(self.valves.KSAMPLER_SCHEDULER)
        if self.valves.KSAMPLER_DENOISE is not None:
            ksampler_options["denoise"] = float(self.valves.KSAMPLER_DENOISE)

        user_message_item = cast(
            Optional[Dict[str, Any]], get_last_user_message_item(messages)
        )
        if not user_message_item:
            return None, None, ksampler_options

        content: Any = user_message_item.get("content") if user_message_item else None

        image_url: Optional[str] = None
        if isinstance(content, list):
            for item in cast(list[Dict[str, Any]], content):
                if item.get("type") == "text":
                    prompt += cast(str, item.get("text", ""))
                elif item.get("type") == "image_url" and item.get("image_url", {}).get(
                    "url"
                ):
                    image_url = cast(str, item["image_url"]["url"])
        elif isinstance(content, str):
            prompt = content

        try:
            if image_url:
                try:
                    if "base64," in image_url:
                        base64_image = image_url.split("base64,", 1)[1]
                    elif image_url.startswith("data:") and "," in image_url:
                        base64_image = image_url.split(",", 1)[1]
                    else:
                        base64_image = image_url
                except Exception as e:
                    logger.warning(f"Unexpected error while extracting base64: {e}")

            if base64_image:
                first10 = base64_image[:10]
                last10 = base64_image[-10:]
                logger.info(
                    f"[setup_inputs] image base64 length={len(base64_image)} "
                    f"first10={first10} last10={last10}"
                )
            else:
                logger.info(
                    "[setup_inputs] no base64 image found in message (base64_raw is None)"
                )

        except Exception as e:
            logger.warning(f"Unexpected error while extracting base64: {e}")

        prompt = re.sub(r"\s+", " ", prompt).strip()

        return prompt, base64_image, ksampler_options

    async def enhance_prompt(
        self,
        prompt: str,
        image: str,
        user: User,
        request: Any,
        event_emitter: Callable[..., Any],
    ) -> str:
        try:
            # Suppress intermediate info emission to avoid UI clutter

            payload: Dict[str, Any] = {
                "model": self.valves.VISION_MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": self.valves.ENHANCER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Enhance the given user prompt based on the given image: "
                                    f"{prompt}, provide only the enhanced AI image edit prompt with no explanations"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                        ],
                    },
                ],
                "stream": False,
            }

            resp_data: Dict[str, Any] = cast(
                Dict[str, Any], await generate_chat_completion(request, payload, user)
            )
            # Suppress intermediate info emission to avoid UI clutter
            enhanced_prompt: str = str(resp_data["choices"][0]["message"]["content"])
            enhanced_prompt_message = f"<details>\n<summary>Enhanced Prompt</summary>\n{enhanced_prompt}\n\n---\n\n</details>"
            await event_emitter(
                {
                    "type": "message",
                    "data": {"content": enhanced_prompt_message},
                }
            )
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}", exc_info=True)

            return prompt

    def prepare_workflow(
        self,
        workflow: Dict[str, Any],
        prompt: str,
        base64_image: Optional[str],
        ksampler_options: Dict[str, Optional[Union[int, float, str]]],
    ) -> Dict[str, Any]:
        prompt_node = self.valves.PROMPT_NODE_ID
        image_node = self.valves.IMAGE_NODE_ID
        ksampler_node = self.valves.KSAMPLER_NODE_ID

        if prompt_node in workflow and "inputs" in workflow[prompt_node]:
            workflow[prompt_node]["inputs"]["text"] = (
                prompt or "A beautiful, high-quality image"
            )
        else:
            logger.warning(f"Prompt node '{prompt_node}' not found in workflow.")

        if image_node in workflow and "inputs" in workflow[image_node]:
            workflow[image_node]["inputs"]["data"] = base64_image
        else:
            logger.warning(f"Image node '{image_node}' not found in workflow.")

        if ksampler_node in workflow and "inputs" in workflow[ksampler_node]:
            for key, value in ksampler_options.items():
                if value is not None:
                    workflow[ksampler_node]["inputs"][key] = value
        else:
            logger.warning(f"ksampler node '{ksampler_node}' not found in workflow.")

        try:
            for _, node in workflow.items():
                ctype = node.get("class_type")
                inputs = node.setdefault("inputs", {})
                if ctype == "DualCLIPLoader":
                    if getattr(self.valves, "CLIP_NAME_1", ""):
                        inputs["clip_name1"] = self.valves.CLIP_NAME_1
                    if getattr(self.valves, "CLIP_NAME_2", ""):
                        inputs["clip_name2"] = self.valves.CLIP_NAME_2
                if ctype in ("UNETLoader", "UnetLoaderGGUF", "UNetLoaderGGUF"):
                    if getattr(self.valves, "UNET_MODEL_NAME", ""):
                        # Standard key in provided workflow
                        inputs["unet_name"] = self.valves.UNET_MODEL_NAME
                if ctype == "VAELoader":
                    if getattr(self.valves, "VAE_NAME", ""):
                        inputs["vae_name"] = self.valves.VAE_NAME
        except Exception as e:
            logger.warning(f"Failed to apply CLIP/UNet valve overrides: {e}")

        _workflow = (
            auto_check_model_loader(workflow)
            if self.valves.AUTO_CHECK_MODEL_LOADER
            else workflow
        )

        return _workflow

    async def queue_prompt(
        self, session: aiohttp.ClientSession, workflow: Dict[str, Any]
    ) -> Optional[str]:

        payload: Dict[str, Any] = {"prompt": workflow, "client_id": self.client_id}
        async with session.post(
            f"{self.valves.COMFYUI_ADDRESS.rstrip('/')}/prompt", json=payload
        ) as response:
            text = await response.text()
            logger.info(f"Queue prompt HTTP {response.status}: {text}")
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Queue prompt JSON response: {data}")
            return data.get("prompt_id")

    async def wait_for_job_signal(
        self, ws_api_url: str, prompt_id: str, event_emitter: Callable[..., Any]
    ) -> bool:
        start_time = asyncio.get_event_loop().time()
        try:
            async with aiohttp.ClientSession().ws_connect(
                f"{ws_api_url}?clientId={self.client_id}"
            ) as ws:
                async for msg in ws:
                    if (
                        asyncio.get_event_loop().time() - start_time
                        > self.valves.MAX_WAIT_TIME
                    ):
                        raise TimeoutError(
                            f"WebSocket wait timed out after {self.valves.MAX_WAIT_TIME}s"
                        )
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.debug(f"WS raw msg: {msg.data}")
                    message = json.loads(msg.data)
                    msg_type, data = message.get("type"), message.get("data", {})

                    if msg_type in {"status", "progress"}:
                        # Silence noisy progress/status updates to avoid UI clutter
                        continue
                    elif msg_type == "executed" and data.get("prompt_id") == prompt_id:
                        logger.info(f"Execution signal received: {data}")
                        return True
                    elif (
                        msg_type == "execution_error"
                        and data.get("prompt_id") == prompt_id
                    ):
                        raise Exception(
                            f"ComfyUI Error: {data.get('exception_message', 'Unknown error')}"
                        )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Operation timed out after {self.valves.MAX_WAIT_TIME}s"
            )
        except Exception as e:
            raise e
        return False

    def extract_image_data(self, outputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        final_image_data: Optional[Dict[str, Any]]
        temp_image_data: Optional[Dict[str, Any]]
        final_image_data, temp_image_data = None, None
        for _, node_output in outputs.items():
            if "ui" in node_output and "images" in node_output.get("ui", {}):
                if node_output["ui"]["images"]:
                    final_image_data = node_output["ui"]["images"][0]
                    break
            elif "images" in node_output and not temp_image_data:
                if node_output["images"]:
                    temp_image_data = node_output["images"][0]
        return final_image_data if final_image_data else temp_image_data

    def _save_image_and_get_public_url(
        self, request: Any, image_data: bytes, content_type: str, user: User
    ) -> str:
        try:
            from open_webui.routers.files import upload_file_handler  # type: ignore

            image_format = mimetypes.guess_extension(content_type)
            if not image_format:

                image_format = ".png"

            file = UploadFile(
                file=io.BytesIO(image_data),
                filename=f"generated-image{image_format}",
            )

            file_item = cast(
                Any,
                upload_file_handler(
                    request=request,
                    file=file,
                    metadata={},
                    process=False,
                    user=user,
                ),
            )
            if not file_item:
                logger.error("Failed to save image to OpenWebUI")
                raise Exception("Failed to save image to OpenWebUI")

            file_id = str(getattr(file_item, "id", ""))
            url = request.app.url_path_for("get_file_content_by_id", id=file_id)
            return str(url)
        except Exception as e:
            logger.error(f"Error saving image to OpenWebUI: {e}", exc_info=True)
            raise e

    async def emit_status(
        self,
        event_emitter: Callable[..., Any],
        level: str,
        description: str,
        done: bool = False,
    ) -> None:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": description,
                    "done": done,
                },
            }
        )

    async def _emit_setup_prompt(self, event_emitter: Callable[..., Any]) -> None:
        """Emit a planner-style input modal prompting the admin to paste a JSON payload for /setup save."""
        v = self.valves
        def _num(val: Optional[Union[int, float]], default: Union[int, float]) -> Union[int, float]:
            return default if val is None else val
        example: Dict[str, Any] = {
            "KSAMPLER_STEPS": _num(v.KSAMPLER_STEPS, 20),
            "KSAMPLER_CFG": _num(v.KSAMPLER_CFG, 1.0),
            "KSAMPLER_SAMPLER_NAME": v.KSAMPLER_SAMPLER_NAME or "euler",
            "KSAMPLER_SCHEDULER": v.KSAMPLER_SCHEDULER or "simple",
            "KSAMPLER_DENOISE": _num(v.KSAMPLER_DENOISE, 1.0),
            "KSAMPLER_SEED": v.KSAMPLER_SEED,
            "CLIP_NAME_1": v.CLIP_NAME_1 or "",
            "CLIP_NAME_2": v.CLIP_NAME_2 or "",
            "UNET_MODEL_NAME": v.UNET_MODEL_NAME or "",
            "VAE_NAME": v.VAE_NAME or "",
            "save_to_backend": True
        }
        placeholder = f"/setup save {json.dumps(example)}"
        await event_emitter(
            {
                "type": "input",
                "data": {
                    "title": "Flux Kontext - Setup",
                    "message": (
                        "Paste a JSON payload to configure sampler and model parameters. "
                        "Edit the example as needed, then send it as a chat message starting with /setup save.\n\n"
                        "Allowed keys: KSAMPLER_STEPS, KSAMPLER_CFG, KSAMPLER_SAMPLER_NAME, KSAMPLER_SCHEDULER, KSAMPLER_DENOISE, KSAMPLER_SEED, CLIP_NAME_1, CLIP_NAME_2, UNET_MODEL_NAME, VAE_NAME, save_to_backend"
                    ),
                    "placeholder": placeholder,
                    "value": placeholder
                }
            }
        )

    async def _emit_current_values(self, event_emitter: Callable[..., Any]) -> None:
        v = self.valves
        eff = self._get_effective_settings()
        lines = [
            "Current settings:",
            f"- KSAMPLER_STEPS: {eff['steps']}",
            f"- KSAMPLER_CFG: {eff['cfg']}",
            f"- KSAMPLER_SAMPLER_NAME: {eff['sampler_name']}",
            f"- KSAMPLER_SCHEDULER: {eff['scheduler']}",
            f"- KSAMPLER_DENOISE: {eff['denoise']}",
            f"- KSAMPLER_SEED: {eff['seed']}",
            f"- CLIP_NAME_1: {v.CLIP_NAME_1}",
            f"- CLIP_NAME_2: {v.CLIP_NAME_2}",
            f"- UNET_MODEL_NAME: {v.UNET_MODEL_NAME}",
            f"- VAE_NAME: {v.VAE_NAME}",
            f"- ENABLE_CONFIG_OVERRIDE: {v.ENABLE_CONFIG_OVERRIDE}",
            f"- CONFIG_BACKEND_PATH: {v.CONFIG_BACKEND_PATH}",
        ]
        await event_emitter({"type": "message", "data": {"content": "\n".join(lines)}})

    def _get_effective_settings(self) -> Dict[str, Union[int, float, str, None]]:
        """Return effective sampler settings using valves overrides if present, otherwise workflow defaults, otherwise hard defaults."""
        # Hard defaults as last resort
        eff: Dict[str, Union[int, float, str, None]] = {
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "seed": None,
        }
        # Try to read from workflow JSON
        try:
            wf = cast(Dict[str, Any], json.loads(self.valves.COMFYUI_WORKFLOW_JSON))
            k_id = self.valves.KSAMPLER_NODE_ID
            node = cast(Dict[str, Any], wf.get(k_id, {}))
            inputs = cast(Dict[str, Any], node.get("inputs", {}))
            eff.update({
                "steps": inputs.get("steps", eff["steps"]),
                "cfg": inputs.get("cfg", eff["cfg"]),
                "sampler_name": inputs.get("sampler_name", eff["sampler_name"]),
                "scheduler": inputs.get("scheduler", eff["scheduler"]),
                "denoise": inputs.get("denoise", eff["denoise"]),
                "seed": inputs.get("seed", eff["seed"]),
            })
        except Exception:
            pass
        # Override with valves if provided
        if self.valves.KSAMPLER_STEPS is not None:
            eff["steps"] = int(self.valves.KSAMPLER_STEPS)
        if self.valves.KSAMPLER_CFG is not None:
            eff["cfg"] = float(self.valves.KSAMPLER_CFG)
        if self.valves.KSAMPLER_SAMPLER_NAME:
            eff["sampler_name"] = str(self.valves.KSAMPLER_SAMPLER_NAME)
        if self.valves.KSAMPLER_SCHEDULER:
            eff["scheduler"] = str(self.valves.KSAMPLER_SCHEDULER)
        if self.valves.KSAMPLER_DENOISE is not None:
            eff["denoise"] = float(self.valves.KSAMPLER_DENOISE)
        if self.valves.KSAMPLER_SEED is not None:
            eff["seed"] = int(self.valves.KSAMPLER_SEED)
        return eff

    async def _ask_input(
        self,
        event_call: Callable[[Dict[str, Any]], Awaitable[Any]],
        title: str,
        message: str,
        placeholder: str,
        value: str,
    ) -> Optional[str]:
        try:
            result = await event_call(
                {
                    "type": "input",
                    "data": {
                        "title": title,
                        "message": message,
                        "placeholder": placeholder,
                        "value": value,
                    },
                }
            )
            if isinstance(result, dict) and "value" in result:
                return str(cast(Any, result)["value"]).strip()
            if isinstance(result, str):
                return result.strip()
        except Exception as e:
            logger.warning(f"Input prompt failed for {title}: {e}")
        return None

    async def _interactive_setup(self) -> None:
        if not hasattr(self, "__event_call__") or self.__event_call__ is None:
            # Fall back to the input prompt if event_call is not available
            await self._emit_setup_prompt(self.__event_emitter__)
            return
        eff = self._get_effective_settings()
        # Ask for each parameter; blank means keep current
        # Steps (int)
        steps_s = await self._ask_input(self.__event_call__, "Steps", "Enter number of steps (leave blank to keep)", str(eff["steps"]), str(eff["steps"]))
        cfg_s = await self._ask_input(self.__event_call__, "CFG", "Enter CFG (leave blank to keep)", str(eff["cfg"]), str(eff["cfg"]))
        sampler_s = await self._ask_input(self.__event_call__, "Sampler Name", "Enter sampler name (leave blank to keep)", str(eff["sampler_name"]), str(eff["sampler_name"]))
        scheduler_s = await self._ask_input(self.__event_call__, "Scheduler", "Enter scheduler (leave blank to keep)", str(eff["scheduler"]), str(eff["scheduler"]))
        denoise_s = await self._ask_input(self.__event_call__, "Denoise", "Enter denoise (leave blank to keep)", str(eff["denoise"]), str(eff["denoise"]))
        seed_s = await self._ask_input(self.__event_call__, "Seed", "Enter seed (-1 for random, leave blank to keep or clear)", str(eff["seed"]) if eff["seed"] is not None else "", str(eff["seed"]) if eff["seed"] is not None else "")
        clip1_s = await self._ask_input(self.__event_call__, "CLIP 1 Filename", "Enter CLIP 1 filename (leave blank to keep)", self.valves.CLIP_NAME_1 or "", self.valves.CLIP_NAME_1 or "")
        clip2_s = await self._ask_input(self.__event_call__, "CLIP 2/T5 Filename", "Enter CLIP 2/T5 filename (leave blank to keep)", self.valves.CLIP_NAME_2 or "", self.valves.CLIP_NAME_2 or "")
        unet_s = await self._ask_input(self.__event_call__, "UNet/Diffusion Filename", "Enter UNet/Diffusion filename (leave blank to keep)", self.valves.UNET_MODEL_NAME or "", self.valves.UNET_MODEL_NAME or "")
        vae_s = await self._ask_input(self.__event_call__, "VAE Filename", "Enter VAE filename (leave blank to keep)", self.valves.VAE_NAME or "", self.valves.VAE_NAME or "")

        applied: Dict[str, Any] = {}
        # Apply conversions and set if provided
        def _apply_num(key: str, raw: Optional[str], conv: Callable[[str], Union[int, float]]):
            nonlocal applied
            if raw is None or raw == "":
                return
            try:
                applied[key] = conv(raw)
                setattr(self.valves, key, applied[key])
            except Exception:
                logger.warning(f"Invalid value for {key}: {raw}")

        def _apply_str(key: str, raw: Optional[str]):
            nonlocal applied
            if raw is None or raw == "":
                return
            applied[key] = raw
            setattr(self.valves, key, raw)

        _apply_num("KSAMPLER_STEPS", steps_s, int)
        _apply_num("KSAMPLER_CFG", cfg_s, float)
        _apply_str("KSAMPLER_SAMPLER_NAME", sampler_s)
        _apply_str("KSAMPLER_SCHEDULER", scheduler_s)
        _apply_num("KSAMPLER_DENOISE", denoise_s, float)
        if seed_s == "":
            # Explicit clear
            applied["KSAMPLER_SEED"] = None
            self.valves.KSAMPLER_SEED = None
        else:
            _apply_num("KSAMPLER_SEED", seed_s, int)
        _apply_str("CLIP_NAME_1", clip1_s)
        _apply_str("CLIP_NAME_2", clip2_s)
        _apply_str("UNET_MODEL_NAME", unet_s)
        _apply_str("VAE_NAME", vae_s)

        # Confirm save to backend defaults
        save_flag = False
        try:
            resp = await self.__event_call__(
                {
                    "type": "confirmation",
                    "data": {
                        "title": "Save Defaults",
                        "message": "Save these settings as backend defaults (config.json)?",
                    },
                }
            )
            if isinstance(resp, bool):
                save_flag = resp
            elif isinstance(resp, dict) and "confirmed" in resp:
                resp_typed = cast(Dict[str, Any], resp)
                conf_val = resp_typed.get("confirmed")
                save_flag = bool(True if conf_val is True else False)
        except Exception:
            pass

        saved_path = None
        if save_flag and applied:
            try:
                saved_path = self._save_config(applied)
            except Exception:
                pass

        msg_lines = ["Applied settings:"] + [f"- {k}: {v}" for k, v in applied.items()] if applied else ["No changes applied"]
        if save_flag:
            msg_lines.append(f"Saved to: {saved_path or '(failed to save)'}")
        await self.__event_emitter__({"type": "message", "data": {"content": "\n".join(msg_lines)}})
        await self.emit_status(self.__event_emitter__, "success", "Setup updated", done=True)

    def _is_admin(self) -> bool:
        try:
            role = str(getattr(self.__user__, "role", "")).lower()
            return role in {"admin", "super_admin", "owner"}
        except Exception:
            return False

    def _extract_last_user_text(self, messages: list[Dict[str, Any]]) -> Optional[str]:
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Concatenate text items
                    texts: list[str] = []
                    for item in cast(list[Dict[str, Any]], content):
                        if item.get("type") == "text":
                            t_val: Any = item.get("text")
                            if isinstance(t_val, str):
                                texts.append(t_val)
                    if texts:
                        return "\n".join(texts)
        return None

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Callable[..., Any],
        __request__: Any = None,
        __task__: Any = None,
        __event_call__: Optional[Callable[[Dict[str, Any]], Awaitable[Any]]] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Main function of the Pipe class.
        Handles prompt enhancement, vision tasks, Ollama unloading, workflow execution, etc...
        """
        self.__event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        # optional event call for interactive prompts
        self.__event_call__ = __event_call__
        # Re-apply backend config (takes precedence) on each run
        try:
            self._load_config_and_apply()
        except Exception:
            pass

        # Handle admin-only /setup commands before any image requirements
        user_text = self._extract_last_user_text(body.get("messages", [])) or ""
        ut_low = user_text.strip().lower()
        if ut_low.startswith("/setup"):
            if not self.valves.ENABLE_SETUP_FORM:
                await self.emit_status(self.__event_emitter__, "warning", "/setup is disabled by valves.", done=True)
                return body
            if not self._is_admin():
                await self.emit_status(self.__event_emitter__, "error", "Only admins can use /setup.", done=True)
                return body

            # /setup save {json}
            if ut_low.startswith("/setup save"):
                # Extract JSON payload after '/setup save'
                try:
                    # Keep original string to preserve case in values
                    m = re.match(r"^/setup\s+save\s+(\{[\s\S]*\})\s*$", user_text.strip(), flags=re.IGNORECASE)
                    if not m:
                        raise ValueError("No JSON payload found after '/setup save'")
                    payload_str = m.group(1)
                    overrides = cast(Dict[str, Any], json.loads(payload_str))

                    # Coerce and apply to valves
                    applied: Dict[str, Any] = {}
                    for k, v in overrides.items():
                        if not hasattr(self.valves, k):
                            continue
                        try:
                            setattr(self.valves, k, v)
                            applied[k] = v
                        except Exception:
                            logger.warning(f"Failed to set valve {k} -> {v}")

                    # Persist if requested
                    save_flag = bool(overrides.get("save_to_backend", False))
                    saved_path = None
                    if save_flag:
                        try:
                            saved_path = self._save_config(applied)
                        except Exception:
                            pass

                    msg_lines = ["Applied settings:"] + [f"- {k}: {v}" for k, v in applied.items()]
                    if save_flag:
                        msg_lines.append(f"Saved to: {saved_path or '(failed to save)'}")
                    await self.__event_emitter__({"type": "message", "data": {"content": "\n".join(msg_lines)}})
                    await self.emit_status(self.__event_emitter__, "success", "Setup updated", done=True)
                except Exception as e:
                    await self.emit_status(self.__event_emitter__, "error", f"Failed to parse/save setup: {e}", done=True)
                return body

            # Plain /setup -> interactive inputs if available; otherwise fallback to single input helper
            await self._emit_current_values(self.__event_emitter__)
            await self._interactive_setup()
            return body
        messages = body.get("messages", [])
        prompt, base64_image, ksampler_options = self.setup_inputs(messages)
        prompt = prompt or ""

        if not base64_image:
            await self.emit_status(
                self.__event_emitter__,
                "error",
                "No valid image provided. Please upload an image.",
                done=True,
            )
            return body

        if self.valves.ENHANCE_PROMPT:
            prompt = await self.enhance_prompt(
                prompt,
                base64_image,
                self.__user__,
                self.__request__,
                self.__event_emitter__,
            )

        if __task__ and __task__ != TASKS.DEFAULT:
            if self.valves.VISION_MODEL_ID:
                resp_data2: Dict[str, Any] = cast(
                    Dict[str, Any],
                    await generate_chat_completion(
                        self.__request__,
                        {
                            "model": self.valves.VISION_MODEL_ID,
                            "messages": body.get("messages"),
                            "stream": False,
                        },
                        user=self.__user__,
                    ),
                )
                return f"{resp_data2['choices'][0]['message']['content']}"
            return "No vision model set for this task."

        if self.valves.UNLOAD_OLLAMA_MODELS:
            unload_all_models(api_url=self.valves.OLLAMA_URL)

        try:
            workflow = json.loads(self.valves.COMFYUI_WORKFLOW_JSON)
        except json.JSONDecodeError:
            await self.emit_status(
                self.__event_emitter__,
                "error",
                "Invalid JSON in the COMFYUI_WORKFLOW_JSON valve.",
                done=True,
            )
            return body

        http_api_url = self.valves.COMFYUI_ADDRESS.rstrip("/")
        ws_api_url = f"{'wss' if http_api_url.startswith('https') else 'ws'}://{http_api_url.split('://', 1)[-1]}/ws"

        workflow = self.prepare_workflow(
            workflow, prompt, base64_image, ksampler_options
        )
        logger.info(f"Generated workflow: {workflow}")

        try:
            async with aiohttp.ClientSession() as session:
                prompt_id = await self.queue_prompt(session, workflow)
                if not prompt_id:
                    await self.emit_status(
                        self.__event_emitter__,
                        "error",
                        "Failed to queue prompt.",
                        done=True,
                    )
                    return body

                await self.emit_status(
                    self.__event_emitter__,
                    "info",
                    "⏳ Generating image...",
                )
                job_done = await self.wait_for_job_signal(
                    ws_api_url, prompt_id, self.__event_emitter__
                )

                if not job_done:
                    raise Exception(
                        "Did not receive a successful execution signal from ComfyUI."
                    )

                job_data = None
                for attempt in range(3):
                    await asyncio.sleep(attempt + 1)
                    logger.info(
                        f"Fetching history for prompt {prompt_id}, attempt {attempt + 1}..."
                    )
                    async with session.get(
                        f"{http_api_url}/history/{prompt_id}"
                    ) as resp:
                        text = await resp.text()
                        logger.info(f"History {resp.status}: {text}")
                        if resp.status == 200:
                            history = await resp.json()
                            logger.info(f"History JSON keys: {list(history.keys())}")
                            if prompt_id in history:
                                job_data = history[prompt_id]
                                break
                    logger.warning(
                        f"Attempt {attempt + 1} to fetch history failed or was incomplete."
                    )

                if not job_data:
                    raise Exception(
                        "Failed to retrieve job data from history after multiple attempts."
                    )

                logger.info(
                    f"Received final job data from history: {json.dumps(job_data, indent=2)}"
                )
                image_to_display = self.extract_image_data(job_data.get("outputs", {}))

                if image_to_display:
                    internal_image_url = f"{http_api_url}/view?filename={image_to_display['filename']}&subfolder={image_to_display.get('subfolder', '')}&type={image_to_display.get('type', 'output')}"
                    # Suppress intermediate download status

                    async with session.get(internal_image_url) as http_response:
                        http_response.raise_for_status()
                        image_data = await http_response.read()
                        content_type = http_response.headers.get(
                            "content-type", "image/png"
                        )

                    # Suppress intermediate embedding status

                    public_image_url = self._save_image_and_get_public_url(
                        request=self.__request__,
                        image_data=image_data,
                        content_type=content_type,
                        user=self.__user__,
                    )

                    alt_text = (
                        prompt if prompt else "Edited image generated by Flux Kontext"
                    )
                    response_content = f"Here is the edited image:\n\n![{alt_text}]({public_image_url})"

                    await self.__event_emitter__(
                        {"type": "message", "data": {"content": response_content}}
                    )
                    await self.emit_status(
                        self.__event_emitter__,
                        "success",
                        "Image processed successfully!",
                        done=True,
                    )

                    body["messages"].append(
                        {"role": "assistant", "content": response_content}
                    )
                    return body

                else:
                    await self.emit_status(
                        self.__event_emitter__,
                        "error",
                        "Execution finished, but no image was found in the output. Please check the workflow.",
                        done=True,
                    )

        except Exception as e:
            logger.error(f"An unexpected error occurred in pipe: {e}", exc_info=True)
            await self.emit_status(
                self.__event_emitter__,
                "error",
                f"An unexpected error occurred: {str(e)}",
                done=True,
            )

        return body


def get_loaded_models(api_url: str = "http://localhost:11434") -> list[Dict[str, Any]]:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        logger.error(f"Error fetching loaded Ollama models: {e}")
        return []


def unload_all_models(api_url: str = "http://localhost:11434") -> None:
    try:
        models = get_loaded_models(api_url)
        if not models:
            return

        logger.info(f"Unloading {len(models)} Ollama models...")
        for model in models:
            model_name = model.get("name")
            if model_name:
                requests.post(
                    f"{api_url.rstrip('/')}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=10,
                )
    except requests.RequestException as e:
        logger.error(f"Error unloading Ollama models: {e}")


def auto_check_model_loader(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    In Flux Kontext, the model exists as:
        - Unquantized (.safetensors) requires UNETLoader
        - Quantized (.gguf) requires UnetLoaderGGUF
    """
    for node_id, node in workflow.items():
        if node.get("class_type") in ["UNETLoader", "UnetLoaderGGUF"]:
            _node = workflow[node_id]
            extention = _node.get("inputs", {}).get("unet_name", "").split(".")[-1]
            if extention == "safetensors":
                if "UNETLoader" not in _node.get("class_type", ""):
                    workflow[node_id]["class_type"] = "UNETLoader"
                    _node["inputs"]["weight_dtype"] = "fp8_e4m3fn_fast"
                    logger.warning(
                        f"Updated model loader '{node_id}' class_type to 'UNETLoader'"
                    )
            elif extention == "gguf":
                if "UnetLoaderGGUF" not in _node.get("class_type", ""):
                    workflow[node_id]["class_type"] = "UnetLoaderGGUF"
                    logger.warning(
                        f"Updated model loader '{node_id}' class_type to 'UnetLoaderGGUF'"
                    )
    return workflow
