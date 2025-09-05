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
version: 4.0
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
import uuid
import aiohttp
import asyncio
import random
from typing import List, Dict, Callable, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import get_last_user_message_item
from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import User, Users

from open_webui.constants import TASKS
import logging
import requests

import io
import mimetypes
from fastapi import UploadFile
from open_webui.routers.files import upload_file_handler
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW_JSON = json.dumps(
    {
        "6": {
            "inputs": {
            "text": "THE PROMPT GOES HERE",
            "clip": [
                "38",
                0
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "CLIP Text Encode (Prompt)"
            }
        },
        "35": {
            "inputs": {
            "guidance": 2.5,
            "conditioning": [
                "177",
                0
            ]
            },
            "class_type": "FluxGuidance",
            "_meta": {
            "title": "FluxGuidance"
            }
        },
        "38": {
            "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp8.safetensors",
            "type": "flux",
            "device": "cpu"
            },
            "class_type": "DualCLIPLoader",
            "_meta": {
            "title": "DualCLIPLoader"
            }
        },
        "39": {
            "inputs": {
            "vae_name": "Flux\\flux_vae.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {
            "title": "Load VAE"
            }
        },
        "42": {
            "inputs": {
            "image": [
                "196",
                0
            ]
            },
            "class_type": "FluxKontextImageScale",
            "_meta": {
            "title": "FluxKontextImageScale"
            }
        },
        "135": {
            "inputs": {
            "conditioning": [
                "6",
                0
            ]
            },
            "class_type": "ConditioningZeroOut",
            "_meta": {
            "title": "ConditioningZeroOut"
            }
        },
        "177": {
            "inputs": {
            "conditioning": [
                "6",
                0
            ],
            "latent": [
                "208",
                0
            ]
            },
            "class_type": "ReferenceLatent",
            "_meta": {
            "title": "ReferenceLatent"
            }
        },
        "194": {
            "inputs": {
            "seed": 241782050172708,
            "steps": 20,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "preview_method": "none",
            "vae_decode": "true (tiled)",
            "model": [
                "197",
                0
            ],
            "positive": [
                "35",
                0
            ],
            "negative": [
                "135",
                0
            ],
            "latent_image": [
                "208",
                0
            ],
            "optional_vae": [
                "39",
                0
            ]
            },
            "class_type": "KSampler (Efficient)",
            "_meta": {
            "title": "KSampler (Efficient)"
            }
        },
        "196": {
            "inputs": {
            "data": ""
            },
            "class_type": "LoadImageFromBase64",
            "_meta": {
            "title": "Load Image (Base64)"
            }
        },
        "197": {
            "inputs": {
            "unet_name": "Image-Diffusion\\Flux\\flux1-kontext-dev-Q5_K_M.gguf"
            },
            "class_type": "UnetLoaderGGUF",
            "_meta": {
            "title": "Unet Loader (GGUF)"
            }
        },
        "205": {
            "inputs": {
            "filename_prefix": "ComfyUI_FLUX_KONTEXT",
            "images": [
                "209",
                0
            ]
            },
            "class_type": "SaveImage",
            "_meta": {
            "title": "Save Image"
            }
        },
        "206": {
            "inputs": {
            "anything": [
                "194",
                5
            ]
            },
            "class_type": "easy cleanGpuUsed",
            "_meta": {
            "title": "Clean VRAM Used"
            }
        },
        "208": {
            "inputs": {
            "pixels": [
                "42",
                0
            ],
            "vae": [
                "39",
                0
            ]
            },
            "class_type": "VAEEncode",
            "_meta": {
            "title": "VAE Encode"
            }
        },
        "209": {
            "inputs": {
            "value": [
                "206",
                0
            ],
            "model": [
                "197",
                0
            ]
            },
            "class_type": "UnloadModel",
            "_meta": {
            "title": "UnloadModel"
            }
        }
    },
    indent=2,
)

class Pipe:
    class Valves(BaseModel):
        # ... (this section remains unchanged)
        COMFYUI_ADDRESS: str = Field(
            title="ComfyUI Address",
            default="http://127.0.0.1:8188",
            description="Address of the running ComfyUI server.",
        )
        COMFYUI_WORKFLOW_JSON: str = Field(
            title="ComfyUI Workflow JSON",
            default=DEFAULT_WORKFLOW_JSON,
            description="The entire ComfyUI workflow in JSON format.",
            extra={"type": "textarea"},
        )
        PROMPT_NODE_ID: str = Field(
            title="Prompt Node ID",
            default="6", 
            description="The ID of the node that accepts the text prompt."
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
            description="Use vision model to enhance prompt"
        )
        VISION_MODEL_ID: str = Field(
            title="Vision Model ID",
            default="", 
            description="Vision model to be used as prompt enhancer"
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
            description="Max wait time for generation (seconds)."
        )
        
        # --- Section Title (not an option, just a label for UI separation) ---
        ADVANCED_OPTIONS_TITLE: str = Field(
            title="◈ [ Advanced Options ] ◈",
            default="NOTHING HERE",
            description="",
            extra={"type": "section_title"},
        )
        # Automatically Check Model Loader (GGUF or SAFETENSORS)
        AUTO_CHECK_MODEL_LOADER: bool = Field(
            title="Auto Check Model Loader",
            default=False,
            description="Automatically check model loader. Enable if you are not sure about the model loader.",
        )
        # KSAMPLER REGEX OPTIONS ENABLED?
        REG_EX: bool = Field(
            title="Inline KSampler parameters / Inline Parameters",
            default=False,
            description="Use regular expressions to extract ksampler values directly from the prompt. Example: {seed=1234, steps=20, cfg=1.7, sampler_name=euler, scheduler=simple, denoise=0.8}",
        )
        KSAMPLER_MIN_STEPS: int = Field(
            title="KSampler Min Steps (if 'Inline parameters' are enabled)",
            default=5,
            description="Minimum number of steps allowed for the sampler.",
        )
        KSAMPLER_MAX_STEPS: int = Field(
            title="KSampler Max Steps (if 'Inline parameters' are enabled)",
            default=60,
            description="Maximum number of steps allowed for the sampler.",
        )
        KSAMPLER_MIN_CFG: float = Field(
            title="KSampler Min CFG (if 'Inline parameters' are enabled)",
            default=0.0,
            description="Minimum CFG value allowed for the sampler.",
        )
        KSAMPLER_MAX_CFG: float = Field(
            title="KSampler Max CFG (if 'Inline parameters' are enabled)",
            default=15.0,
            description="Maximum CFG value allowed for the sampler.",
        )
        KSAMPLER_MIN_DENOISE: float = Field(
            title="KSampler Min Denoise (if 'Inline parameters' are enabled)",
            default=0.0,
            description="Minimum denoise value allowed for the sampler.",
        )
        KSAMPLER_MAX_DENOISE: float = Field(
            title="KSampler Max Denoise (if 'Inline parameters' are enabled)",
            default=1.0,
            description="Maximum denoise value allowed for the sampler.",
        )
        
    def __init__(self):
        self.valves = self.Valves()
        self.client_id = str(uuid.uuid4())

# ----------------------------------------------------------------------------------------------------

    def setup_inputs(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], Optional[str], dict]:
        """
        Setup inputs for the workflow.
        """
        
        # Comfy UI Main Inputs
        prompt = ""
        base64_image = None
        
        # Ksampler Inputs
        # If an option is None, it uses default value from the workflow (in 'prepare_workflow')
        ksampler_options = {
            'seed': random.randint(0, 2**32 - 1),
            'steps': None,
            'cfg': None,
            'sampler_name': None,
            'scheduler': None,
            'denoise': None
        }
        # Ksampler Validation & Limits (for security)
        min_steps, max_steps = self.valves.KSAMPLER_MIN_STEPS, self.valves.KSAMPLER_MAX_STEPS
        min_cfg, max_cfg = self.valves.KSAMPLER_MIN_CFG, self.valves.KSAMPLER_MAX_CFG
        min_denoise, max_denoise = self.valves.KSAMPLER_MIN_DENOISE, self.valves.KSAMPLER_MAX_DENOISE
        
        valid_samplers = ["euler","euler_ancestral","heun","dpm_2","dpm_2_ancestral","lms","dpm_fast","dpm_adaptive","dpmpp_2s_ancestral","dpmpp_sde","dpmpp_sde_gpu","dpmpp_2m","dpmpp_2m_sde","dpmpp_2m_sde_gpu","dpmpp_3m_sde","dpmpp_3m_sde_gpu","ddpm","lcm","ddim","uni_pc","uni_pc_bh2","euler_cfg_pp","euler_ancestral_cfg_pp","heunpp2","ipndm","ipndm_v","deis","oiler","oiler_plus","oiler_plus_ancestral","oiler_cfg_pp","oiler_cfg_pp_alternative","sampler_sonar_euler","bogacki","reversible_bogacki","reversible_heun","reversible_heun_1s","rk4","rkf45","rk_dynamic","solver_diffrax","euler_cycle","euler_dancing","res","adapter","dynamic"]
        valid_schedulers = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal", "AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]
        
        '''
        Extract Inputs from Last User Message (example - beautify for clarity) 
        {'role':'user','content':[{'type':'text','text':'What is in this picture?'},{'type':'image_url','image_url':{'url':'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdYAAAGcCAYAAABk2YF[REDACTED]'}}]}
        '''
        user_message_item = get_last_user_message_item(messages)
        
        if not user_message_item:
            return None, None
        
        content = user_message_item.get("content")

        # Rebuild prompt + capture data image_url from content structure
        image_url =  None
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    prompt += item.get("text", "")
                elif item.get("type") == "image_url" and item.get("image_url", {}).get("url"):
                    image_url = item["image_url"]["url"]
                    
        elif isinstance(content, str):
            prompt = content

        # Extract input image base64 from 'image_url'
        try:
            # Determine the base64 candidate (support data URI or raw base64)
            if image_url:
                try:
                    # 'base64,' marker
                    if "base64," in image_url:
                        base64_image = image_url.split("base64,", 1)[1]
                    elif image_url.startswith("data:") and "," in image_url:
                        base64_image = image_url.split(",", 1)[1]
                    else:
                        base64_image = image_url
                except Exception as e:
                    logger.warning(f"Unexpected error while extracting base64: {e}")

            # Log a compact preview: length, first 10 chars, last 10 chars (avoid logging full data)
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
            
        # Extract ksampler options from user prompt (if present)
        if self.valves.REG_EX:
            pattern = r"(\w+)\s*=\s*([^,}]+)"
            matches = re.findall(pattern, prompt) # [('seed', '21'), ('steps', '28'), ('sampler', 'euler')]
            try:
                options = dict(matches)
                if options != {}:
                    for key, value in options.items():
                        key = key.lower()
                        if key == 'seed':
                            try:
                                int(value)
                                ksampler_options['seed'] = int(value)
                            except ValueError:
                                continue
                        elif key == 'steps' or key == 'step':
                            try:
                                int(value)
                                ksampler_options['steps'] = max(min(int(value), max_steps), min_steps)
                            except ValueError:
                                continue
                        elif key == 'cfg':
                            try:
                                ksampler_options['cfg'] = max(min(float(value), max_cfg), min_cfg)
                            except ValueError:
                                continue
                        elif key == 'sampler_name':
                            if value in valid_samplers:
                                ksampler_options['sampler_name'] = value
                        elif key == 'scheduler':
                            if value in valid_schedulers:
                                ksampler_options['scheduler'] = value
                        elif key == 'denoise':
                            ksampler_options['denoise'] = max(min(float(value), max_denoise), min_denoise)
                            
                # Clean the prompt by removing the options from it
                prompt = re.sub(r"\{[^{}]+\}", "", prompt) # remove everything between {} including {}
                

            except Exception as e:
                logger.warning(f"Unexpected error while extracting ksampler options: {e}")
            
        # Clean the prompt by removing multiple spaces
        prompt = re.sub(r"\s+", " ", prompt).strip()
        
        return prompt, base64_image, ksampler_options

# ----------------------------------------------------------------------------------------------------

    async def enhance_prompt(self, prompt, image, user, request, event_emitter):
        """
        Enhances the prompt based on the given image using a vision model.
        """
        if self.__event_emitter__:
            await self.emit_status(event_emitter, "info", f"Enhancing the prompt...")
        payload = {
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
                            "text": f"Enhance the given user prompt based on the given image: {prompt}, provide only the enhanced AI image edit prompt with no explanations",
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

        response = await generate_chat_completion(request, payload, user)
        await self.emit_status(event_emitter, "info", f"Prompt enhanced")
        enhanced_prompt = response["choices"][0]["message"]["content"]
        enhanced_prompt_message = f"<details>\n<summary>Enhanced Prompt</summary>\n{enhanced_prompt}\n\n---\n\n</details>"
        await event_emitter(
            {
                "type": "message",
                "data": {
                    "content": enhanced_prompt_message,
                },
            }
        )
        return enhanced_prompt

# ----------------------------------------------------------------------------------------------------

    def prepare_workflow(self, workflow: dict, prompt: str, base64_image: str, ksampler_options: dict) -> dict:
        """
        Safely prepares the workflow dictionary by updating the required nodes.
        Logs warnings if required nodes are missing.
        """
        prompt_node = self.valves.PROMPT_NODE_ID
        image_node = self.valves.IMAGE_NODE_ID
        ksampler_node = self.valves.KSAMPLER_NODE_ID

        # Prompt
        if prompt_node in workflow and "inputs" in workflow[prompt_node]:
            workflow[prompt_node]["inputs"]["text"] = prompt or "A beautiful, high-quality image"
        else:
            logger.warning(f"Prompt node '{prompt_node}' not found in workflow.")

        # Image
        if image_node in workflow and "inputs" in workflow[image_node]:
            workflow[image_node]["inputs"]["data"] = base64_image
        else:
            logger.warning(f"Image node '{image_node}' not found in workflow.")

        # KSampler (update the ksampler node with the not None options)
        if ksampler_node in workflow and "inputs" in workflow[ksampler_node]:
            for key, value in ksampler_options.items():
                if value is not None:
                    workflow[ksampler_node]["inputs"][key] = value
        else:
            logger.warning(f"ksampler node '{ksampler_node}' not found in workflow.")

        # Auto check compatible model loader
        _workflow = self.auto_check_model_loader(workflow) if self.valves.AUTO_CHECK_MODEL_LOADER else workflow
        
        return _workflow

# ----------------------------------------------------------------------------------------------------
  
    async def queue_prompt(self, session: aiohttp.ClientSession, workflow: Dict) -> Optional[str]:
        """
        Queues a prompt for execution on ComfyUI.
        """
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with session.post(f"{self.valves.COMFYUI_ADDRESS.rstrip('/')}/prompt", json=payload) as response:
            text = await response.text()
            logger.info(f"Queue prompt HTTP {response.status}: {text}")
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Queue prompt JSON response: {data}")
            return data.get("prompt_id")

# ----------------------------------------------------------------------------------------------------

    async def wait_for_job_signal(self, ws_api_url: str, prompt_id: str, event_emitter: Callable) -> bool:
        """
        Waits for the 'executed' signal from WebSocket without fetching data.
        """
        start_time = asyncio.get_event_loop().time()
        try:
            async with aiohttp.ClientSession().ws_connect(f"{ws_api_url}?clientId={self.client_id}", timeout=30) as ws:
                async for msg in ws:
                    if (asyncio.get_event_loop().time() - start_time > self.valves.MAX_WAIT_TIME):
                        raise TimeoutError(f"WebSocket wait timed out after {self.valves.MAX_WAIT_TIME}s")
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.debug(f"WS raw msg: {msg.data}")
                    message = json.loads(msg.data)
                    msg_type, data = message.get("type"), message.get("data", {})

                    if msg_type == "status":
                        q_remaining = (
                            data.get("status", {})
                            .get("exec_info", {})
                            .get("queue_remaining", 0)
                        )
                        await self.emit_status(
                            event_emitter,
                            "info",
                            f"In queue... {q_remaining} tasks remaining.",
                        )
                    elif msg_type == "progress":
                        progress = int(data.get("value", 0) / data.get("max", 1) * 100)
                        await self.emit_status(
                            event_emitter, "info", f"Processing... {progress}%"
                        )
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

# ----------------------------------------------------------------------------------------------------

    def extract_image_data(self, outputs: Dict) -> Optional[Dict]:
        """
        Extracts the image data from the output dictionary of ComfyUI
        """
        final_image_data, temp_image_data = None, None
        for node_id, node_output in outputs.items():
            if "ui" in node_output and "images" in node_output.get("ui", {}):
                if node_output["ui"]["images"]:
                    final_image_data = node_output["ui"]["images"][0]
                    break
            elif "images" in node_output and not temp_image_data:
                if node_output["images"]:
                    temp_image_data = node_output["images"][0]
        return final_image_data if final_image_data else temp_image_data

# ----------------------------------------------------------------------------------------------------

    def _save_image_and_get_public_url(
        self, request, image_data: bytes, content_type: str, user: User
    ) -> str:
        """
        Saves the image data to OpenWebUI's file storage and returns a publicly accessible URL.
        This logic is adapted from OpenWebUI's native image generation handling.
        """
        try:
            image_format = mimetypes.guess_extension(content_type)
            if not image_format:

                image_format = ".png"

            file = UploadFile(
                file=io.BytesIO(image_data),
                filename=f"generated-image{image_format}",
                headers={"content-type": content_type},
            )

            file_item = upload_file_handler(
                request=request,
                file=file,
                metadata={},
                process=False,
                user=user,
            )
            if not file_item:
                logger.error("Failed to save image to OpenWebUI")
                raise Exception("Failed to save image to OpenWebUI")
            
            url = request.app.url_path_for("get_file_content_by_id", id=file_item.id)
            return url
        except Exception as e:
            logger.error(f"Error saving image to OpenWebUI: {e}", exc_info=True)
            raise e

    async def emit_status(
        self, event_emitter: Callable, level: str, description: str, done: bool = False
    ):
        if event_emitter:
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

    async def pipe(self,body: dict, __user__: dict, __event_emitter__: Callable, __request__=None, __task__=None,) -> dict:
        """
        Main function of the Pipe class.
        Handles prompt enhancement, vision tasks, Ollama unloading, workflow execution, etc...
        """
        self.__event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        messages = body.get("messages", [])
        prompt, base64_image, ksampler_options = self.setup_inputs(messages)
        
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
 
        # This part is ambiguous, so I'm leaving it here for now
        # but the vision model should not be used for image generation in this context
        if __task__ and __task__ != TASKS.DEFAULT:
            if self.valves.VISION_MODEL_ID:
                response = await generate_chat_completion(
                    self.__request__,
                    {
                        "model": self.valves.VISION_MODEL_ID,
                        "messages": body.get("messages"),
                        "stream": False,
                    },
                    user=self.__user__,
                )
                return f"{response['choices'][0]['message']['content']}"
            return "No vision model set for this task." # Fix: replaced return str "Edited Image"

        # Unload Ollama models
        if self.valves.UNLOAD_OLLAMA_MODELS:
            self.unload_all_models(api_url=self.valves.OLLAMA_URL)

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

        workflow = self.prepare_workflow(workflow, prompt, base64_image, ksampler_options)
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
                    f"Workflow queued. Waiting for completion signal...",
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
                    await self.emit_status(
                        self.__event_emitter__,
                        "info",
                        f"Downloading generated image...",
                    )

                    # Download and save image INSIDE the session context
                    async with session.get(internal_image_url) as http_response:
                        http_response.raise_for_status()
                        image_data = await http_response.read()
                        content_type = http_response.headers.get("content-type", "image/png")

                    await self.emit_status(
                        self.__event_emitter__, "info", f"Embedding image into chat..."
                    )

                    public_image_url = self._save_image_and_get_public_url(
                        request=self.__request__,
                        image_data=image_data,
                        content_type=content_type,
                        user=self.__user__,
                    )

                    alt_text = prompt if prompt else "Edited image generated by Flux Kontext"
                    response_content = f"Here is the edited image:\n\n![{alt_text}]({public_image_url})"

                    if self.__event_emitter__:
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

# ==========[Helper Functions]==========
# Ollama Model Management
    def get_loaded_models(self, api_url: str = "http://localhost:11434") -> list:
        try:
            response = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            logger.error(f"Error fetching loaded Ollama models: {e}")
            return []

    def unload_all_models(self, api_url: str = "http://localhost:11434"):
        try:
            models = self.get_loaded_models(api_url)
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
            
# Check and Update Model Loader
    def auto_check_model_loader(self, workflow: dict) -> dict:
        """
        In Flux Kontext, the model exists as:
            - Unquantized (.safetensors)
                - requires UNETLoader

            - Quantized (.gguf)
                - requires UnetLoaderGGUF
        """
        for node_id, node in workflow.items():
            if node["class_type"] in ["UNETLoader", "UnetLoaderGGUF"]:
                _node = workflow[node_id]
                extention = _node["inputs"]["unet_name"].split(".")[-1]
                if extention == "safetensors":
                    if "UNETLoader" not in _node["class_type"]:
                        workflow[node_id]["class_type"] = "UNETLoader"
                        logger.warning(f"Updated model loader '{node_id}' class_type to 'UNETLoader'")
                elif extention == "gguf":
                    if "UnetLoaderGGUF" not in _node["class_type"]:
                        workflow[node_id]["class_type"] = "UnetLoaderGGUF"
                        logger.warning(f"Updated model loader '{node_id}' class_type to 'UnetLoaderGGUF'")
        return workflow