"""
title: ComfyUI Text-to-Image Tool
description: Generate images using ComfyUI Qwen Image workflow. Uses ComfyUI HTTP+WebSocket API, supports unloading Ollama models before run, randomizes seed, and returns the generated image inline.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.0
license: MIT
"""

import io
import json
import logging
import random
import time
import uuid
import aiohttp

from typing import Any, Dict, Optional, Callable, Awaitable, List
from urllib.parse import quote
from pydantic import BaseModel, Field
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from open_webui.routers.files import upload_file_handler
from open_webui.models.users import Users

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DEFAULT_QWEN_T2I_WORKFLOW: Dict[str, Any] = {
    "3": {
        "inputs": {
            "seed": random.randint(0, 2**32 - 1),
            "steps": 4,
            "cfg": 1,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["66", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["58", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "6": {
        "inputs": {
            "text": "",
            "clip": ["38", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "7": {
        "inputs": {
            "text": "",
            "clip": ["38", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "8": {
        "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "37": {
        "inputs": {
            "unet_name": "qwen-image/qwen_image_fp8_e4m3fn.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "38": {
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "39": {
        "inputs": {"vae_name": "qwen/qwen_image_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "58": {
        "inputs": {"width": 1328, "height": 1328, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
        "_meta": {"title": "EmptySD3LatentImage"},
    },
    "60": {
        "inputs": {"filename_prefix": "ComfyUI_T2I", "images": ["75", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "66": {
        "inputs": {"shift": 3.1, "model": ["73", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "73": {
        "inputs": {
            "lora_name": "qwen-image/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
            "strength_model": 1,
            "model": ["37", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoRA Loader (Model Only)"},
    },
    "75": {
        "inputs": {"anything": ["8", 0]},
        "class_type": "easy cleanGpuUsed",
        "_meta": {"title": "Clean VRAM Used"},
    },
}


async def get_loaded_models_async(api_url: str) -> List[Dict[str, Any]]:
    """Get all currently loaded models in VRAM"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_url.rstrip('/')}/api/ps", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return data.get("models", [])
    except Exception:
        logger.debug("Error fetching loaded Ollama models")
        return []


async def unload_all_models_async(api_url: str) -> bool:
    """Unload all currently loaded models from VRAM"""
    try:
        models = await get_loaded_models_async(api_url)
        if not models:
            return True
        logger.debug("Unloading %d Ollama models...", len(models))
        async with aiohttp.ClientSession() as session:
            for model in models:
                model_name = model.get("name") or model.get("model")
                if model_name:
                    payload = {"model": model_name, "keep_alive": 0}
                    try:
                        async with session.post(
                            f"{api_url.rstrip('/')}/api/generate",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            pass
                    except Exception:
                        pass
        return True
    except Exception as e:
        logger.debug(f"Error unloading models: {e}")
        return False


async def wait_for_completion_ws(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_id: str,
    client_id: str,
    max_wait_time: int,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Wait for ComfyUI job completion via WebSocket"""
    start_time = time.monotonic()
    async with aiohttp.ClientSession(headers=headers).ws_connect(
        f"{comfyui_ws_url}?clientId={client_id}"
    ) as ws:
        async for msg in ws:
            if time.monotonic() - start_time > max_wait_time:
                raise TimeoutError(f"Job timed out after {max_wait_time}s")

            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    message = json.loads(msg.data)
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "executed" and data.get("prompt_id") == prompt_id:
                        async with aiohttp.ClientSession(headers=headers) as http_session:
                            async with http_session.get(
                                f"{comfyui_http_url}/history/{prompt_id}"
                            ) as resp:
                                if resp.status == 200:
                                    history = await resp.json()
                                    if prompt_id in history:
                                        return history[prompt_id]
                        raise Exception("Job completed but couldn't retrieve history")

                    elif (
                        msg_type == "execution_error"
                        and data.get("prompt_id") == prompt_id
                    ):
                        error_details = data.get("exception_message", "Unknown error")
                        node_id = data.get("node_id", "N/A")
                        raise Exception(
                            f"ComfyUI job failed on node {node_id}: {error_details}"
                        )

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    if "ComfyUI job" in str(e) or isinstance(e, TimeoutError):
                        raise

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                break

    raise TimeoutError("WebSocket connection closed before job completion.")


def extract_image_files(job_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract image file information from job data"""
    image_files: List[Dict[str, str]] = []
    node_outputs = job_data.get("outputs", {})

    for node_id, node_output in node_outputs.items():
        if isinstance(node_output, dict):
            for key in ["images", "files", "output"]:
                if key in node_output and isinstance(node_output[key], list):
                    for file_info in node_output[key]:
                        if isinstance(file_info, dict):
                            filename = file_info.get("filename")
                            subfolder = file_info.get("subfolder", "")
                            if filename and any(
                                filename.lower().endswith(ext) for ext in IMAGE_EXTS
                            ):
                                image_files.append(
                                    {"filename": filename, "subfolder": subfolder}
                                )

    return image_files


async def download_and_upload_to_owui(
    comfyui_http_url: str,
    filename: str,
    subfolder: str,
    request: Any,
    user: Any,
    headers: Optional[Dict[str, str]] = None,
) -> tuple[str, bool]:
    """Download image from ComfyUI and upload to OpenWebUI"""
    subfolder_param = f"&subfolder={quote(subfolder)}" if subfolder else ""
    comfyui_view_url = f"{comfyui_http_url}/view?filename={quote(filename)}&type=output{subfolder_param}"

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(comfyui_view_url) as response:
                if response.status != 200:
                    return comfyui_view_url, False
                content = await response.read()

        if request and user:
            file = UploadFile(file=io.BytesIO(content), filename=filename)
            file_item = upload_file_handler(
                request=request, file=file, metadata={}, process=False, user=user
            )
            file_id = getattr(file_item, "id", None)

            if file_id:
                return f"/api/v1/files/{file_id}/content", True

    except Exception:
        logger.debug("Failed to download or upload image to OpenWebUI")

    return comfyui_view_url, False


def prepare_workflow(
    base_workflow: Dict[str, Any],
    prompt: str,
) -> Dict[str, Any]:
    """Prepare workflow with prompt and randomized seed"""
    workflow = json.loads(json.dumps(base_workflow))

    # Set prompt in node 6 (positive)
    if "6" in workflow:
        workflow["6"].setdefault("inputs", {})["text"] = prompt

    # Randomize seed in node 3 (KSampler)
    if "3" in workflow and "inputs" in workflow["3"]:
        workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

    return workflow


# --- Main Tool Class ---


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188", description="ComfyUI HTTP API endpoint."
        )
        comfyui_api_key: str = Field(
            default="",
            description="API key for ComfyUI authentication (Bearer token). Leave empty if not required.",
            json_schema_extra={"input": {"type": "password"}},
        )
        custom_workflow: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Custom ComfyUI workflow JSON (if not provided, uses default Qwen T2I workflow).",
        )
        max_wait_time: int = Field(
            default=600, description="Max wait time in seconds for job completion."
        )
        unload_ollama_models: bool = Field(
            default=False, description="Unload Ollama models before calling ComfyUI."
        )
        ollama_api_url: str = Field(
            default="http://localhost:11434",
            description="Ollama API URL for unloading models.",
        )
        return_html_embed: bool = Field(
            default=True,
            description="Return an HTML image embed. If False, returns image URLs as plain text.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def generate_image(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
    ) -> str | HTMLResponse:
        """
        Generate an image from a text prompt using ComfyUI Qwen Image workflow.

        QWEN IMAGE PROMPTING GUIDE:

        Qwen Image excels at generating detailed, photorealistic images with accurate text rendering.

        PROMPT STRUCTURE (use all elements for best results):
        [Main Subject] + [Visual Style/Medium] + [Environment & Background] + [Lighting] + [Camera/Lens Language] + [Atmosphere & Details] + ["Exact Text in Quotes"]

        KEY PROMPTING RULES:

        1. **Text Rendering** (Qwen's Strength):
           - Always enclose exact text in "double quotes"
           - Specify font style, color, size if important
           - Example: "A neon sign reading \"OPEN 24/7\" in bright red letters"

        2. **Five Major Elements**:
           - **Framing/Perspective**: eye-level, bird's eye view, low angle, close-up, wide shot
           - **Lens Type**: 50mm, wide-angle, telephoto, macro
           - **Style**: photorealistic, cinematic, oil painting, watercolor, anime
           - **Lighting**: golden hour, soft morning light, dramatic shadows, neon glow
           - **Atmosphere**: warm, moody, vibrant, ethereal

        3. **Be Specific & Detailed**:
           - Describe subjects thoroughly (age, appearance, clothing, actions)
           - Include scene details (location, weather, time of day)
           - Add sensory details (textures, materials, colors)

        4. **Natural Language**:
           - Write in clear, descriptive sentences
           - Use commas to separate elements
           - Be conversational but precise

        EXAMPLE PROMPTS:

        Good: "A cozy coffee shop interior, warm afternoon lighting streaming through large windows, a wooden sign on the brick wall reading \"DAILY GRIND\", vintage industrial style with exposed beams, potted plants on shelves, shot with 35mm lens at eye level, warm and inviting atmosphere"

        Good: "Photorealistic portrait of a young woman in her 20s with long auburn hair, wearing a denim jacket, standing in front of a street sign that says \"MAIN ST\", golden hour lighting, shallow depth of field, urban background slightly blurred, warm color grading"

        Good: "A vibrant neon-lit street scene in Hong Kong at night, multiple overlapping signs with Chinese and English text including \"龍鳳茶餐廳\" and \"HAPPY KARAOKE\", rain-soaked pavement reflecting colorful lights, shot from eye level, cinematic composition, moody cyberpunk atmosphere"

        :param prompt: Detailed text description following Qwen Image prompting guidelines above
        :return: HTML embed with generated image to the UI and a plain notification, or plain text URLs
        """
        try:
            # Unload Ollama models if enabled
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Unloading Ollama models...",
                                "done": False,
                            },
                        }
                    )
                await unload_all_models_async(self.valves.ollama_api_url)

            # Prepare workflow
            base_workflow = (
                self.valves.custom_workflow
                if self.valves.custom_workflow
                else DEFAULT_QWEN_T2I_WORKFLOW
            )
            workflow = prepare_workflow(base_workflow, prompt)

            # Generate client ID and prepare request
            client_id = str(uuid.uuid4())
            comfyui_http_url = self.valves.comfyui_api_url.rstrip("/")
            comfyui_ws_url = comfyui_http_url.replace("http://", "ws://").replace(
                "https://", "wss://"
            )
            comfyui_ws_url = f"{comfyui_ws_url}/ws"

            # Build auth headers for ComfyUI
            comfyui_headers = {}
            if self.valves.comfyui_api_key:
                comfyui_headers["Authorization"] = f"Bearer {self.valves.comfyui_api_key}"

            payload = {"prompt": workflow, "client_id": client_id}
            async with aiohttp.ClientSession(headers=comfyui_headers) as session:
                async with session.post(
                    f"{comfyui_http_url}/prompt", json=payload
                ) as resp:
                    if resp.status != 200:
                        raise Exception(
                            f"Failed to submit job to ComfyUI: {resp.status}"
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id returned from ComfyUI")

            # Wait for completion
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Generating image...",
                            "done": False,
                        },
                    }
                )

            job_data = await wait_for_completion_ws(
                comfyui_ws_url,
                comfyui_http_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
                comfyui_headers,
            )

            # Extract image files
            image_files = extract_image_files(job_data)
            if not image_files:
                raise Exception("No images found in ComfyUI output")

            image_urls: List[str] = []
            for img_info in image_files:
                url, uploaded = await download_and_upload_to_owui(
                    comfyui_http_url,
                    img_info["filename"],
                    img_info["subfolder"],
                    __request__,
                    Users.get_user_by_id(__user__["id"]) if __user__ else None,
                    comfyui_headers,
                )
                image_urls.append(url)

            # Emit completion status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Image generation complete!",
                            "done": True,
                        },
                    }
                )

            # Return based on valve setting
            if not self.valves.return_html_embed:
                # Return plain text URLs
                urls_text = "\n".join(image_urls)
                return f"Images generated successfully. Download links:\n{urls_text}"

            # Build HTML embed with enhanced styling
            image_url = image_urls[0] if image_urls else ""

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
        .image-wrapper {{
            margin: 0 0 8px 0;
            padding: 0;
            border-radius: 12px;
            overflow: hidden;
            line-height: 0;
        }}
        .image-wrapper img {{
            max-width: 100%;
            height: auto;
            display: block;
            border: none;
            margin: 0;
            padding: 0;
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
        <div class="image-wrapper">
            <img src="{image_url}" alt="Generated Image" />
        </div>
        <div class="prompt-bubble">
            <div class="prompt-label">Prompt</div>
            <div class="prompt-text">{prompt}</div>
        </div>
    </div>
</body>
</html>"""

            return HTMLResponse(
                content=html_content,
                media_type="text/html",
                headers={"content-disposition": "inline"},
            )

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error generating image: {str(e)}"
