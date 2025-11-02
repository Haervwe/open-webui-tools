"""
title: ComfyUI Image-to-Image Tool
description: Edit/transform images using ComfyUI workflows (Flux Kontext or Qwen Edit). Uses ComfyUI HTTP+WebSocket API, supports unloading Ollama models before run, randomizes seed, and returns the edited image inline.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.0
license: MIT
"""

import json
import random
import time
import uuid
import logging
import aiohttp
import io
import requests
from typing import Any, Dict, Optional, Callable, Awaitable, List, Literal
from urllib.parse import quote
from pydantic import BaseModel, Field
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from open_webui.routers.files import upload_file_handler
from open_webui.models.users import Users
from open_webui.utils.misc import get_last_user_message_item

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DEFAULT_FLUX_KONTEXT_WORKFLOW: Dict[str, Any] = {
    "6": {
        "inputs": {"text": "", "clip": ["38", 0]},
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
            "unet_name": "flux-kontext/flux1-dev-kontext_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Unet Loader"},
    },
    "205": {
        "inputs": {"filename_prefix": "ComfyUI_Image2Image", "images": ["209", 0]},
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
}

DEFAULT_QWEN_EDIT_WORKFLOW: Dict[str, Any] = {
    "3": {
        "inputs": {
            "seed": random.randint(0, 2**32 - 1),
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["75", 0],
            "positive": ["111", 0],
            "negative": ["110", 0],
            "latent_image": ["88", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "8": {
        "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "37": {
        "inputs": {
            "unet_name": "qwen-image/Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors",
            "weight_dtype": "default",
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
    "60": {
        "inputs": {"filename_prefix": "Owui_qwen_edit_2509", "images": ["389", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "66": {
        "inputs": {"shift": 3, "model": ["89", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "75": {
        "inputs": {"strength": 1, "model": ["66", 0]},
        "class_type": "CFGNorm",
        "_meta": {"title": "CFGNorm"},
    },
    "88": {
        "inputs": {"pixels": ["93", 0], "vae": ["39", 0]},
        "class_type": "VAEEncode",
        "_meta": {"title": "VAE Encode"},
    },
    "89": {
        "inputs": {
            "lora_name": "qwen-image/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            "strength_model": 1,
            "model": ["37", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoRA Loader (Model Only)"},
    },
    "93": {
        "inputs": {
            "upscale_method": "lanczos",
            "megapixels": 1,
            "image": ["390", 0],
        },
        "class_type": "ImageScaleToTotalPixels",
        "_meta": {"title": "Scale Image to Total Pixels"},
    },
    "110": {
        "inputs": {
            "prompt": "",
            "clip": ["38", 0],
            "vae": ["39", 0],
            "image1": ["93", 0],
            "image2": ["391", 0],
            "image3": ["392", 0],
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "TextEncodeQwenImageEditPlus (Negative)"},
    },
    "111": {
        "inputs": {
            "prompt": "",
            "clip": ["38", 0],
            "vae": ["39", 0],
            "image1": ["93", 0],
            "image2": ["391", 0],
            "image3": ["392", 0],
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "TextEncodeQwenImageEditPlus (Positive)"},
    },
    "389": {
        "inputs": {"anything": ["8", 0]},
        "class_type": "easy cleanGpuUsed",
        "_meta": {"title": "Clean VRAM Used"},
    },
    "390": {
        "inputs": {"image": ""},
        "class_type": "ETN_LoadImageBase64",
        "_meta": {"title": "Load Image (Base64) - Image 1"},
    },
    "391": {
        "inputs": {"image": ""},
        "class_type": "ETN_LoadImageBase64",
        "_meta": {"title": "Load Image (Base64) - Image 2"},
    },
    "392": {
        "inputs": {"image": ""},
        "class_type": "ETN_LoadImageBase64",
        "_meta": {"title": "Load Image (Base64) - Image 3"},
    },
}


def get_loaded_models(api_url: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.RequestException:
        logger.debug("Error fetching loaded Ollama models")
        return []


def unload_all_models(api_url: str) -> None:
    models = get_loaded_models(api_url)
    if not models:
        return
    logger.debug("Unloading %d Ollama models...", len(models))
    for model in models:
        model_name = model.get("name") or model.get("model")
        if model_name:
            try:
                payload = {"model": model_name, "keep_alive": 0}
                requests.post(
                    f"{api_url.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=10,
                )
            except requests.RequestException:
                pass


async def wait_for_completion_ws(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_id: str,
    client_id: str,
    max_wait_time: int,
) -> Dict[str, Any]:
    start_time = time.monotonic()
    async with aiohttp.ClientSession().ws_connect(
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
                        async with aiohttp.ClientSession() as http_session:
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
                                    {
                                        "filename": filename,
                                        "subfolder": str(subfolder).strip("/"),
                                    }
                                )

    return image_files


async def download_and_upload_to_owui(
    comfyui_http_url: str,
    filename: str,
    subfolder: str,
    request: Any,
    user: Any,
) -> tuple[str, bool]:
    subfolder_param = f"&subfolder={quote(subfolder)}" if subfolder else ""
    comfyui_view_url = f"{comfyui_http_url}/view?filename={quote(filename)}&type=output{subfolder_param}"

    try:
        async with aiohttp.ClientSession() as session:
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
    workflow_type: str,
    prompt: str,
    base64_images: List[str],
) -> Dict[str, Any]:
    workflow = json.loads(json.dumps(base_workflow))

    if workflow_type == "Flux_Kontext":
        if "6" in workflow:
            workflow["6"].setdefault("inputs", {})["text"] = prompt
        if "196" in workflow and len(base64_images) > 0:
            workflow["196"].setdefault("inputs", {})["data"] = base64_images[0]

        if "194" in workflow and "inputs" in workflow["194"]:
            workflow["194"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

    elif workflow_type == "QWen_Edit":
        if "111" in workflow:
            workflow["111"].setdefault("inputs", {})["prompt"] = prompt

        image_loader_nodes = ["390", "391", "392"]
        image_input_keys = ["image1", "image2", "image3"]

        num_images = len(base64_images)

        for idx in range(num_images):
            node_id = image_loader_nodes[idx]
            if node_id in workflow:
                workflow[node_id].setdefault("inputs", {})["image"] = base64_images[idx]

        for idx in range(num_images, len(image_loader_nodes)):
            node_id = image_loader_nodes[idx]
            if node_id in workflow:
                del workflow[node_id]

        for conditioning_node_id in ["110", "111"]:
            if conditioning_node_id in workflow and "inputs" in workflow[conditioning_node_id]:
                for idx in range(num_images, len(image_input_keys)):
                    image_key = image_input_keys[idx]
                    if image_key in workflow[conditioning_node_id]["inputs"]:
                        del workflow[conditioning_node_id]["inputs"][image_key]

        if "3" in workflow and "inputs" in workflow["3"]:
            workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

    else:
        for node_id, node in workflow.items():
            if "class_type" in node:
                class_type = node["class_type"]

                if "CLIPTextEncode" in class_type or "TextEncode" in class_type:
                    inputs = node.setdefault("inputs", {})
                    if "text" in inputs or "prompt" in inputs:
                        key = "text" if "text" in inputs else "prompt"
                        inputs[key] = prompt

                if (
                    "LoadImageFromBase64" in class_type
                    or "LoadImageBase64" in class_type
                    or "ETN_LoadImageBase64" in class_type
                ):
                    inputs = node.setdefault("inputs", {})
                    if "data" in inputs or "image" in inputs:
                        key = "data" if "data" in inputs else "image"
                        inputs[key] = base64_images[0] if len(base64_images) > 0 else ""

                if "KSampler" in class_type:
                    inputs = node.setdefault("inputs", {})
                    for seed_key in ("noise_seed", "seed"):
                        if seed_key in inputs:
                            inputs[seed_key] = random.randint(0, 2**32 - 1)

    return workflow


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    last_user_message = None
    for message in reversed(messages):
        if message.get("role") == "user":
            last_user_message = message
            break

    if not last_user_message:
        return []

    base64_images: List[str] = []

    content = last_user_message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_url_obj = item.get("image_url", {})
                url = image_url_obj.get("url") if isinstance(image_url_obj, dict) else None
                if url and isinstance(url, str) and url.startswith("data:image"):
                    base64_data = url.split(",", 1)[1] if "," in url else url
                    base64_images.append(base64_data)

    return base64_images

# --- Main Tool Class ---


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(default="http://localhost:8188", description="ComfyUI HTTP API endpoint.")
        workflow_type: Literal["Flux_Kontext", "QWen_Edit", "Custom"] = Field(default="QWen_Edit", description="Workflow to use for image editing.")
        custom_workflow: Optional[Dict[str, Any]] = Field(default=None, description="Custom ComfyUI workflow JSON (only used if workflow_type='Custom').")
        max_wait_time: int = Field(default=600, description="Max wait time in seconds for job completion.")
        unload_ollama_models: bool = Field(default=False, description="Unload Ollama models before calling ComfyUI.")
        ollama_api_url: str = Field(default="http://localhost:11434", description="Ollama API URL for unloading models.")
        return_html_embed: bool = Field(default=True, description="Return an HTML image embed upon completion.")

    def __init__(self):
        self.valves = self.Valves()

    async def edit_image(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> str | HTMLResponse:
        """
        Edit or transform images using AI-powered workflows. The tool automatically extracts images 
        from your message attachments and applies your text instructions to modify them.
        
        **How it works:**
        - Images are automatically extracted from your message (you don't need to specify them in the prompt)
        - Up to 3 images are passed to the AI model along with your text prompt
        - The AI interprets your instructions and modifies the image(s) accordingly
        - First image is the main subject; additional images provide context, reference, or style guidance
        
        **Prompting Strategies:**
        
        For precise edits (single or multiple images):
        - Direct instructions: "Remove the background", "Change the sky to sunset", "Make the car red"
        - Object-focused: "Remove all text from the image", "Delete the person on the left"
        - Style transfer: "Apply the color grading from image 1 to image 2" (with 2+ images)
        - Contextual edits: "Replace the background with a beach scene while keeping the subject"
        
        For artistic transformations:
        - Style descriptions: "Transform into a watercolor painting", "Make it look like a vintage photograph"
        - Scene reimagining: "Place the subject in a cyberpunk city", "Transform into an anime art style"
        - Atmospheric changes: "Add dramatic lighting", "Make it look like a rainy night"
        
        For multi-image editing (2-3 images):
        - "Blend the composition of image 1 with the style of image 2"
        - "Use the lighting from the first image on the second image"
        - "Transfer the color palette from image 1 to image 2"
        
        **Tips:**
        - Be specific and descriptive in your instructions
        - Describe what to change rather than the entire scene
        - When using multiple images, clearly indicate which image(s) to use as reference
        - Images are automatically processed - no need to mention "the attached image" in your prompt

        :param prompt: Natural language instruction describing the desired image transformation or edit
        """
        try:
            # Extract images from messages
            if not __messages__:
                return "Error: No messages provided. Please attach an image to your message."

            base64_images = extract_images_from_messages(__messages__)
            if not base64_images or len(base64_images) == 0:
                return "Error: No images found in the last message. Please attach at least one image and try again."
            
            # Warn if more than 3 images were provided
            if len(base64_images) > 3 and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"⚠️ Found {len(base64_images)} images, but only the first 3 will be used.",
                            "done": False,
                        },
                    }
                )
            
            # Keep only the first 3 images
            base64_images = base64_images[:3]

            if self.valves.unload_ollama_models:
                unload_all_models(self.valves.ollama_api_url)

            # Select workflow based on type
            if self.valves.workflow_type == "Flux_Kontext":
                base_workflow = DEFAULT_FLUX_KONTEXT_WORKFLOW
            elif self.valves.workflow_type == "QWen_Edit":
                base_workflow = DEFAULT_QWEN_EDIT_WORKFLOW
            elif self.valves.workflow_type == "Custom":
                if not self.valves.custom_workflow:
                    return "Error: Custom workflow selected but no custom workflow JSON provided in valves."
                base_workflow = self.valves.custom_workflow
            else:
                return f"Error: Unknown workflow type '{self.valves.workflow_type}'. Use 'Flux_Kontext', 'QWen_Edit', or 'Custom'."

            # Prepare workflow
            active_workflow = prepare_workflow(
                base_workflow, self.valves.workflow_type, prompt, base64_images
            )

            http_api_url = self.valves.comfyui_api_url.rstrip("/")
            ws_api_url = http_api_url.replace("http", "ws") + "/ws"
            client_id = str(uuid.uuid4())
            payload: Dict[str, Any] = {
                "prompt": active_workflow,
                "client_id": client_id,
            }

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"🎨 Editing image using {self.valves.workflow_type}...",
                            "done": False,
                        },
                    }
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{http_api_url}/prompt",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return f"ComfyUI API error on submission ({resp.status}): {await resp.text()}"
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        return f"Error: No prompt_id from ComfyUI. Response: {json.dumps(result)}"


            job_data = await wait_for_completion_ws(
                ws_api_url,
                http_api_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
            )

            image_files = extract_image_files(job_data)

            if not image_files:
                outputs_json = json.dumps(job_data.get("outputs", {}), indent=2)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "error",
                            "data": {
                                "description": "No output image found.",
                                "done": True,
                            },
                        }
                    )
                return (
                    f"Generation completed (Job: {prompt_id}) but no image files found. "
                    f"Job outputs: ```json\n{outputs_json}\n```"
                )

            image_file_info = image_files[0]
            filename = image_file_info["filename"]
            subfolder = image_file_info.get("subfolder", "")

            # Try to upload to OpenWebUI
            user = None
            if __user__:
                user_id = __user__.get("id")
                if user_id:
                    user = Users.get_user_by_id(user_id)

            image_url, uploaded = await download_and_upload_to_owui(
                http_api_url, filename, subfolder, __request__, user
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "🖼️ Image Generated Successfully!",
                            "done": True,
                        },
                    }
                )

            if self.valves.return_html_embed:
                # Return embedded image
                # Prepare prompt display (truncate if too long)
                prompt_display = prompt[:200] + ('...' if len(prompt) > 200 else '')
                workflow_title = self.valves.workflow_type.replace('_', ' ').title()
                
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edited Image</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
        }}
        body {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .image-container {{
            background: rgba(20, 20, 25, 0.25);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            max-width: 800px;
            width: 100%;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .title {{
            font-size: 20px;
            font-weight: 600;
            color: #f0f0f0;
            margin-bottom: 5px;
        }}
        .subtitle {{
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .image-wrapper {{
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            margin-bottom: 16px;
        }}
        .image-wrapper img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .info {{
            font-size: 13px;
            color: #ccc;
            line-height: 1.6;
            margin-bottom: 16px;
        }}
        .download-btn {{
            display: inline-block;
            width: 100%;
            padding: 11px 18px;
            background: rgba(255, 255, 255, 0.08);
            color: #e0e0e0;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: 13px;
            transition: all 0.2s;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .download-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.2);
        }}
    </style>
</head>
<body>
    <div class="image-container">
        <div class="header">
            <div class="title">Edited Image</div>
            <div class="subtitle">{workflow_title}</div>
        </div>
        <div class="image-wrapper">
            <img src="{image_url}" alt="Edited Image">
        </div>
        <div class="info">
            <strong>Prompt:</strong> {prompt_display}
        </div>
        <a href="{image_url}" download class="download-btn">
            <svg viewBox="0 0 24 24" style="width:14px;height:14px;fill:currentColor;vertical-align:middle;margin-right:6px;">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
            Download Image
        </a>
    </div>
</body>
</html>"""
                return HTMLResponse(
                    content=html_content, headers={"content-disposition": "inline"}
                )
            else:
                return f"✅ Image transformation complete!\n\n**Download:** [Edited Image]({image_url})\n\n**Direct link:** {image_url}"

        except TimeoutError as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"❌ Timeout: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"⏰ Generation timed out: {str(e)}"
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"❌ Error: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"❌ An error occurred: {str(e)}"
