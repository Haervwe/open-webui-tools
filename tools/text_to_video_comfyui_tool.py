"""
title: ComfyUI Text-to-Video Tool
description: Generate video from text prompt via ComfyUI workflow JSON. Uses ComfyUI HTTP+WebSocket API, supports unloading Ollama models before run, randomizes seed when missing, downloads the final video to OpenWebUI cache and emits an HTML video embed.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.1
license: MIT
"""

import aiohttp
import asyncio
import json
import random
import uuid
import os
import logging
import io
from fastapi import UploadFile
from open_webui.routers.files import upload_file_handler  # type: ignore
from typing import Any, Dict, Optional, Callable, Awaitable, List, Tuple
import time
from urllib.parse import quote
from pydantic import BaseModel, Field

import requests
from open_webui.models.users import Users

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Default Workflow ---

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")

DEFAULT_WORKFLOW: Dict[str, Any] = {
    "71": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Cargar CLIP"},
    },
    "72": {
        "inputs": {"text": "", "clip": ["71", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "73": {
        "inputs": {"vae_name": "wan/wan_2.1_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Cargar VAE"},
    },
    "74": {
        "inputs": {"width": 848, "height": 480, "length": 41, "batch_size": 1},
        "class_type": "EmptyHunyuanLatentVideo",
        "_meta": {"title": "EmptyHunyuanLatentVideo"},
    },
    "75": {
        "inputs": {
            "unet_name": "wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Cargar Modelo de Difusión"},
    },
    "76": {
        "inputs": {
            "unet_name": "wan/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Cargar Modelo de Difusión"},
    },
    "78": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
            "model": ["86", 0],
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["81", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Avanzado)"},
    },
    "80": {
        "inputs": {
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "codec": "auto",
            "video-preview": "",
            "video": ["88", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Guardar video"},
    },
    "81": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 742951153577776,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "model": ["82", 0],
            "positive": ["89", 0],
            "negative": ["72", 0],
            "latent_image": ["74", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Avanzado)"},
    },
    "82": {
        "inputs": {"shift": 5.000000000000001, "model": ["83", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "MuestreoDeModeloSD3"},
    },
    "83": {
        "inputs": {
            "lora_name": "wan2_2/14b/lightning/high_noise_model.safetensors",
            "strength_model": 1.0000000000000002,
            "model": ["75", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "CargadorLoRAModeloSolo"},
    },
    "85": {
        "inputs": {
            "lora_name": "wan2_2/14b/lightning/low_noise_model.safetensors",
            "strength_model": 1.0000000000000002,
            "model": ["76", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "CargadorLoRAModeloSolo"},
    },
    "86": {
        "inputs": {"shift": 5.000000000000001, "model": ["85", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "MuestreoDeModeloSD3"},
    },
    "88": {
        "inputs": {"fps": 16, "images": ["114", 0]},
        "class_type": "CreateVideo",
        "_meta": {"title": "Crear video"},
    },
    "89": {
        "inputs": {
            "text": "A golden retriever playing an electric guitar at a concert.",
            "clip": ["71", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "114": {
        "inputs": {
            "tile_size": 128,
            "overlap": 64,
            "temporal_size": 20,
            "temporal_overlap": 8,
            "samples": ["78", 0],
            "vae": ["73", 0],
        },
        "class_type": "VAEDecodeTiled",
        "_meta": {"title": "VAE Decodificar (Mosaico)"},
    },
    "115": {
        "inputs": {"anything": ["88", 0]},
        "class_type": "easy cleanGpuUsed",
        "_meta": {"title": "Clean VRAM Used"},
    },
}

# --- Helper Functions (Module-level) ---


def get_loaded_models(api_url: str) -> List[Dict[str, Any]]:
    """Get loaded Ollama models via the local Ollama API."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.RequestException as e:
        logger.debug("Error fetching loaded Ollama models: %s", e)
        return []


def unload_all_models(api_url: str) -> None:
    """Unload all Ollama models from VRAM."""
    models = get_loaded_models(api_url)
    if not models:
        return
    logger.debug("Unloading %d Ollama models...", len(models))
    for model in models:
        model_name = model.get("name") or model.get("model")
        if model_name:
            try:
                requests.post(
                    f"{api_url.rstrip('/')}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
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
    """Wait for ComfyUI websocket to report job completion and return history."""
    start_time = time.monotonic()
    async with aiohttp.ClientSession().ws_connect(
        f"{comfyui_ws_url}?clientId={client_id}"
    ) as ws:
        async for msg in ws:
            if time.monotonic() - start_time > max_wait_time:
                raise TimeoutError(f"WebSocket wait timed out after {max_wait_time}s")

            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    message = json.loads(msg.data)
                    data = message.get("data", {})
                    msg_type = message.get("type")

                    if (
                        msg_type == "execution_error"
                        and data.get("prompt_id") == prompt_id
                    ):
                        exc = data.get("exception_message", "Unknown error")
                        raise Exception(f"ComfyUI job failed: {exc}")

                    if msg_type == "executed" and data.get("prompt_id") == prompt_id:
                        async with aiohttp.ClientSession() as http_session:
                            async with http_session.get(
                                f"{comfyui_http_url}/history/{prompt_id}"
                            ) as resp:
                                resp.raise_for_status()
                                history = await resp.json()
                                return history.get(prompt_id, {})
                except (json.JSONDecodeError, aiohttp.ClientError):
                    continue
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                raise ConnectionError(
                    "WebSocket connection failed or closed unexpectedly."
                )
    raise TimeoutError("WebSocket connection closed before job completion.")


def extract_video_files(job_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    (Restored from original)
    Robustly extract video filenames from ComfyUI job outputs by recursively searching the data.
    """
    candidates: List[Tuple[str, str]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, str) and any(
            obj.lower().endswith(ext) for ext in VIDEO_EXTS
        ):
            candidates.append((obj, ""))
        elif isinstance(obj, dict):
            filename_val = None
            for k in ("filename", "file", "path", "name"):
                v = obj.get(k)
                if isinstance(v, str) and any(
                    v.lower().endswith(ext) for ext in VIDEO_EXTS
                ):
                    filename_val = v
                    break

            if filename_val:
                subfolder_val = ""
                for sf_key in ("subfolder", "subdir", "folder", "directory"):
                    sf = obj.get(sf_key)
                    if isinstance(sf, str):
                        subfolder_val = sf.strip("/ ")
                        break
                candidates.append((filename_val, subfolder_val))

            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(job_data)

    seen = set()
    video_files: List[Dict[str, str]] = []
    for fname, sub in candidates:
        filename = os.path.basename(str(fname))
        subfolder = str(sub).strip("/ ")
        key = f"{subfolder}/{filename}" if subfolder else filename
        if key not in seen:
            seen.add(key)
            video_files.append({"filename": filename, "subfolder": subfolder})

    return video_files


async def download_and_upload_to_owui(
    comfyui_http_url: str,
    filename: str,
    subfolder: str,
    request: Any,
    user: Any,
) -> Tuple[str, bool]:
    """Download video from ComfyUI and upload to OpenWebUI."""
    subfolder_param = f"&subfolder={quote(subfolder)}" if subfolder else ""
    comfyui_view_url = f"{comfyui_http_url}/api/viewvideo?filename={quote(filename)}&type=output{subfolder_param}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(comfyui_view_url) as response:
                response.raise_for_status()
                content = await response.read()

        if request and user:
            file = UploadFile(file=io.BytesIO(content), filename=filename)
            file_item = upload_file_handler(
                request=request, file=file, metadata={}, process=False, user=user
            )
            file_id = getattr(file_item, "id", None)

            if file_id:
                base_url = str(request.base_url).rstrip("/")
                relative_path = request.app.url_path_for(
                    "get_file_content_by_id", id=str(file_id)
                )
                return f"{base_url}{relative_path}?t={int(time.time())}", True

    except Exception as e:
        logger.debug("Failed to download or upload video to OpenWebUI: %s", e)

    return comfyui_view_url, False  # Fallback to direct ComfyUI URL


def prepare_workflow(
    base_workflow: Dict[str, Any], prompt: str, prompt_node_id: str
) -> Dict[str, Any]:
    """Create a copy of the workflow, inject the prompt, and randomize the seed."""
    workflow = json.loads(json.dumps(base_workflow))  # Deep copy

    if prompt_node_id not in workflow:
        raise ValueError(
            f"Prompt node ID '{prompt_node_id}' not found in the workflow."
        )
    workflow[prompt_node_id].setdefault("inputs", {})["text"] = prompt

    for node in workflow.values():
        if "class_type" in node and "KSampler" in node["class_type"]:
            inputs = node.setdefault("inputs", {})
            for seed_key in ("noise_seed", "seed"):
                if seed_key in inputs and inputs[seed_key] != 0:
                    inputs[seed_key] = random.randint(1, 2**31 - 1)

    return workflow


# --- Main Tool Class ---


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188", description="ComfyUI HTTP API endpoint."
        )
        workflow: Optional[Dict[str, Any]] = Field(
            default=None,
            description="ComfyUI workflow JSON. If empty, a default is used.",
        )
        prompt_node_id: str = Field(
            default="89", description="Node ID for the text prompt input."
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

    def __init__(self):
        self.valves = self.Valves()

    async def generate_video(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
    ) -> str:
        """Generate a video from a text prompt using the provided ComfyUI workflow."""
        try:
            if self.valves.unload_ollama_models:
                unload_all_models(self.valves.ollama_api_url)

            base_workflow = self.valves.workflow or DEFAULT_WORKFLOW
            active_workflow = prepare_workflow(
                base_workflow, prompt, self.valves.prompt_node_id
            )

            http_api_url = self.valves.comfyui_api_url.rstrip("/")
            ws_api_url = http_api_url.replace("http", "ws") + "/ws"
            client_id = str(uuid.uuid4())
            payload = {"prompt": active_workflow, "client_id": client_id}

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "🎬 Submitting to ComfyUI...",
                            "done": False,
                        },
                    }
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{http_api_url}/prompt", json=payload) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        raise RuntimeError(
                            f"ComfyUI did not return a prompt_id. Response: {result}"
                        )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⏳ Waiting for video generation...",
                            "done": False,
                        },
                    }
                )

            job_data = await wait_for_completion_ws(
                ws_api_url,
                http_api_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
            )

            video_files = extract_video_files(job_data)
            if not video_files:
                logger.warning(
                    "No video files extracted from job data: %s",
                    json.dumps(job_data, indent=2),
                )
                return "ComfyUI job completed, but no video output was found in the results."

            video_info = video_files[0]
            current_user = Users.get_user_by_id(__user__["id"]) if __user__ else None

            final_url, _ = await download_and_upload_to_owui(
                http_api_url,
                video_info["filename"],
                video_info["subfolder"],
                __request__,
                current_user,
            )

            if __event_emitter__:
                html_player = f'<video controls src="{final_url}" width="960" style="max-width:100%"></video>'
                await __event_emitter__(
                    {"type": "embeds", "data": {"embeds": [html_player]}}
                )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "✅ Video Generated!", "done": True},
                    }
                )

            return f"🎬 Video generated successfully. Link: {final_url}"

        except Exception as e:
            logger.error(f"Error during video generation: {e}", exc_info=True)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ Error: {e}", "done": True},
                    }
                )
            return f"An error occurred: {e}"
