"""
title: ComfyUI Text-to-Video Tool
description: Generate video from text prompt via ComfyUI workflow JSON. Uses ComfyUI HTTP+WebSocket API with robust fallback polling, supports unloading Ollama models before run, randomizes seed, downloads the final video to OpenWebUI storage and emits an HTML video embed.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.3.0
license: MIT
"""

import aiohttp
import asyncio
import json
import random
import uuid
import os
import io
import logging
import time
from typing import Any, Dict, Optional, Callable, Awaitable, List, Tuple, cast
from urllib.parse import quote
from pydantic import BaseModel, Field
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---

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


async def get_loaded_models_async(
    api_url: str = "http://localhost:11434",
) -> list[Dict[str, Any]]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url.rstrip('/')}/api/ps") as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return cast(list[Dict[str, Any]], data.get("models", []))
    except Exception as e:
        logger.debug("Error fetching loaded models: %s", e)
        return []


async def unload_all_models_async(api_url: str = "http://localhost:11434") -> bool:
    try:
        loaded_models = await get_loaded_models_async(api_url)
        if not loaded_models:
            return True

        async with aiohttp.ClientSession() as session:
            for model in loaded_models:
                model_name = model.get("name", model.get("model", ""))
                if model_name:
                    payload = {"model": model_name, "keep_alive": 0}
                    async with session.post(
                        f"{api_url.rstrip('/')}/api/generate", json=payload
                    ) as resp:
                        pass

        for _ in range(5):
            await asyncio.sleep(1)
            remaining = await get_loaded_models_async(api_url)
            if not remaining:
                logger.info("All models successfully unloaded.")
                return True

        logger.warning("Some models might still be loaded after timeout.")
        return False

    except Exception as e:
        logger.error("Error unloading models: %s", e)
        return False


async def connect_submit_and_wait(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_payload: Dict[str, Any],
    client_id: str,
    max_wait_time: int,
) -> Dict[str, Any]:
    start_time = asyncio.get_event_loop().time()
    prompt_id = None

    async with aiohttp.ClientSession() as session:
        ws_url = f"{comfyui_ws_url}?clientId={client_id}"
        try:
            async with session.ws_connect(ws_url) as ws:
                async with session.post(
                    f"{comfyui_http_url}/prompt", json=prompt_payload
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise Exception(
                            f"Failed to queue prompt: {resp.status} - {err_text}"
                        )
                    resp_json = await resp.json()
                    prompt_id = resp_json.get("prompt_id")
                    if not prompt_id:
                        raise Exception("No prompt_id received from ComfyUI")

                last_poll_time = 0
                poll_interval = 3.0

                while True:
                    execute_poll = False
                    if asyncio.get_event_loop().time() - start_time > max_wait_time:
                        raise TimeoutError(
                            f"Generation timed out after {max_wait_time}s"
                        )

                    if asyncio.get_event_loop().time() - last_poll_time > poll_interval:
                        execute_poll = True
                        last_poll_time = asyncio.get_event_loop().time()

                    if execute_poll and prompt_id:
                        try:
                            async with session.get(
                                f"{comfyui_http_url}/history/{prompt_id}"
                            ) as history_resp:
                                if history_resp.status == 200:
                                    history = await history_resp.json()
                                    if prompt_id in history:
                                        return history[prompt_id]
                        except Exception:
                            pass

                    try:
                        msg = await ws.receive(timeout=1.0)

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            message = json.loads(msg.data)
                            msg_type = message.get("type")
                            data = message.get("data", {})

                            if (
                                msg_type == "execution_cached" or msg_type == "executed"
                            ) and data.get("prompt_id") == prompt_id:
                                async with session.get(
                                    f"{comfyui_http_url}/history/{prompt_id}"
                                ) as final_resp:
                                    if final_resp.status == 200:
                                        history = await final_resp.json()
                                        if prompt_id in history:
                                            return history[prompt_id]

                            elif (
                                msg_type == "execution_error"
                                and data.get("prompt_id") == prompt_id
                            ):
                                error_details = data.get(
                                    "exception_message", "Unknown error"
                                )
                                node_id = data.get("node_id", "N/A")
                                raise Exception(
                                    f"ComfyUI job failed on node {node_id}. Error: {error_details}"
                                )

                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning(
                                "WebSocket connection lost. Switching to pure polling."
                            )
                            break

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning("WS Error: %s", e)
                        break

        except Exception as e:
            if not prompt_id:
                raise e
            logger.warning("WebSocket failed (%s). Fallback to pure polling.", e)

        if prompt_id:
            while asyncio.get_event_loop().time() - start_time <= max_wait_time:
                await asyncio.sleep(2)
                try:
                    async with session.get(
                        f"{comfyui_http_url}/history/{prompt_id}"
                    ) as h_resp:
                        if h_resp.status == 200:
                            history = await h_resp.json()
                            if prompt_id in history:
                                return history[prompt_id]
                except Exception:
                    pass
            raise TimeoutError(f"Generation timed out (polling) after {max_wait_time}s")

        raise Exception("Failed to start generation flow.")


def extract_video_files(job_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Robustly extract video filenames from ComfyUI job outputs
    by recursively searching the data.
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
                relative_path = request.app.url_path_for(
                    "get_file_content_by_id", id=str(file_id)
                )
                return f"{relative_path}?t={int(time.time())}", True

    except Exception as e:
        logger.debug("Failed to download or upload video to OpenWebUI: %s", e)

    return comfyui_view_url, False  # Fallback to direct ComfyUI URL


# --- Main Tool Class ---


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188", description="ComfyUI HTTP API endpoint."
        )
        workflow_json: str = Field(
            default=json.dumps(DEFAULT_WORKFLOW),
            description="ComfyUI Workflow JSON string. If empty, a default is used.",
        )
        prompt_node_id: str = Field(
            default="89", description="Node ID for the text prompt input."
        )
        prompt_field_name: str = Field(
            default="text",
            description="Name of the input field where the prompt text goes.",
        )
        seed_node_id: str = Field(
            default="81",
            description="Node ID containing the seed parameter to randomize.",
        )
        seed_field_name: str = Field(
            default="noise_seed",
            description="Name of the seed input field on the seed node.",
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
            default=True, description="Return an HTML video embed upon completion."
        )

    def __init__(self):
        self.valves = self.Valves()

    async def generate_video(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
    ) -> str | HTMLResponse:
        """Generate a video from a text prompt using the provided ComfyUI workflow."""
        try:
            # --- Unload Ollama models if requested ---
            if self.valves.unload_ollama_models:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "⏳ Unloading Ollama models...",
                                "done": False,
                            },
                        }
                    )
                unloaded = await unload_all_models_async(self.valves.ollama_api_url)
                await asyncio.sleep(2)
                if not unloaded:
                    logger.warning("Ollama models may not have fully unloaded.")

            # --- Parse workflow ---
            try:
                workflow = json.loads(self.valves.workflow_json)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid Workflow JSON in valves: {e}")

            # --- Inject prompt ---
            prompt_node_id = self.valves.prompt_node_id
            prompt_field = self.valves.prompt_field_name
            if prompt_node_id not in workflow:
                raise ValueError(
                    f"Prompt node ID '{prompt_node_id}' not found in the workflow."
                )
            workflow[prompt_node_id].setdefault("inputs", {})[prompt_field] = prompt

            # --- Inject random seed ---
            seed_node_id = self.valves.seed_node_id
            seed_field = self.valves.seed_field_name
            if seed_node_id in workflow:
                workflow[seed_node_id].setdefault("inputs", {})[seed_field] = (
                    random.randint(1, 2**31 - 1)
                )

            # --- Prepare connection ---
            http_api_url = self.valves.comfyui_api_url.rstrip("/")
            ws_api_url = (
                http_api_url.replace("http://", "ws://").replace("https://", "wss://")
                + "/ws"
            )
            client_id = str(uuid.uuid4())
            payload: Dict[str, Any] = {
                "prompt": workflow,
                "client_id": client_id,
            }

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

            # --- Submit and wait (robust WS + polling fallback) ---
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

            result_data = await connect_submit_and_wait(
                ws_api_url,
                http_api_url,
                payload,
                client_id,
                self.valves.max_wait_time,
            )

            # --- Extract video files ---
            video_files = extract_video_files(result_data)
            if not video_files:
                logger.warning(
                    "No video files extracted from job data: %s",
                    json.dumps(result_data, indent=2),
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
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "✅ Video Generated!",
                            "done": True,
                        },
                    }
                )
            if self.valves.return_html_embed:
                html_player = f'<video controls src="{final_url}" width="960" style="max-width:100%"></video>'
                return HTMLResponse(
                    content=html_player,
                    headers={"content-disposition": "inline"},
                )
            return f"🎬 Video generated successfully. Link: {final_url}"

        except Exception as e:
            logger.error("Error during video generation: %s", e, exc_info=True)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"❌ Error: {e}",
                            "done": True,
                        },
                    }
                )
            return f"An error occurred: {e}"
