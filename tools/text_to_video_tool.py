"""
title: ComfyUI Text-to-Video Tool
description: Generate video from text prompt via ComfyUI workflow JSON. Uses ComfyUI HTTP+WebSocket API, supports unloading Ollama models before run, randomizes seed when missing, downloads the final video to OpenWebUI cache and emits an HTML video embed.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.1.0
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

logger = logging.getLogger(__name__)


DEFAULT_WORKFLOW: Dict[str, Any] = {
    "71": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default"
        },
        "class_type": "CLIPLoader",
        "_meta": {
            "title": "Cargar CLIP"
        }
    },
    "72": {
        "inputs": {
            "text": "",
            "clip": [
                "71",
                0
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Negative Prompt)"
        }
    },
    "73": {
        "inputs": {
            "vae_name": "wan/wan_2.1_vae.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {
            "title": "Cargar VAE"
        }
    },
    "74": {
        "inputs": {
            "width": 848,
            "height": 480,
            "length": 41,
            "batch_size": 1
        },
        "class_type": "EmptyHunyuanLatentVideo",
        "_meta": {
            "title": "EmptyHunyuanLatentVideo"
        }
    },
    "75": {
        "inputs": {
            "unet_name": "wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast"
        },
        "class_type": "UNETLoader",
        "_meta": {
            "title": "Cargar Modelo de Difusión"
        }
    },
    "76": {
        "inputs": {
            "unet_name": "wan/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn_fast"
        },
        "class_type": "UNETLoader",
        "_meta": {
            "title": "Cargar Modelo de Difusión"
        }
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
            "model": [
                "86",
                0
            ],
            "positive": [
                "89",
                0
            ],
            "negative": [
                "72",
                0
            ],
            "latent_image": [
                "81",
                0
            ]
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {
            "title": "KSampler (Avanzado)"
        }
    },
    "80": {
        "inputs": {
            "filename_prefix": "video/ComfyUI",
            "format": "auto",
            "codec": "auto",
            "video-preview": "",
            "video": [
                "88",
                0
            ]
        },
        "class_type": "SaveVideo",
        "_meta": {
            "title": "Guardar video"
        }
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
            "model": [
                "82",
                0
            ],
            "positive": [
                "89",
                0
            ],
            "negative": [
                "72",
                0
            ],
            "latent_image": [
                "74",
                0
            ]
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {
            "title": "KSampler (Avanzado)"
        }
    },
    "82": {
        "inputs": {
            "shift": 5.000000000000001,
            "model": [
                "83",
                0
            ]
        },
        "class_type": "ModelSamplingSD3",
        "_meta": {
            "title": "MuestreoDeModeloSD3"
        }
    },
    "83": {
        "inputs": {
            "lora_name": "wan2_2/14b/lightning/high_noise_model.safetensors",
            "strength_model": 1.0000000000000002,
            "model": [
                "75",
                0
            ]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {
            "title": "CargadorLoRAModeloSolo"
        }
    },
    "85": {
        "inputs": {
            "lora_name": "wan2_2/14b/lightning/low_noise_model.safetensors",
            "strength_model": 1.0000000000000002,
            "model": [
                "76",
                0
            ]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {
            "title": "CargadorLoRAModeloSolo"
        }
    },
    "86": {
        "inputs": {
            "shift": 5.000000000000001,
            "model": [
                "85",
                0
            ]
        },
        "class_type": "ModelSamplingSD3",
        "_meta": {
            "title": "MuestreoDeModeloSD3"
        }
    },
    "88": {
        "inputs": {
            "fps": 16,
            "images": [
                "114",
                0
            ]
        },
        "class_type": "CreateVideo",
        "_meta": {
            "title": "Crear video"
        }
    },
    "89": {
        "inputs": {
            "text": "Scene Setting:\nA vibrant, high-energy concert taking place in a massive stadium under a starry night sky. The stage is illuminated with dynamic neon lights, laser beams, and colorful spotlights creating a dazzling visual spectacle.\n\nMain Subject:\nA charismatic golden retriever wearing a leather jacket and sunglasses, passionately playing an electric guitar. The dog's paws are expertly moving along the fretboard, with glowing strings emitting a soft blue hue.\n\nSupporting Elements:\n\nA band of diverse dogs:\nA German shepherd on drums with a snare drum and cymbals\nA poodle on bass guitar with a mini amp\nA dachshund on keyboard with glowing keys\nA crowd of enthusiastic dogs in the audience with paw-print banners and LED bracelets\nConfetti cannons firing multicolored paper confetti\nA giant inflatable guitar backdrop\nAtmosphere:\nEnergetic rock music fills the air, with pyrotechnic sparks shooting from the stage. The dogs' fur glows slightly under the stage lights, and the stadium's massive speakers create visible sound waves.\n\nAdditional Details:\n\nThe lead dog has a custom guitar with a dog bone-shaped headstock\nThe stage has a circular design with a glowing ring of lights\nThe video includes slow-motion shots of the dogs' paws playing the guitar\nThe sky has a gradient from deep indigo to electric purple",
            "clip": [
                "71",
                0
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Positive Prompt)"
        }
    },
    "114": {
        "inputs": {
            "tile_size": 128,
            "overlap": 64,
            "temporal_size": 20,
            "temporal_overlap": 8,
            "samples": [
                "78",
                0
            ],
            "vae": [
                "73",
                0
            ]
        },
        "class_type": "VAEDecodeTiled",
        "_meta": {
            "title": "VAE Decodificar (Mosaico)"
        }
    },
    "115": {
        "inputs": {
            "anything": [
                "88",
                0
            ]
        },
        "class_type": "easy cleanGpuUsed",
        "_meta": {
            "title": "Clean VRAM Used"
        }
    }
}


def get_loaded_models(api_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:

    try:
        resp = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.RequestException as e:
        logger.debug("Error fetching loaded Ollama models: %s", e)
        return []


def unload_all_models(api_url: str = "http://localhost:11434") -> None:

    try:
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
    except requests.RequestException as e:
        logger.debug("Error unloading Ollama models: %s", e)

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


async def wait_for_completion_ws(
    comfyui_ws_url: str,
    comfyui_http_url: str,
    prompt_id: str,
    client_id: str,
    max_wait_time: int,
) -> Dict[str, Any]:

    start_time = asyncio.get_event_loop().time()
    job_data_output = None

    try:
        async with aiohttp.ClientSession().ws_connect(f"{comfyui_ws_url}?clientId={client_id}") as ws:
            async for msg in ws:
                if asyncio.get_event_loop().time() - start_time > max_wait_time:
                    raise TimeoutError(f"WebSocket wait timed out after {max_wait_time}s")

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        message = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "executed" and data.get("prompt_id") == prompt_id:
                        job_data_output = data.get("output", {})
                        async with aiohttp.ClientSession() as http_session:
                            async with http_session.get(f"{comfyui_http_url}/history/{prompt_id}") as resp:
                                if resp.status == 200:
                                    history = await resp.json()
                                    if prompt_id in history:
                                        return history[prompt_id]
                        if job_data_output:
                            return {"outputs": job_data_output}
                        raise Exception("Job executed but failed to retrieve output from history.")

                    if msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        exc = data.get("exception_message", "Unknown error")
                        node_id = data.get("node_id", "N/A")
                        node_type = data.get("node_type", "N/A")
                        raise Exception(f"ComfyUI job failed on node {node_id} ({node_type}). Error: {exc}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise Exception(f"WebSocket connection error: {ws.exception()}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    if not job_data_output:
                        raise Exception("WebSocket closed unexpectedly before completion.")
                    break

            if not job_data_output:
                raise TimeoutError("WebSocket connection closed or timed out without explicit completion event.")

            return {"outputs": job_data_output}

    except asyncio.TimeoutError:
        raise TimeoutError(f"Overall wait/connection timed out after {max_wait_time}s")
    except Exception as e:
        raise Exception(f"Error during WebSocket communication: {e}")


def extract_video_files(job_data: Dict[str, Any]) -> List[Dict[str, str]]:


    candidates: List[Tuple[str, str]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            if any(obj.lower().endswith(ext) for ext in VIDEO_EXTS):
                candidates.append((obj, ""))
        elif isinstance(obj, dict):

            filename_val = None
            for k in ("filename", "file", "path", "name"):
                v = obj.get(k)
                if isinstance(v, str) and any(v.lower().endswith(ext) for ext in VIDEO_EXTS):
                    filename_val = v
                    break

            if filename_val:
                subfolder_val = ""
                for sf_key in ("subfolder", "subdir", "sub_path", "folder", "directory"):
                    sf = obj.get(sf_key)
                    if isinstance(sf, str) and sf:
                        subfolder_val = sf.strip("/ ")
                        break
                    
                if not subfolder_val:
                    norm = filename_val.replace("\\", "/")
                    if "/" in norm:
                        head, tail = os.path.split(norm)
                        subfolder_val = head.strip("/ ")
                        filename_val = os.path.basename(norm)

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

        try:
            filename = os.path.basename(str(fname))
            subfolder = str(sub).strip("/ ") if sub else ""
        except Exception:
            filename = str(fname)
            subfolder = str(sub) if sub else ""

        if not filename or not any(filename.lower().endswith(ext) for ext in VIDEO_EXTS):
            continue

        key = f"{subfolder}/{filename}" if subfolder else filename
        if key in seen:
            continue
        seen.add(key)
        video_files.append({"filename": filename, "subfolder": subfolder})

    return video_files


async def download_and_upload_to_owui(
    comfyui_http_url: str,
    filename: str,
    subfolder: str = "",
    request: Optional[Any] = None,
    user: Optional[Any] = None,
    base_url: str = "",
) -> Tuple[str, bool]:

    enc_filename = quote(filename, safe="")
    enc_subfolder = quote(subfolder, safe="") if subfolder else ""
    subfolder_param = f"&subfolder={enc_subfolder}" if subfolder else ""
    comfyui_file_url = f"{comfyui_http_url}/api/viewvideo?filename={enc_filename}&type=output{subfolder_param}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(comfyui_file_url) as response:
                if response.status != 200:
                    logger.debug("Failed to download video from ComfyUI: HTTP %s (%s)", response.status, comfyui_file_url)
                    return (comfyui_file_url, False)
                content = await response.read()

        comfyui_view_url = comfyui_file_url
        if request and user:
            try:
                file = UploadFile(file=io.BytesIO(content), filename=os.path.basename(str(filename) or f"video_{uuid.uuid4()}.mp4"))
                file_item = upload_file_handler(request=request, file=file, metadata={}, process=False, user=user)
                file_id = None
                try:
                    if isinstance(file_item, dict):
                        file_id = file_item.get("id") or file_item.get("file_id") or (file_item.get("data") or {}).get("id")
                    else:
                        file_id = getattr(file_item, "id", None) or getattr(file_item, "file_id", None)
                except Exception:
                    file_id = None

                if file_id:
                    file_id = str(file_id)
                    base = str(request.base_url).rstrip("/")
                    relative_path = request.app.url_path_for("get_file_content_by_id", id=file_id)
                    timestamp = int(time.time() * 1000)
                    final = f"{base}{relative_path}?t={timestamp}"
                    logger.debug("Uploaded file to OpenWebUI: file_id=%s url=%s", file_id, final)
                    return (final, True)
            except Exception as e:
                logger.debug("Upload via upload_file_handler failed: %s", e)

        logger.debug("Upload to OpenWebUI did not occur; falling back to ComfyUI URL: %s", comfyui_view_url)
        return (comfyui_view_url, False)
    except Exception as e:
        logger.debug("Error downloading/uploading video: %s", e)
        return (f"{comfyui_http_url}/api/viewvideo?filename={quote(filename, safe='')}&type=output{subfolder_param}", False)

def _load_default_workflow() -> Dict[str, Any]:
    return DEFAULT_WORKFLOW.copy()


def ensure_random_seed(workflow: Dict[str, Any], seed_node_id: Optional[str] = None) -> None:

    nodes = workflow.get("nodes") or []
    if isinstance(nodes, dict):
        iterable = nodes.items()
    else:
        iterable = enumerate(nodes)
    for _, node in iterable:
        try:
            if not isinstance(node, dict):
                continue
            ctype = str(node.get("class_type", "")).lower()
            if "ksampler" in ctype:
                inputs = node.setdefault("inputs", {})
                seed = inputs.get("noise_seed") if "noise_seed" in inputs else inputs.get("seed")
                if seed != 0:
                    if "noise_seed" in inputs:
                        inputs["noise_seed"] = random.randint(1, 2**31 - 1)
                    else:
                        inputs["seed"] = random.randint(1, 2**31 - 1)
                    if seed_node_id:
                        inputs.setdefault("seed_mode_id", seed_node_id)
        except Exception:
            continue


class Tools:
    class Valves(BaseModel):
        comfyui_api_url: str = Field(
            default="http://localhost:8188",
            description="ComfyUI HTTP API endpoint.",
        )
        workflow: Dict[str, Any] = Field(
            default_factory=lambda: DEFAULT_WORKFLOW.copy(),
            description="ComfyUI workflow JSON (dict). If empty, a bundled default will be used.",
        )
        prompt_node_id: str = Field(default="89", description="Node ID for text prompt input (string id).")
        output_node_id: str = Field(default="80", description="Node ID expected to hold final video output filename.")
        seed_node_id: str = Field(default="78", description="Seed node id to override (if applicable).")
        max_wait_time: int = Field(default=600, description="Max wait time in seconds for ComfyUI job completion.")
        unload_ollama_models: bool = Field(default=False, description="Unload Ollama models before calling ComfyUI.")
        ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL for unloading models.")
        owui_base_url: str = Field(default="http://localhost:3000", description="OpenWebUI base URL used to build public cache links.")
        prefer_owui_url_emit: bool = Field(
            default=True,
            description="If true, prefer emitting an OpenWebUI URL (uploaded) for the HTML embed. If upload fails and this is True, the tool will attempt one retry before falling back.",
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
        """Generate a video from a text prompt using the provided ComfyUI workflow.

        Returns a user-facing string (status / download link). Emits embeds event with HTML video player when available.
        """

        self.__user__ = Users.get_user_by_id(__user__["id"]) if __user__ else None
        if self.valves.unload_ollama_models:
            try:
                unload_all_models(api_url=self.valves.ollama_url)
            except Exception as e:
                print(f"[DEBUG] Error unloading Ollama models: {e}")

        workflow = self.valves.workflow or _load_default_workflow()
        if not workflow:
            return "Error: No workflow provided and no default found."

        def safe_set_input(wf: Dict[str, Any], node_id: str, input_name: str, value: Any) -> bool:
            if node_id in wf and isinstance(wf[node_id], dict) and "inputs" in wf[node_id]:
                wf[node_id]["inputs"][input_name] = value
                return True
            return False

        try:
            active_workflow = json.loads(json.dumps(workflow))
        except Exception:
            active_workflow = workflow

        if not safe_set_input(active_workflow, str(self.valves.prompt_node_id), "text", prompt):
            if not safe_set_input(active_workflow, str(self.valves.prompt_node_id), "prompt", prompt):
                for nid, node in list(active_workflow.items()):
                    if isinstance(node, dict) and node.get("class_type", "").lower().startswith("cliptextencode"):
                        node.setdefault("inputs", {})["text"] = prompt


        ensure_random_seed(active_workflow, self.valves.seed_node_id)

        if not getattr(self.valves, "comfyui_api_url", None):
            return "Error: ComfyUI API URL not configured in tool valves."
        http_api_url = self.valves.comfyui_api_url.rstrip("/")
        ws_scheme = "wss" if http_api_url.startswith("https") else "ws"
        ws_api_url = f"{ws_scheme}://{http_api_url.split('://', 1)[-1]}/ws"

        client_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {"prompt": active_workflow, "client_id": client_id}

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "🎬 Generating video...", "done": False}})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{http_api_url}/prompt", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return f"ComfyUI API error on submission ({resp.status}): {await resp.text()}"
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        return f"Error: No prompt_id from ComfyUI. Response: {json.dumps(result)}"

            job_data = await wait_for_completion_ws(ws_api_url, http_api_url, prompt_id, client_id, self.valves.max_wait_time)

            video_files = extract_video_files(job_data)

            if not video_files:

                try:
                    async with aiohttp.ClientSession() as probe_session:
                        async with probe_session.get(f"{http_api_url}/history/{prompt_id}") as hresp:
                            if hresp.status == 200:
                                hist = await hresp.json()
                                if prompt_id in hist:
                                    more = hist[prompt_id]
                                else:
                                    more = hist
                                video_files = extract_video_files(more)
                                if video_files:
                                    logger.debug("Found video files in history probe: %s", video_files)
                except Exception as e:
                    logger.debug("Error probing ComfyUI history for filenames: %s", e)

                if not video_files:
                    outputs_obj = job_data.get("outputs", {}) if isinstance(job_data, dict) else {}
                    keys = list(outputs_obj.keys()) if isinstance(outputs_obj, dict) else []
                    hint = f" (outputs keys: {keys[:20]})" if keys else ""
                    return "ComfyUI job completed but no video outputs were detected. Filename must be provided by the API and wasn't found." + hint

            if not video_files:
                outputs_obj = job_data.get("outputs", {}) if isinstance(job_data, dict) else {}
                keys = list(outputs_obj.keys()) if isinstance(outputs_obj, dict) else []
                hint = f" (outputs keys: {keys[:20]})" if keys else ""
                return "ComfyUI job completed but no video outputs were detected." + hint

            video_info = video_files[0]
            filename = video_info.get("filename")
            if not filename:
                return "ComfyUI job completed but video filename was not found in the output metadata."
            subfolder = video_info.get("subfolder", "")

            final_url, uploaded = await download_and_upload_to_owui(
                http_api_url, filename, subfolder, request=__request__, user=self.__user__, base_url=self.valves.owui_base_url
            ) if self.valves.prefer_owui_url_emit else ( "", False)
            
            if subfolder:
                comfyui_url = f"{http_api_url}/api/viewvideo?filename={quote(filename, safe='')}&type=output&subfolder={quote(subfolder, safe='')}"
            else:
                comfyui_url = f"{http_api_url}/api/viewvideo?filename={quote(filename, safe='')}&type=output"

            if not uploaded and self.valves.prefer_owui_url_emit and __request__ and self.__user__:
                logger.debug("Initial upload failed; attempting one retry as `prefer_owui_url_emit` is True")
                final_url_retry, uploaded_retry = await download_and_upload_to_owui(
                    http_api_url, filename, subfolder, request=__request__, user=self.__user__, base_url=self.valves.owui_base_url
                )
                if uploaded_retry:
                    final_url = final_url_retry
                    uploaded = True

            if not uploaded and not self.valves.prefer_owui_url_emit:
                final_url = comfyui_url
            elif not uploaded and self.valves.prefer_owui_url_emit:
                final_url = comfyui_url
            try:
                if subfolder and "api/viewvideo" in str(final_url) and "subfolder=" not in str(final_url):
                    connector = "&" if "?" in final_url else "?"
                    final_url = f"{final_url}{connector}subfolder={quote(subfolder, safe='')}"
                    logger.debug("Appended missing subfolder to final_url: %s", final_url)
            except Exception:
                pass
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "🎬 Video Generated! ", "done": True}})

            html_player = f'<video controls src="{final_url}" width="960" style="max-width:100%"></video>'
            if __event_emitter__:
                await __event_emitter__({"type": "embeds", "data": {"embeds": [html_player]}})

            return f"🎬 Video generated successfully. Download/preview: {final_url}"

        except Exception as e:
            return f"Error during generation: {str(e)}"
