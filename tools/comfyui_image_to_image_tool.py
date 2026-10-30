"""
title: ComfyUI Image-to-Image Tool
description: Edit/transform images using ComfyUI workflows (Flux Kontext or Qwen Edit). Uses ComfyUI HTTP+WebSocket API, supports unloading Ollama models before run, randomizes seed, and returns the edited image inline.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.3.0
required_open_webui_version: 0.9.1
license: MIT
"""

import asyncio
import io
import json
import logging
import mimetypes
import os
import random
import re
import html
import time
import uuid
import aiohttp

from typing import Any, Dict, Iterable, Optional, Callable, Awaitable, List, Literal, Sequence, Tuple, Union
from urllib.parse import urlencode
from pydantic import BaseModel, Field
from fastapi import HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.datastructures import Headers
from open_webui.models.chats import Chats
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import upload_file_handler

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
    start_time = time.monotonic()
    async with aiohttp.ClientSession(headers=headers) as ws_session:
        async with ws_session.ws_connect(
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
    """Extract image outputs from ComfyUI history.

    Each entry carries the ComfyUI output ``type`` (``output``/``temp``/``input``)
    so /view is asked for the right folder. When the workflow produced saved
    (``type == "output"``) images, temp previews (e.g. PreviewImage nodes) are
    dropped, since they are usually duplicates of the saved result.
    """
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
                                        "subfolder": str(subfolder or "").strip("/"),
                                        "type": str(file_info.get("type") or "output"),
                                    }
                                )

    saved_outputs = [item for item in image_files if item["type"] == "output"]
    return saved_outputs or image_files


# --- Open WebUI media storage helpers (canonical, v0.12.5) ---

# Mirrors open_webui.utils.chat_id (v0.12.5). Kept local so the tool also
# imports on older servers that do not ship that module.
TEMPORARY_CHAT_ID_PREFIXES = ("temporary:", "local:")
CHANNEL_CHAT_ID_PREFIX = "channel:"
NON_SAVED_CHAT_ID_PREFIXES = (*TEMPORARY_CHAT_ID_PREFIXES, CHANNEL_CHAT_ID_PREFIX)

# Preferred extensions (mimetypes.guess_extension gives odd ones, e.g. .mpga).
MEDIA_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "image/svg+xml": ".svg",
}

TRANSIENT_HTTP_STATUSES = frozenset({429, 502, 503, 504})


class MediaStorageError(Exception):
    """Saving/downloading media failed. ``str(exc)`` is safe to show the user
    and the LLM: it never contains provider URLs, keys or response bodies."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def is_saved_chat_id(chat_id: Any) -> bool:
    """True for a persisted chat (not temporary/local/channel)."""
    return isinstance(chat_id, str) and bool(chat_id) and not chat_id.startswith(
        NON_SAVED_CHAT_ID_PREFIXES
    )


def normalize_content_type(content_type: Optional[str], default: str = "application/octet-stream") -> str:
    """'Audio/MPEG; charset=x' -> 'audio/mpeg'."""
    value = (content_type or "").split(";", 1)[0].strip().lower()
    return value if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", value or "") else default


def extension_for(content_type: str) -> str:
    ct = normalize_content_type(content_type)
    return MEDIA_EXTENSIONS.get(ct) or mimetypes.guess_extension(ct) or ".bin"


def safe_filename(name: Optional[str], content_type: str, prefix: str = "media") -> str:
    """Basename only, conservative charset, extension that matches the type."""
    ext = extension_for(content_type)
    base = os.path.basename(str(name or "")).strip()
    stem, given_ext = os.path.splitext(base)
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")[:80]
    if not stem:
        clean_prefix = re.sub(r"[^\w-]+", "_", str(prefix or "")).strip("_") or "media"
        stem = f"{clean_prefix}_{uuid.uuid4().hex[:8]}"
    given_ext = given_ext.lower()
    if given_ext and re.fullmatch(r"\.[a-z0-9]{1,5}", given_ext) and (
        mimetypes.guess_type(f"x{given_ext}")[0] == normalize_content_type(content_type)
    ):
        ext = given_ext
    return f"{stem}{ext}"


async def resolve_user(user: Any) -> Optional[UserModel]:
    """Accept the ``__user__`` dict (or a UserModel) and return a UserModel."""
    if isinstance(user, UserModel):
        return user
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not user_id:
        return None
    try:
        return await Users.get_user_by_id(user_id)
    except Exception as exc:  # DB hiccup -> treated as "no user"
        logger.warning("Could not resolve Open WebUI user (%s)", type(exc).__name__)
        return None


def chat_context(
    metadata: Optional[dict],
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """(chat_id, message_id) from ``__metadata__`` or ``__chat_id__``/``__message_id__``."""
    meta = metadata if isinstance(metadata, dict) else {}
    cid = chat_id or meta.get("chat_id")
    mid = message_id or meta.get("message_id")
    return (cid if isinstance(cid, str) and cid else None, mid if isinstance(mid, str) and mid else None)


def _upload_metadata(chat_id: Optional[str], message_id: Optional[str], extra: Optional[dict]) -> dict:
    meta: Dict[str, Any] = {}
    if isinstance(extra, dict):
        # Never let callers smuggle routing keys; we set them below.
        meta.update({k: v for k, v in extra.items() if k not in ("chat_id", "message_id", "channel_id", "file_hash")})
    if chat_id and chat_id.startswith(CHANNEL_CHAT_ID_PREFIX):
        channel_id = chat_id[len(CHANNEL_CHAT_ID_PREFIX):]
        if channel_id:
            meta["channel_id"] = channel_id  # upload_file_handler adds file to channel
    elif chat_id:
        meta["chat_id"] = chat_id
    if message_id:
        meta["message_id"] = message_id
    return meta


def _file_id(file_item: Any) -> Optional[str]:
    if isinstance(file_item, dict):  # process=True returns {'status': True, **model_dump()}
        value = file_item.get("id")
    else:
        value = getattr(file_item, "id", None)
    return value if isinstance(value, str) and value else None


async def save_media_bytes(
    data: bytes,
    *,
    request: Any,
    user: Any,
    content_type: str,
    filename: Optional[str] = None,
    metadata: Optional[dict] = None,
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    prefix: str = "media",
    allow_fallback: bool = False,
    fallback_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Store ``data`` in Open WebUI and link it to the chat message.

    :param request: the tool's ``__request__`` (required).
    :param user: ``__user__`` dict or a UserModel (required).
    :param content_type: real MIME type of ``data`` (e.g. "audio/mpeg").
    :param metadata: the tool's ``__metadata__`` (chat_id/message_id read from it).
    :param allow_fallback/fallback_url: explicit opt-in only. When both are set
        and saving fails, the provider URL is returned instead of raising.
    :returns: {"id", "url" ("/api/v1/files/{id}/content"), "name",
        "content_type", "size", "linked", "fallback"}
    :raises MediaStorageError: on any failure (unless the opt-in fallback applies).
    """
    try:
        return await _save_media_bytes(
            data,
            request=request,
            user=user,
            content_type=content_type,
            filename=filename,
            metadata=metadata,
            chat_id=chat_id,
            message_id=message_id,
            extra_metadata=extra_metadata,
            prefix=prefix,
        )
    except MediaStorageError as exc:
        if allow_fallback and is_http_url(fallback_url):
            logger.warning("Media storage failed, using opt-in provider URL fallback: %s", exc)
            return {
                "id": None,
                "url": fallback_url,
                "name": None,
                "content_type": normalize_content_type(content_type),
                "size": len(data) if isinstance(data, (bytes, bytearray)) else 0,
                "linked": False,
                "fallback": True,
                "error": str(exc),
            }
        raise


async def _save_media_bytes(
    data: bytes,
    *,
    request: Any,
    user: Any,
    content_type: str,
    filename: Optional[str],
    metadata: Optional[dict],
    chat_id: Optional[str],
    message_id: Optional[str],
    extra_metadata: Optional[dict],
    prefix: str,
) -> Dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise MediaStorageError("The generated file is empty.")
    if request is None:
        raise MediaStorageError("Open WebUI request context is unavailable.")
    user_model = await resolve_user(user)
    if user_model is None:
        raise MediaStorageError("The requesting Open WebUI user could not be resolved.")

    ct = normalize_content_type(content_type)
    name = safe_filename(filename, ct, prefix)
    cid, mid = chat_context(metadata, chat_id, message_id)
    upload = UploadFile(
        file=io.BytesIO(bytes(data)),
        filename=name,
        headers=Headers({"content-type": ct}),
    )
    try:
        file_item = await upload_file_handler(
            request,
            file=upload,
            metadata=_upload_metadata(cid, mid, extra_metadata),
            process=False,
            user=user_model,
        )
    except HTTPException as exc:
        if exc.status_code == 413:
            detail = exc.detail if isinstance(exc.detail, str) else ""
            limit = re.search(r"less than ([0-9.]+ ?MB)", detail or "")
            raise MediaStorageError(
                "The generated file ({:.1f} MB) exceeds Open WebUI's upload size limit{}. "
                "Ask an admin to raise Admin Settings > Documents > Max Upload Size.".format(
                    len(data) / (1024 * 1024), f" ({limit.group(1)})" if limit else ""
                ),
                status_code=413,
            ) from None
        raise MediaStorageError(
            f"Open WebUI rejected the file (HTTP {exc.status_code}).", status_code=exc.status_code
        ) from None
    except Exception as exc:  # storage backend / DB error: deterministic here, no retry
        logger.exception("upload_file_handler failed")
        raise MediaStorageError(f"Saving the file to Open WebUI failed ({type(exc).__name__}).") from None

    file_id = _file_id(file_item)
    if not file_id:
        raise MediaStorageError("Open WebUI did not return a file ID.")

    linked = False
    if is_saved_chat_id(cid) and mid:
        # Same as upload_image: makes the file part of the chat (shared-chat access).
        try:
            linked = bool(
                await Chats.insert_chat_files(
                    chat_id=cid, message_id=mid, file_ids=[file_id], user_id=user_model.id
                )
            )
        except Exception as exc:  # not fatal: the owner can still read the file
            logger.warning("Could not link file to chat (%s)", type(exc).__name__)

    meta = (file_item.get("meta") if isinstance(file_item, dict) else getattr(file_item, "meta", None)) or {}
    return {
        "id": file_id,
        "url": f"/api/v1/files/{file_id}/content",
        "name": meta.get("name") or name,
        "content_type": meta.get("content_type") or ct,
        "size": meta.get("size") or len(data),
        "linked": linked,
        "fallback": False,
    }


def is_http_url(url: Any) -> bool:
    return isinstance(url, str) and re.match(r"^https?://[^\s/]+", url, re.I) is not None


async def fetch_media_bytes(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 120.0,
    max_bytes: Optional[int] = 512 * 1024 * 1024,
    retries: int = 2,
    retry_delay: float = 2.0,
    min_bytes: int = 1,
    ssl: Any = None,
    allow_redirects: bool = True,
) -> Tuple[bytes, str]:
    """GET a provider/backend output. Returns (bytes, normalized content_type).

    Retries ONLY transient network failures (connection errors, timeouts,
    HTTP 429/502/503/504) up to ``retries`` extra attempts. Other HTTP errors
    (404, 401, ...) and oversize bodies fail immediately. Error messages never
    contain the URL (signed URLs / keys) or the response body.

    aiohttp drops Authorization/Cookie on cross-origin redirects but NOT custom
    key headers (e.g. "x-api-key"): pass allow_redirects=False when ``headers``
    carry such a key and the output URL may redirect to another host.
    """
    if not is_http_url(url):
        raise MediaStorageError("The provider returned an invalid output URL.")
    attempts = max(0, int(retries)) + 1
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        transient = False
        try:
            own = session is None
            sess = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
            try:
                async with sess.get(url, headers=headers, ssl=ssl, allow_redirects=allow_redirects) as resp:
                    if resp.status != 200:
                        transient = resp.status in TRANSIENT_HTTP_STATUSES
                        last_error = f"HTTP {resp.status}"
                        if not transient:
                            raise MediaStorageError(
                                f"Downloading the generated file failed ({last_error}).", status_code=resp.status
                            )
                    else:
                        if max_bytes and resp.content_length and resp.content_length > max_bytes:
                            raise MediaStorageError("The generated file is too large to download.")
                        buf = bytearray()
                        async for chunk in resp.content.iter_chunked(1 << 16):
                            buf.extend(chunk)
                            if max_bytes and len(buf) > max_bytes:
                                raise MediaStorageError("The generated file is too large to download.")
                        if len(buf) < max(1, min_bytes):
                            # Output not ready yet (some backends 200 with an empty body).
                            transient = True
                            last_error = "empty response"
                        else:
                            return bytes(buf), normalize_content_type(resp.headers.get("Content-Type"))
            finally:
                if own:
                    await sess.close()
        except MediaStorageError:
            raise
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as exc:
            transient = True
            last_error = type(exc).__name__
        except aiohttp.ClientError as exc:  # e.g. InvalidURL, TooManyRedirects
            raise MediaStorageError(f"Downloading the generated file failed ({type(exc).__name__}).") from None
        if transient and attempt < attempts:
            logger.info("Transient download failure (%s), retry %d/%d", last_error, attempt, attempts - 1)
            await asyncio.sleep(retry_delay * attempt)
            continue
        break
    raise MediaStorageError(f"Downloading the generated file failed ({last_error}).")


async def download_to_storage(
    url: str,
    *,
    request: Any,
    user: Any,
    metadata: Optional[dict] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 120.0,
    retries: int = 2,
    prefix: str = "media",
    allow_fallback: bool = False,
    ssl: Any = None,
    allow_redirects: bool = True,
) -> Dict[str, Any]:
    """fetch_media_bytes + save_media_bytes. ``content_type`` overrides the
    response header (use it when the backend sends application/octet-stream).
    With ``allow_fallback=True`` (explicit admin opt-in) a failed download or
    save returns ``url`` itself with ``fallback=True`` instead of raising."""
    try:
        data, served_type = await fetch_media_bytes(
            url, headers=headers, session=session, timeout=timeout, retries=retries, ssl=ssl,
            allow_redirects=allow_redirects,
        )
    except MediaStorageError as exc:
        if allow_fallback:
            logger.warning("Download failed, using opt-in provider URL fallback: %s", exc)
            return {"id": None, "url": url, "name": None, "content_type": content_type,
                    "size": 0, "linked": False, "fallback": True, "error": str(exc)}
        raise
    ct = content_type or served_type
    if ct == "application/octet-stream" and filename:
        ct = mimetypes.guess_type(filename)[0] or ct
    return await save_media_bytes(
        data,
        request=request,
        user=user,
        content_type=ct,
        filename=filename,
        metadata=metadata,
        prefix=prefix,
        allow_fallback=allow_fallback,
        fallback_url=url,
    )


# --- Inline embed builders (canonical, v0.12.5) ---


def esc(value: Any) -> str:
    """HTML-escape any text for element content or a quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


def script_json(value: Any) -> str:
    """JSON for inclusion inside <script>: '<', '>', '&' and U+2028/9 escaped,
    so '</script>' or '<!--' in data can never terminate the block."""
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _clean_urls(urls: Union[str, Iterable[str], None]) -> List[str]:
    if isinstance(urls, str):
        urls = [urls]
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------
# Shared runtime (plain string: no f-string brace escaping, data via JSON)
# --------------------------------------------------------------------------

_RUNTIME_JS = r"""
(function () {
  var M = window.OWUIMedia = {};
  // Same-origin relative Open WebUI API path ("/api/..."), never "//host" or "/\host".
  M.isLocalApi = function (u) {
    return typeof u === 'string' && /^\/api\/[A-Za-z0-9]/.test(u) && u.indexOf('\\') === -1;
  };
  M.isHttp = function (u) { return typeof u === 'string' && /^https?:\/\/[^\s\/]+/i.test(u); };
  M.isSafe = function (u) { return M.isLocalApi(u) || M.isHttp(u); };
  // Document origin. NOT location.origin: for about:srcdoc that is always "null".
  M.origin = (function () {
    try { return typeof self.origin === 'string' ? self.origin : 'null'; } catch (e) { return 'null'; }
  })();
  // allow-same-origin on? (srcdoc then shares the chat page origin; parent is readable)
  M.sameOrigin = (function () {
    try {
      return M.origin !== 'null' && window.parent !== window && window.parent.origin === M.origin &&
        !!window.parent.document;
    } catch (e) { return false; }
  })();
  function resolvesHere(u) {
    try { return new URL(u, document.baseURI).origin === M.origin; } catch (e) { return false; }
  }
  function token() {
    if (!M.sameOrigin) return null;
    try { return window.localStorage.getItem('token') || null; } catch (e) { return null; }
  }
  // Bearer fetch -> blob: URL. Only for same-origin relative /api/ paths.
  M.blobUrl = function (u) {
    if (!M.sameOrigin || !M.isLocalApi(u) || !resolvesHere(u)) {
      return Promise.reject(new Error('not authorized for this URL'));
    }
    var t = token(), opts = { credentials: 'same-origin' };
    if (t) opts.headers = { 'Authorization': 'Bearer ' + t };
    return fetch(u, opts).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.blob();
    }).then(function (b) { return URL.createObjectURL(b); });
  };

  var pending = false;
  M.postHeight = function () {
    if (pending) return;
    pending = true;
    (window.requestAnimationFrame || setTimeout)(function () {
      pending = false;
      var root = document.getElementById('owui-root') || document.body;
      var h = Math.ceil(root.getBoundingClientRect().height +
        (parseFloat(getComputedStyle(document.body).paddingTop) || 0) +
        (parseFloat(getComputedStyle(document.body).paddingBottom) || 0));
      try { window.parent.postMessage({ type: 'iframe:height', height: h }, '*'); } catch (e) {}
    });
  };
  window.addEventListener('load', M.postHeight);
  window.addEventListener('resize', M.postHeight);
  if (window.ResizeObserver) {
    try { new ResizeObserver(M.postHeight).observe(document.getElementById('owui-root') || document.body); } catch (e) {}
  }

  M.fallback = function (box, u, what) {
    if (!box) return;
    box.innerHTML = '';
    box.className += ' owui-fallback';
    var p = document.createElement('div');
    p.className = 'owui-fallback-text';
    p.textContent = 'The ' + (what || 'media') + ' could not be loaded inside this embed.';
    box.appendChild(p);
    if (M.isSafe(u)) {
      var a = document.createElement('a');
      a.href = u; a.target = '_blank'; a.rel = 'noopener noreferrer';
      a.className = 'owui-btn';
      a.textContent = 'Open / download ' + (what || 'file');
      box.appendChild(a);
    }
    if (M.isLocalApi(u) && !M.sameOrigin) {
      var h = document.createElement('div');
      h.className = 'owui-fallback-hint';
      h.textContent = 'To view it inline, enable Settings > Interface > "iframe Sandbox Allow Same Origin".';
      box.appendChild(h);
    }
    M.postHeight();
  };

  // Load url into an <img>/<audio>/<video>. onFail() after every strategy failed.
  M.load = function (el, u, onFail) {
    var tried = false;
    function fail() { if (onFail) onFail(); }
    if (!M.isSafe(u)) { fail(); return; }
    el.addEventListener('error', function () {
      if (tried || !M.isLocalApi(u) || !M.sameOrigin) { fail(); return; }
      tried = true;  // direct src failed (e.g. no cookie) -> Bearer blob
      M.blobUrl(u).then(function (b) { el.dataset.blob = b; el.src = b; }).catch(fail);
    });
    el.addEventListener('load', M.postHeight);
    el.addEventListener('loadedmetadata', M.postHeight);
    el.src = u;  // direct first: streams/range requests, uses the cookie when it is sent
  };

  // Download: same-origin -> blob with Bearer; otherwise a top-level tab (cookie is sent).
  M.download = function (u, name, el) {
    if (!M.isSafe(u)) return;
    function save(href) {
      var a = document.createElement('a');
      a.href = href; a.download = name || ''; a.rel = 'noopener';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }
    function openTab() {
      var a = document.createElement('a');
      a.href = u; a.target = '_blank'; a.rel = 'noopener noreferrer';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }
    if (el && el.dataset && el.dataset.blob) { save(el.dataset.blob); return; }
    if (M.isLocalApi(u) && M.sameOrigin) {
      M.blobUrl(u).then(function (b) { save(b); setTimeout(function () { URL.revokeObjectURL(b); }, 30000); })
        .catch(openTab);
      return;
    }
    openTab();
  };
})();
"""

_BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { background: transparent; overflow: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
.owui-btn {
  display: inline-block; padding: 8px 14px; margin-top: 8px; border-radius: 8px;
  font-size: 12px; font-weight: 600; text-decoration: none; cursor: pointer;
  color: #333; background: rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.12);
}
.owui-btn:hover { background: rgba(0,0,0,0.1); }
.owui-fallback { padding: 14px 16px; border-radius: 12px; line-height: 1.5;
  background: rgba(0,0,0,0.05); border: 1px dashed rgba(0,0,0,0.2); color: #444; font-size: 13px; }
.owui-fallback-hint { margin-top: 8px; font-size: 11px; color: #777; }
.prompt-bubble {
  background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); border-radius: 12px;
  padding: 12px 16px; margin: 8px 0 0 0; font-size: 13px; line-height: 1.5; color: #333;
  word-wrap: break-word; white-space: pre-wrap;
}
.prompt-label { font-weight: 600; color: #666; margin-bottom: 4px; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.5px; white-space: normal; }
.prompt-text { color: #444; }
@media (prefers-color-scheme: dark) {
  .owui-btn { color: #e0e0e0; background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); }
  .owui-btn:hover { background: rgba(255,255,255,0.15); }
  .owui-fallback { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.2); color: #ccc; }
  .owui-fallback-hint { color: #999; }
  .prompt-bubble { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.1); color: #ddd; }
  .prompt-label { color: #aaa; }
  .prompt-text { color: #ccc; }
}
"""


def _page(title: str, css: str, body: str, data: Any, script: str) -> str:
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        f"<title>{esc(title)}</title>\n<style>{_BASE_CSS}{css}</style>\n</head>\n<body>\n"
        f"<div id=\"owui-root\">{body}</div>\n"
        f"<script>{_RUNTIME_JS}</script>\n"
        f"<script>(function (DATA) {{{script}}})({script_json(data)});</script>\n"
        "</body>\n</html>"
    )


def _prompt_bubble(label: str, text: Optional[str]) -> str:
    if not text:
        return ""
    return (
        '<div class="prompt-bubble"><div class="prompt-label">'
        f"{esc(label)}</div><div class=\"prompt-text\">{esc(text)}</div></div>"
    )


# --------------------------------------------------------------------------
# Image gallery (comfyui_text_to_image / muapi look)
# --------------------------------------------------------------------------

_GALLERY_CSS = """
.gallery { display: grid; grid-template-columns: 1fr; gap: 8px; }
.gallery.multi { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.media-wrapper { position: relative; border-radius: 12px; overflow: hidden; line-height: 0; }
.media-wrapper img { max-width: 100%; width: 100%; height: auto; display: block; border: none; }
.media-wrapper .dl {
  position: absolute; right: 8px; bottom: 8px; width: 30px; height: 30px; border-radius: 50%;
  border: none; cursor: pointer; background: rgba(0,0,0,0.55); color: #fff; opacity: 0;
  transition: opacity .15s; display: flex; align-items: center; justify-content: center;
}
.media-wrapper:hover .dl, .media-wrapper .dl:focus { opacity: 1; }
.media-wrapper .dl svg { width: 14px; height: 14px; fill: currentColor; }
.media-wrapper.owui-fallback { line-height: 1.5; }
"""

_GALLERY_JS = r"""
var M = window.OWUIMedia, g = document.getElementById('gallery');
if (DATA.items.length > 1) g.className += ' multi';
DATA.items.forEach(function (it, i) {
  var w = document.createElement('div'); w.className = 'media-wrapper';
  var img = document.createElement('img');
  img.alt = it.alt || ('Generated image ' + (i + 1)); img.decoding = 'async';
  w.appendChild(img);
  var b = document.createElement('button'); b.className = 'dl'; b.title = 'Download';
  b.innerHTML = '<svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>';
  b.addEventListener('click', function () { M.download(it.url, it.name, img); });
  w.appendChild(b);
  g.appendChild(w);
  M.load(img, it.url, function () { M.fallback(w, it.url, 'image'); });
});
"""


def image_gallery_embed(
    urls: Union[str, Sequence[str]],
    *,
    prompt: str = "",
    prompt_label: str = "Prompt",
    title: str = "Generated Image",
    download_names: Optional[Sequence[str]] = None,
) -> str:
    """One or more images (2-column grid for >1), hover download button, prompt bubble."""
    clean = _clean_urls(urls)
    items = []
    for i, u in enumerate(clean):
        name = (download_names[i] if download_names and i < len(download_names) else None) or f"image-{i + 1}.png"
        items.append({"url": u, "name": name, "alt": f"{title} {i + 1}" if len(clean) > 1 else title})
    body = '<div id="gallery" class="gallery"></div>' + _prompt_bubble(prompt_label, prompt)
    return _page(title, _GALLERY_CSS, body, {"items": items}, _GALLERY_JS)


# --------------------------------------------------------------------------
# Return helper
# --------------------------------------------------------------------------


def embed_result(html_content: str, context: Union[str, dict]):
    """Return value for a tool method: (HTMLResponse inline embed, LLM context).

    Open WebUI (utils/middleware.py process_tool_result) renders the HTML as an
    embed and gives ``context`` (str/dict/list) to the model instead of the HTML.
    """
    return (
        HTMLResponse(content=html_content, headers={"Content-Disposition": "inline"}),
        context,
    )


def comfyui_view_url(
    comfyui_http_url: str, filename: str, subfolder: str, file_type: str = "output"
) -> str:
    """Internal ComfyUI /view URL. Only used server-side to download the output;
    it is never shown to the user or the LLM."""
    params = {"filename": filename, "type": file_type or "output"}
    if subfolder:
        params["subfolder"] = subfolder
    return f"{comfyui_http_url}/view?{urlencode(params)}"


async def save_comfyui_image(
    comfyui_http_url: str,
    filename: str,
    subfolder: str,
    *,
    request: Any,
    user: Any,
    metadata: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    prefix: str = "comfyui_image",
    file_type: str = "output",
) -> Dict[str, Any]:
    """Download a ComfyUI output image and store it in Open WebUI, linked to the
    chat message. Raises MediaStorageError (no fallback to the /view URL)."""
    content_type = mimetypes.guess_type(filename)[0] or "image/png"
    return await download_to_storage(
        comfyui_view_url(comfyui_http_url, filename, subfolder, file_type),
        request=request,
        user=user,
        metadata=metadata,
        filename=filename,
        content_type=content_type,
        headers=headers,
        prefix=prefix,
    )


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

    elif workflow_type == "Qwen_Image_Edit":
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
            if (
                conditioning_node_id in workflow
                and "inputs" in workflow[conditioning_node_id]
            ):
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
                url = (
                    image_url_obj.get("url")
                    if isinstance(image_url_obj, dict)
                    else None
                )
                if url and isinstance(url, str) and url.startswith("data:image"):
                    base64_data = url.split(",", 1)[1] if "," in url else url
                    base64_images.append(base64_data)

    return base64_images


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
        workflow_type: Literal["Flux_Kontext", "Qwen_Image_Edit", "Custom"] = Field(
            default="Qwen_Image_Edit", description="Workflow to use for image editing."
        )
        custom_workflow: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Custom ComfyUI workflow JSON (only used if workflow_type='Custom').",
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
            default=True, description="Return an HTML image embed upon completion."
        )

    def __init__(self):
        self.valves = self.Valves()

    async def edit_image(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Any] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Tuple[HTMLResponse, Any]]:
        """
        Edit or transform images using AI-powered workflows. Images are automatically extracted from the user's message.

        **CRITICAL: You MUST enhance and expand the user's prompt before passing it to this tool.**

        **Your Responsibilities:**
        1. Take the user's brief instruction and expand it into a detailed, specific prompt
        2. Add visual details, style descriptions, and technical specifications
        3. Be explicit about what should change and what should remain
        4. For vague requests, infer intent and add descriptive details

        **Prompt Enhancement Examples:**

        User says: "put the cyberpunk dolphin in the beautiful natural background"
        You should pass: "Seamlessly composite the cyberpunk-styled dolphin with neon accents and technological augmentations into a lush natural ocean environment with crystal clear turquoise water, coral reefs, and dappled sunlight filtering through the water surface. Blend the futuristic elements naturally while maintaining the dolphin as the focal point against the pristine underwater scenery. Ensure realistic lighting and color harmony between the cyberpunk subject and organic background."

        User says: "make it look vintage"
        You should pass: "Transform the image into a vintage photograph aesthetic from the 1970s with warm, faded color tones, slight yellow/sepia tint, subtle grain texture, soft vignetting around the edges, and slightly reduced contrast to mimic aged film photography. Add authentic period-appropriate color grading and a nostalgic, timeworn appearance."

        User says: "remove the background"
        You should pass: "Completely remove and replace the background with a clean, transparent background while precisely preserving the main subject. Maintain sharp edges and fine details like hair, fur, or intricate outlines. Ensure professional cutout quality suitable for compositing."

        User says: "add dramatic lighting"
        You should pass: "Add dramatic cinematic lighting with strong directional light source creating deep shadows and bright highlights. Incorporate rim lighting to separate the subject from the background, use warm golden hour tones or cool blue shadows depending on mood, and enhance contrast to create visual depth and emotional impact. Maintain natural light behavior and realistic shadow placement."

        **Enhancement Guidelines:**
        - For object placement: Specify position, scale, blending method, and lighting integration
        - For style changes: Include specific artistic period, technique, color palette, and texture details
        - For removals: Specify what to fill the space with (transparent, similar background, specific content)
        - For atmospheric effects: Detail time of day, weather, mood, color temperature, and quality
        - For multi-image edits: Explicitly state which elements from which image should be combined and how

        **Technical Details to Include:**
        - Lighting direction, quality (soft/hard), and color temperature
        - Desired mood and atmosphere
        - Specific colors, textures, and materials
        - Level of detail and realism expected
        - Composition and framing considerations

        Images are automatically extracted - never mention "the image" or "the attachment" in the prompt.

        :param prompt: Detailed, enhanced instruction with specific visual details, style descriptions, and technical specifications for the image transformation
        """
        try:
            if not __messages__:
                return "Error: No messages provided. Please attach an image to your message."

            base64_images = extract_images_from_messages(__messages__)
            if not base64_images or len(base64_images) == 0:
                return "Error: No images found in the last message. Please attach at least one image and try again."

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

            base64_images = base64_images[:3]

            if self.valves.unload_ollama_models:
                await unload_all_models_async(self.valves.ollama_api_url)

            if self.valves.workflow_type == "Flux_Kontext":
                base_workflow = DEFAULT_FLUX_KONTEXT_WORKFLOW
            elif self.valves.workflow_type == "Qwen_Image_Edit":
                base_workflow = DEFAULT_QWEN_EDIT_WORKFLOW
            elif self.valves.workflow_type == "Custom":
                if not self.valves.custom_workflow:
                    return "Error: Custom workflow selected but no custom workflow JSON provided in valves."
                base_workflow = self.valves.custom_workflow
            else:
                return f"Error: Unknown workflow type '{self.valves.workflow_type}'. Use 'Flux_Kontext', 'Qwen_Image_Edit', or 'Custom'."

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

            comfyui_headers = {}
            if self.valves.comfyui_api_key:
                comfyui_headers["Authorization"] = f"Bearer {self.valves.comfyui_api_key}"

            if __event_emitter__:
                if self.valves.workflow_type == "Custom":
                    status_message = "🎨 Editing image..."
                else:
                    workflow_display = self.valves.workflow_type.replace("_", " ")
                    status_message = f"🎨 Editing image using {workflow_display}..."

                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": status_message,
                            "done": False,
                        },
                    }
                )

            async with aiohttp.ClientSession(headers=comfyui_headers) as session:
                async with session.post(
                    f"{http_api_url}/prompt",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.error(
                            "ComfyUI submission failed (%s): %s",
                            resp.status,
                            (await resp.text())[:2000],
                        )
                        raise Exception(
                            f"ComfyUI API error on submission (HTTP {resp.status})."
                        )
                    result = await resp.json()
                    prompt_id = result.get("prompt_id")
                    if not prompt_id:
                        logger.error("ComfyUI returned no prompt_id: %s", json.dumps(result)[:2000])
                        raise Exception("No prompt_id returned from ComfyUI.")

            job_data = await wait_for_completion_ws(
                ws_api_url,
                http_api_url,
                prompt_id,
                client_id,
                self.valves.max_wait_time,
                headers=comfyui_headers,
            )

            image_files = extract_image_files(job_data)

            if not image_files:
                logger.error(
                    "ComfyUI job %s produced no image outputs: %s",
                    prompt_id,
                    json.dumps(job_data.get("outputs", {}))[:4000],
                )
                raise Exception(
                    "Generation completed but ComfyUI returned no output image."
                )

            image_file_info = image_files[0]
            filename = image_file_info["filename"]
            subfolder = image_file_info.get("subfolder", "")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Saving image to Open WebUI...",
                            "done": False,
                        },
                    }
                )

            # Persist in Open WebUI storage (linked to this chat message). Never
            # fall back to the internal ComfyUI /view URL.
            saved = await save_comfyui_image(
                http_api_url,
                filename,
                subfolder,
                request=__request__,
                user=__user__,
                metadata=__metadata__,
                headers=comfyui_headers,
                prefix="comfyui_i2i",
                file_type=image_file_info.get("type", "output"),
            )
            image_url = saved["url"]

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
                html_content = image_gallery_embed(
                    image_url,
                    prompt=prompt,
                    title="Edited Image",
                    download_names=[saved.get("name") or "edited-image.png"],
                )
                return embed_result(
                    html_content,
                    {
                        "message": (
                            "Image transformation complete and already displayed to the "
                            "user above. Do not repeat it as a link or markdown."
                        ),
                        "files": [image_url],
                    },
                )
            else:
                return f"✅ Image transformation complete!\n\n**Download:** [Edited Image]({image_url})\n\n**Direct link:** {image_url}"

        except MediaStorageError as e:
            message = f"❌ Image edited, but could not save it to Open WebUI: {e}"
        except TimeoutError as e:
            message = f"⏰ Generation timed out: {str(e)}"
        except aiohttp.ClientError as e:
            logger.error("ComfyUI request failed: %s", e)
            message = f"❌ Could not reach ComfyUI ({type(e).__name__})."
        except Exception as e:
            message = f"❌ An error occurred: {str(e)}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": True,
                    },
                }
            )
        return message
