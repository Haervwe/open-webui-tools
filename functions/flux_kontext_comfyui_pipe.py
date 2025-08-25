# title: ComfyUI Universal Pipe
# author: Haervwe
# author_url: https://github.com/Haervwe/open-webui-tools
# funding_url: https://github.com/Haervwe/open-webui-tools
# version: 3.2.0 (Modified)
# required_open_webui_version: 0.5.0


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

import logging
import requests

# [START] MODIFICATION: Added necessary imports for file handling
import io
import mimetypes
from fastapi import UploadFile
from open_webui.routers.files import upload_file_handler
# [END] MODIFICATION

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- OLLAMA VRAM Management Functions ---
# ... (this section remains unchanged)
def get_loaded_models(api_url: str = "http://localhost:11434") -> list:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/ps", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        logger.error(f"Error fetching loaded Ollama models: {e}")
        return []


def unload_all_models(api_url: str = "http://localhost:11434"):
    try:
        for model in get_loaded_models(api_url):
            model_name = model.get("name")
            if model_name:
                requests.post(
                    f"{api_url.rstrip('/')}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=10,
                )
    except requests.RequestException as e:
        logger.error(f"Error unloading Ollama models: {e}")

# --- Default Workflow ---
# ... (this section remains unchanged)
DEFAULT_WORKFLOW_JSON = json.dumps(
    {
        "6": {
            "inputs": {
                "text": "re imagine this man maintainig the facial features as a medieval fantasy king...",
                "clip": ["38", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "35": {
            "inputs": {"guidance": 2.5, "conditioning": ["177", 0]},
            "class_type": "FluxGuidance",
        },
        "37": {
            "inputs": {
                "unet_name": "flux-kontext/flux1-dev-kontext_fp8_scaled.safetensors",
                "weight_dtype": "fp8_e4m3fn_fast",
            },
            "class_type": "UNETLoader",
        },
        "38": {
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "flux/t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "flux",
                "device": "cpu",
            },
            "class_type": "DualCLIPLoader",
        },
        "39": {
            "inputs": {"vae_name": "Flux/ae.safetensors"},
            "class_type": "VAELoader",
        },
        "42": {"inputs": {"image": ["196", 0]}, "class_type": "FluxKontextImageScale"},
        "124": {
            "inputs": {"pixels": ["42", 0], "vae": ["39", 0]},
            "class_type": "VAEEncode",
        },
        "135": {
            "inputs": {"conditioning": ["6", 0]},
            "class_type": "ConditioningZeroOut",
        },
        "136": {
            "inputs": {"filename_prefix": "owui/owui", "images": ["194", 5]},
            "class_type": "SaveImage",
        },
        "177": {
            "inputs": {"conditioning": ["6", 0], "latent": ["124", 0]},
            "class_type": "ReferenceLatent",
        },
        "194": {
            "inputs": {
                "seed": 558680250753563,
                "steps": 20,
                "cfg": 1,
                "sampler_name": "dpmpp_2m",
                "scheduler": "beta",
                "denoise": 1,
                "preview_method": "none",
                "vae_decode": "true (tiled)",
                "model": ["37", 0],
                "positive": ["35", 0],
                "negative": ["135", 0],
                "latent_image": ["124", 0],
                "optional_vae": ["39", 0],
            },
            "class_type": "KSampler (Efficient)",
        },
        "196": {"inputs": {"image": ""}, "class_type": "ETN_LoadImageBase64"},
    },
    indent=2,
)

class Pipe:
    class Valves(BaseModel):
        # ... (this section remains unchanged)
        ComfyUI_Address: str = Field(
            default="http://127.0.0.1:8188",
            description="Address of the running ComfyUI server.",
        )
        ComfyUI_Workflow_JSON: str = Field(
            default=DEFAULT_WORKFLOW_JSON,
            description="The entire ComfyUI workflow in JSON format.",
            extra={"type": "textarea"},
        )
        Prompt_Node_ID: str = Field(
            default="6", description="The ID of the node that accepts the text prompt."
        )
        Image_Node_ID: str = Field(
            default="196",
            description="The ID of the node that accepts the Base64 image.",
        )
        Seed_Node_ID: str = Field(
            default="194",
            description="The ID of the sampler node to apply a random seed to.",
        )
        enhance_prompt: bool = Field(
            default=False, description="Use vision model to enhance prompt"
        )
        vision_model_id: str = Field(
            default="", description="Vision model to be used as prompt enhancer"
        )
        enhancer_system_prompt: str = Field(
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
        unload_ollama_models: bool = Field(
            default=False,
            description="Unload all Ollama models from VRAM before running.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL for unloading models.",
        )
        max_wait_time: int = Field(
            default=1200, description="Max wait time for generation (seconds)."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.client_id = str(uuid.uuid4())

    # [START] MODIFICATION: Added new helper function to save image and create public URL
    def _save_image_and_get_public_url(self, request, image_data: bytes, content_type: str, user: User) -> str:
        """
        Saves the image data to OpenWebUI's file storage and returns a publicly accessible URL.
        This logic is adapted from OpenWebUI's native image generation handling.
        """
        try:
            image_format = mimetypes.guess_extension(content_type)
            if not image_format:
                # Default to .png if the content type is generic (e.g., 'application/octet-stream')
                image_format = ".png"

            file = UploadFile(
                file=io.BytesIO(image_data),
                filename=f"generated-image{image_format}",
                headers={"content-type": content_type},
            )
            
            # Use OpenWebUI's internal file handler to save the file
            file_item = upload_file_handler(
                request=request,
                file=file,
                metadata={}, # Metadata is optional
                process=False,
                user=user,
            )
            
            # Construct the relative URL that OpenWebUI frontend can understand
            url = request.app.url_path_for("get_file_content_by_id", id=file_item.id)
            return url
        except Exception as e:
            logger.error(f"Error saving image to OpenWebUI: {e}", exc_info=True)
            raise

    # [END] MODIFICATION

    async def emit_status(
        # ... (this function remains unchanged)
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

    async def wait_for_job_signal(
        # ... (this function remains unchanged)
        self, ws_api_url: str, prompt_id: str, event_emitter: Callable
    ) -> bool:
        """Waits for the 'executed' signal from WebSocket without fetching data."""
        start_time = asyncio.get_event_loop().time()
        try:
            async with aiohttp.ClientSession().ws_connect(
                f"{ws_api_url}?clientId={self.client_id}", timeout=30
            ) as ws:
                async for msg in ws:
                    if (
                        asyncio.get_event_loop().time() - start_time
                        > self.valves.max_wait_time
                    ):
                        raise TimeoutError(
                            f"WebSocket wait timed out after {self.valves.max_wait_time}s"
                        )
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
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
                        logger.info(
                            f"Execution signal received for prompt {prompt_id}."
                        )
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
                f"Operation timed out after {self.valves.max_wait_time}s"
            )
        except Exception as e:
            raise e
        return False

    def extract_image_data(self, outputs: Dict) -> Optional[Dict]:
        # ... (this function remains unchanged)
        """Extracts the best possible image from the completed job data, prioritizing final images over previews."""
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

    async def queue_prompt(
        # ... (this function remains unchanged)
        self, session: aiohttp.ClientSession, workflow: Dict
    ) -> Optional[str]:
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with session.post(
            f"{self.valves.ComfyUI_Address}/prompt", json=payload
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("prompt_id")

    def parse_input(self, messages: List[Dict]) -> (Optional[str], Optional[str]):
        # ... (this function remains unchanged)
        user_message_item = get_last_user_message_item(messages)
        if not user_message_item:
            return None, None
        prompt, image_url = "", None
        content = user_message_item.get("content")
        print(str(content)[:200])
        print(str(content)[::200])
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    prompt += part.get("text", "")
                elif part.get("type") == "image_url" and part.get("image_url", {}).get(
                    "url"
                ):
                    image_url = part["image_url"]["url"]
        elif isinstance(content, str):
            prompt = content
        if not image_url and user_message_item.get("images"):
            image_url = user_message_item["images"][0]
        base64_image = (
            image_url.split("base64,", 1)[1]
            if image_url and "base64," in image_url
            else None
        )
        return prompt.strip(), base64_image
    
    async def enhance_prompt(self, prompt, image, user, request, event_emitter):
        # ... (this function remains unchanged)
        await self.emit_status(event_emitter, "info", f"Enhancing the prompt...")
        payload = {
            "model": self.valves.vision_model_id,
            "messages": [
                {
                    "role": "system",
                    "content": self.valves.enhancer_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Enhance the given user prompt based on the given image: {prompt}, provide only the enhnaced AI image edit prompt with no explanations",
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

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable,
        __request__=None,
    ) -> dict:
        self.__event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        messages = body.get("messages", [])
        prompt, base64_image = self.parse_input(messages)
        if self.valves.enhance_prompt:
            prompt = await self.enhance_prompt(
                prompt,
                base64_image,
                self.__user__,
                self.__request__,
                self.__event_emitter__,
            )

        if self.valves.unload_ollama_models:
            await self.emit_status(
                self.__event_emitter__, "info", "Unloading Ollama models..."
            )
            unload_all_models(api_url=self.valves.ollama_url)

        if not base64_image:
            await self.emit_status(
                self.__event_emitter__,
                "error",
                "No valid image provided. Please upload an image.",
                done=True,
            )
            # [START] MODIFICATION: Ensure function returns body on early exit
            return body
            # [END] MODIFICATION

        try:
            workflow = json.loads(self.valves.ComfyUI_Workflow_JSON)
        except json.JSONDecodeError:
            await self.emit_status(
                self.__event_emitter__,
                "error",
                "Invalid JSON in the ComfyUI_Workflow_JSON valve.",
                done=True,
            )
            # [START] MODIFICATION: Ensure function returns body on early exit
            return body
            # [END] MODIFICATION

        http_api_url = self.valves.ComfyUI_Address.rstrip("/")
        ws_api_url = f"{'ws' if not http_api_url.startswith('https') else 'wss'}://{http_api_url.split('://', 1)[-1]}/ws"

        prompt_node, image_node, seed_node = (
            self.valves.Prompt_Node_ID,
            self.valves.Image_Node_ID,
            self.valves.Seed_Node_ID,
        )
        workflow[prompt_node]["inputs"]["text"] = (
            prompt if prompt else "A beautiful, high-quality image"
        )
        workflow[image_node]["inputs"]["image"] = base64_image
        workflow[seed_node]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

        try:
            async with aiohttp.ClientSession() as session:
                prompt_id = await self.queue_prompt(session, workflow)
                if not prompt_id:
                    await self.emit_status(
                        self.__event_emitter__,
                        "error",
                        "Failed to queue prompt in ComfyUI.",
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

                # --- RETRY LOGIC FOR HISTORY FETCH ---
                job_data = None
                for attempt in range(3):
                    await asyncio.sleep(attempt + 1)
                    logger.info(
                        f"Fetching history for prompt {prompt_id}, attempt {attempt + 1}..."
                    )
                    async with session.get(
                        f"{http_api_url}/history/{prompt_id}"
                    ) as resp:
                        if resp.status == 200:
                            history = await resp.json()
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

            # [START] MODIFICATION: This is the core of the fix.
            if image_to_display:
                # 1. Construct the internal URL to download the image from ComfyUI
                internal_image_url = f"{http_api_url}/view?filename={image_to_display['filename']}&subfolder={image_to_display.get('subfolder', '')}&type={image_to_display.get('type', 'output')}"
                await self.emit_status(self.__event_emitter__, "info", f"Downloading generated image from ComfyUI...")
                
                # 2. Download the image bytes. Run in an executor to avoid blocking the event loop.
                loop = asyncio.get_event_loop()
                http_response = await loop.run_in_executor(None, requests.get, internal_image_url)
                http_response.raise_for_status()
                image_data = http_response.content
                content_type = http_response.headers.get('content-type', 'image/png')
                
                await self.emit_status(self.__event_emitter__, "info", f"Embedding image into chat...")

                # 3. Use our new helper function to save the image and get a public URL
                public_image_url = self._save_image_and_get_public_url(
                    request=self.__request__,
                    image_data=image_data,
                    content_type=content_type,
                    user=self.__user__
                )

                # 4. Create the response message with the new, correct URL
                response_content = (
                    f"Here is the edited image:\n\n![Generated Image]({public_image_url})"
                )
                
                await self.__event_emitter__(
                    {"type": "message", "data": {"content": response_content}}
                )
                await self.emit_status(
                    self.__event_emitter__,
                    "success",
                    "Image processed successfully!",
                    done=True,
                )
                
                # 5. Append the successful response to the message history and return it
                body["messages"].append(
                    {"role": "assistant", "content": response_content}
                )
                return body
            # [END] MODIFICATION
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
        
        # [START] MODIFICATION: Ensure the body is always returned to maintain chat state
        return body
        # [END] MODIFICATION
