"""
title: Hugging Face API - Flux Pro Image Generator
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.1
license: MIT
description: HuggingFace API implementation for text to image generation
"""

import aiohttp
import uuid
import os
from pydantic import BaseModel, Field
import logging
from aiohttp import ClientTimeout
from typing import Optional

from fastapi import Request, UploadFile
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFException(Exception):
    """Base exception for HuggingFace API related errors."""

    pass


class Tools:
    class Valves(BaseModel):
        HF_API_KEY: str = Field(
            default=None,
            description="HuggingFace API key for accessing the serverless endpoints",
            json_schema_extra={"input": {"type": "password"}},
        )
        HF_API_URL: str = Field(
            default="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo",
            description="HuggingFace API URL for accessing the serverless endpoint of a Text to Image model.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def create_image(
        self,
        prompt: str,
        image_format: str = "default",
        __user__: dict = {},
        __request__: Optional[Request] = None,
        __event_emitter__=None,
    ) -> str:
        """
        Creates visually stunning images with text prompts using text to image models.
        If the user prompt is too general or lacking, embellish it to generate a better illustration.

        :param prompt: The prompt to generate the image.
        :param image_format: Format of the image (default, landscape, portrait, etc.)
        """
        print("[DEBUG] Starting create_flux_image function")

        if not self.valves.HF_API_KEY:
            print("[DEBUG] API key not found")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Error: HuggingFace API key is not set",
                        "done": True,
                    },
                }
            )
            return "HuggingFace API key is not set in the Valves."

        try:
            formats = {
                "default": (1024, 1024),
                "square": (1024, 1024),
                "landscape": (1024, 768),
                "landscape_large": (1440, 1024),
                "portrait": (768, 1024),
                "portrait_large": (1024, 1440),
            }

            print(f"[DEBUG] Validating format: {image_format}")
            if image_format not in formats:
                raise ValueError(
                    f"Invalid format. Must be one of: {', '.join(formats.keys())}"
                )

            width, height = formats[image_format]
            print(f"[DEBUG] Using dimensions: {width}x{height}")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generating image", "done": False},
                }
            )

            headers = {"Authorization": f"Bearer {self.valves.HF_API_KEY}"}
            payload = {
                "inputs": prompt,
                "parameters": {"width": width, "height": height},
            }

            async with aiohttp.ClientSession(
                timeout=ClientTimeout(total=600)
            ) as session:
                async with session.post(
                    self.valves.HF_API_URL, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[DEBUG] API request failed: {error_text}")
                        raise HFException(
                            f"API request failed with status code {response.status}: {error_text}"
                        )

                    image_content = await response.read()

            filename = f"{uuid.uuid4()}.png"
            upload_file = UploadFile(
                file=io.BytesIO(image_content),
                filename=filename,
                headers={"content-type": "image/png"},
            )
            current_user = Users.get_user_by_id(__user__["id"]) if __user__ else None

            file_item = upload_file_handler(
                __request__,
                file=upload_file,
                metadata={},
                process=False,
                user=current_user,
            )

            if file_item and getattr(file_item, "id", None):
                file_id = str(getattr(file_item, "id", ""))
                image_url = __request__.app.url_path_for(
                    "get_file_content_by_id", id=file_id
                )
            else:
                return "Error: Failed to upload image to storage."

            print(f"[DEBUG] Image uploaded successfully, relative URL: {image_url}")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Image generated", "done": True},
                }
            )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"Generated image for prompt: '{prompt}'\n\n![Generated Image]({image_url})"
                    },
                }
            )

            return f"Notify the user that the image has been successfully generated for the prompt: '{prompt}' "

        except aiohttp.ClientError as e:
            error_msg = f"Network error occurred: {str(e)}"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": error_msg, "done": True},
                }
            )
            return error_msg

        except Exception as e:
            print(f"[DEBUG] Unexpected error: {str(e)}")
            error_msg = f"An error occurred: {str(e)}"
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": error_msg, "done": True},
                }
            )
            return error_msg
