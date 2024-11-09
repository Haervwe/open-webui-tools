"""
title: Hugging Face API - Flux Pro Image Generator
author: Haervwe
git: https://github.com/Haervwe/open-webui-tools
version: 0.0.1
license: MIT
description: HuggingFace API implementation for tesxt to image generation
"""

import requests
import base64
from typing import Dict, Any
from pydantic import BaseModel, Field
import logging
from requests.exceptions import Timeout, RequestException

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
        )
        HF_API_URL: str = Field(
            default="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo",
            description="HuggingFace API URL for accessing the serverless endpoint of an Text to image Model.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def create_image(
        self,
        prompt: str,
        image_format: str = "default",
        __user__: dict = {},
        __event_emitter__=None,
    ) -> str:
        """
        Creates visually stunning images with text prompts using text to image models, based on a prompt,
        if the user promt is to general or lacking embelish it to generate a better illustration.

        :param prompt: the prompt to generate the image
        :param image_format: format of the image (default, landscape, portrait)
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

            response = requests.post(
                self.valves.HF_API_URL,
                headers=headers,
                json=payload,
                timeout=(10, 600),
            )

            if response.status_code != 200:
                print(f"[DEBUG] API request failed: {response.text}")
                raise HFException(
                    f"API request failed with status code {response.status_code}: {response.text}"
                )

            # Store the image content first
            image_content = response.content

            # Create URL with the stored content
            image_url = f"data:image/jpeg;base64,{base64.b64encode(image_content).decode('utf-8')}"

            # Send the completion status before the image
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Image generated", "done": True},
                }
            )

            # Send the image in a separate message
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": f"Generated image for prompt: '{prompt}'\n\n![Generated Image]({image_url})"
                    },
                }
            )

            return f"Notify the user that the image has been successfully generated for the promt: '{prompt}' "

        except Timeout as e:
            error_msg = (
                "Request timed out while generating the image. Please try again later."
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": error_msg, "done": True},
                }
            )
            return error_msg

        except RequestException as e:
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
