"""
title: Cloudflare Workers AI Image Generator
author: tan-yong-sheng
author_url: https://github.com/tan-yong-sheng
description: Generate images using Cloudflare Workers AI text-to-image models with preprocessing for different model types
version: 0.1.0
license: MIT
requirements: requests
"""

import requests
import base64
import json
import os
import uuid
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from fastapi import Request, UploadFile
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
import io


def _get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration and preprocessing parameters."""
    model_configs = {
        "@cf/black-forest-labs/flux-1-schnell": {
            "max_prompt_length": 2048,
            "default_steps": 4,
            "max_steps": 8,
            "supports_negative_prompt": False,
            "supports_size": False,
            "supports_guidance": False,
            "supports_seed": True,
            "output_format": "base64",
            "recommended_for": "fast generation, high quality",
        },
        "@cf/stabilityai/stable-diffusion-xl-base-1.0": {
            "max_prompt_length": 1000,
            "default_steps": 15,  # Optimized: quality plateaus around 15-20 steps
            "max_steps": 20,
            "supports_negative_prompt": True,
            "supports_size": True,
            "supports_guidance": True,
            "supports_seed": True,
            "output_format": "binary",
            "recommended_for": "high quality, detailed images",
            "default_guidance": 12.0,  # Optimized: lower than 7.5 default to reduce over-adherence and distortion
            "guidance_range": "10.0-15.0",  # Sweet spot for SDXL quality without distortion
            "recommended_negative": "blurry, low quality, distorted, unrealistic, bad anatomy, poor quality, jpeg artifacts",
            "notes": "SDXL works best with guidance 10-15, detailed prompts, and comprehensive negative prompts",
        },
        "@cf/bytedance/stable-diffusion-xl-lightning": {
            "max_prompt_length": 1000,
            "default_steps": 4,
            "max_steps": 8,
            "supports_negative_prompt": True,
            "supports_size": True,
            "supports_guidance": True,
            "supports_seed": True,
            "output_format": "binary",
            "recommended_for": "fast generation with good quality",
        },
        "@cf/lykon/dreamshaper-8-lcm": {
            "max_prompt_length": 1000,
            "default_steps": 4,  # LCM models work best with 4-8 steps
            "max_steps": 8,  # Not 20! LCM needs fewer steps
            "supports_negative_prompt": True,
            "supports_size": True,
            "supports_guidance": True,
            "supports_seed": True,
            "output_format": "binary",
            "recommended_for": "photorealistic images (LCM - use 4-8 steps, lower guidance)",
            "guidance_range": "1.0-2.0",  # LCM works better with lower guidance
            "notes": "LCM model - requires 4-8 steps and lower guidance (1.0-2.0) for best results",
        },
    }

    # Check for unsupported models and provide helpful error
    unsupported_models = {
        "@cf/runwayml/stable-diffusion-v1-5-img2img": "Image-to-image functionality is not supported in this tool. Please use text-to-image generation instead.",
        "@cf/runwayml/stable-diffusion-v1-5-inpainting": "Image inpainting functionality is not supported in this tool. Please use text-to-image generation instead.",
    }

    if model_name in unsupported_models:
        raise ValueError(unsupported_models[model_name])

    return model_configs.get(
        model_name, model_configs["@cf/black-forest-labs/flux-1-schnell"]
    )


def _preprocess_prompt(prompt: str, model_name: str) -> str:
    """Preprocess prompt based on model requirements."""
    config = _get_model_config(model_name)
    max_length = config.get("max_prompt_length", 2048)

    # Truncate if too long
    if len(prompt) > max_length:
        prompt = prompt[: max_length - 3] + "..."

    # Model-specific prompt enhancements
    if "flux" in model_name.lower():
        # FLUX works well with detailed, artistic descriptions
        if not any(
            word in prompt.lower()
            for word in ["detailed", "high quality", "masterpiece"]
        ):
            prompt = f"detailed, high quality, {prompt}"

    elif "dreamshaper" in model_name.lower():
        # DreamShaper excels at photorealistic content
        if not any(
            word in prompt.lower() for word in ["photorealistic", "realistic", "photo"]
        ):
            prompt = f"photorealistic, {prompt}"

    elif "xl" in model_name.lower():
        # SDXL models benefit from style descriptors and detailed prompts
        if not any(
            word in prompt.lower()
            for word in [
                "cinematic",
                "artistic",
                "professional",
                "detailed",
                "high quality",
                "masterpiece",
                "ultra realistic",
            ]
        ):
            prompt = f"professional, cinematic, highly detailed, {prompt}"

    return prompt


def _build_request_payload(prompt: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """Build the request payload based on model capabilities."""
    config = _get_model_config(model_name)
    payload = {"prompt": _preprocess_prompt(prompt, model_name)}

    # Add parameters based on model support
    if config.get("supports_negative_prompt"):
        negative_prompt = kwargs.get("negative_prompt", "")

        # Auto-enhance negative prompt for SDXL if none provided
        if (
            "stable-diffusion-xl-base-1.0" in model_name.lower()
            and not negative_prompt.strip()
        ):
            negative_prompt = config.get("recommended_negative", "")

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

    if config.get("supports_size"):
        size = kwargs.get("size", "1024x1024")
        if "x" in size:
            width, height = map(int, size.split("x"))
            payload["width"] = max(256, min(2048, width))
            payload["height"] = max(256, min(2048, height))

    if config.get("supports_guidance"):
        guidance_value = kwargs.get("guidance", config.get("default_guidance", 7.5))

        # Special handling for LCM models - use lower guidance
        if "dreamshaper-8-lcm" in model_name.lower():
            guidance_value = min(guidance_value, 2.0)  # Cap at 2.0 for LCM
            if guidance_value > 2.0:
                guidance_value = 1.5  # Use 1.5 as default for LCM
        # Special handling for SDXL - use optimized guidance range
        elif "stable-diffusion-xl-base-1.0" in model_name.lower():
            # Use optimized guidance range for SDXL: 10-15 for better quality
            if guidance_value == 7.5:  # If using default, upgrade to optimized value
                guidance_value = config.get("default_guidance", 12.0)
            guidance_value = max(
                10.0, min(guidance_value, 15.0)
            )  # Clamp to optimal range

        payload["guidance"] = guidance_value

    if config.get("supports_seed") and kwargs.get("seed"):
        payload["seed"] = kwargs["seed"]

    # Steps configuration - use correct parameter name for each model
    steps = kwargs.get("steps", config.get("default_steps", 4))
    if "flux" in model_name.lower():
        payload["steps"] = min(steps, config.get("max_steps", 8))  # FLUX uses "steps"
    else:
        payload["num_steps"] = min(
            steps, config.get("max_steps", 20)
        )  # Others use "num_steps"

    # Handle image input for img2img and inpainting models
    if config.get("supports_image_input") and kwargs.get("image_b64"):
        payload["image_b64"] = kwargs["image_b64"]
        if kwargs.get("strength"):
            payload["strength"] = max(0.1, min(1.0, kwargs["strength"]))

    # Handle mask for inpainting
    if config.get("supports_mask") and kwargs.get("mask"):
        payload["mask"] = kwargs["mask"]

    return payload


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        self.citation = False

    class Valves(BaseModel):
        cloudflare_api_token: str = Field("", description="Your Cloudflare API Token")
        cloudflare_account_id: str = Field("", description="Your Cloudflare Account ID")
        default_model: str = Field(
            "@cf/black-forest-labs/flux-1-schnell",
            description="Default image generation model (always used) - FLUX recommended over SDXL",
        )

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        negative_prompt: str = "",
        steps: int = 4,
        guidance: float = 7.5,
        seed: Optional[int] = None,
        __user__: Dict[str, Any] = {},
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> str:
        """
        Generate an image using Cloudflare Workers AI text-to-image models.

        :param prompt: Text description of the image to generate
        :param size: Image size in format 'widthxheight' (e.g., '1024x1024')
        :param negative_prompt: Text describing what to avoid in the image
        :param steps: Number of diffusion steps (model-dependent limits)
        :param guidance: How closely to follow the prompt (1.0-20.0)
        :param seed: Random seed for reproducible results
        """
        if (
            not self.valves.cloudflare_api_token
            or not self.valves.cloudflare_account_id
        ):
            return "Error: Cloudflare API Token and Account ID must be configured in the tool settings."

        if not prompt.strip():
            return "Error: Please provide a prompt for image generation."

        # Always use the configured default model (ignore any LLM model selection)
        model_name = self.valves.default_model

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating image with {model_name}...",
                        "done": False,
                    },
                }
            )

        try:
            # Validate the default model is supported
            config = _get_model_config(model_name)

            # Build request payload with model-specific preprocessing
            payload = _build_request_payload(
                prompt=prompt,
                model_name=model_name,
                size=size,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                seed=seed,
            )

            # Log a warning if parameters were ignored due to model limitations
            ignored_params = []

            if negative_prompt and not config.get("supports_negative_prompt"):
                ignored_params.append("negative_prompt")
            if size != "1024x1024" and not config.get("supports_size"):
                ignored_params.append("size")
            if guidance != 7.5 and not config.get("supports_guidance"):
                ignored_params.append("guidance")

            if ignored_params and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Note: {', '.join(ignored_params)} not supported by {model_name}",
                            "done": False,
                        },
                    }
                )

            # Make API request
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.valves.cloudflare_account_id}/ai/run/{model_name}"
            headers = {
                "Authorization": f"Bearer {self.valves.cloudflare_api_token}",
                "Content-Type": "application/json",
            }

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Sending request to Cloudflare Workers AI...",
                            "done": False,
                        },
                    }
                )

            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Error: {error_msg}",
                                "done": True,
                            },
                        }
                    )
                return f"Error: {error_msg}"

            # Handle different response formats based on actual Cloudflare API behavior
            content_type = response.headers.get("content-type", "").lower()

            # Process response and save image using upload_file_handler
            filename = f"{uuid.uuid4()}.jpg"
            image_binary = None

            if "application/json" in content_type:
                # JSON response - FLUX models return base64
                try:
                    result = response.json()

                    # Extract image data from Cloudflare's nested structure
                    if "result" in result and isinstance(result["result"], dict):
                        image_data = result["result"].get("image")
                    else:
                        image_data = result.get("image")

                    if not image_data:
                        return f"Error: No image data found in response. Keys: {list(result.keys())}"

                    # Decode base64
                    image_binary = base64.b64decode(image_data)

                except (json.JSONDecodeError, Exception) as e:
                    return f"Error processing JSON response: {str(e)}"

            else:
                # Binary response - other Stable Diffusion models
                image_binary = response.content
                if len(image_binary) < 100:
                    return f"Error: Invalid binary image data received"

            if image_binary:
                upload_file = UploadFile(
                    file=io.BytesIO(image_binary),
                    filename=filename,
                    headers={"content-type": "image/jpeg"},
                )
                current_user = (
                    Users.get_user_by_id(__user__["id"]) if __user__ else None
                )

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

            # Debug: Log the image URL that was generated
            print(f"[DEBUG] Generated image URL: {image_url}")
            print(f"[DEBUG] Event emitter available: {__event_emitter__ is not None}")

            # Try event emitters with error handling
            try:
                if __event_emitter__:
                    print("[DEBUG] Sending status event...")
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "Image generated", "done": True},
                        }
                    )

                    print("[DEBUG] Sending message event...")
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f"Generated image for prompt: '{prompt}'\n\n![Generated Image]({image_url})"
                            },
                        }
                    )
                    print("[DEBUG] Event emitters completed successfully")
                else:
                    print("[DEBUG] No event emitter available")

            except Exception as e:
                print(f"[DEBUG] Event emitter error: {str(e)}")

            # Also return the image URL in the success message for debugging
            return f"Notify the user that the image has been successfully generated for the prompt: '{prompt}'. Image URL: {image_url} and should be always displayed on UI in image markdown format: ![Alt text]({image_url})"

        except ValueError as e:
            # Handle unsupported model errors
            error_msg = str(e)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {error_msg}", "done": True},
                    }
                )
            return f"Error: {error_msg}"

        except requests.exceptions.Timeout:
            error_msg = "Request timed out. The image generation is taking longer than expected."
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return f"Error: {error_msg}"

    async def list_models(self, __event_emitter__: Optional[Callable] = None) -> str:
        """
        List available Cloudflare Workers AI image generation models with their capabilities.
        """
        models_info = {
            "@cf/black-forest-labs/flux-1-schnell": "FLUX.1 Schnell - Fast, high-quality generation (12B params)",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0": "Stable Diffusion XL - High quality, detailed images",
            "@cf/bytedance/stable-diffusion-xl-lightning": "SDXL Lightning - Fast generation with good quality",
            "@cf/lykon/dreamshaper-8-lcm": "DreamShaper 8 LCM - Photorealistic images",
        }

        result = "Available Cloudflare Workers AI Image Generation Models:\n\n"
        for model_id, description in models_info.items():
            config = _get_model_config(model_id)
            result += f"**{model_id}**\n"
            result += f"  Description: {description}\n"
            result += (
                f"  Recommended for: {config.get('recommended_for', 'general use')}\n"
            )
            result += f"  Max steps: {config.get('max_steps', 'N/A')}\n"
            result += f"  Supports negative prompts: {config.get('supports_negative_prompt', False)}\n"
            result += (
                f"  Supports custom size: {config.get('supports_size', False)}\n\n"
            )

        result += f"Current default model: {self.valves.default_model}\n"
        result += "\nNote: Image-to-image and inpainting models are not supported in this tool."

        return result
