"""
title: Veo 3 Video Generation Pipe
authors:
    - Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
description: Generate videos using Google's Veo 3.1 model via Gemini API.
required_open_webui_version: 0.4.0
requirements: google-genai
version: 1.1
license: MIT

This pipe generates videos using Google's Veo 3.1 model through the Gemini API.
It supports text-to-video generation, image-to-video, and reference images.
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Union

from fastapi import UploadFile
from google import genai
from google.genai.types import Image
from open_webui.models.users import User, Users
from open_webui.routers.files import upload_file_handler  # type: ignore
from open_webui.utils.chat import generate_chat_completion  # type: ignore
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(
            title="Google API Key",
            default="",
            description="Google API key for Gemini API access.",
        )
        MODEL: str = Field(
            title="Veo Model",
            default="veo-3.1-generate-preview",
            description="The Veo model to use for video generation.",
        )
        ENHANCE_PROMPT: bool = Field(
            title="Enhance Prompt",
            default=False,
            description="Use vision model to enhance prompt",
        )
        VISION_MODEL_ID: str = Field(
            title="Vision Model ID",
            default="",
            description="Vision model to be used as prompt enhancer",
        )
        ENHANCER_SYSTEM_PROMPT: str = Field(
            title="Enhancer System Prompt",
            default="""
            You are a video prompt engineering assistant.
            For each request, you will receive a user-provided prompt for video generation.
            Generate a single, improved video generation prompt for the Veo model using best practices.
            Be specific and descriptive: use exact color names, detailed adjectives, and clear action verbs.
            Focus on cinematic elements, camera movements, lighting, and visual style.
            Include timing and pacing descriptions where appropriate.
            Output only the final enhanced prompt with no additional explanation or commentary.
            """,
            description="System prompt to be used on the prompt enhancement process",
        )
        MAX_WAIT_TIME: int = Field(
            title="Max Wait Time",
            default=1200,
            description="Max wait time for video generation (seconds).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _extract_last_user_text(self, messages: list[Dict[str, Any]]) -> Optional[str]:
        """Extract the last user message text."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    ]
                    return " ".join(text_parts)
        return None

    def setup_inputs(
        self, messages: list[Dict[str, Any]]
    ) -> tuple[Optional[str], list[Dict[str, str]]]:
        """Extract prompt and base64 images from messages."""
        prompt: Optional[str] = None
        base64_images: list[Dict[str, str]] = []

        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                image_urls: list[str] = []

                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            # Use text from the most recent user message
                            if not prompt:
                                prompt = item.get("text", "")
                        elif item.get("type") == "image_url" and item.get(
                            "image_url", {}
                        ).get("url"):
                            image_urls.append(item["image_url"]["url"])
                elif isinstance(content, str):
                    # Use text from the most recent user message
                    if not prompt:
                        prompt = content

                # Extract base64 images (collect up to 3 images)
                for image_url in image_urls:
                    if len(base64_images) >= 3:
                        break
                    try:
                        if "base64," in image_url:
                            parts = image_url.split("base64,", 1)
                            mime_part = parts[0]
                            if mime_part.startswith("data:"):
                                mime_type = mime_part[5:]  # remove "data:"
                            else:
                                mime_type = "image/jpeg"  # default
                            base64_data = parts[1]
                            base64_images.append(
                                {"base64": base64_data, "mime_type": mime_type}
                            )
                        elif image_url.startswith("data:") and "," in image_url:
                            mime_part, base64_data = image_url.split(",", 1)
                            mime_type = mime_part[5:]  # remove "data:"
                            base64_images.append(
                                {"base64": base64_data, "mime_type": mime_type}
                            )
                        else:
                            # assume it's base64 without prefix
                            base64_images.append(
                                {"base64": image_url, "mime_type": "image/jpeg"}
                            )
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error while extracting base64 image: {e}"
                        )

                # Stop after processing the most recent user message
                if prompt or base64_images:
                    break

        return prompt, base64_images

    def _save_video_and_get_public_url(
        self,
        request: Any,
        video_data: bytes,
        content_type: str,
        user: User,
    ) -> str:
        """Save video data and return public URL."""
        try:
            # Create UploadFile object
            video_file = UploadFile(
                file=io.BytesIO(video_data),
                filename=f"veo3_video_{int(time.time())}.mp4",
            )

            # Upload using the handler
            file_item = upload_file_handler(
                request=request,
                file=video_file,
                metadata={},
                process=False,
                user=user,
            )

            if not file_item:
                raise Exception("Upload failed - no file item returned")

            file_id = str(getattr(file_item, "id", ""))

            relative_path = request.app.url_path_for(
                "get_file_content_by_id", id=file_id
            )

            timestamp = int(time.time() * 1000)
            url_with_cache_bust = f"{relative_path}?t={timestamp}"

            return url_with_cache_bust

        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise

    async def enhance_prompt(
        self,
        prompt: str,
        user: User,
        request: Any,
        event_emitter: Callable[..., Any],
    ) -> str:
        """Enhance the prompt using vision model."""
        try:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "level": "info",
                        "description": "Enhancing prompt...",
                        "done": False,
                    },
                }
            )

            payload: Dict[str, Any] = {
                "model": self.valves.VISION_MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": self.valves.ENHANCER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this video generation prompt: {prompt}",
                    },
                ],
                "stream": False,
            }

            resp_data: Dict[str, Any] = await generate_chat_completion(
                request, payload, user
            )
            enhanced_prompt: str = str(resp_data["choices"][0]["message"]["content"])
            enhanced_prompt_message = f"<details>\n<summary>Enhanced Prompt</summary>\n{enhanced_prompt}\n\n---\n\n</details>"
            await event_emitter(
                {
                    "type": "message",
                    "data": {"content": enhanced_prompt_message},
                }
            )

            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "level": "info",
                        "description": "Prompt enhanced successfully.",
                        "done": False,
                    },
                }
            )

            return enhanced_prompt
        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}", exc_info=True)
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "level": "error",
                        "description": "Failed to enhance prompt.",
                        "done": True,
                    },
                }
            )
            return prompt

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Callable[..., Any],
        __request__: Any = None,
        __task__: Any = None,
        __event_call__: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Main function of the Pipe class.
        Generates videos using Veo 3.1 via Gemini API.
        """
        self.__event_emitter__ = __event_emitter__
        self.__request__ = __request__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        self.__event_call__ = __event_call__

        # Extract prompt and image from messages
        prompt, base64_images = self.setup_inputs(body.get("messages", []))
        prompt = (prompt or "").strip()

        if not prompt:
            await self.__event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "level": "error",
                        "description": "No prompt provided. Please provide a description for the video to generate.",
                    },
                }
            )
            return body

        # Enhance prompt if enabled
        if self.valves.ENHANCE_PROMPT:
            prompt = await self.enhance_prompt(
                prompt,
                self.__user__,
                self.__request__,
                self.__event_emitter__,
            )

        # Check API key
        if not self.valves.GOOGLE_API_KEY:
            await self.__event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "level": "error",
                        "description": "Google API key not configured. Please set GOOGLE_API_KEY in valves.",
                    },
                }
            )
            return body

        await self.__event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "in_progress",
                    "level": "info",
                    "description": "Generating video...",
                    "done": False,
                },
            }
        )

        try:
            client = genai.Client(api_key=self.valves.GOOGLE_API_KEY)

            # Generate video
            if base64_images:
                # Image-to-video generation
                # Note: The Gemini API currently supports one image for video generation.
                try:
                    first_image = base64_images[0]
                    image_bytes = base64.b64decode(first_image["base64"])
                    image_input = Image(
                        image_bytes=image_bytes, mime_type=first_image["mime_type"]
                    )
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    await self.__event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "complete",
                                "level": "error",
                                "description": f"Invalid image format provided: {e}",
                            },
                        }
                    )
                    return body

                operation = client.models.generate_videos(
                    model=self.valves.MODEL,
                    prompt=prompt,
                    image=image_input,
                )
            else:
                # Text-to-video generation
                operation = client.models.generate_videos(
                    model=self.valves.MODEL,
                    prompt=prompt,
                )

            # Poll for completion
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > self.valves.MAX_WAIT_TIME:
                    await self.__event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "complete",
                                "level": "error",
                                "description": f"Video generation timed out after {self.valves.MAX_WAIT_TIME} seconds.",
                            },
                        }
                    )
                    return body

                await asyncio.sleep(10)
                operation = client.operations.get(operation)

            # Calculate total elapsed time
            total_elapsed = int(time.time() - start_time)

            if not operation.response:
                raise Exception(
                    "Video generation operation completed without a response."
                )

            # Download video
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)

            # Save to temporary file to get bytes
            temp_filename = f"temp_veo3_{int(time.time())}.mp4"
            generated_video.video.save(temp_filename)

            # Read the file data
            with open(temp_filename, "rb") as f:
                video_data = f.read()

            # Clean up temp file
            os.remove(temp_filename)

            # Save and get public URL
            public_video_url = self._save_video_and_get_public_url(
                self.__request__,
                video_data,
                "video/mp4",
                self.__user__,
            )

            # Create HTML embed
            video_embed = f"""
<video controls style="max-width: 100%; height: auto;">
    <source src="{public_video_url}" type="video/mp4">
    Your browser does not support the video tag.
</video>
"""

            await self.__event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "level": "success",
                        "description": f"Video generated successfully in {total_elapsed} seconds!",
                        "done": True,
                    },
                }
            )

            # Emit the video embed message
            await self.__event_emitter__(
                {
                    "type": "embeds",
                    "data": {
                        "embeds": [video_embed],
                    },
                }
            )

        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            await self.__event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete",
                        "level": "error",
                        "description": f"Video generation failed: {str(e)}",
                    },
                }
            )

        return body
