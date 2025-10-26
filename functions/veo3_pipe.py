"""
title: Veo 3 Video Generation Pipe
authors:
    - Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
description: Generate videos using Google's Veo 3.1 model via Gemini API.
required_open_webui_version: 0.4.0
requirements: google-genai
version: 1.0
license: MIT

This pipe generates videos using Google's Veo 3.1 model through the Gemini API.
It supports text-to-video generation, image-to-video, and reference images.
"""

import asyncio
import io
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Union

from fastapi import UploadFile
from google import genai
from pydantic import BaseModel, Field
from open_webui.models.users import User, Users
from open_webui.routers.files import upload_file_handler
from open_webui.utils.chat import generate_chat_completion

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
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    return " ".join(text_parts)
        return None

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

            base_url = str(request.base_url).rstrip('/')
            relative_path = request.app.url_path_for("get_file_content_by_id", id=file_id)

            timestamp = int(time.time() * 1000)
            url_with_cache_bust = f"{base_url}{relative_path}?t={timestamp}"

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

            resp_data: Dict[str, Any] = await generate_chat_completion(request, payload, user)
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

        user_text = self._extract_last_user_text(body.get("messages", [])) or ""

        # Extract prompt from messages
        prompt = user_text.strip()
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
            operation = client.models.generate_videos(
                model=self.valves.MODEL,
                prompt=prompt,
            )

            # Poll for completion
            start_time = time.time()
            poll_count = 0
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

                poll_count += 1
                await asyncio.sleep(10)
                operation = client.operations.get(operation)

            # Calculate total elapsed time
            total_elapsed = int(time.time() - start_time)

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
                    },
                }
            )

            # Emit the video embed message
            await self.__event_emitter__(
                {
                    "type": "embeds",
                    "data": {"embeds": [video_embed]},
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