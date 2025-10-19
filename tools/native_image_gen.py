"""
title: Image Gen
author: Haervwe
Based on @justinrahb tool
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.1
required_open_webui_version: 0.6.31
"""

import requests
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Any, Callable, Optional, Dict, List
from open_webui.routers.images import image_generations, GenerateImageForm
from open_webui.models.users import Users


def get_loaded_models(api_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """Get all currently loaded models in VRAM"""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/api/ps")
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as e:
        print(f"Error fetching loaded models: {e}")
        raise


def unload_all_models(api_url: str = "http://localhost:11434") -> dict[str, bool]:
    """Unload all currently loaded models from VRAM"""
    try:
        loaded_models = get_loaded_models(api_url)
        results = {}

        for model in loaded_models:
            if isinstance(model, dict):
                model_name = model.get("name", model.get("model", ""))
            else:
                model_name = str(model)

            if model_name:
                payload: Dict[str, Any] = {"model": model_name, "keep_alive": 0}
                response = requests.post(
                    f"{api_url.rstrip('/')}/api/generate", json=payload
                )
                results[model_name] = response.status_code == 200

        return results
    except requests.RequestException as e:
        print(f"Error unloading models: {e}")
        return {}


class Tools:

    class Valves(BaseModel):

        unload_ollama_models: bool = Field(
            default=False,
            description="Unload all Ollama models before calling ComfyUI.",
        )
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Ollama API URL.",
        )
        emit_embeds: bool = Field(
            default=True,
            description=(
                "When true, emit an 'EMBEDS' event containing the generated images. "
                "When false, skip emitting embeds and only return the concise URLs."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        __request__: Request | None = None,
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Generate an image given a prompt

        :param prompt: prompt to use for image generation
        :param model: model to use, leave empty to use the default model
        """
        if self.valves.unload_ollama_models:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Unloading Ollama models...",
                            "done": False,
                        },
                    }
                )
            unload_all_models(api_url=self.valves.ollama_url)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generating an image", "done": False},
                }
            )

        try:
            if model:
                __request__.app.state.config.IMAGE_GENERATION_MODEL = model
            images = await image_generations(
                request=__request__,
                form_data=GenerateImageForm(prompt=prompt, model=model),
                user=Users.get_user_by_id(__user__["id"]),
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Generated an image", "done": True},
                    }
                )
            bare_urls: list[str] = []
            markdown_attachments: list[str] = []
            for image in images:
                url = f"http://haervwe.ai:3000{image['url']}"
                bare_urls.append(url)
                img_html = f'<img src="{url}" style="max-width:100%; height:auto;" />'
                markdown_attachments.append(img_html)

            if self.valves.emit_embeds and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "embeds",
                        "data": {
                            "description": "Generated images",
                            "embeds": markdown_attachments,
                        },
                    }
                )

            urls_line = " ".join(bare_urls)

            if self.valves.emit_embeds and __event_emitter__:
                return (
                    f"Images were generated and displayed inline. Provide these download links (bare URLs): {urls_line}"
                )

            if self.valves.emit_embeds and not __event_emitter__:
                return (
                    f"Images were generated but could not be displayed inline (no event emitter). Provide these download links (bare URLs): {urls_line}"
                )
                
            return (
                f"Images generated. Provide the following download links (bare URLs): {urls_line}"
            )

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"An error occured: {e}", "done": True},
                    }
                )

            return f"Tell the user: {e}"
