"""
title: Image Gen
author: Haervwe
Based on @justinrahb tool
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.3.1
required_open_webui_version: 0.8.11
"""

import aiohttp
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Any, Callable, Optional, Dict, List, Tuple, Union
from open_webui.routers.images import image_generations, GenerateImageForm
from open_webui.models.users import Users
from fastapi.responses import HTMLResponse


async def get_loaded_models_async(
    api_url: str = "http://localhost:11434",
) -> List[Dict[str, Any]]:
    """Get all currently loaded models in VRAM"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url.rstrip('/')}/api/ps") as response:
                if response.status != 200:
                    return []
                data = await response.json()
                return data.get("models", [])
    except Exception as e:
        print(f"Error fetching loaded models: {e}")
        return []


async def unload_all_models_async(api_url: str = "http://localhost:11434") -> bool:
    """Unload all currently loaded models from VRAM"""
    try:
        loaded_models = await get_loaded_models_async(api_url)
        if not loaded_models:
            return True

        async with aiohttp.ClientSession() as session:
            for model in loaded_models:
                if isinstance(model, dict):
                    model_name = model.get("name", model.get("model", ""))
                else:
                    model_name = str(model)

                if model_name:
                    payload: Dict[str, Any] = {"model": model_name, "keep_alive": 0}
                    async with session.post(
                        f"{api_url.rstrip('/')}/api/generate", json=payload
                    ) as resp:
                        pass

        return True
    except Exception as e:
        print(f"Error unloading models: {e}")
        return False


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
    ) -> Union[str, Tuple[HTMLResponse, str]]:
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
            await unload_all_models_async(api_url=self.valves.ollama_url)
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
                img_html = f'<img src="{url}" style="max-width:100%; height:auto; display:block; border:none; margin:0 0 8px 0; padding:0; border-radius:12px;" />'
                markdown_attachments.append(img_html)

            if self.valves.emit_embeds:
                html_content = f"""<!DOCTYPE html>
<html style="margin:0; padding:0; overflow:hidden;">
<head><meta charset="UTF-8"></head>
<body style="margin:0; padding:0; overflow:hidden;">
<div style="margin:0; padding:0; border:none; line-height:0;">{"".join(markdown_attachments)}</div>
</body>
</html>"""
                urls_line = " ".join(bare_urls)
                return (
                    HTMLResponse(
                        content=html_content,
                        media_type="text/html",
                        headers={"Content-Disposition": "inline"},
                    ),
                    f"Images generated and displayed inline. Download links: {urls_line}",
                )

            urls_line = " ".join(bare_urls)
            return f"Images generated. Provide the following download links (bare URLs): {urls_line}"

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"An error occured: {e}", "done": True},
                    }
                )

            return f"Tell the user: {e}"
