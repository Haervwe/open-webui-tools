"""
title: Image Gen
author: Haervwe
Based on @justinrahb tool
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2
required_open_webui_version: 0.6.18
"""


import requests
from fastapi import Request
from pydantic import BaseModel, Field
from open_webui.routers.images import image_generations, GenerateImageForm
from open_webui.models.users import Users


def get_loaded_models(api_url: str = "http://localhost:11434") -> list[str]:
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
            model_name = model.get("name", model.get("model", ""))
            if model_name:
                payload = {"model": model_name, "keep_alive": 0}
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

    def __init__(self):
        self.valves = self.Valves()

    async def generate_image(
        self,
        prompt: str,
        model: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__=None,
    ) -> str:
        """
        Generate an image given a prompt

        :param prompt: prompt to use for image generation
        :param model: model to use "" if you want to use the default model
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
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Generating an image", "done": False},
            }
        )

        try:
            if model != "":
                __request__.app.state.config.IMAGE_GENERATION_MODEL = model
            images = await image_generations(
                request=__request__,
                form_data=GenerateImageForm(prompt=prompt, model=model),
                user=Users.get_user_by_id(__user__["id"]),
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Generated an image", "done": True},
                }
            )

            images_string = ""
            for image in images:
                images_string = (
                    images_string
                    + f"![Generated Image](http://haervwe.ai:3000{image['url']}) \n"
                )

            return f"Notify the user that the images has been successfully generated, copy this markdown attachments {images_string} ,as is to show them to the user, this {images_string} are the actual generated image urls , use them as is do not add or remove anything form the markdown attachments, they csould be copied AS IS, with no extra parts nor ommited parts "

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occured: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"
