"""
title: Web Image Search using SearXNG
author: Tan Yong Sheng
author_url: https://github.com/tan-yong-sheng
version: 0.1.0
license: MIT
description: Tool to search and retrieve photos from SearXNG image search
"""

import requests
import json
from pydantic import BaseModel, Field
from typing import Callable, Any

class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )

class Tools:
    class Valves(BaseModel):
        SEARXNG_ENGINE_API_BASE_URL: str = Field(
            default="http://searxng:4000/search",
            description="The base URL for the SearXNG search engine API.",
        )
        RETURNED_IMAGES_NO: int = Field(
            default=3,
            description="The number of image results to return.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

    async def search_images(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Searches the web for images and returns the results in Markdown format.
        :param query: The search query for images.
        :return: A string containing the search results as Markdown image links.
        """
        emitter = EventEmitter(__event_emitter__)

        await emitter.emit(f"Initiating image search for: {query}")

        search_engine_url = self.valves.SEARXNG_ENGINE_API_BASE_URL

        params = {
            "q": query,
            "format": "json",
            "categories": "images",
            "number_of_results": self.valves.RETURNED_IMAGES_NO,
        }

        try:
            await emitter.emit("Sending request to SearXNG instance")
            resp = requests.get(
                search_engine_url, params=params, headers=self.headers, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            # Manually limit the number of results to handle cases where the API doesn't respect number_of_results
            limited_results = results[: self.valves.RETURNED_IMAGES_NO]
            await emitter.emit(f"Retrieved {len(results)} image results, processing {len(limited_results)}")

        except requests.exceptions.RequestException as e:
            error_message = f"Error during image search: {str(e)}"
            await emitter.emit(
                status="error",
                description=error_message,
                done=True,
            )
            return json.dumps({"error": error_message})

        markdown_output = ""
        if limited_results:
            await emitter.emit("Processing image results into Markdown format")
            for result in limited_results:
                img_url = result.get("img_src")
                alt_text = result.get("title", "Image")
                if img_url:
                    markdown_output += f"![{alt_text}]({img_url}) "
        
        if not markdown_output:
            markdown_output = "No images found for the given query."

        await emitter.emit(
            status="complete",
            description=f"Image search completed. Found and processed {len(limited_results)} images.",
            done=True,
        )

        return markdown_output.strip()
