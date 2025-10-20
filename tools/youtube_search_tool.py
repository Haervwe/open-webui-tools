"""
title: YouTube Search and Embed Tool
description: Search YouTube videos and display them in an embedded player
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.1
license: MIT
"""

import aiohttp
from typing import Any, Optional, Callable, Awaitable, Literal
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def emit_status(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    description: str,
    done: bool = False,
) -> None:
    """Helper to emit status events"""
    if event_emitter:
        await event_emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )


async def emit_embed(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    video_id: str,
) -> None:
    """Helper to emit embed events for displaying video player"""
    if event_emitter:
        # Create iframe HTML for YouTube embed
        iframe_html = f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'

        await event_emitter(
            {
                "type": "embeds",
                "data": {
                    "embeds": [iframe_html],
                },
            }
        )


class Tools:
    class Valves(BaseModel):
        YOUTUBE_API_KEY: str = Field(
            default="",
            description="YouTube Data API v3 key from https://console.cloud.google.com/apis/credentials",
        )
        MAX_RESULTS: int = Field(
            default=5, description="Maximum number of search results to return (1-10)"
        )
        SHOW_EMBEDDED_PLAYER: bool = Field(
            default=False,
            description="Show embedded YouTube player for the first result of a search instead of waiting for play video tool call",
        )
        REGION_CODE: str = Field(
            default="US",
            description="Region code for search results (e.g., US, GB, JP)",
        )
        SAFE_SEARCH: Literal["none", "moderate", "strict"] = Field(
            default="moderate", description="Safe search filter"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_youtube(
        self,
        query: str,
        max_results: Optional[int] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search YouTube for videos matching the query and display embedded player for first result.

        Args:
            query: Search query string
            max_results: Maximum number of results (default: uses Valves setting)

        Returns:
            Formatted search results with video links
        """
        # Validate API key
        if not self.valves.YOUTUBE_API_KEY:
            return "Error: YouTube Data API key is not set. Please get a free API key from https://console.cloud.google.com/apis/credentials and enable the YouTube Data API v3."

        # Validate and limit max_results
        max_results = max_results or self.valves.MAX_RESULTS
        max_results = min(max(max_results, 1), 10)

        await emit_status(__event_emitter__, f"Searching YouTube for: {query}")

        try:
            # YouTube Data API v3 search endpoint
            search_url = "https://www.googleapis.com/youtube/v3/search"

            search_params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": self.valves.YOUTUBE_API_KEY,
                "regionCode": self.valves.REGION_CODE,
                "safeSearch": self.valves.SAFE_SEARCH,
                "order": "relevance",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status == 403:
                        return "Error: Invalid API key or API quota exceeded. Please check your YouTube Data API key and quota at https://console.cloud.google.com/apis/api/youtube.googleapis.com"
                    elif response.status != 200:
                        return f"Error: YouTube API returned status {response.status}"

                    search_data = await response.json()

                    if "items" not in search_data or not search_data["items"]:
                        return f"No videos found for query: '{query}'"

                    # Build results
                    result = f"**YouTube Search Results for '{query}'**\n\n"

                    for i, item in enumerate(search_data["items"], 1):
                        video_id = item["id"]["videoId"]
                        snippet = item["snippet"]
                        title = snippet.get("title", "Unknown Title")
                        channel = snippet.get("channelTitle", "Unknown Channel")
                        description = snippet.get("description", "")[:150]

                        result += f"**{i}. {title}**\n"
                        result += f"   • Channel: {channel}\n"
                        result += (
                            f"   • URL: https://www.youtube.com/watch?v={video_id}\n"
                        )
                        if description:
                            result += f"   • Description: {description}...\n"
                        result += "\n"

                        # Embed first video if enabled
                        if i == 1 and self.valves.SHOW_EMBEDDED_PLAYER:
                            await emit_embed(__event_emitter__, video_id)

                    await emit_status(__event_emitter__, "Search completed", done=True)
                    return result

        except aiohttp.ClientError as e:
            logger.error(f"Network error during YouTube search: {str(e)}")
            return f"Error: Network error occurred - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during YouTube search: {str(e)}")
            return f"Error: An unexpected error occurred - {str(e)}"

    async def play_video(
        self,
        video_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Play a specific YouTube video by ID in an embedded player.

        Args:
            video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")

        Returns:
            Confirmation message with video link
        """
        await emit_status(__event_emitter__, f"Loading video: {video_id}")

        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Emit embed for the video
            await emit_embed(__event_emitter__, video_id)

            await emit_status(__event_emitter__, "Video loaded", done=True)

            return f"Playing video: {video_url}"

        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            return f"Error: Failed to load video - {str(e)}"
