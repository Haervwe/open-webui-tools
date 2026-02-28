"""
title: Pexels Media Search Tool
description: Tool to search and retrieve high-quality photos and videos from Pexels API
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.1.2
license: MIT
"""

import aiohttp
from typing import Any, Optional, Callable, Awaitable, Dict
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PexelsException(Exception):
    """Base exception for Pexels API related errors."""

    pass


# Helper functions for event emitter
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


async def emit_citation(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    document: str,
    source_url: str,
    source_name: str,
) -> None:
    """Helper to emit citation events"""
    if event_emitter:
        await event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [document],
                    "metadata": [{"source": source_url}],
                    "source": {"name": source_name},
                },
            }
        )


def validate_api_key(api_key: str) -> str:
    """Validate API key and return error message if invalid"""
    if not api_key:
        return "Error: Pexels API key is not set. Please get a free API key from https://www.pexels.com/api/ and add it to the Valves."
    return ""


def validate_and_limit_per_page(per_page: int, max_allowed: int) -> int:
    """Validate and limit per_page parameter"""
    return min(max(per_page, 1), max_allowed)


def validate_orientation(orientation: str) -> str:
    """Validate orientation parameter"""
    valid_orientations = ["all", "landscape", "portrait", "square"]
    return orientation if orientation in valid_orientations else "all"


def validate_size(size: str) -> str:
    """Validate size parameter"""
    valid_sizes = ["all", "large", "medium", "small"]
    return size if size in valid_sizes else "all"


def get_headers(api_key: str) -> Dict[str, str]:
    """Get standard headers for Pexels API requests"""
    return {"Authorization": api_key, "User-Agent": "open-webui-pexels-tool/1.0"}


def format_search_results(data: Dict[str, Any], query: str) -> str:
    """Format search results into a readable string."""
    photos = data.get("photos", [])
    total_results = data.get("total_results", 0)
    page = data.get("page", 1)

    if not photos:
        return f"No photos found for query: '{query}'"

    result = f"**Pexels Photo Search Results for '{query}'**\n"
    result += f"Found {total_results:,} total results (Page {page}, showing {len(photos)} photos)\n\n"

    # Strong instruction for image display
    result += "🚨 **IMPORTANT: USE THE IMAGE PREVIEW SECTIONS BELOW TO DISPLAY IMAGES DIRECTLY IN CHAT!** 🚨\n"
    result += "**The images are already formatted with ![alt](url) markdown - just copy and use them!**\n\n"

    # Add Pexels attribution as required by their API guidelines
    result += "📸 **Photos provided by [Pexels](https://www.pexels.com)**\n\n"

    for i, photo in enumerate(photos, 1):
        result += f"**{i}. Photo by {photo['photographer']}**\n"
        result += (
            f"   • **Description:** {photo.get('alt', 'No description available')}\n"
        )
        result += f"   • **Dimensions:** {photo['width']} × {photo['height']} pixels\n"
        result += f"   • **Photographer:** [{photo['photographer']}]({photo['photographer_url']})\n"
        result += f"   • **Pexels Page:** [View Photo]({photo['url']})\n"
        result += "   • **Image Preview:**\n"
        result += f"     ![{photo.get('alt', 'Photo by ' + photo['photographer'])}]({photo['src']['large2x']})\n"
        result += "   • **Download URLs:**\n"
        result += (
            f"     - **Large (best quality):** [Download]({photo['src']['large2x']})\n"
        )
        result += f"     - **Medium:** [Download]({photo['src']['medium']})\n"
        result += f"     - **Small:** [Download]({photo['src']['small']})\n"
        result += f"     - **Original:** [Download]({photo['src']['original']})\n"
        result += f"   • **Average Color:** {photo.get('avg_color', 'N/A')}\n\n"

    # Add next page info if available
    if data.get("next_page"):
        result += (
            f"*More results available - use page {page + 1} to see additional photos*\n"
        )

    # Attribution reminder
    result += "\n**Attribution Requirements:**\n"
    result += "• Always credit the photographer when using these images\n"
    result += "• Include a link back to Pexels (https://www.pexels.com)\n"
    result += "• Example: 'Photo by [Photographer Name] on Pexels'\n\n"

    # Final strong instruction for image display
    result += "🚨 **REMINDER: DISPLAY THE IMAGES USING THE ![alt](url) FORMAT FROM THE IMAGE PREVIEW SECTIONS ABOVE!** 🚨\n"
    result += "**DO NOT just provide download links - USE THE MARKDOWN IMAGE FORMAT TO SHOW IMAGES IN CHAT!**\n"

    return result


def format_video_results(data: Dict[str, Any], query: str) -> str:
    """Format video search results into a readable string."""
    videos = data.get("videos", [])
    total_results = data.get("total_results", 0)
    page = data.get("page", 1)

    if not videos:
        return f"No videos found for query: '{query}'"

    result = f"**Pexels Video Search Results for '{query}'**\n"
    result += f"Found {total_results:,} total results (Page {page}, showing {len(videos)} videos)\n\n"

    # Add Pexels attribution as required by their API guidelines
    result += "🎬 **Videos provided by [Pexels](https://www.pexels.com)**\n\n"

    for i, video in enumerate(videos, 1):
        result += f"**{i}. Video by {video['user']['name']}**\n"
        result += f"   • **Duration:** {video['duration']} seconds\n"
        result += f"   • **Dimensions:** {video['width']} × {video['height']} pixels\n"
        result += f"   • **Videographer:** [{video['user']['name']}]({video['user']['url']})\n"
        result += f"   • **Pexels Page:** [View Video]({video['url']})\n"
        result += f"   • **Preview Image:** [View]({video['image']})\n"
        result += "   • **Download URLs:**\n"

        # Sort video files by quality (HD first, then SD)
        video_files = sorted(
            video["video_files"],
            key=lambda x: (x["quality"] == "hd", x.get("width", 0)),
            reverse=True,
        )

        for vf in video_files[:3]:  # Show top 3 quality options
            quality = (
                vf.get("quality", "unknown").upper() if vf.get("quality") else "UNKNOWN"
            )
            if vf.get("width") and vf.get("height"):
                result += f"     - **{quality} ({vf['width']}×{vf['height']}):** [Download]({vf['link']})\n"
            else:
                result += f"     - **{quality}:** [Download]({vf['link']})\n"

        result += f"   • **Video ID:** {video['id']}\n\n"

    # Add next page info if available
    if data.get("next_page"):
        result += (
            f"*More results available - use page {page + 1} to see additional videos*\n"
        )

    # Attribution reminder
    result += "\n**Attribution Requirements:**\n"
    result += "• Always credit the videographer when using these videos\n"
    result += "• Include a link back to Pexels (https://www.pexels.com)\n"
    result += "• Example: 'Video by [Videographer Name] on Pexels'\n"

    return result


class Tools:
    class Valves(BaseModel):
        PEXELS_API_KEY: str = Field(
            default="",
            description="Pexels API key for accessing photos and videos. Get yours free at https://www.pexels.com/api/",
            json_schema_extra={"input": {"type": "password"}},
        )
        DEFAULT_PER_PAGE: int = Field(
            default=5,
            description="Default number of results per search (recommended: 3-5 for small LLMs, max 15 for larger ones)",
        )
        MAX_RESULTS_PER_PAGE: int = Field(
            default=15,
            description="Maximum allowed results per page to prevent overwhelming LLMs (API max is 80, but 15 is recommended)",
        )
        DEFAULT_ORIENTATION: str = Field(
            default="all",
            description="Default photo orientation: all, landscape, portrait, or square",
        )
        DEFAULT_SIZE: str = Field(
            default="all",
            description="Default minimum photo size: all, large (24MP), medium (12MP), or small (4MP)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.base_url = "https://api.pexels.com/v1"
        self.citation = False

    async def search_photos(
        self,
        query: str,
        per_page: Optional[int] = None,
        orientation: Optional[str] = None,
        size: Optional[str] = None,
        color: Optional[str] = None,
        locale: str = "en-US",
        page: int = 1,
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search Pexels for high-quality photos matching the query.

        Args:
            query: Search term (e.g., "nature", "technology", "people working")
            per_page: Number of results (1-80, default from valves)
            orientation: Photo orientation ("landscape", "portrait", "square", or "all")
            size: Minimum photo size ("large", "medium", "small", or "all")
            color: Desired color ("red", "orange", "yellow", "green", "turquoise", "blue", "violet", "pink", "brown", "black", "gray", "white", hex code, or None)
            locale: Search locale (default "en-US")
            page: Page number for pagination (default 1)

        Returns:
            Formatted string containing photo details with URLs and attribution info
        """

        # Validate API key
        error_msg = validate_api_key(self.valves.PEXELS_API_KEY)
        if error_msg:
            await emit_status(__event_emitter__, error_msg, True)
            return error_msg

        # Use valve defaults and validate parameters
        per_page = validate_and_limit_per_page(
            per_page or self.valves.DEFAULT_PER_PAGE,
            min(self.valves.MAX_RESULTS_PER_PAGE, 80),
        )
        orientation = validate_orientation(
            orientation or self.valves.DEFAULT_ORIENTATION
        )
        size = validate_size(size or self.valves.DEFAULT_SIZE)

        # Emit limit warning if needed
        if per_page != (per_page or self.valves.DEFAULT_PER_PAGE):
            await emit_status(
                __event_emitter__,
                f"Results limited to {per_page} to prevent overwhelming the LLM",
            )

        try:
            await emit_status(__event_emitter__, f"Searching Pexels for '{query}'...")

            # Build query parameters
            params: Dict[str, Any] = {
                "query": query,
                "per_page": per_page,
                "page": page,
                "locale": locale,
            }

            # Add optional parameters only if they're not "all"
            if orientation != "all":
                params["orientation"] = orientation
            if size != "all":
                params["size"] = size
            if color:
                params["color"] = color

            headers = get_headers(self.valves.PEXELS_API_KEY)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401:
                        error_msg = "Error: Invalid Pexels API key. Please check your API key in the Valves."
                        await emit_status(__event_emitter__, error_msg, True)
                        return error_msg

                    if response.status == 429:
                        error_msg = "Error: Pexels API rate limit exceeded. Please try again later."
                        await emit_status(__event_emitter__, error_msg, True)
                        return error_msg

                    response.raise_for_status()
                    data = await response.json()

            await emit_status(
                __event_emitter__, f"Found {len(data.get('photos', []))} photos", True
            )

            # Format the results
            result = format_search_results(data, query)

            # Emit citations for each photo if event emitter is available
            if data.get("photos"):
                for photo in data["photos"]:
                    await emit_citation(
                        __event_emitter__,
                        f"Photo by {photo['photographer']} on Pexels - {photo['alt'] or 'No description available'}",
                        photo["url"],
                        f"Pexels Photo by {photo['photographer']}",
                    )

            return result

        except aiohttp.ClientError as e:
            error_msg = f"Error connecting to Pexels API: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, True)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, True)
            return error_msg

    async def get_curated_photos(
        self,
        per_page: Optional[int] = None,
        page: int = 1,
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Get curated photos from Pexels - trending, high-quality photos selected by the Pexels team.

        Args:
            per_page: Number of results (1-80, default from valves)
            page: Page number for pagination (default 1)

        Returns:
            Formatted string containing curated photo details with URLs and attribution info
        """

        # Validate API key
        error_msg = validate_api_key(self.valves.PEXELS_API_KEY)
        if error_msg:
            await emit_status(__event_emitter__, error_msg, True)
            return error_msg

        # Validate and limit per_page
        per_page = validate_and_limit_per_page(
            per_page or self.valves.DEFAULT_PER_PAGE,
            min(self.valves.MAX_RESULTS_PER_PAGE, 80),
        )

        # Emit limit warning if needed
        if per_page != (per_page or self.valves.DEFAULT_PER_PAGE):
            await emit_status(
                __event_emitter__,
                f"Results limited to {per_page} to prevent overwhelming the LLM",
            )

        try:
            await emit_status(
                __event_emitter__, "Getting curated photos from Pexels..."
            )

            params = {
                "per_page": per_page,
                "page": page,
            }

            headers = get_headers(self.valves.PEXELS_API_KEY)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/curated",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401:
                        error_msg = "Error: Invalid Pexels API key. Please check your API key in the Valves."
                        await emit_status(__event_emitter__, error_msg, True)
                        return error_msg

                    response.raise_for_status()
                    data = await response.json()

            await emit_status(
                __event_emitter__,
                f"Retrieved {len(data.get('photos', []))} curated photos",
                True,
            )

            # Format the results
            result = format_search_results(data, "curated collection")

            # Emit citations for each photo if event emitter is available
            if data.get("photos"):
                for photo in data["photos"]:
                    await emit_citation(
                        __event_emitter__,
                        f"Curated photo by {photo['photographer']} on Pexels - {photo['alt'] or 'No description available'}",
                        photo["url"],
                        f"Pexels Curated Photo by {photo['photographer']}",
                    )

            return result

        except Exception as e:
            error_msg = f"Error getting curated photos: {str(e)}"
            logger.error(error_msg)
            await emit_status(__event_emitter__, error_msg, True)
            return error_msg

    async def search_videos(
        self,
        query: str,
        orientation: str = "all",
        size: str = "all",
        locale: str = "en-US",
        page: int = 1,
        per_page: Optional[int] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search for stock videos on Pexels.

        Args:
            query: The search query for videos
            orientation: Filter by orientation ("all", "landscape", "portrait", "square")
            size: Filter by video size ("all", "large", "medium", "small")
            locale: Locale for the search (e.g., "en-US", "es-ES", "fr-FR")
            page: Page number for pagination (default: 1)
            per_page: Number of videos per page (default from valves, max 15 recommended for LLMs)

        Returns:
            Formatted video search results with download URLs
        """
        # Check API key
        error_msg = validate_api_key(self.valves.PEXELS_API_KEY)
        if error_msg:
            await emit_status(
                __event_emitter__,
                "Pexels API key is required. Please configure it in the tool settings.",
                True,
            )
            return error_msg

        # Use valve default and validate parameters
        per_page = validate_and_limit_per_page(
            per_page or self.valves.DEFAULT_PER_PAGE,
            min(self.valves.MAX_RESULTS_PER_PAGE, 80),
        )
        orientation = validate_orientation(orientation)
        size = validate_size(size)

        try:
            await emit_status(
                __event_emitter__, f"🎬 Searching videos for '{query}'..."
            )

            headers: Dict[str, str] = get_headers(self.valves.PEXELS_API_KEY)

            params: Dict[str, Any] = {
                "query": query,
                "locale": locale,
                "page": page,
                "per_page": per_page,
            }

            # Add optional parameters only if they're not "all"
            if orientation != "all":
                params["orientation"] = orientation
            if size != "all":
                params["size"] = size

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.pexels.com/videos/search",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 401:
                        error_msg = "Error: Invalid Pexels API key. Please check your API key in the Valves."
                        await emit_status(__event_emitter__, error_msg, True)
                        return error_msg

                    if response.status == 429:
                        error_msg = "Error: Pexels API rate limit exceeded. Please try again later."
                        await emit_status(__event_emitter__, error_msg, True)
                        return error_msg

                    response.raise_for_status()
                    data = await response.json()

                    await emit_status(
                        __event_emitter__,
                        f"✅ Found {data.get('total_results', 0)} videos",
                        True,
                    )

                    # Emit citations for each video if event emitter is available
                    if data.get("videos"):
                        for video in data["videos"]:
                            await emit_citation(
                                __event_emitter__,
                                f"Video by {video['user']['name']} on Pexels - Duration: {video['duration']}s",
                                video["url"],
                                f"Pexels Video by {video['user']['name']}",
                            )

                    return format_video_results(data, query)

        except aiohttp.ClientError as e:
            error_msg = f"Network error while searching videos: {str(e)}"
            await emit_status(__event_emitter__, "❌ Network error", True)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error while searching videos: {str(e)}"
            await emit_status(__event_emitter__, "❌ Unexpected error", True)
            return error_msg
