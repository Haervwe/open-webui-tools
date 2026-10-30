"""
title: YouTube Search and Embed Tool
description: Search YouTube videos and display them in an embedded player
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.1.3
required_open_webui_version: 0.8.11
license: MIT
"""

import aiohttp
import html
import re
from typing import Any, Optional, Callable, Awaitable, Literal, Union, Tuple
from pydantic import BaseModel, Field
import logging
from fastapi.responses import HTMLResponse


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


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize_video_id(value: Any) -> Optional[str]:
    """Return a valid 11-char YouTube video ID (also accepts watch/short URLs), else None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if _VIDEO_ID_RE.match(value):
        return value
    match = re.search(
        r"(?:v=|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])",
        value,
    )
    return match.group(1) if match else None


async def generate_video_embed(
    video_id: str,
) -> HTMLResponse:
    """Helper to generate HTMLResponse for displaying video player.

    ``video_id`` must already be validated with ``normalize_video_id``.
    """
    src = html.escape(f"https://www.youtube.com/embed/{video_id}", quote=True)

    iframe_html = f"""
<style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden;}}</style>
<div id="yt-root" style="width:100%;max-width:1200px;margin:0 auto;">
  <div style="position:relative;width:100%;padding-top:56.25%;height:0;overflow:hidden;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.2);">
    <iframe src="{src}" referrerpolicy="strict-origin-when-cross-origin"
            style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
            allowfullscreen loading="lazy"></iframe>
  </div>
</div>
<script>
(function () {{
  function post() {{
    var r = document.getElementById('yt-root');
    if (!r) return;
    try {{ parent.postMessage({{ type: 'iframe:height', height: Math.ceil(r.getBoundingClientRect().height) }}, '*'); }} catch (e) {{}}
  }}
  window.addEventListener('load', post);
  window.addEventListener('resize', post);
  if (window.ResizeObserver) {{ try {{ new ResizeObserver(post).observe(document.getElementById('yt-root')); }} catch (e) {{}} }}
  post();
}})();
</script>
""".strip()

    return HTMLResponse(
        content=iframe_html,
        media_type="text/html",
        headers={"content-disposition": "inline"},
    )


class Tools:
    class Valves(BaseModel):
        YOUTUBE_API_KEY: str = Field(
            default="",
            description="YouTube Data API v3 key from https://console.cloud.google.com/apis/credentials",
            json_schema_extra={"input": {"type": "password"}},
        )
        MAX_RESULTS: int = Field(
            default=5, description="Maximum number of search results to return (1-10)"
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
    ) -> Union[str, Tuple[HTMLResponse, str]]:
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

            search_params:dict[str, str | int] = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "regionCode": self.valves.REGION_CODE,
                "safeSearch": self.valves.SAFE_SEARCH,
                "order": "relevance",
            }

            # Send the key as a header (not a query param) so it never appears in
            # request URLs, which aiohttp includes in exception messages.
            headers = {"X-Goog-Api-Key": self.valves.YOUTUBE_API_KEY}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    search_url, params=search_params, headers=headers
                ) as response:
                    if response.status == 403:
                        return "Error: Invalid API key or API quota exceeded. Please check your YouTube Data API key and quota at https://console.cloud.google.com/apis/api/youtube.googleapis.com"
                    elif response.status != 200:
                        return f"Error: YouTube API returned status {response.status}"

                    search_data = await response.json()

                    if "items" not in search_data or not search_data["items"]:
                        return f"No videos found for query: '{query}'"

                    # Build the full result list (all results go to the LLM)
                    result = f"YouTube search results for '{query}':\n\n"
                    first_video_id = None
                    first_title = None
                    count = 0

                    for item in search_data["items"]:
                        video_id = normalize_video_id(
                            (item.get("id") or {}).get("videoId")
                        )
                        if not video_id:
                            continue
                        count += 1
                        snippet = item.get("snippet") or {}
                        title = html.unescape(snippet.get("title") or "Unknown Title")
                        channel = html.unescape(
                            snippet.get("channelTitle") or "Unknown Channel"
                        )
                        description = html.unescape(snippet.get("description") or "")[
                            :150
                        ]

                        result += f"{count}. {title}\n"
                        result += f"   - Channel: {channel}\n"
                        result += f"   - Video ID: {video_id}\n"
                        result += f"   - URL: https://www.youtube.com/watch?v={video_id}\n"
                        if description:
                            result += f"   - Description: {description}...\n"
                        result += "\n"

                        if first_video_id is None:
                            first_video_id = video_id
                            first_title = title

                    if not first_video_id:
                        await emit_status(__event_emitter__, "No playable videos found", done=True)
                        return f"No videos found for query: '{query}'"

                    result += (
                        f"The first result ('{first_title}') is already playing in an embedded "
                        "player shown to the user. To play another result, call play_video "
                        "with its Video ID. Do not embed the videos as markdown."
                    )

                    await emit_status(__event_emitter__, "Search completed", done=True)
                    return (await generate_video_embed(first_video_id), result)

        except aiohttp.ClientError as e:
            # Never surface str(e): aiohttp errors embed the request URL.
            logger.error(f"Network error during YouTube search ({type(e).__name__})")
            return f"Error: Network error occurred ({type(e).__name__})"
        except Exception as e:
            logger.error(f"Unexpected error during YouTube search ({type(e).__name__})")
            return f"Error: An unexpected error occurred ({type(e).__name__})"

    async def play_video(
        self,
        video_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Play a specific YouTube video by ID in an embedded player.
        This tool requires a valid YouTube video ID. and DOES NOT use the YouTube Data API for searching.
        First use the Search YouTube tool to find video IDs.
        Args:
            video_id: YouTube video ID (e.g., "dQw4w9WgXcQ")

        Returns:
            Confirmation message with video link
        """
        clean_id = normalize_video_id(video_id)
        if not clean_id:
            await emit_status(__event_emitter__, "Invalid YouTube video ID", done=True)
            return (
                "Error: Invalid YouTube video ID. Use the 11-character Video ID returned by "
                "search_youtube (e.g. dQw4w9WgXcQ)."
            )
        video_id = clean_id

        await emit_status(__event_emitter__, f"Loading video: {video_id}")

        try:

            await emit_status(__event_emitter__, "Video loaded", done=True)

            return (
                await generate_video_embed(video_id),
                f"Video https://www.youtube.com/watch?v={video_id} is now playing in an "
                "embedded player shown to the user. Do not repeat it as a link or markdown.",
            )

        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            return f"Error: Failed to load video - {str(e)}"
