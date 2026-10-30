"""
title: Pexels Media Search Tool
description: Tool to search and retrieve high-quality photos and videos from Pexels API
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.2.0
required_open_webui_version: 0.8.11
license: MIT
"""

import aiohttp
import html
import re
from typing import Any, Optional, Callable, Awaitable, Dict, List, Tuple, Union
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
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


# --------------------------------------------------------------------------
# Inline search-result gallery (sandboxed srcdoc embed, Open WebUI >= 0.8.11)
# Third-party images are loaded directly (they are search results, not
# generated media) with referrerpolicy="no-referrer"; every value is escaped.
# --------------------------------------------------------------------------


def _esc(value: Any) -> str:
    """HTML-escape any text for element content or a quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


def _http_url(value: Any) -> str:
    """Return value if it is an absolute http(s) URL, else ''."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if re.match(r"^https?://[^\s/\\]+", value, re.IGNORECASE) and not re.search(
        r"[\s\"'<>`]", value
    ):
        return value
    return ""


_GALLERY_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { background: transparent; overflow: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; color: #333; }
.hdr { font-size: 12px; font-weight: 600; color: #666; margin: 0 0 8px 2px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.card { border-radius: 12px; overflow: hidden; background: rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.08); }
.card a.media { display: block; position: relative; line-height: 0; background: rgba(0,0,0,0.06); }
.card img { width: 100%; height: 160px; object-fit: cover; display: block; border: none; }
.card .badge { position: absolute; right: 6px; bottom: 6px; padding: 2px 7px; border-radius: 99px;
  font-size: 10px; font-weight: 700; line-height: 1.6; color: #fff; background: rgba(0,0,0,0.6); }
.card .cap { padding: 7px 10px 9px; font-size: 11px; line-height: 1.4; color: #555; word-wrap: break-word; }
.card .cap a { color: inherit; font-weight: 600; text-decoration: none; }
.card .cap a:hover { text-decoration: underline; }
.card.broken img { display: none; }
.card.broken a.media { line-height: 1.5; padding: 24px 10px; font-size: 11px; color: #777; text-align: center; }
.foot { margin: 8px 0 0 2px; font-size: 11px; color: #777; }
.foot a { color: inherit; }
@media (prefers-color-scheme: dark) {
  body { color: #ddd; }
  .hdr { color: #aaa; }
  .card { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.1); }
  .card a.media { background: rgba(255,255,255,0.06); }
  .card .cap { color: #bbb; }
  .foot, .card.broken a.media { color: #999; }
}
"""

_GALLERY_JS = r"""
(function () {
  var pending = false;
  function post() {
    if (pending) return;
    pending = true;
    (window.requestAnimationFrame || setTimeout)(function () {
      pending = false;
      var r = document.getElementById('owui-root') || document.body;
      try { parent.postMessage({ type: 'iframe:height', height: Math.ceil(r.getBoundingClientRect().height) }, '*'); } catch (e) {}
    });
  }
  var imgs = document.querySelectorAll('.card img');
  for (var i = 0; i < imgs.length; i++) {
    imgs[i].addEventListener('load', post);
    imgs[i].addEventListener('error', function (e) {
      var card = e.target.closest('.card');
      if (card) {
        card.className += ' broken';
        var a = card.querySelector('a.media');
        if (a) a.appendChild(document.createTextNode('Image unavailable - open source'));
      }
      post();
    });
  }
  window.addEventListener('load', post);
  window.addEventListener('resize', post);
  if (window.ResizeObserver) { try { new ResizeObserver(post).observe(document.getElementById('owui-root')); } catch (e) {} }
  post();
})();
"""


def search_gallery_embed(
    items: list,
    *,
    title: str = "Image results",
    footer: str = "",
    footer_url: str = "",
) -> str:
    """Build the gallery HTML.

    :param items: dicts {"image", "link"?, "caption"?, "caption_url"?, "badge"?, "alt"?}
        (all URLs must be absolute http(s); anything else is dropped).
    """
    cards = []
    for it in items:
        img = _http_url(it.get("image"))
        if not img:
            continue
        link = _http_url(it.get("link")) or img
        alt = it.get("alt") or it.get("caption") or title
        badge = it.get("badge")
        caption = it.get("caption") or ""
        caption_url = _http_url(it.get("caption_url"))
        cap_html = ""
        if caption:
            cap_inner = (
                f'<a href="{_esc(caption_url)}" target="_blank" rel="noopener noreferrer">{_esc(caption)}</a>'
                if caption_url
                else _esc(caption)
            )
            cap_html = f'<div class="cap">{cap_inner}</div>'
        cards.append(
            '<div class="card">'
            f'<a class="media" href="{_esc(link)}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{_esc(img)}" alt="{_esc(alt)}" loading="lazy" decoding="async" referrerpolicy="no-referrer">'
            + (f'<span class="badge">{_esc(badge)}</span>' if badge else "")
            + "</a>"
            + cap_html
            + "</div>"
        )
    foot = ""
    if footer:
        footer_link = _http_url(footer_url)
        foot = (
            f'<div class="foot"><a href="{_esc(footer_link)}" target="_blank" rel="noopener noreferrer">{_esc(footer)}</a></div>'
            if footer_link
            else f'<div class="foot">{_esc(footer)}</div>'
        )
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        '<meta name="referrer" content="no-referrer">\n'
        f"<title>{_esc(title)}</title>\n<style>{_GALLERY_CSS}</style>\n</head>\n<body>\n"
        f'<div id="owui-root"><div class="hdr">{_esc(title)}</div>'
        f'<div class="grid">{"".join(cards)}</div>{foot}</div>\n'
        f"<script>{_GALLERY_JS}</script>\n</body>\n</html>"
    )


def gallery_response(html_content: str, context: Any) -> Tuple[HTMLResponse, Any]:
    """(inline HTMLResponse embed, LLM context) - see Open WebUI process_tool_result."""
    return (
        HTMLResponse(content=html_content, headers={"Content-Disposition": "inline"}),
        context,
    )


def format_search_results(data: Dict[str, Any], query: str) -> str:
    """Compact text for the LLM (the images themselves are shown in the gallery embed)."""
    photos = data.get("photos", []) or []
    total_results = data.get("total_results", 0) or 0
    page = data.get("page", 1) or 1

    if not photos:
        return f"No photos found for query: '{query}'"

    result = (
        f"Pexels photos for '{query}': showing {len(photos)} of {total_results:,} "
        f"(page {page}). They are already displayed to the user in an image gallery "
        "with photographer credits; do not repeat them as markdown images or links.\n\n"
    )
    for i, photo in enumerate(photos, 1):
        photographer = photo.get("photographer") or "Unknown"
        alt = (photo.get("alt") or "").strip() or "No description available"
        result += (
            f"{i}. Photo by {photographer} - {alt} "
            f"({photo.get('width', '?')}x{photo.get('height', '?')}) - {photo.get('url', '')}\n"
        )

    if data.get("next_page"):
        result += f"\nMore results are available on page {page + 1}.\n"
    result += (
        "\nAttribution: credit the photographer and Pexels when using these photos "
        "(e.g. 'Photo by <name> on Pexels').\n"
    )
    return result


def format_video_results(data: Dict[str, Any], query: str) -> str:
    """Compact text for the LLM (previews are shown in the gallery embed)."""
    videos = data.get("videos", []) or []
    total_results = data.get("total_results", 0) or 0
    page = data.get("page", 1) or 1

    if not videos:
        return f"No videos found for query: '{query}'"

    result = (
        f"Pexels videos for '{query}': showing {len(videos)} of {total_results:,} "
        f"(page {page}). Preview thumbnails linking to each video page are already "
        "displayed to the user; do not repeat them as markdown images or links.\n\n"
    )
    for i, video in enumerate(videos, 1):
        user = video.get("user") or {}
        result += (
            f"{i}. Video by {user.get('name') or 'Unknown'} - {video.get('duration', '?')}s "
            f"({video.get('width', '?')}x{video.get('height', '?')}) - ID {video.get('id', '?')} "
            f"- {video.get('url', '')}\n"
        )

    if data.get("next_page"):
        result += f"\nMore results are available on page {page + 1}.\n"
    result += (
        "\nAttribution: credit the videographer and Pexels when using these videos "
        "(e.g. 'Video by <name> on Pexels').\n"
    )
    return result


def photo_gallery(data: Dict[str, Any], title: str) -> str:
    items = []
    for photo in data.get("photos", []) or []:
        src = photo.get("src") or {}
        photographer = photo.get("photographer") or "Unknown"
        items.append(
            {
                "image": src.get("large") or src.get("medium") or src.get("large2x"),
                "link": photo.get("url"),
                "alt": photo.get("alt") or f"Photo by {photographer}",
                "caption": f"Photo by {photographer}",
                "caption_url": photo.get("photographer_url"),
            }
        )
    return search_gallery_embed(
        items, title=title, footer="Photos provided by Pexels", footer_url="https://www.pexels.com"
    )


def video_gallery(data: Dict[str, Any], title: str) -> str:
    items = []
    for video in data.get("videos", []) or []:
        user = video.get("user") or {}
        name = user.get("name") or "Unknown"
        duration = video.get("duration")
        items.append(
            {
                "image": video.get("image"),
                "link": video.get("url"),
                "alt": f"Video by {name}",
                "badge": f"{duration}s" if duration is not None else None,
                "caption": f"Video by {name}",
                "caption_url": user.get("url"),
            }
        )
    return search_gallery_embed(
        items, title=title, footer="Videos provided by Pexels", footer_url="https://www.pexels.com"
    )


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
    ) -> Union[str, Tuple[HTMLResponse, str]]:
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
            Photos shown to the user in an inline gallery, plus a short list (photographer, description, Pexels page) for you
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
                        f"Photo by {photo['photographer']} on Pexels - {photo.get('alt') or 'No description available'}",
                        photo["url"],
                        f"Pexels Photo by {photo['photographer']}",
                    )

                return gallery_response(
                    photo_gallery(data, f"Pexels photos: {query}"), result
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
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Get curated photos from Pexels - trending, high-quality photos selected by the Pexels team.

        Args:
            per_page: Number of results (1-80, default from valves)
            page: Page number for pagination (default 1)

        Returns:
            Curated photos shown to the user in an inline gallery, plus a short list for you
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
                        f"Curated photo by {photo['photographer']} on Pexels - {photo.get('alt') or 'No description available'}",
                        photo["url"],
                        f"Pexels Curated Photo by {photo['photographer']}",
                    )

                return gallery_response(
                    photo_gallery(data, "Pexels curated photos"), result
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
    ) -> Union[str, Tuple[HTMLResponse, str]]:
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
            Video previews shown to the user in an inline gallery, plus a short list (videographer, duration, ID, Pexels page) for you
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

                    result = format_video_results(data, query)
                    if data.get("videos"):
                        return gallery_response(
                            video_gallery(data, f"Pexels videos: {query}"), result
                        )
                    return result

        except aiohttp.ClientError as e:
            error_msg = f"Network error while searching videos: {str(e)}"
            await emit_status(__event_emitter__, "❌ Network error", True)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error while searching videos: {str(e)}"
            await emit_status(__event_emitter__, "❌ Unexpected error", True)
            return error_msg
