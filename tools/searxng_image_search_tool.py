"""
title: Web Image Search using SearXNG
author: Tan Yong Sheng
author_url: https://github.com/tan-yong-sheng
version: 0.3.0
required_open_webui_version: 0.8.11
license: MIT
description: Tool to search and retrieve photos from SearXNG image search
"""

import aiohttp
import html
import json
import re
from pydantic import BaseModel, Field
from typing import Callable, Any, Tuple, Union
from fastapi.responses import HTMLResponse


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
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Searches the web for images and shows them to the user in an inline gallery.
        :param query: The search query for images.
        :return: The images are displayed to the user; you receive a short list of titles and source pages.
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
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    search_engine_url,
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            results = data.get("results", [])
            # Manually limit the number of results to handle cases where the API doesn't respect number_of_results
            limited_results = results[: self.valves.RETURNED_IMAGES_NO]
            await emitter.emit(
                f"Retrieved {len(results)} image results, processing {len(limited_results)}"
            )

        except aiohttp.ClientResponseError as e:
            # Do not echo the (possibly internal) SearXNG URL to the model.
            error_message = f"Error during image search: SearXNG returned HTTP {e.status}"
            await emitter.emit(
                status="error",
                description=error_message,
                done=True,
            )
            return json.dumps({"error": error_message})
        except Exception as e:
            error_message = f"Error during image search: {type(e).__name__}"
            await emitter.emit(
                status="error",
                description=error_message,
                done=True,
            )
            return json.dumps({"error": error_message})

        items = []
        lines = []
        limit = max(1, int(self.valves.RETURNED_IMAGES_NO or 1))
        # Skip results without a usable http(s) image (data:, relative, ...) and
        # fill up to the configured count from the remaining results.
        for result in results:
            if len(items) >= limit:
                break
            img_url = _http_url(result.get("img_src"))
            if not img_url:
                continue
            title = (result.get("title") or "").strip() or "Image"
            source = _http_url(result.get("url"))
            thumb = _http_url(result.get("thumbnail_src"))
            items.append(
                {
                    "image": img_url if not thumb else thumb,
                    "link": source or img_url,
                    "alt": title,
                    "caption": title,
                    "caption_url": source,
                }
            )
            lines.append(f"{len(items)}. {title}" + (f" - {source}" if source else ""))

        if not items:
            await emitter.emit(
                status="complete",
                description="Image search completed. No images found.",
                done=True,
            )
            return "No images found for the given query."

        await emitter.emit(
            status="complete",
            description=f"Image search completed. Found and processed {len(items)} images.",
            done=True,
        )

        context = (
            f"Found {len(items)} images for '{query}'. They are already displayed to the "
            "user in an image gallery; do not repeat them as markdown images or links.\n"
            + "\n".join(lines)
        )
        return gallery_response(
            search_gallery_embed(items, title=f"Image results: {query}"), context
        )
