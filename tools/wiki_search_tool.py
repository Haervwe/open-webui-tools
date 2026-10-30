"""
title: Wikipedia Search Tool
description: Tool to search Wikipedia and retrieve comprehensive article information including images and references
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
requirements:wikipedia-api
version: 0.2.0
required_open_webui_version: 0.8.11
"""

import asyncio
import html
import re
import wikipediaapi
import aiohttp
from typing import Optional, Any, Callable, Awaitable, Tuple, Union
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
import logging
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


async def _emit(event_emitter, event: dict) -> None:
    if event_emitter:
        await event_emitter(event)


async def _status(event_emitter, description: str, done: bool = False) -> None:
    await _emit(
        event_emitter,
        {"type": "status", "data": {"description": description, "done": done}},
    )


async def _init_wiki(wiki_instance, valves):
    """Initialize Wikipedia API client if not already initialized"""
    if not wiki_instance.wiki or getattr(wiki_instance, "_wiki_key", None) != (
        valves.user_agent,
        valves.language,
    ):
        wiki_instance.wiki = wikipediaapi.Wikipedia(
            user_agent=valves.user_agent,
            language=valves.language,
            extract_format=wikipediaapi.ExtractFormat.HTML
        )
        wiki_instance._wiki_key = (valves.user_agent, valves.language)


def _fetch_article(wiki, query: str, max_results: int) -> Optional[dict]:
    """All blocking wikipediaapi (requests-based) calls. Run via asyncio.to_thread."""
    page = wiki.page(query)
    if not page.exists():
        return None

    summary = BeautifulSoup(page.summary, "html.parser").get_text()

    sections = []
    for section in page.sections:
        if len(section.text.strip()) > 0:
            clean_text = BeautifulSoup(section.text, "html.parser").get_text()
            if len(clean_text) > 0:
                sections.append({"title": section.title, "text": clean_text})

    related = []
    for link_title, link_page in list(page.links.items())[: max(0, int(max_results))]:
        if link_title.startswith("File:"):
            continue
        if link_page.exists():
            related.append({"title": link_title, "url": link_page.fullurl})

    return {
        "title": page.title,
        "url": page.fullurl,
        "summary": summary,
        "sections": sections,
        "related": related,
    }

async def _get_page_images(base_api_url: str, title: str, user_agent: str = "") -> list[dict]:
    """Fetch images for a Wikipedia page (url = original, thumb = ~480px thumbnail)"""
    headers = {"User-Agent": user_agent} if user_agent else None
    async with aiohttp.ClientSession(
        headers=headers, timeout=aiohttp.ClientTimeout(total=20)
    ) as session:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "images|pageimages",
            "piprop": "original",
            "imlimit": "5"
        }
        
        async with session.get(base_api_url, params=params) as response:
            data = await response.json()
            
        page_id = list(data["query"]["pages"].keys())[0]
        page_data = data["query"]["pages"][page_id]
        
        images = []
        if "images" in page_data:
            image_titles = [img["title"] for img in page_data["images"]
                          if not img["title"].lower().endswith((".svg", ".gif"))]
            
            if image_titles:
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(image_titles),
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "iiurlwidth": "480"
                }
                
                async with session.get(base_api_url, params=params) as response:
                    img_data = await response.json()
                    
                for page in img_data["query"]["pages"].values():
                    if "imageinfo" in page:
                        info = page["imageinfo"][0]
                        images.append({
                            "title": page["title"],
                            "url": info.get("url"),
                            "thumb": info.get("thumburl") or info.get("url"),
                            "page": info.get("descriptionurl"),
                        })
        
        return images

class Tools:
    class Valves(BaseModel):
        """Configuration for Wikipedia tool"""
        user_agent: str = Field(
            default="open-webui-tools/1.0",
            description="User agent for Wikipedia API requests"
        )
        language: str = Field(
            default="en",
            description="Language for Wikipedia articles"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.wiki = None
        self.citation = False
        self.base_api_url = "https://en.wikipedia.org/w/api.php"

    def _api_url(self) -> str:
        lang = (self.valves.language or "en").strip().lower()
        if re.fullmatch(r"[a-z][a-z0-9-]{1,15}", lang):
            return f"https://{lang}.wikipedia.org/w/api.php"
        return self.base_api_url

    async def search_wiki(
        self,
        query: str,
        max_results: int = 3,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Search Wikipedia and return comprehensive article information including content, images, and references.

        Args:
            query: Search term to find Wikipedia articles
            max_results: Maximum number of related articles to include (default: 3)

        Returns:
            The article text with references and related links. Article images are shown to the user in a small gallery.
        """
        try:
            await _init_wiki(self, self.valves)

            await _status(__event_emitter__, f"Searching Wikipedia for: {query}")

            # wikipediaapi is synchronous (requests): keep it off the event loop
            article = await asyncio.to_thread(
                _fetch_article, self.wiki, query, max_results
            )

            if not article:
                await _status(__event_emitter__, "No Wikipedia article found", True)
                return "No Wikipedia article found for the given query."

            # Get images (non-fatal)
            try:
                images = await _get_page_images(
                    self._api_url(), article["title"], self.valves.user_agent
                )
            except Exception as e:
                logger.warning(f"Could not fetch Wikipedia images: {e}")
                images = []
            images = [img for img in images if _http_url(img.get("url"))]

            # Build response
            result = f"# {article['title']}\n\n"
            result += f"{article['summary']}\n\n"

            if article["sections"]:
                result += "## Article Contents\n\n"
                for section in article["sections"]:
                    result += f"### {section['title']}\n{section['text']}\n\n"

            if images:
                result += "## Images\n"
                result += (
                    "These article images are already shown to the user in a gallery; "
                    "do not embed them as markdown images. You cannot see them, so only "
                    "refer to them by title.\n"
                )
                for img in images:
                    result += f"- {img['title']}: {img.get('page') or img['url']}\n"
                result += "\n"

            # Add references and external links
            result += "## References & Links\n"
            result += f"- Wikipedia Article: [{article['title']}]({article['url']})\n"

            # Add related pages
            result += "\n## Related Articles\n"
            for rel in article["related"]:
                result += f"- [{rel['title']}]({rel['url']})\n"

            # Emit citation data
            full_text = article["summary"] + "\n" + "\n".join(
                [s["text"] for s in article["sections"]]
            )
            await _emit(__event_emitter__, {
                "type": "citation",
                "data": {
                    "document": [full_text],
                    "metadata": [{"source": article["url"]}],
                    "source": {"name": article["title"]}
                }
            })

            await _status(
                __event_emitter__, f"Found Wikipedia article: {article['title']}", True
            )

            if images:
                gallery = search_gallery_embed(
                    [
                        {
                            "image": img.get("thumb") or img["url"],
                            "link": img.get("page") or img["url"],
                            "alt": img["title"],
                            "caption": re.sub(r"^[^:]+:", "", img["title"]).rsplit(".", 1)[0],
                            "caption_url": img.get("page"),
                        }
                        for img in images
                    ],
                    title=f"Wikipedia: {article['title']}",
                    footer="Images from Wikimedia Commons / Wikipedia",
                    footer_url=article["url"],
                )
                return gallery_response(gallery, result)

            return result

        except Exception as e:
            error_msg = f"Error during Wikipedia search: {str(e)}"
            logger.error(error_msg)
            await _status(__event_emitter__, error_msg, True)
            return error_msg
