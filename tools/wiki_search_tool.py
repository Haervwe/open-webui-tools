"""
title: Wikipedia Search Tool
description: Tool to search Wikipedia and retrieve comprehensive article information including images and references
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
requirements:wikipedia-api
version: 0.1.0
"""

import wikipediaapi
import aiohttp
import json
from typing import Optional, Dict, Any, Callable, Awaitable
from pydantic import BaseModel, Field
import logging
import re
from bs4 import BeautifulSoup
from functools import partial

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def _init_wiki(wiki_instance, valves):
    """Initialize Wikipedia API client if not already initialized"""
    if not wiki_instance.wiki:
        wiki_instance.wiki = wikipediaapi.Wikipedia(
            user_agent=valves.user_agent,
            language=valves.language,
            extract_format=wikipediaapi.ExtractFormat.HTML
        )

async def _get_page_images(base_api_url: str, title: str) -> list[dict]:
    """Fetch images for a Wikipedia page"""
    async with aiohttp.ClientSession() as session:
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
                    "iiprop": "url"
                }
                
                async with session.get(base_api_url, params=params) as response:
                    img_data = await response.json()
                    
                for page in img_data["query"]["pages"].values():
                    if "imageinfo" in page:
                        images.append({
                            "title": page["title"],
                            "url": page["imageinfo"][0]["url"]
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
        self.base_api_url = "https://en.wikipedia.org/w/api.php"
        self.image_prompt = """IMPORTANT INSTRUCTION FOR USING IMAGES:
1. The text after this contains Wikipedia article images in markdown format.
2. Images are formatted like this example: ![Socrates](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Socrate_du_Louvre.jpg/1024px-Socrate_du_Louvre.jpg)
3. You MUST use these images to:
   - Analyze visual details in the images
   - Reference specific images when discussing related content
   - Include visual descriptions in your responses
   - Help users better understand the topic through visual context
4. When you see an image, treat it as if you can fully see and analyze it
5. Images may show historical figures, objects, places, diagrams, or other relevant visuals
6. Make your responses more engaging by referring to what is shown in the images

Example of how to reference images:
"As we can see in the image, the marble bust shows Socrates with his characteristic snub nose and beard..."
"""

    async def search_wiki(
        self,
        query: str,
        max_results: int = 3,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> str:
        """
        Search Wikipedia and return comprehensive article information including content, images, and references.
        
        Args:
            query: Search term to find Wikipedia articles
            max_results: Maximum number of related articles to include (default: 3)
        
        Returns:
            Formatted string containing article information with images and links in markdown format
        """
        try:
            await _init_wiki(self, self.valves)
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Searching Wikipedia for: {query}", "done": False}
            })

            # Get main page
            page = self.wiki.page(query)
            
            if not page.exists():
                await __event_emitter__({
                    "type": "status", 
                    "data": {"description": "No Wikipedia article found", "done": True}
                })
                return "No Wikipedia article found for the given query."

            # Get images
            images = await _get_page_images(self.base_api_url, page.title)
            
            # Build response
            result = f"# {page.title}\n\n"

            # Add main image if available
            if images and len(images) > 0:
                result += f"![{page.title}]({images[0]['url']})\n\n"

            # Add full summary
            summary = BeautifulSoup(page.summary, "html.parser").get_text()
            result += f"{summary}\n\n"

            # Add complete sections without truncation
            sections = []
            for section in page.sections:
                if len(section.text.strip()) > 0:
                    clean_text = BeautifulSoup(section.text, "html.parser").get_text()
                    if len(clean_text) > 0:
                        sections.append({
                            "title": section.title,
                            "text": clean_text
                        })

            if sections:
                result += "## Article Contents\n\n"
                for section in sections:
                    result += f"### {section['title']}\n{section['text']}\n\n"

            # Add additional images in a gallery
            if len(images) > 1:
                result += "## Gallery\n\n"
                for img in images[1:]:
                    result += f"![{img['title']}]({img['url']})\n\n"

            # Add references and external links
            result += f"## References & Links\n"
            result += f"- Wikipedia Article: [{page.title}]({page.fullurl})\n"
            
            # Add related pages
            result += "\n## Related Articles\n"
            for link_title, link_page in list(page.links.items())[:max_results]:
                if link_page.exists() and not link_title.startswith("File:"):
                    result += f"- [{link_title}]({link_page.fullurl})\n"

            # Emit citation data
            if __event_emitter__:
                full_text = summary + "\n" + "\n".join([s["text"] for s in sections])
                await __event_emitter__({
                    "type": "citation",
                    "data": {
                        "document": [full_text],
                        "metadata": [{"source": page.fullurl}],
                        "source": {"name": page.title}
                    }
                })

            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Found Wikipedia article: {page.title}",
                    "done": True
                }
            })

            return result

        except Exception as e:
            error_msg = f"Error during Wikipedia search: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            return error_msg
