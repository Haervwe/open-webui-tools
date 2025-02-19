"""
title: searchthearxiv.com Tool
description: Tool to perform semantic search for relevant journals on arXiv via searchthearxiv.com
author: Haervwe, Tan Yong Sheng
author_urls:
  - https://github.com/Haervwe/
  - https://github.com/tan-yong-sheng/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.1
"""

import aiohttp
import asyncio
from typing import Any, Optional, Callable, Awaitable
from pydantic import BaseModel
import urllib.parse


class Tools:
    class UserValves(BaseModel):
        """No API keys required for arXiv search"""

        pass

    def __init__(self):
        self.base_url = "https://searchthearxiv.com/search"
        self.max_results = 5

    async def search_papers(
        self,
        topic: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search searchthearxiv.com for papers on a given topic and return formatted results.

        Args:
            topic: Topic to search for (e.g., "quantum computing", "transformer models")

        Returns:
            Formatted string containing paper details including titles, authors, dates,
            URLs and abstracts.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching arXiv...",
                        "done": False,
                    },
                }
            )

        try:
            # Construct search query
            search_query = topic
            encoded_query = urllib.parse.quote(search_query)
            params = {"query": encoded_query}

            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36",
                "x-requested-with": "XMLHttpRequest",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url, params=params, headers=headers, timeout=30
                ) as response:
                    response.raise_for_status()
                    # Use content_type=None to bypass MIME type checking.
                    root = await response.json(content_type=None)

            entries = root.get("papers", [])
            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "No papers found", "done": True},
                        }
                    )
                return f"No papers found on arXiv related to '{topic}'"

            results = ""
            # Loop over each paper entry.
            for i, entry in enumerate(entries, 1):
                # Extract paper details with fallbacks
                title = entry.get("title")
                title_text = title.strip() if title else "Unknown Title"

                authors_str = entry.get("authors", "Unknown Authors")

                summary = entry.get("abstract")
                summary_text = summary.strip() if summary else "No summary available"

                link = entry.get("id")
                link_text = (
                    f"https://arxiv.org/abs/{link}" if link else "No link available"
                )
                pdf_link = (
                    f"https://arxiv.org/pdf/{link}" if link else "No link available"
                )

                year = entry.get("year")
                month = entry.get("month")
                pub_date = f"{month}-{int(year)}" if year and month else "Unknown Date"

                # Format paper entry
                results += f"{i}. {title_text}\n"
                results += f"   Authors: {authors_str}\n"
                results += f"   Published: {pub_date}\n"
                results += f"   URL: {link_text}\n"
                results += f"   PDF URL: {pdf_link}\n"
                results += f"   Summary: {summary_text}\n\n"

                # Emit citation data as provided.
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [summary_text],
                                "metadata": [{"source": pdf_link}],
                                "source": {"name": title_text},
                            },
                        }
                    )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Search completed", "done": True},
                    }
                )

            return results

        except aiohttp.ClientError as e:
            error_msg = f"Error searching arXiv: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg
