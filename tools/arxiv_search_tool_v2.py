"""
title: searchthearxiv.com Tool
description: Tool to search searchthearxiv.org for relevant papers on a topic
author: Tan Yong Sheng
website: https://tanyongsheng.com
"""

# Adapted from the arXiv Search Tool by Tan Yong Sheng 
# Source: https://github.com/Haervwe/open-webui-tools/

import requests
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
        Search searchteharxiv.org for papers on a given topic and return formatted results.

        Args:
            topic: Topic to search for (e.g., "quantum computing", "transformer models")

        Returns:
            Formatted string containing paper details including titles, authors, dates,
            URLs and abstracts
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching searchthearxiv.com database...",
                        "done": False,
                    },
                }
            )

        try:
            # Construct search query
            search_query = topic
            encoded_query = urllib.parse.quote(search_query)

            params = {
                "query": encoded_query,
            }
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "x-requested-with": "XMLHttpRequest",
            }

            # Make request to arXiv API
            response = requests.get(self.base_url, params=params, 
                                    headers=headers, timeout=30)
            response.raise_for_status()

            # Parse JSON response
            root = response.json()
            entries = root.get("papers", []) 
            
            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "No papers found", "done": True},
                        }
                    )
                return f"No papers found on searchthearxiv.com related to '{topic}'"

            # Format results
            results = (
                f"Found {len(entries)} recent papers on searchthearxiv.com about '{topic}':\n\n"
            )

            for i, entry in enumerate(entries, 1):
                # Extract paper details with fallbacks
                title = entry.get("title", None)
                title_text = (
                    title.strip() if title is not None else "Unknown Title"
                )

                authors_str = entry.get("authors", "Unknown Authors")

                summary = entry.get("abstract", None)
                summary_text = (
                    summary.strip()
                    if summary is not None
                    else "No summary available"
                )

                link = entry.get("id", None)
                link_text = f"https://arxiv.org/abs/{link}" if link is not None else "No link available"


                # Extract publication date
                year = entry.get("year", None)
                
                month = entry.get("month", None)
                if year is not None and month is not None:
                    pub_date = f"{month}-{str(int(year))}"
                else:
                    pub_date = "Unknown Date"

                # Format paper entry
                results += f"{i}. {title_text}\n"
                results += f"   Authors: {authors_str}\n"
                results += f"   Published: {pub_date}\n"
                results += f"   URL: {link_text}\n"
                results += f"   Summary: {summary_text}\n\n"

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Search completed", "done": True},
                    }
                )

            return results

        except requests.RequestException as e:
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
