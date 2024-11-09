"""
title: arXiv Search Tool
description: Tool to search arXiv.org for relevant papers on a topic
author: Haervwe
git: https://github.com/Haervwe/open-webui-tools  
version: 0.1.3
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable
from pydantic import BaseModel
import urllib.parse


class Tools:
    class UserValves(BaseModel):
        """No API keys required for arXiv search"""

        pass

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_results = 5

    async def search_papers(
        self,
        topic: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search arXiv.org for papers on a given topic and return formatted results.

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
                        "description": "Searching arXiv database...",
                        "done": False,
                    },
                }
            )

        try:
            # Construct search query
            search_query = f'all:"{topic}" OR abs:"{topic}" OR ti:"{topic}"'
            encoded_query = urllib.parse.quote(search_query)

            params = {
                "search_query": encoded_query,
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            # Make request to arXiv API
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")

            if not entries:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": "No papers found", "done": True},
                        }
                    )
                return f"No papers found on arXiv related to '{topic}'"

            # Format results
            results = (
                f"Found {len(entries)} recent papers on arXiv about '{topic}':\n\n"
            )

            for i, entry in enumerate(entries, 1):
                # Extract paper details with fallbacks
                title = entry.find("{http://www.w3.org/2005/Atom}title")
                title_text = (
                    title.text.strip() if title is not None else "Unknown Title"
                )

                authors = entry.findall("{http://www.w3.org/2005/Atom}author")
                author_names = []
                for author in authors:
                    name = author.find("{http://www.w3.org/2005/Atom}name")
                    if name is not None and name.text:
                        author_names.append(name.text)
                authors_str = (
                    ", ".join(author_names) if author_names else "Unknown Authors"
                )

                summary = entry.find("{http://www.w3.org/2005/Atom}summary")
                summary_text = (
                    summary.text.strip()
                    if summary is not None
                    else "No summary available"
                )

                link = entry.find("{http://www.w3.org/2005/Atom}id")
                link_text = link.text if link is not None else "No link available"

                published = entry.find("{http://www.w3.org/2005/Atom}published")
                if published is not None and published.text:
                    try:
                        pub_date = datetime.strptime(
                            published.text, "%Y-%m-%dT%H:%M:%SZ"
                        ).strftime("%Y-%m-%d")
                    except ValueError:
                        pub_date = "Unknown Date"
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
