"""
title: searchthearxiv.com Tool
description: Tool to perform semantic search for relevant journals on arXiv via searchthearxiv.com
author: Haervwe, Tan Yong Sheng
author_urls:
  - https://github.com/Haervwe/
  - https://github.com/tan-yong-sheng/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.3.0
"""

import aiohttp
import asyncio
from typing import Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
import urllib.parse
import re
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return ''.join(self.text)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def _cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

class Tools:
    class Valves(BaseModel):
        max_results: int = Field(
            default=5, description="Maximum number of papers to display"
        )
        enable_semantic_ranking: bool = Field(
            default=False, 
            description="Use Open WebUI's embedding routines to semantically rank results."
        )
        max_chars: int = Field(
            default=40000, 
            description="Maximum characters to return when retrieving a paper."
        )

    def __init__(self):
        self.base_url = "https://searchthearxiv.com/search"
        self.citation = False
        self.valves = self.Valves()

    async def search_papers(
        self,
        topic: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
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
            
            # Semantic ranking if enabled
            if self.valves.enable_semantic_ranking and __request__:
                try:
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "Semantically ranking abstracts...",
                                    "done": False,
                                },
                            }
                        )
                    embedding_func = __request__.app.state.EMBEDDING_FUNCTION
                    query_embedding = await embedding_func(topic)
                    if isinstance(query_embedding, dict) and "result" in query_embedding:
                        query_embedding = query_embedding["result"]
                    elif isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                         query_embedding = query_embedding[0]
                         
                    for entry in entries:
                        abstract = entry.get("abstract", "")
                        if abstract:
                            entry_embedding = await embedding_func(abstract)
                            # Handle different response shapes from EMBEDDING_FUNCTION
                            if isinstance(entry_embedding, dict) and "result" in entry_embedding:
                                entry_embedding = entry_embedding["result"]
                            elif isinstance(entry_embedding, list) and len(entry_embedding) > 0 and isinstance(entry_embedding[0], list):
                                entry_embedding = entry_embedding[0]
                                
                            entry["_score"] = _cosine_similarity(query_embedding, entry_embedding)
                        else:
                            entry["_score"] = -1.0
                            
                    entries = sorted(entries, key=lambda x: x.get("_score", 0), reverse=True)
                except Exception as e:
                    print(f"Failed to rank using embeddings: {e}")

            # Apply max_results limit
            entries = entries[:self.valves.max_results]

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
            
    async def read_paper(
        self,
        arxiv_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Retrieve the full text (HTML version) of an arXiv paper by its ID.
        
        Args:
            arxiv_id: The arXiv ID of the paper (e.g., "2401.00001")
            
        Returns:
            The full text of the paper if available in HTML format, otherwise an error message indicating where to find the PDF.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Retrieving full text for {arxiv_id}...",
                        "done": False,
                    },
                }
            )
            
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(html_url, timeout=30) as response:
                    if response.status != 200:
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"HTML format not available for {arxiv_id}, attempting to read PDF...", 
                                        "done": False
                                    },
                                }
                            )
                            
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                        async with session.get(pdf_url, timeout=60) as pdf_res:
                            if pdf_res.status != 200:
                                return f"Failed to retrieve HTML or PDF format for {arxiv_id}."
                            pdf_bytes = await pdf_res.read()
                            
                        text = ""
                        try:
                            import fitz # PyMuPDF
                            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                            for page in doc:
                                text += page.get_text() + "\n"
                        except ImportError:
                            try:
                                import pypdf
                                import io
                                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                                for page in reader.pages:
                                    text += (page.extract_text() or "") + "\n"
                            except ImportError:
                                try:
                                    import PyPDF2
                                    import io
                                    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                                    for page in reader.pages:
                                        text += (page.extract_text() or "") + "\n"
                                except ImportError:
                                    return f"Failed to retrieve HTML format for {arxiv_id}. The paper might be older or not formatted for HTML yet. PDF extraction libraries (PyMuPDF, pypdf, PyPDF2) are not installed in the environment. You can view the PDF at {pdf_url}"
                    else:
                        html_text = await response.text()
                
                        # Simple fallback to strip tags
                        text = strip_tags(html_text)
                        text = re.sub(r'\n\s*\n', '\n\n', text)
            
            max_chars = self.valves.max_chars
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n... (text truncated due to length)"
                
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Successfully retrieved paper {arxiv_id}",
                            "done": True,
                        },
                    }
                )
                
            return text
            
        except Exception as e:
            error_msg = f"Error retrieving paper {arxiv_id}: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return error_msg
