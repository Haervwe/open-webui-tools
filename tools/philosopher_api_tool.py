"""
title: Philosophy API Tool
description: Tool to interact with the Philosophy API using GraphQL to search philosophical content
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.0
"""

import aiohttp # type: ignore
from typing import Optional, Dict, Any, Callable, Awaitable
from pydantic import BaseModel, Field # type: ignore
import logging
import asyncio
from aiohttp import ClientTimeout, TCPConnector # type: ignore

# Setup logger
logger = logging.getLogger("philosophy_api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

GRAPHQL_URL = "https://philosophersapi.com/graphql"
TIMEOUT = ClientTimeout(total=10)  # 10 seconds total timeout
BACKUP_URLS = [
    "https://philosophyapi.com/graphql",
    "https://api.philosophy.io/graphql"
]

# Add REST API endpoints
BASE_URL = "https://philosophersapi.com/api"
REST_ENDPOINTS = {
    "search": f"{BASE_URL}/philosophers/search",
    "by_name": f"{BASE_URL}/philosophers/name",
    "ideas": f"{BASE_URL}/keyideas/search",
    "categories": f"{BASE_URL}/categories/search"
}

# Helper function moved outside the class
async def _try_graphql_request(session, url: str, query_data: dict) -> Optional[dict]:
    """Helper function to make GraphQL requests with error handling"""
    try:
        async with session.post(url, json=query_data) as response:
            if response.status == 200:
                data = await response.json()
                logger.debug(f"GraphQL Response Data: {data}")
                return data
            logger.debug(f"Request failed with status: {response.status}")
            return None
    except Exception as e:
        logger.debug(f"Request exception: {str(e)}")
        return None

# Helper function for REST API requests
async def _try_rest_request(session, url: str, params: dict) -> Optional[dict]:
    """Helper function to make REST requests with error handling"""
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                logger.debug(f"REST Response Data: {data}")
                return data
            logger.debug(f"REST request failed with status: {response.status}")
            return None
    except Exception as e:
        logger.debug(f"REST request exception: {str(e)}")
        return None

class Tools:
    class Valves(BaseModel):
        """Configuration valves for Philosophy API"""
        timeout: int = Field(default=10, description="Request timeout in seconds")
        max_retries: int = Field(default=2, description="Maximum number of retry attempts")

    def __init__(self):
        self.valves = self.Valves()
        logger.setLevel(logging.DEBUG)
        logger.debug("Philosophy API Tool initialized")

    async def search_philosophy(
        self,
        query: str,
        search_type: str = "all",
        limit: int = 5,
        __user__: dict = {},
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search the Philosophy API using GraphQL to find information about philosophers, ideas, 
        schools of thought, and more.

        Args:
            :param query: Search term to filter results (e.g., "Plato", "ethics", "existence")
            :param search_type: Type of search to perform ("all", "philosophers", "ideas", "categories", "quotes")
            :param limit: Maximum number of results to return per category (default 5)

        Returns:
            A markdown-formatted string containing:
            1. Title section with search query
            2. For philosophers:
                - Name and life dates
                - High-quality portrait image (if available)
                - Birth location and school of thought
                - Key ideas with references
                - Notable quotes with sources and dates
                - Major works with links
                - Links to encyclopedias and further reading
            3. For ideas:
                - Full idea text
                - Author attribution
                - Academic references
                - Category classifications
            4. For categories:
                - Category name and description
                - List of associated philosophers
                - Related concepts
            5. For quotes:
                - Full quote text
                - Author attribution
                - Source work and year
            All sections properly formatted with markdown headers, lists, and quotation blocks.
        """
        try:
            logger.debug(f"Searching philosophy with query: {query}, type: {search_type}")
            
            # Initial status message
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Searching for '{query}' in philosophical {search_type}...", "done": False},
            }) # type: ignore

            # Format search query
            search_query = query.replace(" ", "+")
            
            async with aiohttp.ClientSession() as session:
                # Try REST API first
                if search_type in ["philosophers", "all"]:
                    # Try exact name match first
                    result_data = await _try_rest_request(session, 
                        f"{REST_ENDPOINTS['by_name']}/{search_query}", {})
                    
                    if not result_data:
                        # Fall back to search
                        result_data = await _try_rest_request(session,
                            REST_ENDPOINTS['search'], {"keyword": search_query})
                        
                    # If still empty, try GraphQL fallback
                    if not result_data:
                        result_data = await _try_graphql_request(session, GRAPHQL_URL, {
                            'query': """
                            query SearchPhilosophers($search: String!) {
                                philosophers: searchPhilosophers(search: $search) {
                                    name
                                    life
                                    school
                                    interests
                                    birthYear
                                    deathYear
                                    birthDate
                                    deathDate
                                    speLink
                                    iepLink
                                    wikiTitle
                                    topicalDescription
                                    birthLocation {
                                        name
                                    }
                                    works {
                                        title
                                        link
                                    }
                                    keyIdeas {
                                        text
                                        reference
                                        categoryAbbrevs
                                    }
                                    quotes {
                                        quote
                                        work
                                        year
                                    }
                                }
                            }""",
                            'variables': {'search': query}
                        })

                # Normalize the response so that we always work with a dict with a "data" key.
                if isinstance(result_data, list):
                    # Assume REST response returns list of philosophers.
                    result_data = {"data": {"philosophers": result_data}}
                elif isinstance(result_data, dict) and not result_data.get("data"):
                    # In case GraphQL returns dict without "data", wrap it.
                    result_data = {"data": result_data}

                # Process the REST API response and treat it as a single philosopher result if it's a direct match
                if isinstance(result_data, dict) and not result_data.get('data'):
                    # Direct philosopher lookup response
                    philosopher_data = result_data
                    result_data = {
                        'data': {
                            'philosophers': [],  # Initialize empty list for search results
                            'philosopher': philosopher_data  # Store direct lookup result
                        }
                    }

                if not result_data or not result_data.get('data'):
                    logger.debug(f"No data in response: {result_data}")
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"No results found for '{query}'", "done": True},
                    }) # type: ignore
                    return "No results found. Try different search terms."

                data = result_data.get('data', {})
                logger.debug(f"Processing data: {data}")

                if not result_data or not result_data.get('data'):
                    logger.debug(f"No data in response: {result_data}")
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "No results found", "done": True},
                    }) # type: ignore
                    return "No results found. Try different search terms."

                result = """# Philosophy Search Results

> **Note on Image Links:** When displaying philosopher images, use the full URL in markdown format: 
> https://philosophersapi.com/Images/[image-path]
> Example: ![Inage](https://philosophersapi.com/Images/PlatoFull-iPhone@3x.jpg)
> MAke sure to use the provided images if possible and contextually correct.
> **Note on External Links:** Format all links as markdown: [Text](URL)
> Example: [Stanford Encyclopedia](https://plato.stanford.edu/entries/plato)

"""
                data = result_data['data']
                logger.debug(f"Processing data: {data}")

                has_results = False

                if search_type in ["philosophers", "all"]:
                    philosopher = data.get('philosopher')
                    philosophers = data.get('philosophers', [])
                    
                    if philosopher or philosophers:  # Direct name match or search results
                        has_results = True
                        phil = philosopher if philosopher else philosophers[0]  # Use first result if search
                        
                        result += f"## {phil.get('name', 'Unknown Philosopher')}\n\n"
                        
                        # Add full-size image if available
                        if phil.get('images', {}).get('fullImages', {}).get('full1260x1680'):
                            image_url = f"https://philosophersapi.com{phil['images']['fullImages']['full1260x1680']}"
                            result += f"![{phil.get('name')}]({image_url})\n\n"
                        
                        if phil.get('life'):
                            result += f"**Life:** {phil['life']}"
                            if phil.get('birthLocation'):
                                result += f" • Born in {phil['birthLocation'].get('name', 'Unknown location')}"
                            result += "\n"
                        
                        if phil.get('school'):
                            result += f"**School:** {phil['school']}\n"
                        if phil.get('interests'):
                            result += f"**Interests:** {phil['interests']}\n"
                        
                        # Clean up encyclopedia links
                        if phil.get('wikiTitle'):
                            result += f"\n**Wikipedia:** [Read more](https://en.wikipedia.org/wiki/{phil['wikiTitle']})\n"
                        if phil.get('speLink'):
                            result += f"**Stanford Encyclopedia:** [Read more]({phil['speLink']})\n"
                        if phil.get('iepLink'):
                            result += f"**Internet Encyclopedia:** [Read more]({phil['iepLink']})\n"

                        if phil.get('topicalDescription'):
                            result += f"\n{phil['topicalDescription']}\n"
                        
                        if phil.get('keyIdeas'):
                            result += "\n### Key Ideas\n"
                            for idea in phil['keyIdeas'][:limit]:
                                result += f"> {idea['text']}\n"
                                result += f"*Reference: {idea.get('reference', 'N/A')}*\n\n"

                        if phil.get('quotes'):
                            result += "### Notable Quotes\n"
                            for quote in phil['quotes'][:3]:
                                result += f"> {quote['quote']}\n"
                                if quote.get('work'):
                                    result += f"*From: {quote['work']}"
                                    if quote.get('year'):
                                        result += f" ({quote['year']})"
                                    result += "*\n"
                                result += "\n"

                        if phil.get('works'):
                            result += "### Major Works\n"
                            for work in phil['works']:
                                if work.get('link'):
                                    result += f"- [{work['title']}]({work['link']})\n"
                                else:
                                    result += f"- {work['title']}\n"
                            result += "\n"

                        result += "---\n\n"

                        # If we have more search results, list them briefly
                        if philosophers and len(philosophers) > 1:
                            result += "### Related Philosophers\n"
                            for related in philosophers[1:limit]:
                                result += f"- {related.get('name')}"
                                if related.get('life'):
                                    result += f" {related['life']}"
                                result += "\n"
                            result += "\n"

                if search_type in ["ideas", "all"]:
                    ideas = data.get('ideas', [])
                    if ideas:
                        has_results = True
                        result += "## Key Ideas\n\n"
                        for idea in ideas[:limit]:
                            result += f"> {idea.get('text', 'No text available')}\n\n"
                            # Safely access nested philosopher name
                            philosopher = idea.get('philosopher', {})
                            result += f"**By:** {philosopher.get('name', 'Unknown Philosopher')}\n"
                            result += f"**Reference:** {idea.get('reference', 'N/A')}\n"
                            # Add categories if available
                            if idea.get('categoryAbbrevs'):
                                result += f"**Categories:** {', '.join(idea['categoryAbbrevs'])}\n"
                            result += "\n---\n\n"

                if search_type in ["categories", "all"]:
                    categories = data.get('categories', [])
                    if categories:
                        has_results = True
                        result += "## Categories/Schools\n\n"
                        for cat in categories[:limit]:
                            result += f"### {cat['name']}\n"
                            result += f"{cat.get('description', '')}\n\n"
                            if cat.get('associatedPhilosophers'):
                                result += "**Notable Philosophers:**\n"
                                for phil in cat['associatedPhilosophers'][:5]:
                                    result += f"- {phil['name']}\n"
                            result += "\n---\n\n"

                if search_type in ["quotes", "all"]:
                    quotes = data.get('quotes', [])
                    if quotes:
                        has_results = True
                        result += "## Notable Quotes\n\n"
                        for quote in quotes[:limit]:
                            result += f"> {quote.get('quote', 'No quote available')}\n"
                            # Safely access nested philosopher name
                            philosopher = quote.get('philosopher', {})
                            result += f"**By:** {philosopher.get('name', 'Unknown Philosopher')}\n"
                            work = quote.get('work', 'Unknown work')
                            year = quote.get('year', '')
                            if work or year:
                                result += f"*From: {work}"
                                if year:
                                    result += f" ({year})"
                                result += "*\n"
                            result += "\n---\n\n"

                # Count results for status message
                result_counts = {
                    'philosophers': len(data.get('philosophers', [])) + (1 if data.get('philosopher') else 0),
                    'ideas': len(data.get('ideas', [])),
                    'categories': len(data.get('categories', [])),
                    'quotes': len(data.get('quotes', []))
                }
                
                total_results = sum(result_counts.values())
                
                if not total_results:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"No results found for '{query}'", "done": True},
                    }) # type: ignore
                    return "No results found. Try different search terms."

                # Final status message with result counts
                result_types = []
                if result_counts['philosophers']:
                    result_types.append(f"{result_counts['philosophers']} philosopher(s)")
                if result_counts['ideas']:
                    result_types.append(f"{result_counts['ideas']} key idea(s)")
                if result_counts['categories']:
                    result_types.append(f"{result_counts['categories']} category(ies)")
                if result_counts['quotes']:
                    result_types.append(f"{result_counts['quotes']} quote(s)")

                status_msg = f"Found {total_results} result(s) for '{query}'"
                if result_types:
                    status_msg += f": {', '.join(result_types)}"
                
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": status_msg, "done": True},
                }) # type: ignore
                
                return result

        except Exception as e:
            error_msg = f"Error searching for '{query}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Search failed for '{query}'", "done": True},
            }) # type: ignore
            return "Error: Search failed due to technical difficulties. Please try again later."
