"""
title: Xquik X Data Tool
description: Search and look up X posts, users, timelines, and trends using the Xquik API
author: Xquik
author_url: https://github.com/Xquik-dev/x-twitter-scraper
funding_url: https://xquik.com
requirements:aiohttp
version: 0.1.0
license: MIT
"""

import json
from typing import Any, Awaitable, Callable, Dict, Optional

import aiohttp
from pydantic import BaseModel, Field


async def emit_status(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    description: str,
    done: bool = False,
) -> None:
    """Emit Open WebUI status events when an event emitter is available."""
    if event_emitter:
        await event_emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )


class Tools:
    class Valves(BaseModel):
        XQUIK_API_KEY: str = Field(
            default="",
            description="Xquik API key for authenticated read requests",
            json_schema_extra={"input": {"type": "password"}},
        )
        BASE_URL: str = Field(
            default="https://xquik.com/api/v1",
            description="Xquik API base URL",
        )
        DEFAULT_LIMIT: int = Field(
            default=10,
            description="Default result limit for list endpoints",
        )
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=30,
            description="HTTP request timeout in seconds",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.citation = False

    def _headers(self) -> Dict[str, str]:
        headers = {
            "accept": "application/json",
            "xquik-api-contract": "2026-04-29",
        }
        if self.valves.XQUIK_API_KEY:
            headers["x-api-key"] = self.valves.XQUIK_API_KEY
        return headers

    async def _get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        base_url = self.valves.BASE_URL.rstrip("/")
        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT_SECONDS)

        clean_params = {
            key: value
            for key, value in params.items()
            if value is not None and value != ""
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{base_url}{path}",
                headers=self._headers(),
                params=clean_params,
            ) as response:
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    data = await response.json()
                else:
                    data = {"message": await response.text()}

                if response.status >= 400:
                    return {
                        "ok": False,
                        "status": response.status,
                        "error": data,
                    }

                return {"ok": True, "status": response.status, "data": data}

    def _format_response(self, title: str, payload: Dict[str, Any]) -> str:
        if not payload["ok"]:
            return (
                f"{title} failed with HTTP {payload['status']}.\n\n"
                f"{json.dumps(payload['error'], indent=2, ensure_ascii=False)}"
            )

        return (
            f"{title}\n\n"
            f"```json\n{json.dumps(payload['data'], indent=2, ensure_ascii=False)}\n```"
        )

    async def search_tweets(
        self,
        query: str,
        limit: Optional[int] = None,
        query_type: str = "Latest",
        cursor: Optional[str] = None,
        since_time: Optional[str] = None,
        until_time: Optional[str] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search X posts with X search operators and cursor pagination.

        Args:
            query: Search query, such as keywords, hashtags, from:user, or exact phrases.
            limit: Maximum posts to return. Defaults to the configured limit.
            query_type: Sort order. Use Latest or Top.
            cursor: Pagination cursor from a prior response.
            since_time: ISO 8601 timestamp for the lower time bound.
            until_time: ISO 8601 timestamp for the upper time bound.
        """
        result_limit = max(1, min(limit or self.valves.DEFAULT_LIMIT, 200))
        sort_order = "Top" if query_type.lower() == "top" else "Latest"

        await emit_status(__event_emitter__, f"Searching X posts for: {query}")
        payload = await self._get_json(
            "/x/tweets/search",
            {
                "q": query,
                "limit": result_limit,
                "queryType": sort_order,
                "cursor": cursor,
                "sinceTime": since_time,
                "untilTime": until_time,
            },
        )
        await emit_status(__event_emitter__, "X post search finished", done=True)
        return self._format_response(f"X post search results for '{query}'", payload)

    async def lookup_tweet(
        self,
        tweet_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Look up one X post by numeric post ID.

        Args:
            tweet_id: Numeric X post ID.
        """
        await emit_status(__event_emitter__, f"Looking up X post: {tweet_id}")
        payload = await self._get_json(f"/x/tweets/{tweet_id}", {})
        await emit_status(__event_emitter__, "X post lookup finished", done=True)
        return self._format_response(f"X post {tweet_id}", payload)

    async def search_users(
        self,
        query: str,
        cursor: Optional[str] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Search X users by name or username.

        Args:
            query: Name, username, or keyword to search.
            cursor: Pagination cursor from a prior response.
        """
        await emit_status(__event_emitter__, f"Searching X users for: {query}")
        payload = await self._get_json(
            "/x/users/search",
            {"q": query, "cursor": cursor},
        )
        await emit_status(__event_emitter__, "X user search finished", done=True)
        return self._format_response(f"X user search results for '{query}'", payload)

    async def get_user(
        self,
        user_id_or_username: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Look up an X user by numeric ID or username.

        Args:
            user_id_or_username: Numeric user ID or username without @.
        """
        await emit_status(
            __event_emitter__, f"Looking up X user: {user_id_or_username}"
        )
        payload = await self._get_json(f"/x/users/{user_id_or_username}", {})
        await emit_status(__event_emitter__, "X user lookup finished", done=True)
        return self._format_response(f"X user {user_id_or_username}", payload)

    async def get_user_tweets(
        self,
        user_id_or_username: str,
        cursor: Optional[str] = None,
        include_replies: bool = False,
        include_parent_tweet: bool = False,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        List recent posts for an X user.

        Args:
            user_id_or_username: Numeric user ID or username without @.
            cursor: Pagination cursor from a prior response.
            include_replies: Include reply posts.
            include_parent_tweet: Include parent post data for replies.
        """
        await emit_status(
            __event_emitter__, f"Fetching posts for X user: {user_id_or_username}"
        )
        payload = await self._get_json(
            f"/x/users/{user_id_or_username}/tweets",
            {
                "cursor": cursor,
                "includeReplies": include_replies,
                "includeParentTweet": include_parent_tweet,
            },
        )
        await emit_status(__event_emitter__, "X user posts finished", done=True)
        return self._format_response(f"Posts for X user {user_id_or_username}", payload)

    async def get_trends(
        self,
        woeid: int = 1,
        count: int = 30,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Get trending X topics for a region.

        Args:
            woeid: Yahoo WOEID region code. Use 1 for worldwide.
            count: Number of trends to return, from 1 to 50.
        """
        trend_count = max(1, min(count, 50))
        await emit_status(__event_emitter__, f"Fetching X trends for WOEID {woeid}")
        payload = await self._get_json(
            "/trends",
            {"woeid": woeid, "count": trend_count},
        )
        await emit_status(__event_emitter__, "X trends finished", done=True)
        return self._format_response(f"X trends for WOEID {woeid}", payload)
