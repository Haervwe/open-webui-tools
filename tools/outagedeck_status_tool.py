"""
title: OutageDeck Provider Status
description: Check provider and service health, active incidents, and incident timelines without an API key
author: OutageDeck
author_url: https://outagedeck.com
funding_url: https://outagedeck.com/pricing?utm_source=open_webui&utm_medium=integration&utm_campaign=open_webui_tool
requirements:aiohttp
version: 0.1.0
license: MIT
"""

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import urlencode

import aiohttp
from pydantic import BaseModel, Field


BASE_URL = "https://outagedeck.com/api/v1"
CAMPAIGN_PARAMS = {
    "utm_source": "open_webui",
    "utm_medium": "integration",
    "utm_campaign": "open_webui_tool",
}
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PROVIDER_STATUSES = {
    "operational",
    "degraded",
    "partial_outage",
    "major_outage",
    "maintenance",
    "unknown",
}
PROVIDER_CATEGORIES = {
    "cloud",
    "hosting",
    "ai",
    "devtools",
    "data",
    "monitoring",
    "auth",
    "security",
    "email",
    "comms",
    "telecom",
    "productivity",
    "fintech",
}
INCIDENT_STATES = {"active", "resolved"}
INCIDENT_SEVERITIES = {"minor", "major", "critical", "maintenance"}
PROVIDER_SORTS = {"severity", "name"}


async def emit_status(
    event_emitter: Optional[Callable[[Any], Awaitable[None]]],
    description: str,
    done: bool = False,
) -> None:
    """Emit an Open WebUI status event when an emitter is available."""
    if event_emitter:
        await event_emitter(
            {"type": "status", "data": {"description": description, "done": done}}
        )


def validate_slug(value: str, label: str, required: bool = True) -> str:
    """Return a safe OutageDeck slug or raise a model-readable validation error."""
    slug = (value or "").strip()
    if not slug and not required:
        return ""
    if not slug or not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            f"{label} must contain lowercase letters, numbers, and single hyphens only"
        )
    return slug


def validate_choice(value: str, label: str, choices: set) -> str:
    """Validate an optional enum-like tool argument."""
    choice = (value or "").strip()
    if choice and choice not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{label} must be one of: {allowed}")
    return choice


def validate_integer(value: int, label: str, minimum: int, maximum: int) -> int:
    """Validate a bounded integer without accepting booleans as integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def trim_text(value: Any, maximum: int = 500) -> Any:
    """Bound long text fields while leaving non-string values unchanged."""
    if isinstance(value, str) and len(value) > maximum:
        return value[: maximum - 1].rstrip() + "…"
    return value


def incident_summary(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Select the incident fields most useful to an LLM during triage."""
    return {
        "slug": incident.get("slug"),
        "title": incident.get("title"),
        "summary": trim_text(incident.get("summary")),
        "status": incident.get("status"),
        "severity": incident.get("severity"),
        "startedAt": incident.get("startedAt"),
        "updatedAt": incident.get("updatedAt"),
        "resolvedAt": incident.get("resolvedAt"),
        "provider": incident.get("provider"),
        "affectedServices": (incident.get("affectedServices") or [])[:20],
    }


def provider_summary(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Select bounded provider health, source, service, and incident fields."""
    source = provider.get("source") or {}
    current_status = provider.get("currentStatus") or {}
    return {
        "slug": provider.get("slug"),
        "name": provider.get("name"),
        "status": {
            "code": current_status.get("code"),
            "label": current_status.get("label"),
            "headline": current_status.get("headline"),
            "summary": trim_text(current_status.get("summary")),
            "capturedAt": current_status.get("capturedAt"),
        },
        "categories": (provider.get("categories") or [])[:20],
        "counts": provider.get("counts"),
        "source": {
            "name": source.get("name"),
            "checkedAt": source.get("checkedAt"),
            "officialUrl": source.get("officialUrl"),
        },
        "services": [
            {
                "slug": service.get("slug"),
                "name": service.get("name"),
                "status": service.get("status"),
                "category": service.get("category"),
            }
            for service in (provider.get("services") or [])[:30]
        ],
        "activeIncidents": [
            incident_summary(incident)
            for incident in (provider.get("activeIncidents") or [])[:20]
        ],
    }


def campaign_url(path: str) -> str:
    """Create an attributable human-readable OutageDeck source URL."""
    return f"https://outagedeck.com{path}?{urlencode(CAMPAIGN_PARAMS)}"


class Tools:
    """Read-only Open WebUI tools for OutageDeck provider and incident status."""

    class Valves(BaseModel):
        """Administrator-controlled request and response bounds."""

        REQUEST_TIMEOUT_SECONDS: int = Field(
            default=15,
            ge=5,
            le=60,
            description="OutageDeck request timeout in seconds",
        )
        MAX_PROVIDER_RESULTS: int = Field(
            default=20,
            ge=1,
            le=50,
            description="Maximum providers returned by one search",
        )
        MAX_TIMELINE_UPDATES: int = Field(
            default=20,
            ge=1,
            le=50,
            description="Maximum updates returned for one incident",
        )

    def __init__(self) -> None:
        """Initialize default Open WebUI valve values."""
        self.valves = self.Valves()
        self.citation = False

    async def _get_json(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fetch one fixed-origin API path and normalize transport failures."""
        clean_params = {
            key: value
            for key, value in (params or {}).items()
            if value is not None and value != ""
        }
        clean_params.update(CAMPAIGN_PARAMS)
        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT_SECONDS)
        headers = {
            "accept": "application/json",
            "user-agent": "OutageDeck-OpenWebUI-Tool/0.1.0",
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{BASE_URL}{path}",
                    headers=headers,
                    params=clean_params,
                    allow_redirects=False,
                ) as response:
                    try:
                        payload = await response.json(content_type=None)
                    except (json.JSONDecodeError, ValueError, aiohttp.ContentTypeError):
                        payload = {"message": trim_text(await response.text())}

                    if response.status < 200 or response.status >= 300:
                        return {
                            "ok": False,
                            "status": response.status,
                            "error": payload,
                        }
                    if not isinstance(payload, dict):
                        return {
                            "ok": False,
                            "status": response.status,
                            "error": {
                                "message": "OutageDeck returned an invalid response"
                            },
                        }
                    return {"ok": True, "status": response.status, "data": payload}
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "status": 0,
                "error": {"message": "The OutageDeck request timed out"},
            }
        except aiohttp.ClientError as exc:
            return {
                "ok": False,
                "status": 0,
                "error": {"message": f"Could not reach OutageDeck: {exc}"},
            }

    def _format_result(
        self,
        title: str,
        payload: Dict[str, Any],
        normalized: Any,
        source_url: str,
    ) -> str:
        """Format a bounded result and its attributable human-facing source."""
        if not payload["ok"]:
            status = payload.get("status") or "network"
            error = json.dumps(payload.get("error"), ensure_ascii=False)
            return f"{title} failed ({status}): {error}"
        return (
            f"{title}\n\n"
            f"```json\n{json.dumps(normalized, indent=2, ensure_ascii=False)}\n```\n\n"
            f"Source: {source_url}"
        )

    async def find_providers(
        self,
        query: str = "",
        status: str = "",
        category: str = "",
        sort: str = "severity",
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Find monitored vendors and see their current status.

        Args:
            query: Provider name or keyword, up to 100 characters.
            status: Optional operational, degraded, partial_outage, major_outage,
                maintenance, or unknown filter.
            category: Optional cloud, hosting, ai, devtools, data, monitoring,
                auth, security, email, comms, telecom, productivity, or fintech.
            sort: Sort providers by severity or name.
        """
        clean_query = (query or "").strip()
        if len(clean_query) > 100:
            raise ValueError("query must be 100 characters or fewer")
        clean_status = validate_choice(status, "status", PROVIDER_STATUSES)
        clean_category = validate_choice(category, "category", PROVIDER_CATEGORIES)
        clean_sort = validate_choice(sort, "sort", PROVIDER_SORTS) or "severity"

        await emit_status(__event_emitter__, "Checking monitored providers")
        try:
            payload = await self._get_json(
                "/providers",
                {
                    "q": clean_query,
                    "status": clean_status,
                    "category": clean_category,
                    "sort": clean_sort,
                },
            )
            data = payload.get("data", {}).get("data", {}) if payload["ok"] else {}
            providers = (data.get("providers") or [])[
                : self.valves.MAX_PROVIDER_RESULTS
            ]
            normalized = {
                "count": len(providers),
                "totalMatches": data.get("count", len(providers)),
                "providers": [provider_summary(item) for item in providers],
            }
            source_path = "/providers"
            return self._format_result(
                "OutageDeck provider search",
                payload,
                normalized,
                campaign_url(source_path),
            )
        finally:
            await emit_status(__event_emitter__, "Provider check finished", done=True)

    async def get_provider_status(
        self,
        provider_slug: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Get one provider's status, freshness, services, and active incidents.

        Args:
            provider_slug: Lowercase OutageDeck provider slug, such as github,
                openai, anthropic, aws, or cloudflare.
        """
        slug = validate_slug(provider_slug, "provider_slug")
        await emit_status(__event_emitter__, f"Checking provider: {slug}")
        try:
            payload = await self._get_json(f"/providers/{slug}")
            data = payload.get("data", {}).get("data", {}) if payload["ok"] else {}
            normalized = provider_summary(data) if data else {}
            return self._format_result(
                f"OutageDeck provider status: {slug}",
                payload,
                normalized,
                campaign_url(f"/providers/{slug}"),
            )
        finally:
            await emit_status(__event_emitter__, "Provider status finished", done=True)

    async def list_incidents(
        self,
        provider_slug: str = "",
        state: str = "active",
        severity: str = "",
        page: int = 1,
        limit: int = 10,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        List active or resolved incidents, optionally filtered by provider.

        Args:
            provider_slug: Optional lowercase provider slug, such as github.
            state: Optional active or resolved filter. Defaults to active.
            severity: Optional minor, major, critical, or maintenance filter.
            page: Results page, starting at 1.
            limit: Incidents per page, from 1 to 50.
        """
        slug = validate_slug(provider_slug, "provider_slug", required=False)
        clean_state = validate_choice(state, "state", INCIDENT_STATES)
        clean_severity = validate_choice(severity, "severity", INCIDENT_SEVERITIES)
        clean_page = validate_integer(page, "page", 1, 10000)
        clean_limit = validate_integer(limit, "limit", 1, 50)

        target = slug or "all providers"
        await emit_status(__event_emitter__, f"Checking incidents for {target}")
        try:
            payload = await self._get_json(
                "/incidents",
                {
                    "provider": slug,
                    "state": clean_state,
                    "severity": clean_severity,
                    "page": clean_page,
                    "limit": clean_limit,
                },
            )
            data = payload.get("data", {}).get("data", {}) if payload["ok"] else {}
            normalized = {
                "count": data.get("count", 0),
                "page": data.get("page", clean_page),
                "totalPages": data.get("totalPages"),
                "totalIncidents": data.get("totalIncidents"),
                "incidents": [
                    incident_summary(item) for item in (data.get("incidents") or [])
                ],
            }
            source_path = f"/providers/{slug}/incidents" if slug else "/incidents"
            return self._format_result(
                f"OutageDeck incidents: {target}",
                payload,
                normalized,
                campaign_url(source_path),
            )
        finally:
            await emit_status(__event_emitter__, "Incident check finished", done=True)

    async def get_incident_details(
        self,
        incident_slug: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Get an incident's status, affected services, and update timeline.

        Args:
            incident_slug: Lowercase incident slug returned by list_incidents or a
                provider status lookup.
        """
        slug = validate_slug(incident_slug, "incident_slug")
        await emit_status(__event_emitter__, f"Checking incident: {slug}")
        try:
            payload = await self._get_json(f"/incidents/{slug}")
            data = payload.get("data", {}).get("data", {}) if payload["ok"] else {}
            normalized = incident_summary(data) if data else {}
            if data:
                normalized["impactSummary"] = trim_text(data.get("impactSummary"))
                normalized["updates"] = [
                    {
                        "status": update.get("status"),
                        "body": trim_text(update.get("body"), 1000),
                        "createdAt": update.get("createdAt"),
                    }
                    for update in (data.get("updates") or [])[
                        -self.valves.MAX_TIMELINE_UPDATES :
                    ]
                ]
            return self._format_result(
                f"OutageDeck incident: {slug}",
                payload,
                normalized,
                campaign_url(f"/incidents/{slug}"),
            )
        finally:
            await emit_status(__event_emitter__, "Incident details finished", done=True)

    async def get_service_status(
        self,
        service_slug: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Get component-level status and incident history for one service.

        Args:
            service_slug: Lowercase service slug returned in provider status,
                such as github-actions, openai-api, or cloudflare-workers.
        """
        slug = validate_slug(service_slug, "service_slug")
        await emit_status(__event_emitter__, f"Checking service: {slug}")
        try:
            payload = await self._get_json(f"/services/{slug}")
            data = payload.get("data", {}).get("data", {}) if payload["ok"] else {}
            normalized = {
                "slug": data.get("slug"),
                "name": data.get("name"),
                "status": data.get("status"),
                "summary": trim_text(data.get("summary")),
                "description": trim_text(data.get("description")),
                "category": data.get("category"),
                "provider": data.get("provider"),
                "counts": data.get("counts"),
                "incidents": [
                    incident_summary(item)
                    for item in (data.get("incidents") or [])[:20]
                ],
            }
            return self._format_result(
                f"OutageDeck service status: {slug}",
                payload,
                normalized,
                campaign_url(f"/services/{slug}"),
            )
        finally:
            await emit_status(__event_emitter__, "Service status finished", done=True)
