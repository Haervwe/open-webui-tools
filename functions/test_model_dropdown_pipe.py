"""
title: Test Model Dropdown Pipe
author: haervwe
author_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.0
license: MIT
description: >
    Test pipe: fetches active BASE models at import time and exposes them
    as a Literal valve dropdown in the Open WebUI settings UI.
"""

import logging
from typing import List, Optional, Literal, get_args, Annotated
from pydantic import BaseModel, Field, create_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run at module import time — DB is fully available at this point in OWUI
# ---------------------------------------------------------------------------


def _fetch_base_model_ids() -> List[str]:
    """
    Primary: app.state.BASE_MODELS (populated at OWUI startup, always available).
    Cross-reference with DB to exclude models explicitly deactivated by the admin.
    Also filters out pipe/pipeline manifold entries.

    Fallback: app.state.MODELS (lazy, may be empty at import time).
    """
    try:
        from open_webui.main import app  # type: ignore
        from open_webui.models.models import Models  # type: ignore

        # IDs that have been explicitly deactivated in the DB
        inactive_ids: set = {m.id for m in Models.get_base_models() if not m.is_active}
        log.warning("[ModelDropdown] inactive_ids from DB: %s", inactive_ids)

        base_models = getattr(app.state, "BASE_MODELS", None)
        if base_models:
            ids = [
                m["id"]
                for m in base_models
                if m.get("id")
                and m["id"] not in inactive_ids  # skip deactivated
                and "pipe" not in m  # skip function pipe manifolds
                and "pipeline" not in m  # skip Pipelines-server models
            ]
            log.warning("[ModelDropdown] BASE_MODELS filtered ids: %s", ids)
            return ids or ["(none)"]

        # Fallback: MODELS dict (populated lazily, may be empty at import time)
        models_state = getattr(app.state, "MODELS", None)
        if models_state:
            items = (
                list(models_state.values())
                if isinstance(models_state, dict)
                else list(models_state)
            )
            ids = [
                m["id"]
                for m in items
                if m.get("id")
                and not m.get("preset")
                and not m.get("arena")
                and "pipe" not in m
                and "pipeline" not in m
            ]
            log.warning("[ModelDropdown] MODELS fallback ids: %s", ids)
            return ids or ["(none)"]

    except Exception as e:
        log.warning("[ModelDropdown] error: %s", e)
        return [f"error: {e}"]

    return ["(no models found)"]


def _build_valves_class(model_ids: List[str]) -> type:
    unique_ids = list(dict.fromkeys(model_ids)) or ["(none)"]
    literal_type = Literal[tuple(unique_ids)]  # type: ignore[valid-type]
    return create_model(
        "Valves",
        __base__=BaseModel,
        selected_model=(
            Annotated[
                literal_type,
                Field(
                    default=unique_ids[0],
                    description="Base model — fetched from active Open WebUI providers.",
                ),
            ],
            unique_ids[0],
        ),
    )


# Built once at import time → OWUI reads the correct schema for the settings UI
_BASE_MODEL_IDS = _fetch_base_model_ids()
_DynamicValves = _build_valves_class(_BASE_MODEL_IDS)


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------


class Pipe:
    Valves = _DynamicValves  # class-level: schema visible to OWUI immediately

    def __init__(self):
        self.id = "test_model_dropdown"
        self.name = "Test Model Dropdown"
        self.type = "pipe"
        self.valves = self.Valves()

    def pipes(self) -> List[dict]:
        return [{"id": "test-model-dropdown", "name": "Test Model Dropdown"}]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        **kwargs,
    ) -> str:
        selected = self.valves.selected_model
        choices: List[str] = list(
            get_args(self.Valves.__annotations__.get("selected_model", str))
        )

        lines = [
            f"**Selected model:** `{selected}`",
            "",
            f"**All available base models ({len(choices)}):**",
        ]
        for mid in choices:
            marker = "✅" if mid == selected else "•"
            lines.append(f"{marker} `{mid}`")

        return "\n".join(lines)
