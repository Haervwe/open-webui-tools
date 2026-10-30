"""Tests for functions/resume.py: the pipe must return the text it streamed.

Open WebUI >= 0.12 rebuilds the final reply from the pipe's own output when the
stream ends, so text sent only through "message" events would be wiped. These
tests check that the returned value is exactly what was emitted (no double
rendering) and that status events are still sent.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

# Import the real third-party deps before patching sys.modules, so patch.dict's
# restore does not unload half-initialised numpy/pandas modules.
import aiofiles  # noqa: F401
import aiohttp  # noqa: F401
import fastapi  # noqa: F401
import pandas  # noqa: F401

RESUME_PATH = os.path.join(os.path.dirname(__file__), "..", "functions", "resume.py")


class _FakeStreamingResponse:
    def __init__(self, parts):
        self._parts = parts

    @property
    def body_iterator(self):
        async def gen():
            for p in self._parts:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n".encode()
            yield b"data: [DONE]"

        return gen()


def _load_resume(generate_mock):
    owui = types.ModuleType("open_webui")
    constants = types.ModuleType("open_webui.constants")
    constants.TASKS = types.SimpleNamespace(DEFAULT="default")
    main = types.ModuleType("open_webui.main")
    main.generate_chat_completions = generate_mock
    models = types.ModuleType("open_webui.models")
    users = types.ModuleType("open_webui.models.users")
    users.User = object
    users.Users = types.SimpleNamespace(get_user_by_id=AsyncMock(return_value={"id": "u1"}))
    stubs = {
        "open_webui": owui,
        "open_webui.constants": constants,
        "open_webui.main": main,
        "open_webui.models": models,
        "open_webui.models.users": users,
    }
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("resume_under_test", RESUME_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _completion(text):
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "resumes.csv"
    path.write_text(
        "Category,Resume\n"
        "Data Science,python pandas machine learning\n"
        "Data Science,deep learning statistics\n"
        "HR,recruiting onboarding\n"
    )
    return str(path)


def _make_pipe(module, dataset):
    pipe = module.Pipe()
    pipe.valves.Model = "test-model"
    pipe.valves.Dataset_path = dataset
    pipe.valves.web_search = False
    return pipe


def _collect():
    events = []

    async def emitter(event):
        events.append(event)

    return events, emitter


def test_full_analysis_returns_emitted_text(dataset):
    responses = iter(
        [
            _completion("Data Science"),  # tags
            _completion("Looks solid."),  # first impression
            _completion("Needs metrics."),  # adversarial analysis
            _completion("Tell me about a model."),  # interview questions
        ]
    )
    gen = AsyncMock(side_effect=lambda **kw: next(responses))
    module = _load_resume(gen)
    pipe = _make_pipe(module, dataset)
    events, emitter = _collect()

    result = asyncio.run(
        pipe.pipe(
            body={"messages": [{"role": "user", "content": "my resume text"}]},
            __user__={"id": "u1"},
            __event_emitter__=emitter,
            __request__=None,
        )
    )

    emitted = "".join(e["data"]["content"] for e in events if e["type"] == "message")
    assert result == emitted
    assert "**First Impression:**\nLooks solid." in result
    assert "**Adversarial Analysis:**\nNeeds metrics." in result
    assert "Tell me about a model." in result
    statuses = [e for e in events if e["type"] == "status"]
    assert statuses and statuses[-1]["data"]["done"] is True
    # every sub-call forwards the request by keyword
    for call in gen.call_args_list:
        assert set(call.kwargs) == {"request", "form_data", "user"}


def test_career_advisor_follow_up_returns_streamed_text(dataset):
    gen = AsyncMock(return_value=_FakeStreamingResponse(["Hello ", "there"]))
    module = _load_resume(gen)
    pipe = _make_pipe(module, dataset)
    events, emitter = _collect()

    result = asyncio.run(
        pipe.pipe(
            body={
                "messages": [
                    {"role": "user", "content": "resume"},
                    {"role": "assistant", "content": "analysis"},
                ]
            },
            __user__={"id": "u1"},
            __event_emitter__=emitter,
            __request__=None,
        )
    )

    assert result == "Hello there"
    assert [e["data"]["content"] for e in events if e["type"] == "message"] == ["Hello ", "there"]


def test_state_reset_between_calls_and_partial_text_kept_on_error(dataset):
    responses = iter(
        [
            _completion("Data Science"),
            _completion("First."),
            RuntimeError("boom"),
        ]
    )

    def side_effect(**kw):
        r = next(responses)
        if isinstance(r, Exception):
            raise r
        return r

    gen = AsyncMock(side_effect=side_effect)
    module = _load_resume(gen)
    pipe = _make_pipe(module, dataset)
    pipe.response_text = "stale text from a previous call"
    events, emitter = _collect()

    result = asyncio.run(
        pipe.pipe(
            body={"messages": [{"role": "user", "content": "resume"}]},
            __user__={"id": "u1"},
            __event_emitter__=emitter,
            __request__=None,
        )
    )

    assert result == "**First Impression:**\nFirst."
    assert any(e["type"] == "status" and e["data"]["level"] == "error" for e in events)
