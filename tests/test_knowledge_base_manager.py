"""Tests for the Knowledge Base Manager tool against a local mock Open WebUI API."""

import asyncio
import importlib.util
from pathlib import Path

import pytest
from aiohttp import web

MODULE_PATH = Path(__file__).parents[1] / "tools" / "knowledge_base_manager.py"
SPEC = importlib.util.spec_from_file_location("knowledge_base_manager", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Tools = MODULE.Tools

KB = "11111111-1111-4111-8111-111111111111"
DIR = "22222222-2222-4222-8222-222222222222"
FILE = "33333333-3333-4333-8333-333333333333"
NEW_FILE = "44444444-4444-4444-8444-444444444444"


class FakeRequest:
    """Minimal stand-in for the Starlette request OWUI passes as __request__."""

    def __init__(self, port, host="0.0.0.0", root_path=""):
        self.headers = {"authorization": "Bearer USER-JWT"}
        self.cookies = {"token": "USER-JWT"}
        self.scope = {"server": (host, port), "root_path": root_path, "scheme": "http"}
        self.base_url = "https://public.example.invalid/"


class MockOWUI:
    """Records every request; responses are driven by simple state."""

    def __init__(self):
        self.seen = []
        self.file_status = ["completed"]
        self.link = True
        self.redirect_to = None

    async def handle(self, req):
        self.seen.append((req.method, req.path_qs, req.headers.get("Authorization")))
        path = req.path
        if self.redirect_to:
            raise web.HTTPFound(self.redirect_to)
        if req.method == "POST" and path == "/api/v1/files/":
            await req.read()
            return web.json_response(
                {"status": True, "id": NEW_FILE, "filename": "note.md"}
            )
        if req.method == "GET" and path == f"/api/v1/files/{NEW_FILE}":
            status = (
                self.file_status.pop(0)
                if len(self.file_status) > 1
                else self.file_status[0]
            )
            data = {"status": status}
            if status == "failed":
                data["error"] = "embedding failed"
            return web.json_response(
                {"id": NEW_FILE, "filename": "note.md", "data": data}
            )
        if req.method == "GET" and path == f"/api/v1/files/{FILE}":
            return web.json_response({"id": FILE, "filename": "old.md", "data": {}})
        if req.method == "GET" and path == f"/api/v1/knowledge/{KB}/files":
            items = []
            if self.link and "directory_id" not in req.query:
                items = [{"id": NEW_FILE, "filename": "note.md", "directory_id": DIR}]
            breadcrumbs = []
            if req.query.get("directory_id") == DIR:
                breadcrumbs = [{"id": DIR, "name": "Docs"}]
            return web.json_response(
                {
                    "items": items,
                    "total": len(items),
                    "directories": [],
                    "breadcrumbs": breadcrumbs,
                }
            )
        if req.method == "GET" and path == f"/api/v1/knowledge/{KB}":
            return web.json_response({"id": KB, "name": "My KB"})
        if req.method == "DELETE":
            return web.json_response({"message": "ok"})
        if path == "/api/v1/knowledge/search/files":
            return web.json_response({"items": [], "total": 0})
        return web.json_response({"items": [], "total": 0})


async def _serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def run(coro_fn):
    """Start a mock server, run coro_fn(tool, request, mock), stop the server."""

    async def main():
        mock = MockOWUI()
        runner, port = await _serve(mock.handle)
        try:
            tool = Tools()
            return await coro_fn(tool, FakeRequest(port), mock)
        finally:
            await runner.cleanup()

    return asyncio.run(main())


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def no_sleep(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr(MODULE.asyncio, "sleep", no_sleep)


# --------------------------------------------------------------------------- ids


@pytest.mark.parametrize(
    "bad_id", ["all", "../../users/all", "x/../../configs/export#", "a?b", "search", ""]
)
def test_injected_file_ids_never_reach_the_api(bad_id):
    async def body(tool, req, mock):
        async def approve(_):
            return True

        results = [
            await tool.delete_knowledge_file(
                file_id=bad_id, confirm=True, __request__=req, __event_call__=approve
            ),
            await tool.read_knowledge_file(file_id=bad_id, __request__=req),
            await tool.update_knowledge_file(
                file_id=bad_id, content="x", __request__=req
            ),
            await tool.rename_knowledge_file(
                file_id=bad_id, new_filename="a.md", __request__=req
            ),
        ]
        return results, mock.seen

    results, seen = run(body)
    assert all(r.startswith("Error") for r in results), results
    assert seen == []


@pytest.mark.parametrize(
    "bad_id", ["x/../../configs/export#", "create", "reindex", f"{KB}/delete"]
)
def test_injected_knowledge_and_directory_ids_rejected(bad_id):
    async def body(tool, req, mock):
        results = [
            await tool.list_knowledge_files(
                knowledge_id=bad_id, directory_id="", __request__=req
            ),
            await tool.list_knowledge_files(
                knowledge_id=KB, directory_id=bad_id, __request__=req
            ),
            await tool.get_knowledge_tree(knowledge_id=bad_id, __request__=req),
            await tool.update_knowledge_base(
                knowledge_id=bad_id, new_name="n", __request__=req
            ),
            await tool.create_knowledge_markdown(
                knowledge_id=KB,
                filename="a.md",
                content="x",
                directory_id=bad_id,
                __request__=req,
            ),
            await tool.move_knowledge_directory(
                knowledge_id=KB,
                directory_id=DIR,
                target_parent_id=bad_id,
                __request__=req,
            ),
            await tool.rename_knowledge_directory(
                knowledge_id=KB, directory_id=bad_id, new_name="n", __request__=req
            ),
        ]
        return results, mock.seen

    results, seen = run(body)
    assert all(r.startswith("Error") for r in results), results
    assert seen == []


# ---------------------------------------------------------------------- base url


def test_base_url_from_server_socket_and_credentials_forwarded():
    async def body(tool, req, mock):
        out = await tool.list_knowledge_files(
            knowledge_id=KB, directory_id="", __request__=req
        )
        return out, mock.seen

    out, seen = run(body)
    assert "FILES:" in out
    assert seen and seen[0][0] == "GET"
    assert seen[0][1].startswith(f"/api/v1/knowledge/{KB}/files")
    assert seen[0][2] == "Bearer USER-JWT"


def test_base_url_resolution_rules():
    tool = Tools()
    req = FakeRequest(8080, host="0.0.0.0", root_path="/owui/")
    assert tool._resolve_base_url(req) == ("http://127.0.0.1:8080/owui/api/v1", None)
    req = FakeRequest(3000, host="::")
    assert tool._resolve_base_url(req) == ("http://[::1]:3000/api/v1", None)
    req.scope = {}
    assert tool._resolve_base_url(req) == (
        "https://public.example.invalid/api/v1",
        None,
    )
    tool.valves.OWUI_API_BASE_URL = "http://owui:8080/api/v1/"
    assert tool._resolve_base_url(req) == ("http://owui:8080/api/v1", None)
    for bad in (
        "ftp://x/api/v1",
        "http://u:p@x/api/v1",
        "http://x/api/v1?a=b",
        "/api/v1",
    ):
        tool.valves.OWUI_API_BASE_URL = bad
        base, err = tool._resolve_base_url(req)
        assert base is None and err.startswith("Error")


def test_redirects_are_not_followed():
    async def main():
        other = MockOWUI()
        other_runner, other_port = await _serve(other.handle)
        mock = MockOWUI()
        mock.redirect_to = f"http://127.0.0.1:{other_port}/steal"
        runner, port = await _serve(mock.handle)
        try:
            out = await Tools().read_knowledge_file(
                file_id=FILE, __request__=FakeRequest(port)
            )
        finally:
            await runner.cleanup()
            await other_runner.cleanup()
        return out, other.seen

    out, other_seen = asyncio.run(main())
    assert out.startswith("Open WebUI API error 302")
    assert other_seen == []


# ------------------------------------------------------------------ confirmation


def _deletes(seen):
    return [s for s in seen if s[0] == "DELETE"]


@pytest.mark.parametrize(
    "answer",
    [None, False, {"error": "Client session disconnected."}, {"status": "cancelled"}],
)
def test_delete_file_needs_human_confirmation(answer):
    async def body(tool, req, mock):
        prompts = []

        async def event_call(event):
            prompts.append(event)
            return answer

        out = await tool.delete_knowledge_file(
            file_id=FILE, confirm=True, __request__=req, __event_call__=event_call
        )
        return out, mock.seen, prompts

    out, seen, prompts = run(body)
    assert out.startswith("Deletion cancelled")
    assert _deletes(seen) == []
    assert prompts[0]["type"] == "confirmation"
    assert "old.md" in prompts[0]["data"]["message"]


def test_delete_without_event_call_is_refused():
    async def body(tool, req, mock):
        out = await tool.delete_knowledge_file(
            file_id=FILE, confirm=True, __request__=req
        )
        out2 = await tool.delete_knowledge_directory(
            knowledge_id=KB, directory_id=DIR, __request__=req
        )
        out3 = await tool.delete_knowledge_base(
            knowledge_id=KB, confirm=True, __request__=req
        )
        return [out, out2, out3], mock.seen

    outs, seen = run(body)
    assert all(o.startswith("Deletion cancelled") for o in outs), outs
    assert _deletes(seen) == []


def test_delete_confirmed_by_user():
    async def body(tool, req, mock):
        async def yes(event):
            return True

        out = await tool.delete_knowledge_file(
            file_id=FILE, confirm=True, __request__=req, __event_call__=yes
        )
        out2 = await tool.delete_knowledge_directory(
            knowledge_id=KB, directory_id=DIR, __request__=req, __event_call__=yes
        )
        return out, out2, mock.seen

    out, out2, seen = run(body)
    assert "deleted successfully" in out
    assert "deleted successfully" in out2
    paths = [s[1] for s in _deletes(seen)]
    assert paths == [
        f"/api/v1/files/{FILE}",
        f"/api/v1/knowledge/{KB}/dirs/{DIR}/delete?move_files=true",
    ]


def test_confirm_flag_still_required_and_valve_can_disable_ui_prompt():
    async def body(tool, req, mock):
        out = await tool.delete_knowledge_file(
            file_id=FILE, confirm=False, __request__=req
        )
        tool.valves.REQUIRE_UI_CONFIRMATION = False
        out2 = await tool.delete_knowledge_file(
            file_id=FILE, confirm=True, __request__=req
        )
        return out, out2, mock.seen

    out, out2, seen = run(body)
    assert out == "Deletion cancelled: confirm must be true."
    assert "deleted successfully" in out2
    assert len(_deletes(seen)) == 1


# ----------------------------------------------------------------- upload checks


def test_upload_success_is_verified():
    async def body(tool, req, mock):
        mock.file_status = ["pending", "pending", "completed"]
        out = await tool.create_knowledge_markdown(
            knowledge_id=KB, filename="note", content="# hi", __request__=req
        )
        return out, mock.seen

    out, seen = run(body)
    assert out.startswith("Created and indexed knowledge file successfully.")
    assert f"file_id={NEW_FILE}" in out
    assert f"directory_id={DIR}" in out
    assert sum(1 for s in seen if s[1] == f"/api/v1/files/{NEW_FILE}") == 3
    assert _deletes(seen) == []


def test_failed_processing_is_reported_and_orphan_removed():
    async def body(tool, req, mock):
        mock.file_status = ["failed"]
        out = await tool.create_knowledge_markdown(
            knowledge_id=KB, filename="note.md", content="# hi", __request__=req
        )
        return out, mock.seen

    out, seen = run(body)
    assert out.startswith("Error: Open WebUI failed to process or link")
    assert "embedding failed" in out
    assert [s[1] for s in _deletes(seen)] == [f"/api/v1/files/{NEW_FILE}"]


def test_unlinked_upload_is_reported_and_orphan_removed():
    async def body(tool, req, mock):
        mock.link = False
        out = await tool.upsert_knowledge_markdown_at_path(
            knowledge_id=KB, path="note.md", content="# hi", __request__=req
        )
        return out, mock.seen

    out, seen = run(body)
    assert out.startswith("Error: the file was uploaded but Open WebUI did not add it")
    assert [s[1] for s in _deletes(seen)] == [f"/api/v1/files/{NEW_FILE}"]


# ------------------------------------------------------------------ input checks


def test_empty_inputs_rejected_without_calls():
    async def body(tool, req, mock):
        outs = [
            await tool.find_and_read_knowledge_file(query="  ", __request__=req),
            await tool.rename_knowledge_directory(
                knowledge_id=KB, directory_id=DIR, new_name=" ", __request__=req
            ),
        ]
        return outs, mock.seen

    outs, seen = run(body)
    assert all(o.startswith("Error") for o in outs)
    assert seen == []
