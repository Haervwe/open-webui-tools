"""Unit tests for the 3D Model Viewer tool (offline, Open WebUI stubbed)."""

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _install_open_webui_stubs() -> None:
    """Stand-ins for the Open WebUI modules the tools import (setdefault per
    module so it coexists with stubs installed by other test modules)."""
    defaults = {
        "open_webui": {},
        "open_webui.models": {},
        "open_webui.models.chats": {"Chats": SimpleNamespace(insert_chat_files=AsyncMock())},
        "open_webui.models.files": {"Files": SimpleNamespace(get_file_by_id=AsyncMock(return_value=None))},
        "open_webui.models.users": {
            "Users": SimpleNamespace(get_user_by_id=AsyncMock(return_value=None)),
            "UserModel": type("UserModel", (), {}),
        },
        "open_webui.routers": {},
        "open_webui.routers.files": {"upload_file_handler": AsyncMock()},
        "open_webui.storage": {},
        "open_webui.storage.provider": {"Storage": SimpleNamespace(get_file=lambda path: path)},
        "open_webui.utils": {},
        "open_webui.utils.access_control": {},
        "open_webui.utils.access_control.files": {"has_access_to_file": AsyncMock(return_value=False)},
    }
    for name, attrs in defaults.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for attr, value in attrs.items():
            if isinstance(module, types.ModuleType) and not hasattr(module, attr):
                setattr(module, attr, value)


_install_open_webui_stubs()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import atlascloud_media_tool  # noqa: E402
import model_3d_viewer_tool as viewer  # noqa: E402
from model_3d_viewer_tool import (  # noqa: E402
    MODEL_VIEWER_INTEGRITY,
    MODEL_VIEWER_MODULE_URL,
    Tools,
    find_attached_model,
    model_3d_viewer_embed,
    parse_file_reference,
)

FILE_ID = "3f2c9a1e-1111-4222-8333-944455556666"
OTHER_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
USER = {"id": "u1", "name": "Alice", "role": "user"}


def _file(file_id, *, user_id="u1", name="robot.glb", content_type="model/gltf-binary", path=None, size=2048):
    return SimpleNamespace(
        id=file_id,
        user_id=user_id,
        filename=name,
        path=path if path is not None else f"/data/uploads/{file_id}_{name}",
        meta={"name": name, "content_type": content_type, "size": size},
    )


@pytest.fixture
def env(monkeypatch):
    """Fake file table, user lookup, access check and storage for the viewer module."""
    files = {}
    users = {"u1": SimpleNamespace(id="u1", role="user"), "admin": SimpleNamespace(id="admin", role="admin")}
    access = AsyncMock(return_value=False)

    async def get_file_by_id(file_id, *args, **kwargs):
        return files.get(file_id)

    async def get_user_by_id(user_id, *args, **kwargs):
        return users.get(user_id)

    # Other test modules may have stubbed open_webui.models.users with a MagicMock first.
    monkeypatch.setattr(viewer, "UserModel", type("UserModel", (), {}))
    monkeypatch.setattr(viewer, "Files", SimpleNamespace(get_file_by_id=get_file_by_id))
    monkeypatch.setattr(viewer, "Users", SimpleNamespace(get_user_by_id=get_user_by_id))
    monkeypatch.setattr(viewer, "Storage", SimpleNamespace(get_file=lambda path: path))
    access_module = types.ModuleType("open_webui.utils.access_control.files")
    access_module.has_access_to_file = access
    monkeypatch.setitem(sys.modules, "open_webui.utils.access_control.files", access_module)
    return SimpleNamespace(files=files, users=users, access=access)


def _html(result):
    assert isinstance(result, tuple), result
    response, message = result
    return response.body.decode("utf-8"), message


# --------------------------------------------------------------------------
# Viewer embed parity
# --------------------------------------------------------------------------


def test_viewer_embed_is_identical_to_atlascloud():
    args = ("/api/v1/files/x/content", "https://cdn.example/p.png", "Robot <b>", "a robot")
    kwargs = {"download_name": "robot.glb", "model_label": "GLB - 2 KB", "auto_rotate": False, "height": 500}
    assert model_3d_viewer_embed(*args, **kwargs) == atlascloud_media_tool.model_3d_viewer_embed(*args, **kwargs)
    assert viewer._RUNTIME_JS == atlascloud_media_tool._RUNTIME_JS
    assert viewer._MODEL_3D_JS == atlascloud_media_tool._MODEL_3D_JS
    assert viewer._MODEL_3D_CSS == atlascloud_media_tool._MODEL_3D_CSS
    assert viewer._BASE_CSS == atlascloud_media_tool._BASE_CSS
    assert MODEL_VIEWER_MODULE_URL == atlascloud_media_tool.MODEL_VIEWER_MODULE_URL
    assert MODEL_VIEWER_INTEGRITY == atlascloud_media_tool.MODEL_VIEWER_INTEGRITY


def test_parse_file_reference_forms():
    assert parse_file_reference(FILE_ID) == FILE_ID
    assert parse_file_reference(f"/api/v1/files/{FILE_ID}") == FILE_ID
    assert parse_file_reference(f"/api/v1/files/{FILE_ID}/content") == FILE_ID
    assert parse_file_reference(f"https://owui.example/api/v1/files/{FILE_ID}/content?x=1") == FILE_ID
    assert parse_file_reference("/etc/passwd") is None
    assert parse_file_reference("file:///api/v1/files/abc") is None
    assert parse_file_reference("../../secret") is None
    assert parse_file_reference("") is None


# --------------------------------------------------------------------------
# view_3d_model
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_file_id_renders_viewer(env):
    env.files[FILE_ID] = _file(FILE_ID)
    emitter = AsyncMock()

    html_doc, message = _html(await Tools().view_3d_model(FILE_ID, __user__=USER, __event_emitter__=emitter))

    assert f"/api/v1/files/{FILE_ID}/content" in html_doc
    assert "model-viewer" in html_doc
    assert '"autoRotate": true' in html_doc
    assert '"meta": "GLB - 2.0 KB"' in html_doc
    assert "robot.glb" in message and f"/api/v1/files/{FILE_ID}/content" in message
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True
    env.access.assert_not_awaited()  # owner: no extra access query


@pytest.mark.asyncio
async def test_link_form_and_auto_rotate_valve(env):
    env.files[FILE_ID] = _file(FILE_ID)
    tool = Tools()
    tool.valves.AUTO_ROTATE = False

    html_doc, _ = _html(await tool.view_3d_model(f"/api/v1/files/{FILE_ID}/content", __user__=USER))

    assert f"/api/v1/files/{FILE_ID}/content" in html_doc
    assert '"autoRotate": false' in html_doc


@pytest.mark.asyncio
async def test_attached_file_discovery_prefers_most_recent(env):
    env.files[FILE_ID] = _file(FILE_ID)
    env.files[OTHER_ID] = _file(OTHER_ID, name="old.glb")
    files = [
        {"type": "file", "id": OTHER_ID, "url": f"/api/v1/files/{OTHER_ID}", "name": "old.glb"},
        {"type": "file", "id": "img-1", "name": "photo.png", "file": {"meta": {"content_type": "image/png"}}},
        {"type": "file", "id": FILE_ID, "url": f"/api/v1/files/{FILE_ID}",
         "file": {"id": FILE_ID, "filename": "robot.glb", "meta": {"name": "robot.glb"}}},
        {"type": "collection", "id": "kb-1", "name": "kb.glb"},
    ]

    html_doc, _ = _html(await Tools().view_3d_model(__user__=USER, __files__=files))

    assert f"/api/v1/files/{FILE_ID}/content" in html_doc
    assert OTHER_ID not in html_doc


def test_discovery_falls_back_to_messages_then_metadata():
    messages = [
        {"role": "user", "content": "here", "files": [{"type": "file", "id": OTHER_ID, "name": "a.glb"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again", "files": [{"type": "file", "id": FILE_ID, "name": "b.gltf"}]},
    ]
    assert find_attached_model([], messages, None) == FILE_ID
    meta = {"files": [{"type": "file", "url": f"/api/v1/files/{OTHER_ID}/content",
                       "content_type": "model/gltf-binary"}]}
    assert find_attached_model(None, [{"role": "user", "content": "hi"}], meta) == OTHER_ID
    assert find_attached_model([{"type": "file", "id": FILE_ID, "name": "doc.pdf"}], [], {}) is None


@pytest.mark.asyncio
async def test_no_attachment_reports_clearly(env):
    result = await Tools().view_3d_model(__user__=USER, __files__=[], __messages__=[])
    assert isinstance(result, str)
    assert "No GLB" in result


@pytest.mark.asyncio
async def test_access_denied_for_other_users_file(env):
    env.files[FILE_ID] = _file(FILE_ID, user_id="someone-else")

    result = await Tools().view_3d_model(FILE_ID, __user__=USER)

    assert isinstance(result, str)
    assert "not found, or you do not have access" in result
    assert FILE_ID not in result
    env.access.assert_awaited_once()
    assert env.access.await_args.args[:2] == (FILE_ID, "read")


@pytest.mark.asyncio
async def test_missing_file_uses_same_message_as_denied(env):
    result = await Tools().view_3d_model(FILE_ID, __user__=USER)
    assert "not found, or you do not have access" in result


@pytest.mark.asyncio
async def test_shared_access_and_admin_are_allowed(env):
    env.files[FILE_ID] = _file(FILE_ID, user_id="someone-else")
    env.access.return_value = True
    assert isinstance(await Tools().view_3d_model(FILE_ID, __user__=USER), tuple)

    env.access.return_value = False
    env.access.reset_mock()
    result = await Tools().view_3d_model(FILE_ID, __user__={"id": "admin"})
    assert isinstance(result, tuple)
    env.access.assert_not_awaited()


@pytest.mark.asyncio
async def test_access_check_error_denies(env):
    env.files[FILE_ID] = _file(FILE_ID, user_id="someone-else")
    env.access.side_effect = RuntimeError("db down")
    result = await Tools().view_3d_model(FILE_ID, __user__=USER)
    assert "not found, or you do not have access" in result


@pytest.mark.asyncio
async def test_no_user_is_rejected(env):
    env.files[FILE_ID] = _file(FILE_ID)
    result = await Tools().view_3d_model(FILE_ID, __user__=None)
    assert isinstance(result, str) and "signed-in user" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,content_type",
    [("photo.png", "image/png"), ("notes.txt", "text/plain"), ("robot.glb", "image/png"), ("mesh.obj", "")],
)
async def test_non_glb_file_rejected(env, name, content_type):
    env.files[FILE_ID] = _file(FILE_ID, name=name, content_type=content_type)
    result = await Tools().view_3d_model(FILE_ID, __user__=USER)
    assert isinstance(result, str)
    assert "not a GLB / glTF" in result


@pytest.mark.asyncio
async def test_glb_magic_bytes_checked_for_local_files(env, tmp_path):
    good = tmp_path / "good.glb"
    good.write_bytes(b"glTF\x02\x00\x00\x00" + b"\x00" * 16)
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    env.files[FILE_ID] = _file(FILE_ID, name="good.glb", content_type="application/octet-stream", path=str(good))
    env.files[OTHER_ID] = _file(OTHER_ID, name="bad.glb", content_type="", path=str(bad))

    assert isinstance(await Tools().view_3d_model(FILE_ID, __user__=USER), tuple)
    result = await Tools().view_3d_model(OTHER_ID, __user__=USER)
    assert isinstance(result, str) and "not valid GLB" in result


@pytest.mark.asyncio
async def test_remote_storage_skips_header_check(env, monkeypatch):
    calls = []
    monkeypatch.setattr(viewer, "Storage", SimpleNamespace(get_file=lambda p: calls.append(p) or p))
    env.files[FILE_ID] = _file(FILE_ID, path="s3://bucket/robot.glb")
    assert isinstance(await Tools().view_3d_model(FILE_ID, __user__=USER), tuple)
    assert calls == []


@pytest.mark.asyncio
async def test_gltf_accepted_with_note(env):
    env.files[FILE_ID] = _file(FILE_ID, name="scene.gltf", content_type="model/gltf+json", size=None)
    html_doc, message = _html(await Tools().view_3d_model(FILE_ID, __user__=USER))
    assert '"meta": "GLTF"' in html_doc
    assert "external" in message


@pytest.mark.asyncio
async def test_unsupported_reference_rejected(env):
    result = await Tools().view_3d_model("https://evil.example/model.glb", __user__=USER)
    assert isinstance(result, str) and "Unsupported file reference" in result


@pytest.mark.asyncio
async def test_html_escaping_of_file_name(env):
    evil = '<script>alert(1)</script>"><img src=x onerror=alert(2)>.glb'
    env.files[FILE_ID] = _file(FILE_ID, name=evil, content_type="model/gltf-binary")

    html_doc, _ = _html(await Tools().view_3d_model(FILE_ID, __user__=USER))

    assert "<script>alert(1)" not in html_doc
    assert "<img src=x" not in html_doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_doc  # <title>
    assert "\\u003cscript\\u003e" in html_doc  # DATA JSON
