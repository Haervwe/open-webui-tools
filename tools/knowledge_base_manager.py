"""
title: Knowledge Base Manager
description: Manage Open WebUI knowledge bases, nested directories and Markdown files from chat
(list, tree, create, rename, move, upsert by path, read, search, delete). Every call goes through
Open WebUI's own /api/v1 routes as the signed-in user, so Open WebUI's access control applies.
Destructive operations require a confirmation click in the chat UI.
author: izyazev
author_url: https://github.com/izyazev/open-webui-knowledge-base-manager
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.1.1
license: MIT
required_open_webui_version: 0.9.6
"""

import asyncio
import ipaddress
import json
import os
import re
from typing import Any, Optional
from urllib.parse import quote, urlsplit

import aiohttp
from pydantic import BaseModel, Field

# Open WebUI knowledge bases, knowledge directories and files all use uuid4 ids.
# Anything else (e.g. "all", "search", "../users", "a?b", "x#") is rejected before
# it can reach a URL path, so the model cannot be steered onto other API routes.
_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class _Conn:
    """aiohttp session bound to one resolved Open WebUI API base URL."""

    def __init__(self, base_url, headers, cookies, error=None):
        self.base_url = base_url
        self.error = error
        self.session = (
            aiohttp.ClientSession(headers=headers, cookies=cookies)
            if error is None
            else None
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        if self.session is not None:
            await self.session.close()
        return False


class Tools:
    """Open WebUI knowledge base management tool.

    File indexing remains synchronous: the file creation operation waits
    until Open WebUI finishes processing and indexing.
    """

    class Valves(BaseModel):
        OWUI_API_BASE_URL: str = Field(
            default="",
            description=(
                "Base URL of this Open WebUI instance's API, e.g. http://127.0.0.1:8080/api/v1. "
                "Leave empty to use the local socket Open WebUI is serving on (or 127.0.0.1:$PORT). The user's "
                "Authorization header and cookies are only ever sent to this base."
            ),
        )
        REQUIRE_UI_CONFIRMATION: bool = Field(
            default=True,
            description=(
                "Ask the user to confirm deletions in the chat UI. When disabled, only the "
                "model-supplied confirm flag gates deletions."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        self._short_timeout = aiohttp.ClientTimeout(total=30)
        self._long_timeout = aiohttp.ClientTimeout(total=120)

    def _auth(self, request):
        if request is None:
            return {}, {}
        headers = {}
        authorization = request.headers.get("authorization")
        if authorization:
            headers["Authorization"] = authorization
        return headers, dict(request.cookies)

    def _resolve_base_url(self, request):
        """Return (base_url, err). base_url ends with /api/v1 and has no trailing slash."""
        configured = (self.valves.OWUI_API_BASE_URL or "").strip().rstrip("/")
        if configured:
            parts = urlsplit(configured)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                return None, "Error: OWUI_API_BASE_URL must be an absolute http(s) URL."
            if parts.query or parts.fragment or parts.username or parts.password:
                return (
                    None,
                    "Error: OWUI_API_BASE_URL must not contain credentials, query or fragment.",
                )
            return configured, None

        scope = getattr(request, "scope", None) or {}
        root_path = (scope.get("root_path") or "").rstrip("/")
        server = scope.get("server")
        if server and server[0] and server[1]:
            # The socket uvicorn accepted this request on: Open WebUI itself,
            # without hair-pinning through a public reverse proxy.
            host, port = str(server[0]), int(server[1])
            try:
                ip = ipaddress.ip_address(host)
                if ip.is_unspecified:
                    ip = ipaddress.ip_address("::1" if ip.version == 6 else "127.0.0.1")
                host = f"[{ip}]" if ip.version == 6 else str(ip)
            except ValueError:
                pass
            headers = getattr(request, "headers", None) or {}
            scheme = (
                "https"
                if scope.get("scheme") == "https" and "x-forwarded-proto" not in headers
                else "http"
            )
            return f"{scheme}://{host}:{port}{root_path}/api/v1", None

        # Never derive it from request.base_url: that comes from the Host header,
        # and the user's credentials are forwarded to this URL.
        port = os.environ.get("PORT", "8080")
        if not port.isdigit():
            return (
                None,
                "Error: cannot determine the Open WebUI API URL; set the OWUI_API_BASE_URL valve.",
            )
        return f"http://127.0.0.1:{port}{root_path}/api/v1", None

    def _check_id(self, value, name, allow_empty=False):
        """Validate an Open WebUI id (uuid4). Returns an error string or None."""
        if value is None or value == "":
            return None if allow_empty else f"Error: {name} is required."
        if not isinstance(value, str) or not _ID_RE.match(value):
            return f"Error: invalid {name}: expected an Open WebUI id (uuid), got {str(value)[:80]!r}."
        return None

    def _check_ids(self, *, optional=(), **ids):
        """Validate several ids at once; names listed in optional may be empty."""
        for name, value in ids.items():
            err = self._check_id(value, name, allow_empty=name in optional)
            if err:
                return err
        return None

    def _build_path(self, path, ids):
        """Fill {placeholders} in path with validated, URL-quoted ids."""
        ids = ids or {}
        for name in _PLACEHOLDER_RE.findall(path):
            if name not in ids:
                return None, f"Error: internal path parameter {name} missing."
            err = self._check_id(ids[name], name)
            if err:
                return None, err
        built = _PLACEHOLDER_RE.sub(lambda m: quote(ids[m.group(1)], safe=""), path)
        if not built.startswith("/") or "//" in built or "#" in built:
            return None, "Error: invalid API path."
        return built, None

    def _api_error(self, status, body):
        detail = body
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("detail") is not None:
                detail = parsed["detail"]
                if not isinstance(detail, str):
                    detail = json.dumps(detail)
        except (ValueError, TypeError):
            pass
        return f"Open WebUI API error {status}: {str(detail)[:500]}"

    async def _request(
        self,
        conn,
        method,
        path,
        *,
        ids=None,
        timeout=None,
        expected=(200,),
        **kwargs,
    ):
        """Unified HTTP request sending and response handling.

        path may contain {placeholders}; their values come from ids and are
        validated as Open WebUI ids and URL-quoted before use.
        """
        if conn.error:
            return None, conn.error
        path, err = self._build_path(path, ids)
        if err:
            return None, err
        url = f"{conn.base_url}{path}"
        try:
            async with conn.session.request(
                method,
                url,
                timeout=timeout or self._short_timeout,
                # Never follow redirects: the forwarded credentials must only
                # reach the Open WebUI base URL.
                allow_redirects=False,
                **kwargs,
            ) as response:
                if response.status not in expected:
                    body = await response.text()
                    return None, self._api_error(response.status, body)
                if response.status == 204:
                    return {}, None
                try:
                    return await response.json(content_type=None), None
                except (ValueError, aiohttp.ContentTypeError):
                    text = await response.text()
                    return {"_text": text}, None
        except asyncio.TimeoutError:
            return None, "Error: Open WebUI API request timed out."
        except aiohttp.ClientError as exc:
            return None, f"Error: Open WebUI connection failed ({type(exc).__name__})."

    async def _confirm(self, event_call, title, message, confirm=True):
        """Ask the user to confirm a destructive action. Returns an error string or None."""
        if confirm is not True:
            return "Deletion cancelled: confirm must be true."
        if not self.valves.REQUIRE_UI_CONFIRMATION:
            return None
        if event_call is None:
            return (
                "Deletion cancelled: user confirmation is required but the chat UI is not "
                "available for this call."
            )
        try:
            result = await event_call(
                {"type": "confirmation", "data": {"title": title, "message": message}}
            )
        except Exception as exc:
            return f"Deletion cancelled: confirmation request failed ({type(exc).__name__})."
        if isinstance(result, dict) and result.get("error"):
            return f"Deletion cancelled: confirmation not received ({result.get('error')})."
        if result is not True:
            return "Deletion cancelled by the user."
        return None

    def _path_parts(self, path):
        path = (path or "").strip().replace("\\", "/").strip("/")
        if not path:
            return None, "Error: path is empty."
        parts = [p.strip() for p in path.split("/") if p.strip()]
        if not parts or any(p in (".", "..") for p in parts):
            return None, "Error: invalid path; '.' and '..' are not allowed."
        return parts, None

    def _file_item(self, item):
        return item.get("file", item) if isinstance(item, dict) else {}

    async def _directory_listing(self, session, knowledge_id, directory_id=""):
        """Fetches all pages of a single directory's contents."""
        if directory_id is None:
            return None, "Error: directory id missing in Open WebUI response."
        err = self._check_ids(
            knowledge_id=knowledge_id,
            directory_id=directory_id or "",
            optional=("directory_id",),
        )
        if err:
            return None, err
        page = 1
        all_items = []
        directories = []
        breadcrumbs = []
        while True:
            data, err = await self._request(
                session,
                "GET",
                "/knowledge/{knowledge_id}/files",
                ids={"knowledge_id": knowledge_id},
                params={"directory_id": directory_id, "page": page},
            )
            if err:
                return None, err
            data = data or {}
            if page == 1:
                directories = data.get("directories", [])
                breadcrumbs = data.get("breadcrumbs", [])
            batch = data.get("items", [])
            all_items.extend(batch)
            total = data.get("total")
            if not batch or total is None or len(all_items) >= total:
                break
            page += 1
            if page > 1000:
                return (
                    None,
                    "Error: directory listing exceeded 1000 pages. Possible API issue.",
                )
        return {
            "directories": directories,
            "items": all_items,
            "breadcrumbs": breadcrumbs,
        }, None

    async def _resolve_directory(self, session, knowledge_id, parts):
        current = ""
        for part in parts:
            listing, err = await self._directory_listing(session, knowledge_id, current)
            if err:
                return None, err
            matches = [
                d
                for d in listing["directories"]
                if str(d.get("name", "")).casefold() == part.casefold()
            ]
            if not matches:
                return None, f"Directory not found: {part}"
            if len(matches) > 1:
                return None, f"Multiple matching directories found: {part}"
            current = matches[0].get("id")
            err = self._check_id(current, "directory_id")
            if err:
                return None, err
        return current, None

    async def _find_file_in_directory(
        self, session, knowledge_id, directory_id, filename
    ):
        listing, err = await self._directory_listing(
            session, knowledge_id, directory_id
        )
        if err:
            return None, err
        matches = [
            self._file_item(x)
            for x in listing["items"]
            if self._file_item(x).get("filename", "").casefold() == filename.casefold()
        ]
        if not matches:
            return None, f"File not found: {filename}"
        if len(matches) > 1:
            return None, f"Multiple matching files found: {filename}"
        return matches[0], None

    async def _knowledge_has_content(
        self, session, knowledge_id, directory_id="", visited=None
    ):
        """Checks whether any files exist at any nesting level of the knowledge base."""
        if visited is None:
            visited = set()
        key = directory_id or "ROOT"
        if key in visited:
            return False, None
        visited.add(key)

        listing, err = await self._directory_listing(
            session, knowledge_id, directory_id
        )
        if err:
            return None, err
        if listing["items"]:
            return True, None
        for directory in listing["directories"]:
            child_id = directory.get("id")
            if not child_id:
                continue
            has_content, err = await self._knowledge_has_content(
                session, knowledge_id, child_id, visited
            )
            if err or has_content:
                return has_content, err
        return False, None

    async def _open_session(self, request):
        headers, cookies = self._auth(request)
        base_url, err = self._resolve_base_url(request)
        return _Conn(base_url, headers, cookies, error=err)

    async def list_knowledge_bases(self, __request__=None) -> str:
        """
        List knowledge bases available to the current Open WebUI user,
        including their IDs and write access.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as session:
            data, err = await self._request(session, "GET", "/knowledge/")
        if err:
            return err
        items = (data or {}).get("items", [])
        if not items:
            return "No knowledge bases found."
        return "\n".join(
            f"- {x.get('name', 'Unnamed')} (id={x.get('id')}, write_access={x.get('write_access')})"
            for x in items
        )

    async def list_knowledge_files(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        directory_id: str = "",
        __request__=None,
    ) -> str:
        """
        List files and directories inside an Open WebUI knowledge base.

        directory_id:
        - empty string = root directory
        - directory ID = show contents of that directory

        Returns directory IDs, file IDs and breadcrumbs so the model
        can navigate the knowledge base without creating duplicate folders.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as session:
            data, err = await self._directory_listing(
                session, knowledge_id, directory_id
            )
        if err:
            return err
        out = []
        if data["breadcrumbs"]:
            out.append("PATH:")
            out += [
                f"- {x.get('name', 'Unnamed')} (directory_id={x.get('id')})"
                for x in data["breadcrumbs"]
            ]
        out.append("\nDIRECTORIES:")
        out += [
            f"- {x.get('name', 'Unnamed')} (directory_id={x.get('id')}, parent_id={x.get('parent_id')})"
            for x in data["directories"]
        ] or ["- none"]
        out.append("\nFILES:")
        out += [
            f"- {self._file_item(x).get('filename', 'Unnamed')} (file_id={self._file_item(x).get('id')})"
            for x in data["items"]
        ] or ["- none"]
        return "\n".join(out)

    async def create_knowledge_base(
        self,
        name: str = Field(
            ..., description="Name of the new Open WebUI knowledge base."
        ),
        description: str = "",
        __request__=None,
    ) -> str:
        """
        Create a new Open WebUI knowledge base for the current user.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        name = name.strip()
        if not name:
            return "Error: knowledge base name is empty."
        async with await self._open_session(__request__) as session:
            data, err = await self._request(
                session,
                "POST",
                "/knowledge/create",
                expected=(200, 201),
                json={
                    "name": name,
                    "description": description.strip(),
                    "access_grants": None,
                },
                timeout=self._short_timeout,
            )
        if err:
            return err
        return f"Knowledge base created successfully.\nname={data.get('name', name)}\nknowledge_id={data.get('id')}\ndescription={data.get('description', description)}"

    async def update_knowledge_base(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        new_name: str = "",
        new_description: str = "",
        __request__=None,
    ) -> str:
        """
        Update the name and/or description of an Open WebUI knowledge base.

        Empty new_name keeps the existing name.
        Empty new_description keeps the existing description.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as session:
            current, err = await self._request(
                session,
                "GET",
                "/knowledge/{knowledge_id}",
                ids={"knowledge_id": knowledge_id},
            )
            if err:
                return err
            payload = {
                "name": new_name.strip() or current.get("name", ""),
                "description": new_description.strip()
                or current.get("description", ""),
                "access_grants": current.get("access_grants"),
            }
            data, err = await self._request(
                session,
                "POST",
                "/knowledge/{knowledge_id}/update",
                ids={"knowledge_id": knowledge_id},
                expected=(200, 201),
                json=payload,
                timeout=self._short_timeout,
            )
        if err:
            return err
        return f"Knowledge base updated successfully.\nknowledge_id={knowledge_id}\nname={data.get('name', payload['name'])}\ndescription={data.get('description', payload['description'])}"

    async def create_knowledge_directory(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        name: str = Field(..., description="Name of the directory to create."),
        parent_id: str = "",
        __request__=None,
    ) -> str:
        """
        Create a directory inside an Open WebUI knowledge base.
        Supports nested directories using parent_id.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        name = name.strip()
        if not name:
            return "Error: directory name is empty."
        err = self._check_ids(
            knowledge_id=knowledge_id, parent_id=parent_id, optional=("parent_id",)
        )
        if err:
            return err
        async with await self._open_session(__request__) as session:
            data, err = await self._request(
                session,
                "POST",
                "/knowledge/{knowledge_id}/dirs/create",
                ids={"knowledge_id": knowledge_id},
                expected=(200, 201),
                json={"name": name, "parent_id": parent_id or None},
            )
        if err:
            return err
        return f"Knowledge directory created successfully.\nname={data.get('name', name)}\ndirectory_id={data.get('id')}\nparent_id={data.get('parent_id')}"

    async def ensure_knowledge_directory_path(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        path: str = Field(
            ...,
            description="Directory path to create or reuse, for example Servers/VPN/Notes.",
        ),
        __request__=None,
    ) -> str:
        """
        Ensure that a nested directory path exists inside a knowledge base.

        Existing directories are reused.
        Missing directories are created.
        Example: Servers/VPN/Notes
        """
        parts, err = self._path_parts(path)
        if err:
            return err
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        created = []
        reused = []
        current = ""
        async with await self._open_session(__request__) as session:
            for part in parts:
                listing, err = await self._directory_listing(
                    session, knowledge_id, current
                )
                if err:
                    return err
                matches = [
                    d
                    for d in listing["directories"]
                    if d.get("name", "").casefold() == part.casefold()
                ]
                if len(matches) > 1:
                    return f"Multiple matching directories found: {part}"
                if matches:
                    current = matches[0].get("id")
                    err = self._check_id(current, "directory_id")
                    if err:
                        return err
                    reused.append(part)
                    continue
                data, err = await self._request(
                    session,
                    "POST",
                    "/knowledge/{knowledge_id}/dirs/create",
                    ids={"knowledge_id": knowledge_id},
                    expected=(200, 201),
                    json={"name": part, "parent_id": current or None},
                )
                if err:
                    return err
                current = (data or {}).get("id")
                err = self._check_id(current, "directory_id")
                if err:
                    return err
                created.append(part)
        return f"Knowledge directory path ready.\npath={'/'.join(parts)}\ndirectory_id={current}\ncreated={', '.join(created) or 'none'}\nreused={', '.join(reused) or 'none'}"

    async def _upload(self, session, knowledge_id, directory_id, filename, content):
        """Upload a Markdown file into a knowledge base and verify the result.

        Open WebUI answers POST /files/ with the pre-processing file record even
        when extraction, embedding or the knowledge link failed (those errors are
        only stored in file.data.status/error), so the stored status and the
        knowledge link are checked before reporting success. A failed upload is
        deleted so it does not linger as an orphan in the user's files.

        Returns (data, err); data is the file record plus "directory_id"
        (where Open WebUI actually linked it; None = root).
        """
        err = self._check_ids(
            knowledge_id=knowledge_id,
            directory_id=directory_id or "",
            optional=("directory_id",),
        )
        if err:
            return None, err
        form = aiohttp.FormData()
        form.add_field(
            "file",
            content.encode("utf-8"),
            filename=filename,
            content_type="text/markdown; charset=utf-8",
        )
        meta = {"knowledge_id": knowledge_id}
        if directory_id:
            meta["directory_id"] = directory_id
        form.add_field("metadata", json.dumps(meta), content_type="application/json")
        data, err = await self._request(
            session,
            "POST",
            "/files/?process=true&process_in_background=false",
            expected=(200, 201),
            data=form,
            timeout=self._long_timeout,
        )
        if err:
            if "timed out" in err:
                return None, (
                    "Error: the upload timed out. The file may still have been created and "
                    "indexed; list the directory before retrying to avoid a duplicate."
                )
            return None, err
        data = data or {}
        file_id = data.get("id")
        err = self._check_id(file_id, "file_id")
        if err:
            return (
                None,
                f"Error: upload response did not contain a valid file id. {err}",
            )

        status, error = await self._file_status(session, file_id)
        if status != "completed":
            reason = error or f"processing status is {status!r}"
            cleanup = await self._delete_orphan(session, file_id)
            return None, (
                f"Error: Open WebUI failed to process or link the uploaded file: {reason[:500]}"
                f"{cleanup}"
            )

        linked, err = await self._find_linked_file(
            session, knowledge_id, file_id, data.get("filename") or filename
        )
        if err:
            return None, (
                f"Error: the file was uploaded (file_id={file_id}) but its knowledge link "
                f"could not be verified: {err}"
            )
        if linked is None:
            cleanup = await self._delete_orphan(session, file_id)
            return None, (
                "Error: the file was uploaded but Open WebUI did not add it to the knowledge "
                f"base (missing write access?).{cleanup}"
            )
        data["directory_id"] = linked.get("directory_id")
        return data, None

    async def _file_status(self, session, file_id, attempts=30, delay=1.0):
        """Return (status, error) from the stored file record, waiting while pending."""
        status, error = "unknown", None
        for attempt in range(attempts):
            record, err = await self._request(
                session, "GET", "/files/{file_id}", ids={"file_id": file_id}
            )
            if err:
                return "unknown", err
            file_data = (record or {}).get("data") or {}
            status = file_data.get("status") or "pending"
            error = file_data.get("error")
            if status in ("completed", "failed"):
                return status, error
            if attempt < attempts - 1:
                await asyncio.sleep(delay)
        return status, error

    async def _find_linked_file(self, session, knowledge_id, file_id, filename):
        """Return (item, err): the knowledge file entry for file_id, or None if not linked.

        Searches by filename first (cheap), then scans the whole knowledge base so a
        filename normalised by the server never causes a false "not linked".
        """
        for query in (filename, None):
            page = 1
            seen = 0
            while page <= 1000:
                params = {"page": page}
                if query:
                    params["query"] = query
                data, err = await self._request(
                    session,
                    "GET",
                    "/knowledge/{knowledge_id}/files",
                    ids={"knowledge_id": knowledge_id},
                    params=params,
                )
                if err:
                    return None, err
                batch = (data or {}).get("items", [])
                for item in batch:
                    if self._file_item(item).get("id") == file_id:
                        return item, None
                seen += len(batch)
                total = (data or {}).get("total")
                if not batch or total is None or seen >= total:
                    break
                page += 1
            if not filename:
                break
        return None, None

    async def _delete_orphan(self, session, file_id):
        _, err = await self._request(
            session,
            "DELETE",
            "/files/{file_id}",
            ids={"file_id": file_id},
            expected=(200, 204),
        )
        if err:
            return (
                f"\nThe uploaded file could not be removed (file_id={file_id}): {err}"
            )
        return "\nThe uploaded file was removed again."

    def _dir_label(self, directory_id):
        return directory_id or "ROOT"

    async def create_knowledge_markdown(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the target Open WebUI knowledge base."
        ),
        filename: str = Field(
            ...,
            description="Filename for the new Markdown document, for example server-notes.md.",
        ),
        content: str = Field(
            ..., description="Complete Markdown content to write into the new document."
        ),
        directory_id: str = "",
        __request__=None,
    ) -> str:
        """
        Create a new Markdown file in an Open WebUI knowledge base
        and automatically index it for retrieval.

        directory_id:
        - empty string = create in knowledge base root
        - directory ID = create directly inside that directory
        """
        filename = filename.strip()
        if not filename:
            return "Error: filename is empty."
        if "/" in filename or "\\" in filename:
            return "Error: filename must not contain directory separators."
        if not filename.lower().endswith(".md"):
            filename += ".md"
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as session:
            data, err = await self._upload(
                session, knowledge_id, directory_id, filename, content
            )
        if err:
            return err
        return f"Created and indexed knowledge file successfully.\nfilename={data.get('filename',filename)}\nfile_id={data.get('id')}\nknowledge_id={knowledge_id}\ndirectory_id={self._dir_label(data.get('directory_id'))}"

    async def create_knowledge_markdown_at_path(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the target Open WebUI knowledge base."
        ),
        path: str = Field(
            ...,
            description="Full path including filename, for example Projects/Servers/VPN/Notes/readme.md.",
        ),
        content: str = Field(
            ..., description="Complete Markdown content for the new file."
        ),
        __request__=None,
    ) -> str:
        """
        Create a Markdown file at a nested path inside a knowledge base.

        Existing directories are reused.
        Missing directories are created automatically.
        The file is created directly in the final directory.
        """
        parts, err = self._path_parts(path)
        if err:
            return err
        filename = parts[-1]
        if not filename.lower().endswith(".md"):
            filename += ".md"
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as session:
            directory, err = await self._ensure_dirs(session, knowledge_id, parts[:-1])
            if err:
                return err
            data, err = await self._upload(
                session, knowledge_id, directory, filename, content
            )
        if err:
            return err
        return f"Knowledge Markdown file created successfully.\npath={'/'.join(parts[:-1]+[filename])}\nfilename={data.get('filename',filename)}\nfile_id={data.get('id')}\ndirectory_id={self._dir_label(data.get('directory_id'))}"

    async def _ensure_dirs(self, session, knowledge_id, parts):
        current = ""
        for part in parts:
            listing, err = await self._directory_listing(session, knowledge_id, current)
            if err:
                return None, err
            matches = [
                d
                for d in listing["directories"]
                if d.get("name", "").casefold() == part.casefold()
            ]
            if len(matches) > 1:
                return None, f"Multiple matching directories found: {part}"
            if matches:
                current = matches[0].get("id")
                err = self._check_id(current, "directory_id")
                if err:
                    return None, err
                continue
            data, err = await self._request(
                session,
                "POST",
                "/knowledge/{knowledge_id}/dirs/create",
                ids={"knowledge_id": knowledge_id},
                expected=(200, 201),
                json={"name": part, "parent_id": current or None},
            )
            if err:
                return None, err
            current = (data or {}).get("id")
            err = self._check_id(current, "directory_id")
            if err:
                return None, err
        return current, None

    async def _read_or_update_path(self, knowledge_id, path, content, update, request):
        parts, err = self._path_parts(path)
        if err:
            return err
        if request is None:
            return "Error: Open WebUI request context is unavailable."
        filename = parts[-1]
        async with await self._open_session(request) as session:
            directory, err = await self._resolve_directory(
                session, knowledge_id, parts[:-1]
            )
            if err:
                return err
            item, err = await self._find_file_in_directory(
                session, knowledge_id, directory, filename
            )
            if err:
                return err
            if update:
                _, err = await self._request(
                    session,
                    "POST",
                    "/files/{file_id}/data/content/update",
                    ids={"file_id": item.get("id")},
                    expected=(200, 201),
                    json={"content": content},
                    timeout=self._long_timeout,
                )
                if err:
                    return err
                return f"Knowledge file updated successfully.\npath={path}\nfile_id={item.get('id')}\ndirectory_id={directory or 'ROOT'}"
            data, err = await self._request(
                session,
                "GET",
                "/files/{file_id}/data/content",
                ids={"file_id": item.get("id")},
            )
        if err:
            return err
        return f"path={path}\nfilename={filename}\nfile_id={item.get('id')}\ndirectory_id={directory or 'ROOT'}\n\n{(data or {}).get('content','')}"

    async def update_knowledge_file(
        self,
        file_id: str = Field(
            ..., description="ID of the existing Open WebUI knowledge file."
        ),
        content: str = Field(
            ...,
            description="Complete new Markdown/text content that will replace the existing file content.",
        ),
        __request__=None,
    ) -> str:
        """
        Replace the content of an existing Open WebUI knowledge file
        and reindex it for retrieval.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            _, e = await self._request(
                s,
                "POST",
                "/files/{file_id}/data/content/update",
                ids={"file_id": file_id},
                expected=(200, 201),
                json={"content": content},
                timeout=self._long_timeout,
            )
        return (
            e
            or f"Knowledge file updated and reindexed successfully.\nfile_id={file_id}"
        )

    async def read_knowledge_file(
        self,
        file_id: str = Field(
            ..., description="ID of the Open WebUI knowledge file to read."
        ),
        __request__=None,
    ) -> str:
        """
        Read the extracted/indexed text content of an Open WebUI knowledge file.
        Use this before editing an existing document.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            data, e = await self._request(
                s, "GET", "/files/{file_id}/data/content", ids={"file_id": file_id}
            )
        if e:
            return e
        c = (data or {}).get("content", "")
        return c if c else "File exists, but no extracted text content was found."

    async def update_knowledge_file_at_path(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        path: str = Field(
            ...,
            description="Full path to the existing file, for example Projects/Servers/VPN/Notes/readme.md.",
        ),
        content: str = Field(
            ...,
            description="Complete new content that will replace the existing file content.",
        ),
        __request__=None,
    ) -> str:
        """
        Find an existing knowledge file by its exact path
        and replace its complete text content.

        The file is reindexed automatically.
        Does not create missing directories or files.
        """
        return await self._read_or_update_path(
            knowledge_id, path, content, True, __request__
        )

    async def read_knowledge_file_at_path(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        path: str = Field(
            ...,
            description="Exact file path, for example Projects/Servers/VPN/Notes/readme.md.",
        ),
        __request__=None,
    ) -> str:
        """
        Read an existing knowledge file by its exact path.
        Does not create or modify anything.
        """
        return await self._read_or_update_path(
            knowledge_id, path, None, False, __request__
        )

    async def upsert_knowledge_markdown_at_path(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the target Open WebUI knowledge base."
        ),
        path: str = Field(
            ...,
            description="Full Markdown file path, for example Projects/Servers/VPN/Notes/readme.md.",
        ),
        content: str = Field(
            ..., description="Complete Markdown content to create or replace."
        ),
        __request__=None,
    ) -> str:
        """
        Create or update a Markdown file at an exact knowledge-base path.

        - Existing directories are reused.
        - Missing directories are created.
        - Existing file is updated and reindexed.
        - Missing file is created and indexed.
        - Ambiguous directory or file matches are never guessed.
        """
        parts, err = self._path_parts(path)
        if err:
            return err
        filename = parts[-1]
        if not filename.lower().endswith(".md"):
            filename += ".md"
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            directory, err = await self._ensure_dirs(s, knowledge_id, parts[:-1])
            if err:
                return err
            item, finderr = await self._find_file_in_directory(
                s, knowledge_id, directory, filename
            )
            if finderr and finderr != f"File not found: {filename}":
                # API errors or ambiguous matches must not lead to creating
                # a new file and a possible duplicate.
                return finderr
            if not finderr:
                _, err = await self._request(
                    s,
                    "POST",
                    "/files/{file_id}/data/content/update",
                    ids={"file_id": item.get("id")},
                    expected=(200, 201),
                    json={"content": content},
                    timeout=self._long_timeout,
                )
                if err:
                    return err
                return f"Knowledge Markdown file updated successfully.\naction=updated\npath={'/'.join(parts[:-1]+[filename])}\nfile_id={item.get('id')}"
            data, err = await self._upload(
                s, knowledge_id, directory, filename, content
            )
        if err:
            return err
        return f"Knowledge Markdown file created successfully.\naction=created\npath={'/'.join(parts[:-1]+[filename])}\nfile_id={data.get('id')}"

    async def rename_knowledge_file(
        self,
        file_id: str = Field(
            ..., description="ID of the Open WebUI knowledge file to rename."
        ),
        new_filename: str = Field(
            ...,
            description="New filename including extension, for example vpn-guide.md.",
        ),
        __request__=None,
    ) -> str:
        """
        Rename an existing Open WebUI knowledge file.
        """
        new_filename = new_filename.strip()
        if not new_filename:
            return "Error: new filename is empty."
        if "/" in new_filename or "\\" in new_filename:
            return "Error: filename must not contain directory separators."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            _, e = await self._request(
                s,
                "POST",
                "/files/{file_id}/rename",
                ids={"file_id": file_id},
                expected=(200, 201),
                json={"filename": new_filename},
            )
        return (
            e
            or f"Knowledge file renamed successfully.\nfile_id={file_id}\nnew_filename={new_filename}"
        )

    async def rename_knowledge_directory(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        directory_id: str = Field(..., description="ID of the directory to rename."),
        new_name: str = Field(..., description="New directory name."),
        __request__=None,
    ) -> str:
        """
        Rename a directory inside an Open WebUI knowledge base.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            return "Error: new directory name is empty."
        if "/" in new_name or "\\" in new_name:
            return "Error: directory name must not contain directory separators."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            d, e = await self._request(
                s,
                "POST",
                "/knowledge/{knowledge_id}/dirs/{directory_id}/update",
                ids={"knowledge_id": knowledge_id, "directory_id": directory_id},
                expected=(200, 201),
                json={"name": new_name},
            )
        return (
            e
            or f"Knowledge directory renamed successfully.\ndirectory_id={directory_id}\nnew_name={(d or {}).get('name',new_name)}"
        )

    async def move_knowledge_directory(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        directory_id: str = Field(..., description="ID of the directory to move."),
        target_parent_id: str = Field(
            ...,
            description="ID of the target parent directory. Use an empty string to move the directory to the knowledge base root.",
        ),
        __request__=None,
    ) -> str:
        """
        Move a directory inside an Open WebUI knowledge base.

        Use an empty target_parent_id to move the directory to the root.
        """
        err = self._check_ids(
            target_parent_id=target_parent_id, optional=("target_parent_id",)
        )
        if err:
            return err
        if target_parent_id and target_parent_id == directory_id:
            return "Error: a directory cannot be moved into itself."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            d, e = await self._request(
                s,
                "POST",
                "/knowledge/{knowledge_id}/dirs/{directory_id}/update",
                ids={"knowledge_id": knowledge_id, "directory_id": directory_id},
                expected=(200, 201),
                json={"parent_id": target_parent_id or None},
            )
        return (
            e
            or f"Knowledge directory moved successfully.\ndirectory_id={directory_id}\nparent_id={(d or {}).get('parent_id')}"
        )

    async def delete_knowledge_file(
        self,
        file_id: str = Field(
            ..., description="ID of the Open WebUI file to permanently delete."
        ),
        confirm: bool = Field(
            ...,
            description="Must be true. Set true only when the user explicitly requested permanent deletion of the file.",
        ),
        __request__=None,
        __event_call__=None,
    ) -> str:
        """
        Permanently delete an Open WebUI knowledge file.

        This removes the file itself, its knowledge-base associations,
        and its indexed embeddings.

        Use only when the user explicitly asks to delete the file.
        The user is asked to confirm the deletion in the chat UI.
        """
        if confirm is not True:
            return "Deletion cancelled: confirm must be true."
        err = self._check_id(file_id, "file_id")
        if err:
            return err
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            record, e = await self._request(
                s, "GET", "/files/{file_id}", ids={"file_id": file_id}
            )
            if e:
                return e
            name = (record or {}).get("filename") or "Unnamed"
            e = await self._confirm(
                __event_call__,
                "Delete file permanently?",
                f'Permanently delete the file "{name}" (id {file_id})? It is removed '
                "from every knowledge base that uses it, together with its embeddings. "
                "This cannot be undone.",
                confirm,
            )
            if e:
                return e
            _, e = await self._request(
                s,
                "DELETE",
                "/files/{file_id}",
                ids={"file_id": file_id},
                expected=(200, 204),
            )
        return (
            e or f"Knowledge file permanently deleted successfully.\nfile_id={file_id}"
        )

    async def delete_knowledge_file_at_path(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        path: str = Field(
            ..., description="Exact path of the file to permanently delete."
        ),
        confirm: bool = Field(
            ...,
            description="Must be true. Use true only when the user explicitly requested permanent deletion of this file.",
        ),
        __request__=None,
        __event_call__=None,
    ) -> str:
        """
        Permanently delete an existing knowledge file by its exact path.

        Safety:
        - confirm must be true and the user confirms in the chat UI;
        - exact directory path and filename must resolve;
        - ambiguous matches are never deleted.
        """
        if confirm is not True:
            return "Deletion cancelled: confirm must be true."
        parts, err = self._path_parts(path)
        if err:
            return err
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            directory, err = await self._resolve_directory(s, knowledge_id, parts[:-1])
            if err:
                return err
            item, err = await self._find_file_in_directory(
                s, knowledge_id, directory, parts[-1]
            )
            if err:
                return err
            err = self._check_id(item.get("id"), "file_id")
            if err:
                return err
            err = await self._confirm(
                __event_call__,
                "Delete file permanently?",
                f"Permanently delete \"{'/'.join(parts)}\" (id {item.get('id')})? It is "
                "removed from every knowledge base that uses it, together with its "
                "embeddings. This cannot be undone.",
                confirm,
            )
            if err:
                return err
            _, err = await self._request(
                s,
                "DELETE",
                "/files/{file_id}",
                ids={"file_id": item.get("id")},
                expected=(200, 204),
            )
        return (
            err
            or f"Knowledge file permanently deleted successfully.\npath={path}\nfile_id={item.get('id')}"
        )

    async def delete_knowledge_directory(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        directory_id: str = Field(..., description="ID of the directory to delete."),
        __request__=None,
        __event_call__=None,
    ) -> str:
        """
        Delete a directory from an Open WebUI knowledge base.

        Nested subdirectories are removed too; all files inside are moved to
        the parent directory instead of being deleted. The user is asked to confirm in the chat UI.
        """
        err = self._check_ids(knowledge_id=knowledge_id, directory_id=directory_id)
        if err:
            return err
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            listing, e = await self._directory_listing(s, knowledge_id, directory_id)
            if e:
                return e
            name = (
                next(
                    (
                        b.get("name")
                        for b in listing["breadcrumbs"]
                        if b.get("id") == directory_id
                    ),
                    None,
                )
                or directory_id
            )
            e = await self._confirm(
                __event_call__,
                "Delete knowledge directory?",
                f'Delete the directory "{name}" (id {directory_id}) and its '
                f"{len(listing['directories'])} direct subdirectory(ies)? Files inside "
                "(including nested ones) are kept and moved to the parent directory.",
            )
            if e:
                return e
            _, e = await self._request(
                s,
                "DELETE",
                "/knowledge/{knowledge_id}/dirs/{directory_id}/delete",
                ids={"knowledge_id": knowledge_id, "directory_id": directory_id},
                expected=(200, 204),
                params={"move_files": "true"},
            )
        return (
            e
            or f"Knowledge directory deleted successfully.\ndirectory_id={directory_id}"
        )

    async def search_knowledge_files(
        self,
        query: str = Field(
            ...,
            description="Filename or text to search for across all accessible knowledge bases.",
        ),
        include_content: bool = False,
        page: int = 1,
        max_content_items: int = 10,
        __request__=None,
    ) -> str:
        """
        Search files across all Open WebUI knowledge bases available
        to the current user.

        By default searches filenames.
        Set include_content=true to also match the extracted file text and to
        return a content preview for found files.
        max_content_items limits how many files will have their content fetched (default: 10).
        """
        query = query.strip()
        if not query:
            return "Error: search query is empty."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        page = max(1, page)
        max_content_items = max(1, min(max_content_items, 50))

        async with await self._open_session(__request__) as s:
            data, error = await self._request(
                s,
                "GET",
                "/knowledge/search/files",
                params={
                    "query": query,
                    "include_content": str(include_content).lower(),
                    "page": page,
                },
            )
            if error:
                return error

            items = (data or {}).get("items", [])
            if not items:
                return f"No matching knowledge files found.\nquery={query}\npage={page}"

            result = [f"Search results for: {query}", f"page={page}", ""]
            for i, item in enumerate(items):
                file_data = self._file_item(item)
                filename = file_data.get("filename", file_data.get("name", "Unnamed"))
                file_id = file_data.get("id")
                result.append(f"- {filename} (file_id={file_id})")

                if include_content and file_id and i < max_content_items:
                    # Some Open WebUI versions do not return the text in the search
                    # response. Therefore, with include_content we fetch it with
                    # a separate request.
                    content_data, content_error = await self._request(
                        s,
                        "GET",
                        "/files/{file_id}/data/content",
                        ids={"file_id": file_id},
                    )
                    if content_error:
                        result.append(f"  content_preview_error={content_error}")
                    else:
                        content = (content_data or {}).get("content", "")
                        result.append(
                            f"  content_preview={content[:1000]}"
                            if content
                            else "  content_preview=<empty>"
                        )

        return "\n".join(result)

    async def find_and_read_knowledge_file(
        self,
        query: str = Field(
            ..., description="Filename or search text used to find the knowledge file."
        ),
        exact_filename: str = "",
        __request__=None,
    ) -> str:
        """
        Find a file across all accessible Open WebUI knowledge bases
        and immediately return its full extracted/indexed text content.

        If exact_filename is provided, prefer an exact filename match.
        If several files still match, return their IDs instead of guessing.
        """
        query = (query or "").strip()
        exact_filename = (exact_filename or "").strip()
        if not query:
            return "Error: search query is empty."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        # Search results come 30 per page; when an exact filename is requested,
        # look through more pages so the match is not missed.
        max_pages = 20 if exact_filename else 1
        async with await self._open_session(__request__) as s:
            items = []
            for page in range(1, max_pages + 1):
                d, e = await self._request(
                    s,
                    "GET",
                    "/knowledge/search/files",
                    params={
                        "query": query,
                        "include_content": "false",
                        "page": page,
                    },
                )
                if e:
                    return e
                batch = (d or {}).get("items", [])
                items.extend(batch)
                total = (d or {}).get("total")
                if not batch or total is None or len(items) >= total:
                    break
            if exact_filename:
                exact = [
                    x
                    for x in items
                    if x.get("filename", "").casefold() == exact_filename.casefold()
                ]
                if exact:
                    items = exact
            if not items:
                return "No matching knowledge file found."
            if len(items) > 1:
                return "Multiple matching files found:\n" + "\n".join(
                    f"- {x.get('filename','Unnamed')} (file_id={x.get('id')})"
                    for x in items
                )
            x = items[0]
            data, e = await self._request(
                s, "GET", "/files/{file_id}/data/content", ids={"file_id": x.get("id")}
            )
        return (
            e
            or f"filename={x.get('filename','Unnamed')}\nfile_id={x.get('id')}\n\n{(data or {}).get('content','')}"
        )

    async def get_knowledge_tree(
        self,
        knowledge_id: str = Field(
            ..., description="ID of the Open WebUI knowledge base."
        ),
        max_depth: int = 20,
        max_nodes: int = 1000,
        __request__=None,
    ) -> str:
        """
        Return the complete directory/file tree of an Open WebUI knowledge base.

        Includes directory IDs and file IDs.
        Traverses nested directories recursively.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        max_depth = max(1, min(max_depth, 50))
        max_nodes = max(1, min(max_nodes, 5000))
        lines = []
        count = 0
        visited = set()
        async with await self._open_session(__request__) as s:
            meta, e = await self._request(
                s,
                "GET",
                "/knowledge/{knowledge_id}",
                ids={"knowledge_id": knowledge_id},
            )
            if e:
                return e
            lines.append(
                f"{meta.get('name','Knowledge Base')} (knowledge_id={knowledge_id})"
            )

            async def walk(directory, depth):
                nonlocal count
                if count >= max_nodes or depth > max_depth:
                    return
                key = directory or "ROOT"
                if key in visited:
                    return
                visited.add(key)
                listing, e = await self._directory_listing(s, knowledge_id, directory)
                if e:
                    lines.append("  " * depth + e)
                    return
                for x in listing["items"]:
                    if count >= max_nodes:
                        return
                    count += 1
                    f = self._file_item(x)
                    lines.append(
                        "  " * depth
                        + f"📄 {f.get('filename','Unnamed')} (file_id={f.get('id')})"
                    )
                for x in listing["directories"]:
                    if count >= max_nodes:
                        return
                    count += 1
                    lines.append(
                        "  " * depth
                        + f"📁 {x.get('name','Unnamed')} (directory_id={x.get('id')})"
                    )
                    await walk(x.get("id"), depth + 1)

            await walk("", 1)
        if count >= max_nodes:
            lines.append(f"\nTree truncated after {max_nodes} nodes.")
        return "\n".join(lines)

    async def delete_knowledge_base(
        self,
        knowledge_id: str = Field(
            ...,
            description="ID of the Open WebUI knowledge base to permanently delete.",
        ),
        confirm: bool = Field(
            ...,
            description="Must be true. Use true only when the user explicitly requested deletion of the knowledge base.",
        ),
        allow_nonempty: bool = False,
        __request__=None,
        __event_call__=None,
    ) -> str:
        """
        Permanently delete an Open WebUI knowledge base.

        Safety rules:
        - confirm must be true and the user confirms in the chat UI;
        - by default refuses to delete a knowledge base containing files;
        - set allow_nonempty=true only when the user explicitly wants
          a non-empty knowledge base deleted.
        """
        if confirm is not True:
            return "Deletion cancelled: confirm must be true."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            meta, e = await self._request(
                s,
                "GET",
                "/knowledge/{knowledge_id}",
                ids={"knowledge_id": knowledge_id},
            )
            if e:
                return e
            has_content, e = await self._knowledge_has_content(s, knowledge_id)
            if e:
                return e
            if has_content and not allow_nonempty:
                return "Deletion refused: knowledge base contains files. Use allow_nonempty=true only when explicitly intended."
            e = await self._confirm(
                __event_call__,
                "Delete knowledge base permanently?",
                f"Permanently delete the knowledge base \"{meta.get('name', 'Unnamed')}\" "
                f"(id {knowledge_id})"
                + (
                    " including all of its directories and file links"
                    if has_content
                    else " (it contains no files)"
                )
                + "? Its search index is removed. This cannot be undone.",
                confirm,
            )
            if e:
                return e
            _, e = await self._request(
                s,
                "DELETE",
                "/knowledge/{knowledge_id}/delete",
                ids={"knowledge_id": knowledge_id},
                expected=(200, 204),
            )
        return (
            e
            or f"Knowledge base permanently deleted successfully.\nknowledge_id={knowledge_id}\nname={meta.get('name','Unnamed')}"
        )
