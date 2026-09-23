import asyncio
import json
from typing import Any, Optional

import aiohttp
from pydantic import Field


class Tools:
    """Open WebUI knowledge base management tool.

    File indexing remains synchronous: the file creation operation waits
    until Open WebUI finishes processing and indexing.
    """

    def __init__(self):
        self.base_url = "http://127.0.0.1:8080/api/v1"
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

    async def _request(
        self, session, method, path, *, timeout=None, expected=(200,), **kwargs
    ):
        """Unified HTTP request sending and response handling."""
        try:
            async with session.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout or self._short_timeout,
                **kwargs,
            ) as response:
                if response.status not in expected:
                    body = await response.text()
                    return (
                        None,
                        f"Open WebUI API error {response.status}: {body[:2000]}",
                    )
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
            return None, f"Error: Open WebUI connection failed: {exc}"

    def _error(self, message):
        return message

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
        page = 1
        all_items = []
        directories = []
        breadcrumbs = []
        while True:
            data, err = await self._request(
                session,
                "GET",
                f"/knowledge/{knowledge_id}/files",
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
        return aiohttp.ClientSession(headers=headers, cookies=cookies)

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
                session, "GET", f"/knowledge/{knowledge_id}"
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
                f"/knowledge/{knowledge_id}/update",
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
        async with await self._open_session(__request__) as session:
            data, err = await self._request(
                session,
                "POST",
                f"/knowledge/{knowledge_id}/dirs/create",
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
                    reused.append(part)
                    continue
                data, err = await self._request(
                    session,
                    "POST",
                    f"/knowledge/{knowledge_id}/dirs/create",
                    expected=(200, 201),
                    json={"name": part, "parent_id": current or None},
                )
                if err:
                    return err
                current = data.get("id")
                created.append(part)
        return f"Knowledge directory path ready.\npath={'/'.join(parts)}\ndirectory_id={current}\ncreated={', '.join(created) or 'none'}\nreused={', '.join(reused) or 'none'}"

    async def _upload(self, session, knowledge_id, directory_id, filename, content):
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
        return await self._request(
            session,
            "POST",
            "/files/?process=true&process_in_background=false",
            expected=(200, 201),
            data=form,
            timeout=self._long_timeout,
        )

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
        return f"Created and indexed knowledge file successfully.\nfilename={data.get('filename',filename)}\nfile_id={data.get('id')}\nknowledge_id={knowledge_id}\ndirectory_id={directory_id or 'ROOT'}"

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
        return f"Knowledge Markdown file created successfully.\npath={'/'.join(parts[:-1]+[filename])}\nfilename={data.get('filename',filename)}\nfile_id={data.get('id')}\ndirectory_id={directory or 'ROOT'}"

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
                continue
            data, err = await self._request(
                session,
                "POST",
                f"/knowledge/{knowledge_id}/dirs/create",
                expected=(200, 201),
                json={"name": part, "parent_id": current or None},
            )
            if err:
                return None, err
            current = data.get("id")
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
                    f"/files/{item.get('id')}/data/content/update",
                    expected=(200, 201),
                    json={"content": content},
                    timeout=self._long_timeout,
                )
                if err:
                    return err
                return f"Knowledge file updated successfully.\npath={path}\nfile_id={item.get('id')}\ndirectory_id={directory or 'ROOT'}"
            data, err = await self._request(
                session, "GET", f"/files/{item.get('id')}/data/content"
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
                f"/files/{file_id}/data/content/update",
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
            data, e = await self._request(s, "GET", f"/files/{file_id}/data/content")
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
                    f"/files/{item.get('id')}/data/content/update",
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
                f"/files/{file_id}/rename",
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
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            d, e = await self._request(
                s,
                "POST",
                f"/knowledge/{knowledge_id}/dirs/{directory_id}/update",
                expected=(200, 201),
                json={"name": new_name.strip()},
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
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            d, e = await self._request(
                s,
                "POST",
                f"/knowledge/{knowledge_id}/dirs/{directory_id}/update",
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
    ) -> str:
        """
        Permanently delete an Open WebUI knowledge file.

        This removes the file itself, its knowledge-base associations,
        and its indexed embeddings.

        Use only when the user explicitly asks to delete the file.
        """
        if confirm is not True:
            return "Deletion cancelled: confirm must be true."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            _, e = await self._request(
                s, "DELETE", f"/files/{file_id}", expected=(200, 204)
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
    ) -> str:
        """
        Permanently delete an existing knowledge file by its exact path.

        Safety:
        - confirm must be true;
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
            _, err = await self._request(
                s, "DELETE", f"/files/{item.get('id')}", expected=(200, 204)
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
    ) -> str:
        """
        Delete a directory from an Open WebUI knowledge base.

        Files inside the directory are moved to its parent directory
        instead of being deleted.
        """
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            _, e = await self._request(
                s,
                "DELETE",
                f"/knowledge/{knowledge_id}/dirs/{directory_id}/delete",
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

        By default searches metadata/filenames.
        Set include_content=true to return file content preview for found files.
        max_content_items limits how many files will have their content fetched (default: 10).

        Note: Open WebUI search API searches by filename, not by file content.
        Full-text search within file content is not supported by the current API.
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
                        s, "GET", f"/files/{file_id}/data/content"
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
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            d, e = await self._request(
                s,
                "GET",
                "/knowledge/search/files",
                params={
                    "query": query.strip(),
                    "include_content": "false",
                    "page": 1,
                },
            )
            if e:
                return e
            items = (d or {}).get("items", [])
            if exact_filename:
                exact = [
                    x
                    for x in items
                    if x.get("filename", "").casefold()
                    == exact_filename.strip().casefold()
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
                s, "GET", f"/files/{x.get('id')}/data/content"
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
            meta, e = await self._request(s, "GET", f"/knowledge/{knowledge_id}")
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
    ) -> str:
        """
        Permanently delete an Open WebUI knowledge base.

        Safety rules:
        - confirm must be true;
        - by default refuses to delete a knowledge base containing files;
        - set allow_nonempty=true only when the user explicitly wants
          a non-empty knowledge base deleted.
        """
        if confirm is not True:
            return "Deletion cancelled: confirm must be true."
        if __request__ is None:
            return "Error: Open WebUI request context is unavailable."
        async with await self._open_session(__request__) as s:
            meta, e = await self._request(s, "GET", f"/knowledge/{knowledge_id}")
            if e:
                return e
            has_content, e = await self._knowledge_has_content(s, knowledge_id)
            if e:
                return e
            if has_content and not allow_nonempty:
                return "Deletion refused: knowledge base contains files. Use allow_nonempty=true only when explicitly intended."
            _, e = await self._request(
                s,
                "DELETE",
                f"/knowledge/{knowledge_id}/delete",
                expected=(200, 204),
            )
        return (
            e
            or f"Knowledge base permanently deleted successfully.\nknowledge_id={knowledge_id}\nname={meta.get('name','Unnamed')}"
        )
