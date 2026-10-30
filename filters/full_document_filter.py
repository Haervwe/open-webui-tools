"""
title: Full Document Filter
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.1
required_open_webui_version: 0.5.0
"""

import logging
import re

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        max_turns: int = Field(
            default=8, description="Maximum allowable conversation turns for a user."
        )

    class UserValves(BaseModel):
        max_turns: int = Field(
            default=4, description="Maximum allowable conversation turns for a user."
        )

    def __init__(self):
        self.file_handler = True
        self.valves = self.Valves()

    def clean_text(self, text: str) -> str:
        # Remove multiple consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove newlines between words (keep paragraph structure)
        text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)

        # Trim leading and trailing whitespace
        return text.strip()

    def inlet(self, body: dict, user=None) -> dict:
        # Never log the body: it holds the whole chat and the document text.
        log.debug(
            "full_document inlet: %d message(s), %d file(s)",
            len(body.get("messages") or []),
            len(body.get("files") or []),
        )

        # Check if files exist
        if body.get("files"):
            for file_info in body["files"]:
                file_data = file_info.get("file", {}).get("data", {})
                content = file_data.get("content")

                if content and body.get("messages"):
                    # Clean the file content
                    cleaned_content = self.clean_text(content)

                    # Prepend cleaned file content to the first message
                    original_content = body["messages"][0].get("content", "")
                    if isinstance(original_content, list):
                        # Multimodal message: prepend a text part instead of
                        # stringifying the parts list.
                        body["messages"][0]["content"] = [
                            {"type": "text", "text": cleaned_content},
                            *original_content,
                        ]
                    else:
                        body["messages"][0][
                            "content"
                        ] = f"{cleaned_content}\n\n{original_content}"

                    # Remove the files from the body
                    body["files"] = []
                    break

        return body

    def outlet(self, body: dict, user=None) -> dict:
        return body
