"""
title: Full Document Filter
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.0
"""

from pydantic import BaseModel, Field
import re


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
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")

        # Check if files exist
        if body.get("files"):
            for file_info in body["files"]:
                file_data = file_info.get("file", {}).get("data", {})
                content = file_data.get("content")

                if content and body.get("messages"):
                    # Clean the file content
                    cleaned_content = self.clean_text(content)

                    # Prepend cleaned file content to the first message
                    original_content = body["messages"][0]["content"]
                    body["messages"][0][
                        "content"
                    ] = f"{cleaned_content}\n\n{original_content}"

                    # Remove the files from the body
                    body["files"] = []
                    break

        ## print(f"modified body:{body}")
        return body

    def outlet(self, body: dict, user=None) -> dict:
        return body
