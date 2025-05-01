"""
title: Clear thinking tags
description: Checks if the thinking tag was not closed in the final message and puts the thinking content as a message instead of leaving it as an incomplete thought.
author: Haervwe
author_url: https://github.com/haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2
"""

import re
from typing import Optional, Dict, Any


class Filter:

    def __init__(self):
        pass

    def _clean_extracted_content(self, text: str) -> str:
        """Helper function to clean the extracted final message content."""

        cleaned_text = re.sub(r"</think>", "", text, flags=re.IGNORECASE).strip()

        lines = cleaned_text.splitlines()
        unquoted_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith(">"):

                while stripped_line.startswith(">"):
                    stripped_line = stripped_line[1:].strip()
                unquoted_lines.append(stripped_line)
            else:
                unquoted_lines.append(stripped_line)

        return "\n".join(unquoted_lines).strip()

    def outlet(
        self, body: Dict[str, Any], __user__: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        print(f"outlet:{__name__}")

        if (
            not isinstance(body, dict)
            or "messages" not in body
            or not isinstance(body.get("messages"), list)
        ):
            print(
                "Outlet filter: Invalid body structure or missing/invalid messages list."
            )
            return body

        messages = body["messages"]
        if not messages:

            return body

        last_message = messages[-1]

        message_content = last_message.get("content", "")
        is_assistant = last_message.get("role") == "assistant"
        is_string_content = isinstance(message_content, str)
        faulty_tag_marker = '<details type="reasoning" done="false">'
        has_faulty_tag = faulty_tag_marker in message_content

        if is_assistant and is_string_content and has_faulty_tag:
            print(
                "Outlet filter: Candidate message found - Assistant role, string content, contains 'reasoning done=false'."
            )

            last_faulty_tag_start_index = message_content.rfind(faulty_tag_marker)

            if last_faulty_tag_start_index != -1:

                pattern = r"<summary>Thinking…</summary>(.*?)(?:</details>)?$"

                substring_to_search = message_content[last_faulty_tag_start_index:]
                match = re.search(
                    pattern, substring_to_search, re.DOTALL | re.IGNORECASE
                )

                if match:
                    print(
                        "Outlet filter: Regex matched expected structure within the last faulty block."
                    )
                    extracted_content_raw = match.group(1)
                    cleaned_final_content = self._clean_extracted_content(
                        extracted_content_raw
                    )

                    content_before_block = message_content[
                        :last_faulty_tag_start_index
                    ].strip()

                    if content_before_block and cleaned_final_content:
                        new_content = (
                            content_before_block + "\n\n" + cleaned_final_content
                        )
                    elif cleaned_final_content:
                        new_content = cleaned_final_content
                    else:
                        new_content = content_before_block

                    print(
                        f"Outlet filter: Original content length: {len(message_content)}"
                    )

                    print(f"Outlet filter: New content length: {len(new_content)}")

                    last_message["content"] = new_content.strip()
                    print("Outlet filter: Last message content updated successfully.")

                else:

                    print(
                        "Outlet filter: Regex did not match expected structure *after* the last 'reasoning done=false' tag. No modification performed."
                    )

        return body
