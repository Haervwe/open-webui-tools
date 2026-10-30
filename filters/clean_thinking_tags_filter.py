"""
title: Clear thinking tags
description: Checks if the thinking tag was not closed in the final message and puts the thinking content as a message instead of leaving it as an incomplete thought.
author: Haervwe
author_url: https://github.com/haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.2.1
required_open_webui_version: 0.5.0
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

    def _fix_output_items(self, message: Dict[str, Any]) -> bool:
        """
        Open WebUI 0.12+ stores reasoning as separate `output` items and renders
        from `output`. If the output ends in a reasoning item with no visible
        answer after it, turn that reasoning item into a message item.
        """
        output = message.get("output")
        if not isinstance(output, list) or not output:
            return False

        def _item_text(item: Dict[str, Any]) -> str:
            parts = item.get("content") or []
            if not isinstance(parts, list):
                return ""
            return "".join(
                str(part.get("text"))
                for part in parts
                if isinstance(part, dict) and part.get("text") is not None
            )

        # Ignore trailing empty message items when looking for the last real item
        last_index = len(output) - 1
        while (
            last_index >= 0
            and isinstance(output[last_index], dict)
            and output[last_index].get("type") == "message"
            and not _item_text(output[last_index]).strip()
        ):
            last_index -= 1

        if last_index < 0:
            return False
        last_item = output[last_index]
        if not isinstance(last_item, dict) or last_item.get("type") != "reasoning":
            return False

        # Reasoning items hold plain text (no blockquote), so only drop stray
        # </think> tags; keep per-line indentation and real markdown quotes.
        cleaned_text = re.sub(
            r"</think>", "", _item_text(last_item), flags=re.IGNORECASE
        ).strip()
        if not cleaned_text:
            return False

        print("Outlet filter: Trailing reasoning output item with no answer found.")
        output[last_index:] = [
            {
                "type": "message",
                "id": last_item.get("id") or "msg-clean-thinking",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": cleaned_text}],
            }
        ]
        # Keep content in sync so OWUI persists it and the frontend picks it up
        message["content"] = "\n".join(
            text
            for text in (
                _item_text(item)
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
            )
            if text.strip()
        )
        return True

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

        if last_message.get("role") == "assistant" and last_message.get("output"):
            if self._fix_output_items(last_message):
                print("Outlet filter: Last message output updated successfully.")
            return body

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
