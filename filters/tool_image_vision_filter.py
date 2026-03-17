import logging
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field

from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.config import UPLOAD_DIR
from open_webui.constants import TASKS

logger = logging.getLogger(__name__)

class Filter:
    class Valves(BaseModel):
        max_turns: int = Field(
            default=-1,
            description="How many past messages to scan for tool images (-1 for all)",
        )
        show_status: bool = Field(
            default=True,
            description="Show a status message with the count of attached images",
        )

    def __init__(self):
        self.valves = self.Valves()

    def get_base64_from_file_id(self, file_id: str) -> Optional[str]:
        """Resolves a file ID to a base64 data URL."""
        try:
            file = Files.get_file_by_id(file_id)
            if not file or not file.path:
                return None
            
            file_path = Path(file.path)
            if not file_path.is_absolute():
                file_path = UPLOAD_DIR / file_path
                
            if not file_path.exists():
                return None
                
            with open(file_path, "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode("utf-8")
                mime_type = file.meta.get("content_type", "image/png") if file.meta else "image/png"
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logging.error(f"ToolImageVisionFilter: Failed to resolve base64 for {file_id}: {e}")
            return None

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __chat_id__: Optional[str] = None,
        __task__: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info(f"ToolImageVisionFilter: Ingress - task={__task__}, chat_id={__chat_id__}")
        
        # Only run for default task
        if __task__ and __task__ != TASKS.DEFAULT:
            return body

        if not __chat_id__:
            return body

        # Check for vision support in the model metadata
        metadata = body.get("metadata", {})
        model_info = metadata.get("model", {})
        meta = model_info.get("info", {}).get("meta", {})
        capabilities = meta.get("capabilities", {})
        
        if not capabilities.get("vision", False):
            logger.info(f"ToolImageVisionFilter: Model {body.get('model')} does not support vision, skipping projection")
            return body

        try:
            # Fetch message map once to identify files attached to messages in history
            messages_map = Chats.get_messages_map_by_chat_id(__chat_id__)
            logger.info(f"ToolImageVisionFilter: Fetched {len(messages_map)} messages from DB")
        except Exception as e:
            logger.warning(f"ToolImageVisionFilter: Failed to get messages map: {e}")
            return body

        # 1. Reconstruct the active branch using parentId chain
        body_messages = body.get("messages", [])
        if not body_messages:
            logger.info("ToolImageVisionFilter: No body messages, skipping")
            return body
            
        logger.info(f"ToolImageVisionFilter: Body messages count: {len(body_messages)}")
        for i, m in enumerate(body_messages[-3:]): # Check last few
            logger.info(f"ToolImageVisionFilter: Body message[{len(body_messages)-3+i}] id={m.get('id')}, role={m.get('role')}")

        # The last message in body (the new input) often has no ID yet.
        # We search from the end of body_messages for the first message with an ID.
        start_id = None
        for i in range(len(body_messages) - 1, -1, -1):
            mid = body_messages[i].get("id")
            if mid and mid in messages_map:
                start_id = mid
                break
        
        if not start_id and messages_map:
            logger.info("ToolImageVisionFilter: No body ID found in map. Falling back to latest DB message.")
            # Fallback to the message with the latest timestamp or just one that has no children
            try:
                # Simplest heuristic: latest message by timestamp
                sorted_msgs = sorted(
                    messages_map.values(), 
                    key=lambda x: x.get("timestamp", 0), 
                    reverse=True
                )
                if sorted_msgs:
                    start_id = sorted_msgs[0].get("id")
            except Exception as e:
                logger.error(f"ToolImageVisionFilter: Fallback failed: {e}")

        logger.info(f"ToolImageVisionFilter: Final Chain start_id={start_id}")
        
        active_chain = []
        visited = set()
        current_id = start_id
        while current_id and current_id in messages_map and current_id not in visited:
            visited.add(current_id)
            msg = messages_map[current_id]
            active_chain.append(msg)
            current_id = msg.get("parentId")
            
        # Reverse because we built it bottom-up
        active_chain.reverse() 
        logger.info(f"ToolImageVisionFilter: Reconstructed active branch with {len(active_chain)} messages")

        # 2. Combine for full scan list
        # We want to scan images from BOTH the volatile body (new turns) and the DB chain (history)
        scan_list = []
        seen_ids = set()
        
        # Add body messages (highest priority)
        for m in reversed(body_messages):
            mid = m.get("id")
            if mid: seen_ids.add(mid)
            scan_list.append(m)
            
        # Add DB chain messages (history)
        for m in reversed(active_chain):
            mid = m.get("id")
            if mid and mid not in seen_ids:
                scan_list.append(m)
                
        logger.info(f"ToolImageVisionFilter: Total messages to scan: {len(scan_list)}")

        # 2. Identify all user messages in the body to receive projected vision context
        body_user_messages = [m for m in body_messages if m.get("role") == "user"]
        for user_msg in body_user_messages:
            # Ensure content is in multimodal format (list of parts)
            content = user_msg.get("content", "")
            if isinstance(content, str):
                user_msg["content"] = [{"type": "text", "text": content}] if content else []
            elif not isinstance(content, list):
                user_msg["content"] = []

        max_scan = self.valves.max_turns
        total_images_synced = 0
        scanned = 0
        
        # Track already injected image URLs for each user message in the body
        # (Using msg['id'] if available, or its object ID as a fallback)
        injected_urls_per_msg = {}
        
        # 3. Scan and Attach
        for i, msg in enumerate(scan_list):
            if max_scan != -1 and scanned >= max_scan:
                break
            scanned += 1
            
            role = msg.get("role")
            msg_id = msg.get("id")
            
            if role not in ["assistant", "tool"]:
                continue
                
            # Check for images
            files = msg.get("files") or []
            image_files = [
                f for f in files 
                if f.get("type") == "image" or (f.get("content_type") or "").startswith("image/")
            ]
            
            if not image_files:
                continue
                
            logger.info(f"ToolImageVisionFilter: FOUND {len(image_files)} images in message {msg_id} (role={role})")
                
            # A. Retroactive DB Sync: Find the original user turn preceding these images
            orig_user_msg = None
            for j in range(i + 1, len(scan_list)):
                if scan_list[j].get("role") == "user":
                    orig_user_msg = scan_list[j]
                    break
            
            if orig_user_msg:
                orig_msg_id = orig_user_msg.get("id")
                db_files = []
                if orig_msg_id and orig_msg_id in messages_map:
                    db_files = messages_map[orig_msg_id].get("files") or []
                
                db_ids = {f.get("id") for f in db_files if f.get("id")}
                files_to_db = [f for f in image_files if f.get("id") and f.get("id") not in db_ids]
                
                if files_to_db and orig_msg_id:
                    try:
                        logger.info(f"ToolImageVisionFilter: Syncing {len(files_to_db)} files to DB message {orig_msg_id}")
                        Chats.add_message_files_by_id_and_message_id(__chat_id__, orig_msg_id, files_to_db)
                    except Exception as e:
                        logger.error(f"ToolImageVisionFilter: DB sync failed: {e}")

            # B. Body Multimodal Injection: Project into the most appropriate USER message
            # logic: closest FOLLOWER user message, fallback to closest PRECEDER user message
            target_user_msg = None
            
            # 1. Search for nearest follower (indices 0 to i-1)
            for j in range(i - 1, -1, -1):
                m = scan_list[j]
                if m.get("role") == "user" and m in body_messages:
                    target_user_msg = m
                    break
            
            # 2. Search for nearest preceder (indices i+1 to len-1)
            if not target_user_msg:
                for j in range(i + 1, len(scan_list)):
                    m = scan_list[j]
                    if m.get("role") == "user" and m in body_messages:
                        target_user_msg = m
                        break

            if target_user_msg:
                msg_key = id(target_user_msg)
                if msg_key not in injected_urls_per_msg:
                    injected_urls_per_msg[msg_key] = set()
                    # Also include existing image URLs in that message
                    for part in target_user_msg["content"]:
                        if part.get("type") == "image_url" and "image_url" in part:
                            injected_urls_per_msg[msg_key].add(part["image_url"]["url"])

                content_parts = target_user_msg["content"]
                
                for f in image_files:
                    f_url = f.get("url", "")
                    f_id = f.get("id")
                    
                    # If ID is missing, try to extract it from the URL (/api/v1/files/{id}/content)
                    if not f_id and "/api/v1/files/" in f_url:
                        parts = f_url.split("/")
                        for idx, p in enumerate(parts):
                            if p == "files" and idx + 1 < len(parts):
                                f_id = parts[idx+1]
                                break

                    if not f_id:
                        logger.warning(f"ToolImageVisionFilter: Could not find/extract ID for {f_url}, skipping vision projection")
                        continue

                    # Resolve to base64 for LLM perception
                    base64_url = self.get_base64_from_file_id(f_id)
                    if not base64_url:
                        logger.warning(f"ToolImageVisionFilter: Could not resolve base64 for {f_id}")
                        continue

                    if base64_url not in injected_urls_per_msg[msg_key]:
                        logger.info(f"ToolImageVisionFilter: Injecting image as base64 into content of user message")
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": base64_url}
                        })
                        total_images_synced += 1
                        injected_urls_per_msg[msg_key].add(base64_url)
                    else:
                        logger.info(f"ToolImageVisionFilter: Image {f_id} already in content, skipping")


        if total_images_synced > 0:
            system_msg = None
            for m in body.get("messages", []):
                if m.get("role") == "system":
                    system_msg = m
                    break
            
            notice = "Images generated on the previous turns are attached to the following user message by default."
            if system_msg:
                content = system_msg.get("content", "")
                if isinstance(content, str):
                    if notice not in content:
                        system_msg["content"] = f"{content}\n\nNOTE: {notice}"
                elif isinstance(content, list):
                    found_text = False
                    for part in content:
                        if part.get("type") == "text":
                            if notice not in part.get("text", ""):
                                part["text"] = f"{part.get('text', '')}\n\nNOTE: {notice}"
                            found_text = True
                            break
                    if not found_text:
                        content.append({"type": "text", "text": f"NOTE: {notice}"})
            else:
                body.get("messages", []).insert(0, {
                    "role": "system",
                    "content": f"NOTE: {notice}"
                })

        if self.valves.show_status and total_images_synced > 0 and __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Linked {total_images_synced} tool image{'s' if total_images_synced > 1 else ''} to vision context", "done": True}
            })

        # Final Summary for Logs
        logger.info("ToolImageVisionFilter: Final body summary:")
        for i, m in enumerate(body.get("messages", [])):
            c = m.get("content", "")
            c_type = "str" if isinstance(c, str) else f"list[{len(c)}]"
            logger.info(f"  msg[{i}] role={m.get('role')}, content={c_type}, files={len(m.get('files') or [])}")

        return body
