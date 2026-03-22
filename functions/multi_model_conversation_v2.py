"""
title: Multi Model Conversations v2
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 2.7.0
"""

import logging
import json
import re
import html as html_module
import ast
from uuid import uuid4
from typing import Callable, Awaitable, Any, Optional
import time
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import User, Users
from open_webui.models.models import Models
from open_webui.utils.tools import get_tools, get_builtin_tools
from open_webui.utils.middleware import process_tool_result
from open_webui.utils.chat import (
    generate_chat_completion as generate_raw_chat_completion,
)

name = "Conversation"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()

THINK_OPEN_PATTERN = re.compile(
    r"<(?:think|thinking|reason|reasoning|thought|Thought)>|\|begin_of_thought\|",
    re.IGNORECASE,
)
THINK_CLOSE_PATTERN = re.compile(
    r"</(?:think|thinking|reason|reasoning|thought|Thought)>|\|end_of_thought\|",
    re.IGNORECASE,
)

SPEAKER_COLORS = ["🔴", "🔵", "🟢", "🟡", "🟣", "🟠", "🟤", "⚫", "⚪"]


def clean_thinking_tags(message: str) -> str:
    complete_pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|"
        r"|"
        r"<details\s+type=[\"']reasoning[\"'][^>]*>.*?</details>",
        re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(complete_pattern, "", message)

    orphan_close_pattern = re.compile(
        r"</(?:think|thinking|reason|reasoning|thought|Thought)>"
        r"|"
        r"\|end_of_thought\|",
        re.IGNORECASE,
    )

    last_match_end = -1
    for match in orphan_close_pattern.finditer(cleaned):
        last_match_end = match.end()

    if last_match_end != -1:
        cleaned = cleaned[last_match_end:]

    orphan_open_pattern = re.compile(
        r"<(?:think|thinking|reason|reasoning|thought|Thought)>"
        r"|"
        r"\|begin_of_thought\|"
        r"|"
        r"<details[^>]*>",
        re.IGNORECASE,
    )
    cleaned = re.sub(orphan_open_pattern, "", cleaned)

    return cleaned.strip()


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: Optional[User]
    __model__: str

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the pipe operations.",
        )

    class UserValves(BaseModel):
        NUM_PARTICIPANTS: int = Field(
            default=2,
            description="Number of participants in the conversation (1-5)",
            ge=1,
            le=5,
        )
        ROUNDS_PER_CONVERSATION: int = Field(
            default=3, description="Number of rounds in the entire conversation", ge=1
        )
        Participant1Model: str = Field(
            default="", description="Model ID for Participant 1"
        )
        Participant1Alias: str = Field(
            default="", description="Alias for Participant 1"
        )
        Participant1SystemMessage: str = Field(
            default="", description="System Message for Participant 1"
        )
        Participant2Model: str = Field(
            default="", description="Model ID for Participant 2"
        )
        Participant2Alias: str = Field(
            default="", description="Alias for Participant 2"
        )
        Participant2SystemMessage: str = Field(
            default="", description="System Message for Participant 2"
        )
        Participant3Model: str = Field(
            default="", description="Model ID for Participant 3"
        )
        Participant3Alias: str = Field(
            default="", description="Alias for Participant 3"
        )
        Participant3SystemMessage: str = Field(
            default="", description="System Message for Participant 3"
        )
        Participant4Model: str = Field(
            default="", description="Model ID for Participant 4"
        )
        Participant4Alias: str = Field(
            default="", description="Alias for Participant 4"
        )
        Participant4SystemMessage: str = Field(
            default="", description="System Message for Participant 4"
        )
        Participant5Model: str = Field(
            default="", description="Model ID for Participant 5"
        )
        Participant5Alias: str = Field(
            default="", description="Alias for Participant 5"
        )
        Participant5SystemMessage: str = Field(
            default="", description="System Message for Participant 5"
        )
        AllParticipantsApendedMessage: str = Field(
            default="Respond only as your specified character and never use your name as title, just output the response as if you really were talking(no one says his name before a phrase), do not go off character in any situation, Your acted response as",
            description="Appended message to all participants internally to prime them properly",
        )
        UseGroupChatManager: bool = Field(
            default=False,
            description="Use Group Chat Manager to select speakers dynamically",
        )
        ManagerModel: str = Field(
            default="",
            description="Model for the Manager (leave empty to use user's default model)",
        )
        ManagerSystemMessage: str = Field(
            default="You are a group chat manager. Your role is to decide who should speak next in a multi-participant conversation. You will be given the conversation history and a list of participant aliases. Choose the alias of the participant who is most likely to provide a relevant and engaging response to the latest message. Consider the context of the conversation, the personalities of the participants, and avoid repeatedly selecting the same participant.",
            description="System message for the Manager",
        )
        ManagerSelectionPrompt: str = Field(
            default="Conversation History:\n{history}\n\nThe last speaker was '{last_speaker}'. Based on the flow of the conversation, who should speak next? Choose exactly one from the following list of participants: {participant_list}\n\nRespond with ONLY the alias of your choice, and nothing else.",
            description="Template for the Manager's selection prompt. Use {history}, {last_speaker}, and {participant_list}.",
        )
        Temperature: float = Field(default=1, description="Models temperature")
        Top_k: int = Field(default=50, description="Models top_k")
        Top_p: float = Field(default=0.8, description="Models top_p")

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    def _extract_config_from_metadata(self, body: dict) -> Optional[dict]:
        containers = []

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            containers.append(metadata)

        chat_metadata = body.get("chat_metadata")
        if isinstance(chat_metadata, dict):
            containers.append(chat_metadata)

        params = body.get("params")
        if isinstance(params, dict):
            params_metadata = params.get("metadata")
            if isinstance(params_metadata, dict):
                containers.append(params_metadata)

        for container in containers:
            config = container.get("multi_model_config")
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except json.JSONDecodeError:
                    continue
            if isinstance(config, dict):
                return config
        return None

    def _persist_config_to_metadata(self, body: dict, config: dict) -> None:
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            body["metadata"] = metadata
        metadata["multi_model_config"] = config

    def _build_default_config_from_valves(self, valves) -> dict:
        participants = []
        for i in range(1, valves.NUM_PARTICIPANTS + 1):
            model = getattr(valves, f"Participant{i}Model", "")
            if model:
                participants.append(
                    {
                        "model": model,
                        "alias": getattr(valves, f"Participant{i}Alias", "") or model,
                        "system_message": getattr(
                            valves, f"Participant{i}SystemMessage", ""
                        ),
                    }
                )

        return {
            "rounds": valves.ROUNDS_PER_CONVERSATION,
            "use_manager": valves.UseGroupChatManager,
            "manager_model": valves.ManagerModel,
            "participants": participants,
        }

    def _sanitize_config(self, config: Optional[dict], valves) -> dict:
        if not isinstance(config, dict):
            return self._build_default_config_from_valves(valves)

        participants = []
        for participant in config.get("participants", []):
            if not isinstance(participant, dict):
                continue
            model = str(participant.get("model", "")).strip()
            if not model:
                continue
            alias = str(participant.get("alias", "")).strip() or model
            system_message = str(participant.get("system_message", "")).strip()
            participants.append(
                {
                    "model": model,
                    "alias": alias,
                    "system_message": system_message,
                }
            )

        rounds = config.get("rounds", valves.ROUNDS_PER_CONVERSATION)
        try:
            rounds = int(rounds)
        except (TypeError, ValueError):
            rounds = valves.ROUNDS_PER_CONVERSATION
        rounds = max(1, rounds)

        use_manager = bool(config.get("use_manager", valves.UseGroupChatManager))

        manager_model = str(config.get("manager_model", "") or valves.ManagerModel or "").strip()

        return {
            "rounds": rounds,
            "use_manager": use_manager,
            "manager_model": manager_model,
            "participants": participants,
        }

    def _normalize_alias(self, alias: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", (alias or "").lower()).strip()
        return re.sub(r"\s+", " ", cleaned)

    def _build_config_js(self, default_valves: dict) -> str:
        # NOTE: This is a plain string, NOT an f-string.
        # We inject the defaults JSON via simple string replace to avoid Python
        # misinterpreting JS object literals like {id: ''} as f-string expressions
        # (since `id` is a Python builtin, f"{id: ''}" raises a TypeError).
        defaults_json = json.dumps(default_valves)
        js_code = r"""
return (function() {
  return new Promise(async (resolve) => {
    // Fetch models
    let availableModels = [];
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/models', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const json = await res.json();
        availableModels = json.data || [];
      }
    } catch(e) {
      console.error('Failed to fetch models', e);
    }

    const defaults = __DEFAULTS_JSON__;
    const MAX_PARTICIPANTS = 5;

    // Create UI overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed; inset: 0; z-index: 999999;
      background: rgba(0,0,0,0.6); backdrop-filter: blur(12px);
      display: flex; align-items: center; justify-content: center;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    `;

    const panel = document.createElement('div');
    panel.style.cssText = `
      background: rgba(20, 20, 25, 0.7); backdrop-filter: blur(20px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 24px; width: 95vw; max-width: 720px;
      max-height: 90vh; overflow-y: auto;
      box-shadow: 0 16px 48px rgba(0,0,0,0.4);
      display: flex; flex-direction: column; gap: 20px;
      color: #e2e8f0; scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;
    `;
    overlay.appendChild(panel);

    const header = document.createElement('div');
    header.style.cssText = 'display: flex; flex-direction: column; gap: 4px;';
    const title = document.createElement('h2');
    title.textContent = '\u2728 Multi-Model Conversation';
    title.style.cssText = 'margin: 0; font-size: 20px; font-weight: 600; color: #fff; letter-spacing: -0.5px;';
    const subtitle = document.createElement('p');
    subtitle.textContent = 'Configure participants and conversation rules.';
    subtitle.style.cssText = 'margin: 0; font-size: 13px; color: #94a3b8;';
    header.appendChild(title);
    header.appendChild(subtitle);
    panel.appendChild(header);

    const form = document.createElement('div');
    form.style.cssText = 'display: flex; flex-direction: column; gap: 16px;';
    panel.appendChild(form);

    function createInputGrp(labelText, inputEl) {
      const grp = document.createElement('div');
      grp.style.cssText = 'display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 140px;';
      const lbl = document.createElement('label');
      lbl.textContent = labelText;
      lbl.style.cssText = 'font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;';
      grp.appendChild(lbl);
      inputEl.style.cssText = 'background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: #f8fafc; padding: 10px 14px; border-radius: 8px; font-size: 14px; outline: none; transition: all 0.2s; font-family: inherit; width: 100%; box-sizing: border-box;';
      inputEl.onfocus = () => { inputEl.style.borderColor = 'rgba(255,255,255,0.3)'; inputEl.style.background = 'rgba(0,0,0,0.4)'; };
      inputEl.onblur = () => { inputEl.style.borderColor = 'rgba(255,255,255,0.1)'; inputEl.style.background = 'rgba(0,0,0,0.3)'; };
      grp.appendChild(inputEl);
      return grp;
    }

    function createSelectGrp(labelText, options, defaultVal) {
      const sel = document.createElement('select');
      options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt.id; o.textContent = opt.name;
        sel.appendChild(o);
      });
      if (defaultVal) sel.value = defaultVal;
      return createInputGrp(labelText, sel);
    }

    const globalRow = document.createElement('div');
    globalRow.style.cssText = 'display: flex; gap: 16px; flex-wrap: wrap; background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); align-items: flex-end;';

    const roundsInp = document.createElement('input');
    roundsInp.type = 'number'; roundsInp.min = 1; roundsInp.max = 20; roundsInp.value = defaults.ROUNDS_PER_CONVERSATION || 3;
    globalRow.appendChild(createInputGrp('Rounds', roundsInp));

    const numPartsInp = document.createElement('input');
    numPartsInp.type = 'number'; numPartsInp.min = 1; numPartsInp.max = MAX_PARTICIPANTS; numPartsInp.value = defaults.NUM_PARTICIPANTS || 2;
    globalRow.appendChild(createInputGrp('Total Participants', numPartsInp));

    const managerDiv = document.createElement('div');
    managerDiv.style.cssText = 'display: flex; align-items: center; gap: 8px; height: 40px;';
    const managerChk = document.createElement('input');
    managerChk.type = 'checkbox'; managerChk.checked = defaults.UseGroupChatManager || false;
    managerChk.style.cssText = 'width: 18px; height: 18px; accent-color: #3b82f6; cursor: pointer;';
    const managerLbl = document.createElement('label');
    managerLbl.textContent = 'Auto-pilot (Use Manager)';
    managerLbl.style.cssText = 'font-size: 14px; font-weight: 500; color: #cbd5e1; cursor: pointer;';
    managerLbl.onclick = () => { managerChk.checked = !managerChk.checked; toggleManagerModel(); };
    managerDiv.appendChild(managerChk); managerDiv.appendChild(managerLbl);
    globalRow.appendChild(managerDiv);

    form.appendChild(globalRow);

    const modelOptions = [{id: '', name: 'Select Model...'}, ...availableModels.map(m => ({id: m.id, name: m.name}))];

    // Manager model selector (shown only when manager is enabled)
    const managerModelRow = document.createElement('div');
    managerModelRow.style.cssText = 'display: none; gap: 16px; flex-wrap: wrap; background: rgba(59,130,246,0.08); padding: 16px; border-radius: 12px; border: 1px solid rgba(59,130,246,0.2); align-items: flex-end;';
    const managerModelGrp = createSelectGrp('Manager Model (selects next speaker)', modelOptions, defaults.ManagerModel || '');
    const managerModelSel = managerModelGrp.querySelector('select');
    managerModelRow.appendChild(managerModelGrp);
    form.appendChild(managerModelRow);

    function toggleManagerModel() {
      managerModelRow.style.display = managerChk.checked ? 'flex' : 'none';
    }
    managerChk.onchange = toggleManagerModel;
    toggleManagerModel();

    const partsCont = document.createElement('div');
    partsCont.style.cssText = 'display: flex; flex-direction: column; gap: 12px;';
    form.appendChild(partsCont);

    const partUIs = [];

    function renderParticipants() {
      partsCont.innerHTML = '';
      partUIs.length = 0;
      let count = parseInt(numPartsInp.value);
      if (isNaN(count) || count < 1) count = 1;
      if (count > MAX_PARTICIPANTS) count = MAX_PARTICIPANTS;

      for (let i = 1; i <= count; i++) {
        const pbox = document.createElement('div');
        pbox.style.cssText = 'background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 16px; border-radius: 12px; display: flex; flex-direction: column; gap: 16px; transition: all 0.2s;';

        const ptit = document.createElement('div');
        ptit.textContent = `Participant ${i}`;
        ptit.style.cssText = 'font-size: 13px; font-weight: 600; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.8;';
        pbox.appendChild(ptit);

        const row = document.createElement('div');
        row.style.cssText = 'display: flex; gap: 16px; flex-wrap: wrap;';

        const defModel = defaults[`Participant${i}Model`] || '';
        const selGrp = createSelectGrp('Model', modelOptions, defModel);
        row.appendChild(selGrp);

        const defAlias = defaults[`Participant${i}Alias`] || '';
        const aliasInp = document.createElement('input');
        aliasInp.type = 'text'; aliasInp.value = defAlias; aliasInp.placeholder = 'e.g. Alice';
        row.appendChild(createInputGrp('Alias / Name', aliasInp));

        pbox.appendChild(row);

        const defSys = defaults[`Participant${i}SystemMessage`] || '';
        const sysInp = document.createElement('textarea');
        sysInp.value = defSys; sysInp.rows = 2;
        sysInp.placeholder = 'You are a helpful assistant...';
        sysInp.style.resize = 'vertical';
        pbox.appendChild(createInputGrp('System Prompt / Character Sheet', sysInp));

        partsCont.appendChild(pbox);
        partUIs.push({
          sel: selGrp.querySelector('select'),
          alias: aliasInp,
          sys: sysInp
        });
      }
    }
    numPartsInp.oninput = renderParticipants;
    renderParticipants();

    const actions = document.createElement('div');
    actions.style.cssText = 'display: flex; gap: 12px; margin-top: 8px; justify-content: flex-end;';

    const btnCancel = document.createElement('button');
    btnCancel.textContent = 'Skip / Use Defaults';
    btnCancel.style.cssText = 'background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.1); color: #fff; padding: 10px 20px; border-radius: 8px; font-size: 14px; cursor: pointer; transition: all 0.2s; font-weight: 500; font-family: inherit;';
    btnCancel.onmouseenter = () => { btnCancel.style.background = 'rgba(255,255,255,0.15)'; btnCancel.style.borderColor = 'rgba(255,255,255,0.2)'; };
    btnCancel.onmouseleave = () => { btnCancel.style.background = 'rgba(255,255,255,0.1)'; btnCancel.style.borderColor = 'rgba(255,255,255,0.1)'; };
    btnCancel.onclick = () => { cleanup(); resolve(null); };
    actions.appendChild(btnCancel);

    const btnStart = document.createElement('button');
    btnStart.textContent = 'Start Conversation \u2728';
    btnStart.style.cssText = 'background: linear-gradient(135deg, #3b82f6, #2563eb); border: none; color: #fff; padding: 10px 24px; border-radius: 8px; font-size: 14px; cursor: pointer; transition: all 0.2s; font-weight: 600; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); font-family: inherit; letter-spacing: 0.2px;';
    btnStart.onmouseenter = () => { btnStart.style.transform = 'translateY(-1px)'; btnStart.style.boxShadow = '0 6px 16px rgba(59, 130, 246, 0.5)'; };
    btnStart.onmouseleave = () => { btnStart.style.transform = 'none'; btnStart.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)'; };
    btnStart.onclick = () => {
      const config = {
        rounds: parseInt(roundsInp.value) || 3,
        use_manager: managerChk.checked,
        manager_model: managerChk.checked ? managerModelSel.value : '',
        participants: partUIs.map(ui => ({
          model: ui.sel.value,
          alias: ui.alias.value.trim() || (ui.sel.selectedIndex > 0 ? ui.sel.options[ui.sel.selectedIndex].text : 'Participant'),
          system_message: ui.sys.value.trim()
        })).filter(p => p.model !== '')
      };
      cleanup();
      resolve(config);
    };
    actions.appendChild(btnStart);
    panel.appendChild(actions);

    function cleanup() {
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }
    document.body.appendChild(overlay);
  });
})()
""".replace("__DEFAULTS_JSON__", defaults_json)
        return js_code

    def _parse_sse_events(self, buffer: str) -> tuple[list[dict], str, bool]:
        events = []
        done = False

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                stripped = line.strip()
                if stripped.startswith("data:"):
                    data_lines.append(stripped[5:].lstrip())

            if not data_lines:
                continue

            payload = "\n".join(data_lines).strip()
            if not payload:
                continue

            if payload == "[DONE]":
                done = True
                break

            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    events.append(parsed)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed SSE payload: {payload[:200]}")

        return events, buffer, done

    def _extract_stream_events(self, event_payload: dict):
        choices = event_payload.get("choices", [])
        if not choices:
            return

        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta", {}) or {}

        reasoning_keys = ["reasoning", "reasoning_content", "thinking"]
        for reasoning_key in reasoning_keys:
            reasoning_text = delta.get(reasoning_key)
            if isinstance(reasoning_text, str) and reasoning_text:
                yield {"type": "reasoning", "text": reasoning_text}

        content = delta.get("content")
        if isinstance(content, str) and content:
            yield {"type": "content", "text": content}

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            yield {"type": "tool_calls", "data": tool_calls}

    def _replace_thinking_tags(self, text: str) -> str:
        """Replace raw thinking tags with <details> HTML the frontend understands."""
        text = THINK_OPEN_PATTERN.sub(
            '<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n',
            text,
        )
        text = THINK_CLOSE_PATTERN.sub("\n</details>\n\n", text)
        return text

    async def get_streaming_completion(
        self, messages, model: str, valves, tools_specs=None
    ):
        try:
            form_data = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": valves.Temperature,
                "top_k": valves.Top_k,
                "top_p": valves.Top_p,
            }
            if tools_specs:
                form_data["tools"] = tools_specs
            response = await generate_raw_chat_completion(
                self.__request__,
                form_data,
                user=self.__user__,
                bypass_filter=True,
                bypass_system_prompt=True,
            )
            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")

            sse_buffer = ""
            async for chunk in response.body_iterator:
                decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                sse_buffer += decoded
                events, sse_buffer, done = self._parse_sse_events(sse_buffer)
                for event_payload in events:
                    for event in self._extract_stream_events(event_payload):
                        yield event
                if done:
                    break
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            yield {"type": "error", "text": f"\n\n**Error:** {e}\n\n"}

    async def get_completion(self, messages, model: str, valves) -> str:
        form_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": valves.Temperature,
            "top_k": valves.Top_k,
            "top_p": valves.Top_p,
        }
        response = await generate_chat_completions(
            self.__request__,
            form_data,
            user=self.__user__,
        )
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return content if isinstance(content, str) else ""

    # ── Tool calling helpers ──────────────────────────────────────────

    def _check_model_native_fc(self, model_id: str) -> bool:
        """Return True if the model is configured for native function calling."""
        model_info = Models.get_model_by_id(model_id)
        if model_info and model_info.params:
            params = (
                model_info.params.model_dump()
                if hasattr(model_info.params, "model_dump")
                else {}
            )
            if params.get("function_calling") == "native":
                return True
        # Fall back to runtime MODELS state
        models = getattr(self.__request__.app.state, "MODELS", {})
        model = models.get(model_id, {})
        info = model.get("info", {})
        info_params = info.get("params", {})
        if isinstance(info_params, dict):
            return info_params.get("function_calling") == "native"
        if hasattr(info_params, "model_dump"):
            return info_params.model_dump().get("function_calling") == "native"
        return False

    async def _load_tools(self, tool_ids: list[str], extra_params: dict) -> dict:
        """Load tools using Open WebUI's get_tools(), returns tools_dict."""
        if not tool_ids:
            return {}
        return await get_tools(
            self.__request__,
            tool_ids,
            self.__user__,
            extra_params,
        )

    def _build_tool_call_details(
        self,
        call_id: str,
        name: str,
        arguments: str,
        done: bool = False,
        result=None,
        files=None,
        embeds=None,
    ) -> str:
        """Build <details type='tool_calls'> HTML matching Open WebUI's serialize_output()."""
        # arguments is already a JSON string, just escape for HTML attribute
        args_escaped = html_module.escape(arguments)
        if done:
            # result may be a string or other type
            result_text = (
                result
                if isinstance(result, str)
                else json.dumps(result or "", ensure_ascii=False)
            )
            result_escaped = html_module.escape(
                json.dumps(result_text, ensure_ascii=False)
            )
            files_escaped = html_module.escape(json.dumps(files)) if files else ""
            embeds_escaped = html_module.escape(json.dumps(embeds)) if embeds else ""
            return (
                f'<details type="tool_calls" done="true" id="{call_id}" '
                f'name="{name}" arguments="{args_escaped}" '
                f'result="{result_escaped}" files="{files_escaped}" '
                f'embeds="{embeds_escaped}">\n'
                f"<summary>Tool Executed</summary>\n</details>\n"
            )
        return (
            f'<details type="tool_calls" done="false" id="{call_id}" '
            f'name="{name}" arguments="{args_escaped}">\n'
            f"<summary>Executing...</summary>\n</details>\n"
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict],
        tools_dict: dict,
        metadata: dict,
        total_emitted: str,
    ) -> tuple[list[dict], str]:
        """Execute tool calls, emit <details> tags live. Returns (results, updated_total_emitted)."""
        results = []
        for tc in tool_calls:
            call_id = tc.get("id", str(uuid4()))
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            # Parse arguments
            params = {}
            try:
                params = ast.literal_eval(args_str)
            except Exception:
                try:
                    params = json.loads(args_str)
                except Exception:
                    logger.error(f"Failed to parse tool args for {name}: {args_str}")
                    results.append(
                        {
                            "tool_call_id": call_id,
                            "content": f"Error: malformed arguments for {name}",
                        }
                    )
                    continue

            # Emit "Executing..." tag — ensure newline before (middleware L369-370)
            executing_tag = self._build_tool_call_details(
                call_id, name, args_str, done=False
            )
            if total_emitted and not total_emitted.endswith("\n"):
                total_emitted += "\n"
            total_emitted += executing_tag
            await self.emit_replace(total_emitted)

            # Execute the tool
            tool_result = None
            tool = tools_dict.get(name)
            tool_type = tool.get("type", "") if tool else ""

            if tool and "callable" in tool:
                spec = tool.get("spec", {})
                allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
                filtered_params = {
                    k: v for k, v in params.items() if k in allowed_params
                }
                try:
                    tool_result = await tool["callable"](**filtered_params)
                except Exception as e:
                    tool_result = str(e)
            else:
                tool_result = f"Error: tool '{name}' not found"

            # Process result using Open WebUI's process_tool_result
            tool_return = process_tool_result(
                self.__request__,
                name,
                tool_result,
                tool_type,
                False,
                metadata,
                self.__user__,
            )
            result_str = tool_return[0] if len(tool_return) > 0 else ""
            result_files = tool_return[1] if len(tool_return) > 1 else None
            result_embeds = tool_return[2] if len(tool_return) > 2 else None

            # Replace executing tag with completed tag
            total_emitted = total_emitted.replace(executing_tag, "")
            done_tag = self._build_tool_call_details(
                call_id,
                name,
                args_str,
                done=True,
                result=result_str,
                files=result_files,
                embeds=result_embeds,
            )
            if total_emitted and not total_emitted.endswith("\n"):
                total_emitted += "\n"
            total_emitted += done_tag
            await self.emit_replace(total_emitted)

            results.append(
                {
                    "tool_call_id": call_id,
                    "content": str(result_str) if result_str else "",
                }
            )
        return results, total_emitted

    async def _emit_accumulated_tool_files(self):
        """Emit a single combined chat:message:files event with all accumulated tool files."""
        if self._accumulated_tool_files and self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {
                    "type": "chat:message:files",
                    "data": {"files": list(self._accumulated_tool_files)},
                }
            )
            logger.debug(
                f"Emitted combined chat:message:files with {len(self._accumulated_tool_files)} file(s)"
            )
            # Clear after emitting so the next participant doesn't re-emit previous files
            self._accumulated_tool_files = []

    async def emit_message(self, message: str):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {"type": "message", "data": {"content": message}}
            )

    async def emit_replace(self, content: str):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {"type": "replace", "data": {"content": content}}
            )

    async def emit_status(self, level: str, message: str, done: bool):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )

    async def emit_model_title(self, model_name: str):
        await self.emit_message(f"\n\n### {model_name}\n\n")

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __event_call__: Callable[[Any], Awaitable[Any]] = None,
        __task__=None,
        __model__=None,
        __request__=None,
        __metadata__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        self.__model__ = __model__
        self.__request__ = __request__
        self.__metadata__ = __metadata__ or {}

        valves = __user__.get("valves", self.UserValves())
        raw_history = body.get("messages", [])

        conversation_history = []
        for msg in raw_history:
            cleaned_msg = msg.copy()
            if "content" in cleaned_msg and isinstance(cleaned_msg["content"], str):
                cleaned_content = clean_thinking_tags(cleaned_msg["content"])

                if cleaned_msg.get("role") == "assistant" and not cleaned_msg.get(
                    "_speaker"
                ):
                    # Split concatenated multi-model messages back into individual speaker turns
                    # Optionally match the color circle emoji if present
                    parts = re.split(
                        r"(?:\n\n|^)### (?:(?:🔴|🔵|🟢|🟡|🟣|🟠|🟤|⚫|⚪)\s+)?(.+?)\n\n",
                        cleaned_content,
                    )

                    if len(parts) > 1:
                        if parts[0].strip():
                            conversation_history.append(
                                {"role": "assistant", "content": parts[0].strip()}
                            )

                        for i in range(1, len(parts), 2):
                            speaker_alias = parts[i].strip()
                            speaker_content = (
                                parts[i + 1].strip() if i + 1 < len(parts) else ""
                            )

                            if speaker_content:
                                conversation_history.append(
                                    {
                                        "role": "assistant",
                                        "content": speaker_content,
                                        "_speaker": speaker_alias,
                                    }
                                )
                        continue

                cleaned_msg["content"] = cleaned_content

            conversation_history.append(cleaned_msg)

        if not conversation_history:
            return "Error: No message history found."

        if __task__ and __task__ != TASKS.DEFAULT:
            model = valves.Participant1Model or self.__model__
            response = await generate_chat_completions(
                self.__request__,
                {"model": model, "messages": conversation_history, "stream": False},
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        # 1. Determine Configuration
        config = self._extract_config_from_metadata(body)
        if not config:
            if __event_call__:
                default_valves_dict = {}
                for attr, fieldInfo in valves.model_fields.items():
                    default_valves_dict[attr] = getattr(valves, attr)

                js_code = self._build_config_js(default_valves_dict)
                await self.emit_status(
                    "info", "Waiting for configuration setup...", False
                )
                try:
                    result = await __event_call__(
                        {
                            "type": "execute",
                            "data": {
                                "code": js_code,
                            },
                        }
                    )
                    if isinstance(result, dict) and "participants" in result:
                        config = result
                except Exception as e:
                    logger.error(f"Event call failed: {e}")

            if config:
                self._persist_config_to_metadata(body, config)

        config = self._sanitize_config(config, valves)
        self._persist_config_to_metadata(body, config)

        # Validate Participants
        participants = config.get("participants", [])
        if not participants:
            await self.emit_status("error", "No valid participants configured.", True)
            return "Error: No participants configured. Please set at least one participant."

        rounds = config.get("rounds", 3)
        use_manager = config.get("use_manager", False)
        manager_model = config.get("manager_model", "") or ""
        last_speaker = None

        # 2. Load tools (per-participant instead of global-only)
        # tool_ids come from metadata (which is popped from form_data by functions.py
        # before the pipe receives body, so we must read from __metadata__)
        global_tool_ids = self.__metadata__.get("tool_ids") or []
        if not global_tool_ids:
            # Fallback: check body in case it wasn't popped (e.g. direct API calls)
            global_tool_ids = body.get("tool_ids") or []

        logger.debug(f"[MultiModelTools] Global tool_ids from request: {global_tool_ids}")

        # Proxy event emitter for tools: intercepts 'chat:message:files'
        self._accumulated_tool_files = []

        async def _tool_event_proxy(event):
            event_type = event.get("type", "")
            if event_type == "chat:message:files":
                files = event.get("data", {}).get("files", [])
                self._accumulated_tool_files.extend(files)
                logger.debug(
                    f"Captured {len(files)} file(s) from tool, "
                    f"total accumulated: {len(self._accumulated_tool_files)}"
                )
            else:
                if __event_emitter__:
                    await __event_emitter__(event)

        extra_params = {
            "__event_emitter__": _tool_event_proxy,
            "__event_call__": __event_call__,
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": self.__metadata__,
            "__chat_id__": self.__metadata__.get("chat_id"),
            "__message_id__": self.__metadata__.get("message_id"),
        }

        # Pre-load tools for each participant model
        participant_tools_map = {}
        models_state = getattr(self.__request__.app.state, "MODELS", {})

        for p in participants:
            p_model_id = p["model"]
            if p_model_id in participant_tools_map:
                continue  # already loaded for this model

            p_tools_dict = {}
            p_tools_specs = None

            # Fetch model info to get default tools (toolIds) and features
            model_info = models_state.get(p_model_id, {})
            model_db_info = Models.get_model_by_id(p_model_id)

            logger.debug(
                f"[MultiModelTools] Loading tools for participant model: {p_model_id}"
            )

            default_tool_ids = []
            p_features = {}

            # 1. Start with global features from request body
            global_features = (
                self.__metadata__.get("features") or body.get("features") or {}
            )
            p_features.update(global_features)

            if model_db_info:
                meta = (
                    model_db_info.meta.model_dump()
                    if hasattr(model_db_info.meta, "model_dump")
                    else model_db_info.meta
                )
                params = (
                    model_db_info.params.model_dump()
                    if hasattr(model_db_info.params, "model_dump")
                    else model_db_info.params
                )

                if isinstance(meta, dict):
                    default_tool_ids.extend(meta.get("toolIds", []))
                    if "features" in meta and isinstance(meta["features"], dict):
                        p_features.update(meta["features"])
                    for f_id in meta.get("defaultFeatureIds", []):
                        p_features[f_id] = True

                if isinstance(params, dict):
                    default_tool_ids.extend(
                        params.get("toolIds", []) or params.get("tools", [])
                    )
                    if "features" in params and isinstance(params["features"], dict):
                        p_features.update(params["features"])
                    for f_id in params.get("defaultFeatureIds", []):
                        p_features[f_id] = True

            if model_info:
                info_meta = model_info.get("info", {}).get("meta", {})
                info_params = model_info.get("info", {}).get("params", {})

                default_tool_ids.extend(info_meta.get("toolIds", []))
                default_tool_ids.extend(
                    info_params.get("toolIds", []) or info_params.get("tools", [])
                )

                if isinstance(info_meta.get("features"), dict):
                    p_features.update(info_meta["features"])
                for f_id in info_meta.get("defaultFeatureIds", []):
                    p_features[f_id] = True

                if isinstance(info_params.get("features"), dict):
                    p_features.update(info_params["features"])
                for f_id in info_params.get("defaultFeatureIds", []):
                    p_features[f_id] = True

            # Clean and deduplicate tool IDs (handling cases where UI might inadvertently store dicts)
            clean_default_ids = []
            for t in default_tool_ids:
                if isinstance(t, str) and t.strip():
                    clean_default_ids.append(t.strip())
                elif isinstance(t, dict) and "id" in t:
                    clean_default_ids.append(str(t["id"]))

            # Combine global tools and model's default tools uniquely
            combined_tool_ids = list(set(global_tool_ids + clean_default_ids))

            # Load user-imported tools
            if combined_tool_ids:
                try:
                    p_tools_dict = await self._load_tools(
                        combined_tool_ids, extra_params
                    )
                except Exception as e:
                    logger.error(
                        f"[MultiModelTools] Failed to load imported tools for {p_model_id}: {e}"
                    )

            # Load built-in tools (web search, knowledge, etc.) based on the specific model.
            # Normalize legacy knowledge items so query_knowledge_files recognizes them
            # (it expects {"type": ..., "id": ...} but DB may have {"collection_name": ...}).
            p_model_info = model_info
            try:
                raw_knowledge = (
                    model_info.get("info", {}).get("meta", {}).get("knowledge", [])
                    or []
                )

                # Fallback: if MODELS state has no knowledge, try reading from DB directly
                if not raw_knowledge and model_db_info and model_db_info.meta:
                    m = (
                        model_db_info.meta.model_dump()
                        if hasattr(model_db_info.meta, "model_dump")
                        else model_db_info.meta
                    )
                    if isinstance(m, dict):
                        raw_knowledge = m.get("knowledge", []) or []

                if raw_knowledge:
                    normalized = []
                    needs_normalization = False
                    for item in raw_knowledge:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") and item.get("id"):
                            normalized.append(item)
                        elif item.get("collection_name"):
                            normalized.append(
                                {
                                    "type": "collection",
                                    "id": item["collection_name"],
                                    "name": item.get("name", ""),
                                }
                            )
                            needs_normalization = True
                        else:
                            normalized.append(item)

                    # Create a shallow copy to avoid mutating shared MODELS state
                    p_model_info = {**model_info}
                    p_model_info["info"] = {**p_model_info.get("info", {})}
                    p_model_info["info"]["meta"] = {
                        **p_model_info["info"].get("meta", {})
                    }
                    p_model_info["info"]["meta"]["knowledge"] = normalized
                    if needs_normalization:
                        logger.debug(
                            f"[MultiModelTools] Normalized legacy knowledge items for {p_model_id}"
                        )
            except Exception as e:
                logger.error(
                    f"[MultiModelTools] Knowledge normalization failed for {p_model_id}: {e}"
                )
                p_model_info = model_info  # fall back to original

            logger.debug(
                f"[MultiModelTools] Features for {p_model_id}: {p_features}"
            )
            logger.debug(
                f"[MultiModelTools] ENABLE_IMAGE_GENERATION={getattr(self.__request__.app.state.config, 'ENABLE_IMAGE_GENERATION', 'NOT_SET')}"
            )

            try:
                builtin_tools = get_builtin_tools(
                    self.__request__,
                    extra_params,
                    features=p_features,
                    model=p_model_info,
                )
                if builtin_tools:
                    p_tools_dict.update(builtin_tools)
                    logger.debug(
                        f"[MultiModelTools] Builtin tools loaded for {p_model_id}: {list(builtin_tools.keys())}"
                    )
                else:
                    logger.debug(
                        f"[MultiModelTools] No builtin tools returned for {p_model_id}"
                    )
            except Exception as e:
                logger.error(
                    f"[MultiModelTools] Failed to load built-in tools for {p_model_id}: {e}"
                )

            # Get native function calling flag
            native_fc = self._check_model_native_fc(p_model_id)

            # Get system message
            model_system_message = ""
            if model_db_info and model_db_info.params:
                p_params = (
                    model_db_info.params.model_dump()
                    if hasattr(model_db_info.params, "model_dump")
                    else model_db_info.params
                )
                if isinstance(p_params, dict):
                    model_system_message = p_params.get("system", "") or ""

            # Build OpenAI-format tool specs
            if p_tools_dict:
                p_tools_specs = [
                    {"type": "function", "function": t.get("spec", {})}
                    for t in p_tools_dict.values()
                ]

            participant_tools_map[p_model_id] = {
                "dict": p_tools_dict,
                "specs": p_tools_specs,
                "native_fc": native_fc,
                "system_message": model_system_message,
            }

        MAX_TOOL_CALL_RETRIES = 5
        MAX_MALFORMED_RETRIES = 2  # Silent retries for malformed tool calls

        # 3. Run Conversation Rounds
        # total_emitted tracks ALL content sent to the frontend across
        # all participants so that replace events preserve prior output
        #
        # One "round" = len(participants) turns.
        # Without manager: each participant speaks once per round in order.
        # With manager: the manager picks who speaks each turn (no consecutive repeats).
        # Total turns = len(participants) * rounds.
        total_emitted = ""
        turns_per_round = len(participants)

        # Resolve the manager model once (fall back to first participant's model)
        if use_manager:
            resolved_manager_model = (
                manager_model
                or valves.ManagerModel
                or participants[0]["model"]
            )
            if not resolved_manager_model:
                await self.emit_status(
                    "error",
                    "No manager model configured. Please set a Manager Model.",
                    True,
                )
                return "Error: No manager model configured."
            logger.debug(f"[MultiModelTools] Using manager model: {resolved_manager_model}")

        for round_num in range(rounds):
            # With manager: len(participants) turns per round, manager picks each.
            # Without manager: 1 "turn" that runs all participants sequentially.
            num_turns = turns_per_round if use_manager else 1

            for turn in range(num_turns):
                if use_manager:
                    participant_aliases = [p["alias"] for p in participants]
                    # Build a clean conversation dump for the manager:
                    # only user messages and final assistant text (with speaker labels),
                    # no tool calls, files, embeds, or internal metadata.
                    history_lines = []
                    for msg in conversation_history:
                        speaker = msg.get("_speaker")
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if not content or not content.strip():
                            continue
                        if speaker:
                            # Participant response — label with alias
                            history_lines.append(f"{speaker}: {content.strip()}")
                        elif role == "user":
                            history_lines.append(f"User: {content.strip()}")
                        elif role == "system":
                            continue  # skip system prompts
                        else:
                            history_lines.append(f"{role}: {content.strip()}")
                    history_str = "\n".join(history_lines)
                    manager_prompt = valves.ManagerSelectionPrompt.format(
                        history=history_str,
                        last_speaker=last_speaker or "None",
                        participant_list=", ".join(participant_aliases),
                    )
                    manager_messages = [
                        {"role": "system", "content": valves.ManagerSystemMessage},
                        {"role": "user", "content": manager_prompt},
                    ]

                    await self.emit_status(
                        "info",
                        f"Round {round_num + 1}/{rounds} — Turn {turn + 1}/{turns_per_round}: Manager selecting next speaker...",
                        False,
                    )

                    try:
                        manager_response = await generate_chat_completions(
                            self.__request__,
                            {
                                "model": resolved_manager_model,
                                "messages": manager_messages,
                                "stream": False,
                            },
                            user=self.__user__,
                        )
                        selected_alias = manager_response["choices"][0]["message"][
                            "content"
                        ].strip()
                    except Exception as e:
                        logger.error(f"Manager failed to select: {e}")
                        selected_alias = ""

                    normalized_selected_alias = self._normalize_alias(selected_alias)
                    selected_participant = next(
                        (
                            p
                            for p in participants
                            if self._normalize_alias(p["alias"])
                            == normalized_selected_alias
                        ),
                        None,
                    )

                    if not selected_participant or self._normalize_alias(
                        selected_participant["alias"]
                    ) == self._normalize_alias(last_speaker or ""):
                        await self.emit_status(
                            "info",
                            "Manager selection was invalid/repeated. Falling back.",
                            False,
                        )
                        last_speaker_index = next(
                            (
                                i
                                for i, p in enumerate(participants)
                                if p["alias"] == last_speaker
                            ),
                            -1,
                        )
                        fallback_index = (last_speaker_index + 1) % len(participants)
                        selected_participant = participants[fallback_index]

                    participants_to_run = [selected_participant]
                    last_speaker = selected_participant["alias"]
                else:
                    participants_to_run = participants

                for participant in participants_to_run:
                    model = participant["model"]
                    alias = participant["alias"]

                    # Determine if this model can use tools
                    participant_tools_specs = None
                    p_tools = participant_tools_map.get(model, {})
                    tools_dict = p_tools.get("dict", {})
                    tools_specs = p_tools.get("specs", None)
                    native_fc = p_tools.get("native_fc", False)
                    model_system_message = p_tools.get("system_message", "")

                    if tools_dict and tools_specs:
                        if native_fc:
                            participant_tools_specs = tools_specs
                            logger.debug(
                                f"[MultiModelTools] Sending {len(tools_specs)} tool specs to {alias} ({model})"
                            )
                        else:
                            logger.warning(
                                f"Model {model} does not have native function calling enabled. "
                                f"Tools will be skipped for {alias}."
                            )

                    # Build system prompt: model's system message + participant config + conversation flow
                    system_parts = []
                    if model_system_message.strip():
                        system_parts.append(model_system_message.strip())
                    if participant["system_message"].strip():
                        system_parts.append(participant["system_message"].strip())
                    system_parts.append(
                        f"{valves.AllParticipantsApendedMessage} {alias}\n\n"
                        "This is a multi-participant conversation. Messages from other "
                        "participants appear as user messages labeled with their name. "
                        "Respond naturally in character without prefixing your name."
                    )

                    # If this participant has tools, add explicit tool-use guidance
                    if participant_tools_specs:
                        tool_names = [t["function"]["name"] for t in participant_tools_specs if "function" in t]
                        system_parts.append(
                            "TOOL USE INSTRUCTIONS:\n"
                            f"You have access to the following tools: {', '.join(tool_names)}.\n"
                            "When the user's request can be fulfilled by a tool, call the appropriate tool.\n"
                            "CRITICAL: After every tool call, once you receive the tool results, "
                            "you MUST produce a visible text response summarizing or presenting the result to the user. "
                            "Never stop at just the tool call — always follow up with a written reply."
                        )

                    system_prompt = "\n\n".join(system_parts)

                    # Build messages with role-based separation:
                    # - Other participants' responses → user role with speaker label
                    # - This participant's own past responses → assistant role
                    #   (including private tool call messages if any)
                    adapted_history = []
                    for msg in conversation_history:
                        speaker = msg.get("_speaker")
                        if speaker and speaker != alias:
                            # Other participant's response → present as user message
                            # (tool calls are private — only show final text)
                            adapted_history.append(
                                {
                                    "role": "user",
                                    "content": f"{speaker} says: {msg['content']}",
                                }
                            )
                        elif speaker and speaker == alias:
                            # This participant's own past response → assistant role
                            # Include private tool interaction chain if any
                            tool_msgs = msg.get("_tool_messages", [])
                            for tm in tool_msgs:
                                adapted_history.append(
                                    {k: v for k, v in tm.items() if not k.startswith("_")}
                                )
                            adapted_history.append(
                                {"role": "assistant", "content": msg["content"]}
                            )
                        else:
                            # Original user/system messages → keep as-is
                            adapted_history.append(msg)

                    messages = [
                        {"role": "system", "content": system_prompt}
                    ] + adapted_history

                    await self.emit_status(
                        "info", f"Getting response from: {alias} ({model})...", False
                    )

                    try:
                        p_idx = participants.index(participant) % len(SPEAKER_COLORS)
                        color = SPEAKER_COLORS[p_idx]
                        title_text = f"\n\n### {color} {alias}\n\n"
                        total_emitted += title_text
                        await self.emit_replace(total_emitted)

                        # Emit warning if tools configured but model lacks native FC
                        if tools_dict and not participant_tools_specs:
                            warn_tag = (
                                '<details type="tool_calls" done="true" id="warn" '
                                f'name="⚠️ Tools Unavailable" '
                                f'arguments="&quot;&quot;" '
                                f'result="&quot;Model {html_module.escape(model)} does not have '
                                f'native function calling enabled. Tools skipped for {html_module.escape(alias)}.&quot;" '
                                f'files="" embeds="">\n'
                                f"<summary>Tools Unavailable</summary>\n</details>\n"
                            )
                            total_emitted += warn_tag
                            await self.emit_replace(total_emitted)

                        full_response = ""
                        reasoning_buffer = ""
                        reasoning_start_time = None
                        accumulated_tool_calls = []
                        metadata = body.get("metadata", {})

                        # ── Streaming + tool call accumulation helper ────
                        async def _stream_and_accumulate(stream_messages, tc_specs):
                            nonlocal full_response, reasoning_buffer
                            nonlocal reasoning_start_time, total_emitted
                            nonlocal accumulated_tool_calls

                            accumulated_tool_calls = []
                            last_emit_time = 0.0
                            emit_interval = 0.1

                            async def throttled_emit_replace(html_content, force=False):
                                nonlocal last_emit_time
                                now = time.monotonic()
                                if force or (now - last_emit_time >= emit_interval):
                                    await self.emit_replace(html_content)
                                    last_emit_time = now

                            async for event in self.get_streaming_completion(
                                stream_messages,
                                model=model,
                                valves=valves,
                                tools_specs=tc_specs,
                            ):
                                event_type = event.get("type")
                                if event_type == "error":
                                    total_emitted += event.get("text", "")
                                    await throttled_emit_replace(total_emitted, force=True)
                                    continue

                                if event_type == "tool_calls":
                                    # Accumulate tool call deltas (same as middleware.py)
                                    for delta_tc in event.get("data", []):
                                        tc_index = delta_tc.get("index")
                                        if tc_index is not None:
                                            existing = None
                                            for atc in accumulated_tool_calls:
                                                if atc.get("index") == tc_index:
                                                    existing = atc
                                                    break
                                            if existing is None:
                                                delta_tc.setdefault("function", {})
                                                delta_tc["function"].setdefault("name", "")
                                                delta_tc["function"].setdefault(
                                                    "arguments", ""
                                                )
                                                accumulated_tool_calls.append(delta_tc)
                                            else:
                                                dn = delta_tc.get("function", {}).get(
                                                    "name"
                                                )
                                                da = delta_tc.get("function", {}).get(
                                                    "arguments"
                                                )
                                                if dn:
                                                    existing["function"]["name"] += dn
                                                if da:
                                                    existing["function"]["arguments"] += da
                                    continue

                                if event_type == "reasoning":
                                    reasoning_piece = event.get("text", "")
                                    if reasoning_piece:
                                        if reasoning_start_time is None:
                                            reasoning_start_time = time.monotonic()
                                        reasoning_buffer += reasoning_piece
                                        # Format with blockquote prefix (> ) like middleware
                                        display = "\n".join(
                                            f"> {line}"
                                            if not line.startswith(">")
                                            else line
                                            for line in reasoning_buffer.splitlines()
                                        )
                                        await throttled_emit_replace(
                                            total_emitted
                                            + '<details type="reasoning" done="false">\n'
                                            + "<summary>Thinking...</summary>\n"
                                            + display
                                            + "\n</details>\n\n"
                                        )
                                    continue

                                # Finalize thinking block when transitioning
                                if reasoning_buffer:
                                    reasoning_duration = (
                                        round(time.monotonic() - reasoning_start_time)
                                        if reasoning_start_time
                                        else 1
                                    )
                                    display = "\n".join(
                                        f"> {line}" if not line.startswith(">") else line
                                        for line in reasoning_buffer.splitlines()
                                    )
                                    total_emitted += (
                                        f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                        f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                        + display
                                        + "\n</details>\n\n"
                                    )
                                    reasoning_buffer = ""
                                    await throttled_emit_replace(total_emitted, force=True)

                                if event_type == "content":
                                    chunk_text = event.get("text", "")
                                    if not chunk_text:
                                        continue
                                    full_response += chunk_text
                                    formatted_chunk = self._replace_thinking_tags(chunk_text)
                                    total_emitted += formatted_chunk
                                    await self.emit_message(formatted_chunk)
                                    continue

                            # Flush reasoning if stream ended during thinking
                            if reasoning_buffer:
                                reasoning_duration = (
                                    round(time.monotonic() - reasoning_start_time)
                                    if reasoning_start_time
                                    else 1
                                )
                                display = "\n".join(
                                    f"> {line}" if not line.startswith(">") else line
                                    for line in reasoning_buffer.splitlines()
                                )
                                total_emitted += (
                                    f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                    f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                    + display
                                    + "\n</details>\n\n"
                                )
                                reasoning_buffer = ""
                            
                            # Final flush at the end of the stream
                            await throttled_emit_replace(total_emitted, force=True)

                        # ── Initial streaming call ──────────────────────
                        await _stream_and_accumulate(messages, participant_tools_specs)

                        # ── Tool call execution + re-prompt loop ────────
                        tool_interaction_messages = []
                        if accumulated_tool_calls and participant_tools_specs:
                            tool_call_retries = 0
                            malformed_retries = 0
                            current_messages = list(messages)

                            while (
                                accumulated_tool_calls
                                and tool_call_retries < MAX_TOOL_CALL_RETRIES
                            ):
                                tool_call_retries += 1

                                # ── Silent retry for malformed tool calls ───
                                malformed = []
                                valid = []
                                for tc in accumulated_tool_calls:
                                    func = tc.get("function", {})
                                    tc_name = func.get("name", "").strip()
                                    tc_args = func.get("arguments", "{}")

                                    # Check: name must exist and be a known tool
                                    if not tc_name or tc_name not in tools_dict:
                                        malformed.append(
                                            f"Unknown tool '{tc_name}'" if tc_name
                                            else "Tool call with empty name"
                                        )
                                        continue

                                    # Check: arguments must be parseable
                                    try:
                                        ast.literal_eval(tc_args)
                                    except Exception:
                                        try:
                                            json.loads(tc_args)
                                        except Exception:
                                            malformed.append(
                                                f"Malformed arguments for '{tc_name}': {tc_args[:100]}"
                                            )
                                            continue

                                    valid.append(tc)

                                if malformed and malformed_retries < MAX_MALFORMED_RETRIES:
                                    malformed_retries += 1
                                    logger.warning(
                                        f"[MultiModelTools] Silent retry {malformed_retries}/{MAX_MALFORMED_RETRIES} "
                                        f"for {alias}: {'; '.join(malformed)}"
                                    )

                                    # Build correction messages without showing anything to user
                                    available_tools = list(tools_dict.keys())
                                    correction_msg = {
                                        "role": "user",
                                        "content": (
                                            f"Your previous tool call was malformed: {'; '.join(malformed)}. "
                                            f"Available tools are: {', '.join(available_tools)}. "
                                            "Please try again with a valid tool name and properly formatted JSON arguments. "
                                            "After the tool call, always provide a text response."
                                        ),
                                    }
                                    current_messages.append(correction_msg)

                                    # Reset and re-call model
                                    full_response = ""
                                    accumulated_tool_calls = []
                                    await _stream_and_accumulate(
                                        current_messages, participant_tools_specs
                                    )
                                    continue

                                # If some valid and some malformed (past retry limit),
                                # just execute the valid ones
                                if malformed and valid:
                                    logger.warning(
                                        f"[MultiModelTools] Skipping {len(malformed)} malformed tool call(s), "
                                        f"executing {len(valid)} valid one(s) for {alias}"
                                    )
                                    accumulated_tool_calls = valid
                                elif malformed and not valid:
                                    logger.warning(
                                        f"[MultiModelTools] All tool calls malformed for {alias}, "
                                        "skipping tool execution"
                                    )
                                    accumulated_tool_calls = []
                                    break

                                # Execute the accumulated tool calls
                                results, total_emitted = await self._execute_tool_calls(
                                    accumulated_tool_calls,
                                    tools_dict,
                                    metadata,
                                    total_emitted,
                                )
                                # Emit all accumulated files as single combined events
                                await self._emit_accumulated_tool_files()

                                # Build re-prompt messages (OpenAI format):
                                # 1. Assistant message with tool_calls
                                assistant_tc_msg = {
                                    "role": "assistant",
                                    "content": full_response or None,
                                    "tool_calls": [
                                        {
                                            "id": tc.get("id", str(uuid4())),
                                            "type": "function",
                                            "function": tc.get("function", {}),
                                        }
                                        for tc in accumulated_tool_calls
                                    ],
                                }
                                current_messages.append(assistant_tc_msg)
                                tool_interaction_messages.append(assistant_tc_msg)

                                # 2. Tool result messages
                                for result in results:
                                    tool_result_msg = {
                                        "role": "tool",
                                        "tool_call_id": result["tool_call_id"],
                                        "content": result.get("content", ""),
                                    }
                                    current_messages.append(tool_result_msg)
                                    tool_interaction_messages.append(tool_result_msg)

                                # Reset for next streaming round
                                full_response = ""

                                # On last retry, don't pass tools so model is forced to produce text
                                reprompt_tools = (
                                    participant_tools_specs
                                    if tool_call_retries < MAX_TOOL_CALL_RETRIES
                                    else None
                                )

                                # Re-call model with tool results
                                await _stream_and_accumulate(
                                    current_messages, reprompt_tools
                                )

                        # ── Fallback for empty response ─────────────────
                        if not full_response.strip():
                            await self.emit_status(
                                "info",
                                f"Empty stream from {alias} ({model}). Retrying once (non-stream).",
                                False,
                            )
                            # Use current_messages (includes tool results) if available,
                            # otherwise fall back to original messages
                            fallback_msgs = (
                                current_messages
                                if tool_interaction_messages
                                else messages
                            )
                            fallback_response = await self.get_completion(
                                fallback_msgs, model=model, valves=valves
                            )
                            if fallback_response.strip():
                                full_response = fallback_response
                                total_emitted += fallback_response
                                await self.emit_replace(total_emitted)
                            else:
                                await self.emit_status(
                                    "warning",
                                    f"No response produced by {alias} ({model}).",
                                    False,
                                )

                        cleaned_response = clean_thinking_tags(full_response)
                        if cleaned_response.strip():
                            conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": cleaned_response.strip(),
                                    "_speaker": alias,
                                    "_tool_messages": tool_interaction_messages,
                                }
                            )

                    except Exception as e:
                        error_message = (
                            f"Error getting response from {alias} ({model}): {e}"
                        )
                        await self.emit_status("error", error_message, True)
                        await self.emit_message(f"\n\n**ERROR**: {error_message}\n\n")

        await self.emit_status("success", "Conversation round completed.", True)

        # When returning from pipe directly, we yield an empty string because we used emit_message
        return ""
