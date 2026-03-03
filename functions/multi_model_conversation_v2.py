"""
title: Multi Model Conversations v2
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 2.1.0
"""

import logging
import json
import re
from typing import Callable, Awaitable, Any, Optional
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import User, Users

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


def clean_thinking_tags(message: str) -> str:
    complete_pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
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
        r"\|begin_of_thought\|",
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

        return {
            "rounds": rounds,
            "use_manager": use_manager,
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
    managerLbl.onclick = () => managerChk.checked = !managerChk.checked;
    managerDiv.appendChild(managerChk); managerDiv.appendChild(managerLbl);
    globalRow.appendChild(managerDiv);

    form.appendChild(globalRow);

    const partsCont = document.createElement('div');
    partsCont.style.cssText = 'display: flex; flex-direction: column; gap: 12px;';
    form.appendChild(partsCont);

    const partUIs = [];
    const modelOptions = [{id: '', name: 'Select Model...'}, ...availableModels.map(m => ({id: m.id, name: m.name}))];

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

    def _replace_thinking_tags(self, text: str) -> str:
        """Replace raw thinking tags with <details> HTML the frontend understands."""
        text = THINK_OPEN_PATTERN.sub(
            '<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n',
            text,
        )
        text = THINK_CLOSE_PATTERN.sub("\n</details>\n\n", text)
        return text

    async def get_streaming_completion(self, messages, model: str, valves):
        try:
            form_data = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": valves.Temperature,
                "top_k": valves.Top_k,
                "top_p": valves.Top_p,
            }
            response = await generate_chat_completions(
                self.__request__,
                form_data,
                user=self.__user__,
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

    async def emit_message(self, message: str):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {"type": "message", "data": {"content": message}}
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
        await self.emit_message(f"\n\n### 🗣️ {model_name}\n\n")

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __event_call__: Callable[[Any], Awaitable[Any]] = None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = Users.get_user_by_id(__user__["id"])
        self.__model__ = __model__
        self.__request__ = __request__

        valves = __user__.get("valves", self.UserValves())
        raw_history = body.get("messages", [])

        conversation_history = []
        for msg in raw_history:
            cleaned_msg = msg.copy()
            if "content" in cleaned_msg:
                if isinstance(cleaned_msg["content"], str):
                    cleaned_msg["content"] = clean_thinking_tags(cleaned_msg["content"])
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
        last_speaker = None

        # 2. Run Conversation Rounds
        for round_num in range(rounds):
            if use_manager:
                participant_aliases = [p["alias"] for p in participants]
                history_str = "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in conversation_history
                )
                manager_prompt = valves.ManagerSelectionPrompt.format(
                    history=history_str,
                    last_speaker=last_speaker or "None",
                    participant_list=", ".join(participant_aliases),
                )
                manager_messages = [
                    {"role": "system", "content": valves.ManagerSystemMessage},
                    {"role": "user", "content": manager_prompt},
                ]

                manager_model = valves.ManagerModel or self.__model__
                await self.emit_status(
                    "info", "Group Chat Manager selecting next speaker...", False
                )

                try:
                    manager_response = await generate_chat_completions(
                        self.__request__,
                        {
                            "model": manager_model,
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

                system_prompt = (
                    f"{participant['system_message']}\n\n"
                    f"{valves.AllParticipantsApendedMessage} {alias}\n\n"
                    "This is a multi-participant conversation. Messages from other "
                    "participants appear as user messages labeled with their name. "
                    "Respond naturally in character without prefixing your name."
                )

                # Build messages with role-based separation:
                # - Other participants' responses → user role with speaker label
                # - This participant's own past responses → assistant role
                adapted_history = []
                for msg in conversation_history:
                    speaker = msg.get("_speaker")
                    if speaker and speaker != alias:
                        # Other participant's response → present as user message
                        adapted_history.append(
                            {
                                "role": "user",
                                "content": f"{speaker} says: {msg['content']}",
                            }
                        )
                    elif speaker and speaker == alias:
                        # This participant's own past response → assistant role
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
                    await self.emit_model_title(alias)

                    full_response = ""
                    reasoning_buffer = ""

                    async for event in self.get_streaming_completion(
                        messages, model=model, valves=valves
                    ):
                        event_type = event.get("type")
                        if event_type == "error":
                            await self.emit_message(event.get("text", ""))
                            continue

                        if event_type == "reasoning":
                            reasoning_piece = event.get("text", "")
                            if reasoning_piece:
                                if not reasoning_buffer:
                                    await self.emit_status(
                                        "info",
                                        f"{alias} is thinking...",
                                        False,
                                    )
                                reasoning_buffer += reasoning_piece
                            continue

                        # Flush accumulated reasoning as a complete block
                        if reasoning_buffer:
                            await self.emit_message(
                                '<details type="reasoning" done="true">\n'
                                "<summary>Thought</summary>\n"
                                + reasoning_buffer
                                + "\n</details>\n\n"
                            )
                            reasoning_buffer = ""

                        if event_type == "content":
                            chunk_text = event.get("text", "")
                            if not chunk_text:
                                continue

                            full_response += chunk_text
                            await self.emit_message(
                                self._replace_thinking_tags(chunk_text)
                            )
                            continue

                    # Flush reasoning if stream ended during thinking
                    if reasoning_buffer:
                        await self.emit_message(
                            '<details type="reasoning" done="true">\n'
                            "<summary>Thought</summary>\n"
                            + reasoning_buffer
                            + "\n</details>\n\n"
                        )
                        reasoning_buffer = ""

                    if not full_response.strip():
                        await self.emit_status(
                            "info",
                            f"Empty stream from {alias} ({model}). Retrying once (non-stream).",
                            False,
                        )
                        fallback_response = await self.get_completion(
                            messages, model=model, valves=valves
                        )
                        if fallback_response.strip():
                            full_response = fallback_response
                            await self.emit_message(fallback_response)
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
