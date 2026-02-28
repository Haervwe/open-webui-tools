"""
title: MCP Pipe
description: Funtion Pipe made to use python mcp servers on open-webui.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
git: https://github.com/Haervwe/open-webui-tools  
version: 0.1.1
requirements: mcp, mcp-tavily, mcp_server_time
"""
import logging
import json
import asyncio
from typing import Optional, Callable, Awaitable, Dict, List, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from contextlib import AsyncExitStack
from mcp import ClientSession as Cls
from mcp import StdioServerParameters
from mcp.types import ListToolsResult, CallToolResult
from mcp.client.stdio import stdio_client
from open_webui.main import generate_chat_completions
from open_webui.constants import TASKS
from open_webui.utils.misc import get_last_user_message
import inspect
import aiohttp
import os

name = "MCP Pipe"


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


def parse_server_args(args: str | List[str]) -> List[str]:
    """Parse server arguments from string or list into proper format"""
    if isinstance(args, str):
        parsed = [arg.strip().strip("\"'") for arg in args.split(",")]
        return parsed
    elif isinstance(args, list):
        return args
    else:
        logger.error(f"Invalid server args format: {type(args)} - {args}")
        return []


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str


class MCPClient:
    def __init__(
        self,
        mcp_config_path: str,  # Path to the mcp_config.json file
        **kwargs,
    ):
        self.mcp_config_path = mcp_config_path
        self.sessions: Dict[str, Cls] = {}  # Store sessions by server name
        self.exit_stack = AsyncExitStack()
        self.pipe = kwargs.get("pipe")
        self.available_tools = []
        self.available_prompts = []  # Change from dict to list to track server info

    async def connect_to_servers(self):
        """Connect to multiple MCP servers defined in mcp_config.json"""
        try:
            with open(self.mcp_config_path, "r") as f:
                mcp_config = json.load(f)

            mcp_servers = mcp_config.get("mcpServers", {})
            logger.info(f"Found {len(mcp_servers)} MCP servers in config")

            for server_name, server_config in mcp_servers.items():
                logger.info(f"Connecting to server: {server_name}")
                command = server_config.get("command")
                args = parse_server_args(server_config.get("args", []))
                env = server_config.get("env")

                server_params = StdioServerParameters(
                    command=command, args=args, env=env
                )

                try:
                    stdio_transport = await self.exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    stdio, write = stdio_transport

                    session = await self.exit_stack.enter_async_context(
                        Cls(stdio, write)
                    )
                    await session.initialize()

                    # Store session and aggregate tools
                    self.sessions[server_name] = session
                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        self.available_tools.append(
                            {
                                "id": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                                "server": server_name,  # Add server information
                            }
                        )

                    tool_names = [
                        tool["id"]
                        for tool in self.available_tools
                        if tool["server"] == server_name
                    ]
                    logger.info(f"Connected to {server_name} with tools: {tool_names}")
                    await self.pipe.emit_status(
                        "info",
                        f"Connected to {server_name} with tools: {tool_names}",
                        False,
                    )

                    # Add prompts handling similar to tools
                    try:
                        prompts_response = await session.list_prompts()
                        for prompt in prompts_response.prompts:
                            # Convert PromptArgument objects to dict for JSON serialization
                            serialized_arguments = []
                            if hasattr(prompt, "arguments") and prompt.arguments:
                                for arg in prompt.arguments:
                                    serialized_arguments.append(
                                        {
                                            "name": arg.name,
                                            "description": (
                                                arg.description
                                                if hasattr(arg, "description")
                                                else None
                                            ),
                                            "required": (
                                                arg.required
                                                if hasattr(arg, "required")
                                                else False
                                            ),
                                        }
                                    )

                            self.available_prompts.append(
                                {
                                    "name": prompt.name,
                                    "description": prompt.description,
                                    "arguments": serialized_arguments,
                                    "server": server_name,
                                }
                            )
                        prompt_names = [
                            prompt["name"]
                            for prompt in self.available_prompts
                            if prompt["server"] == server_name
                        ]
                        logger.info(f"Added prompts from {server_name}: {prompt_names}")
                    except Exception as e:
                        logger.warning(
                            f"Server {server_name} doesn't support prompts: {str(e)}"
                        )

                except Exception as e:
                    logger.error(f"Failed to connect to server {server_name}: {str(e)}")
                    await self.pipe.emit_status(
                        "error", f"Failed to connect to {server_name}: {str(e)}", False
                    )

        except Exception as e:
            error_msg = f"Failed to connect to MCP servers: {str(e)}"
            logger.exception(error_msg)
            await self.pipe.emit_status("error", error_msg, True)
            raise

    async def call_tool(self, tool_name: str, tool_args: Dict) -> str:
        """Calls a tool on the correct MCP server based on tool information."""
        # Find the server for this tool
        tool_server = next(
            (
                tool["server"]
                for tool in self.available_tools
                if tool["id"] == tool_name
            ),
            None,
        )

        if not tool_server or tool_server not in self.sessions:
            error_msg = f"Tool {tool_name} not available or server not connected."
            logger.error(error_msg)
            return error_msg

        session = self.sessions[tool_server]
        try:
            logger.debug(f"Calling tool {tool_name} on server {tool_server}")
            tool_result = await session.call_tool(tool_name, tool_args)

            if isinstance(tool_result, CallToolResult):
                tool_content = "".join([msg.text for msg in tool_result.content])
            else:
                tool_content = str(tool_result)
            return tool_content

        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            return f"Tool call {tool_name} failed: {e}"

    async def list_available_tools(self) -> ListToolsResult:
        """Lists available tools from the MCP server."""
        if self.session is None:
            raise RuntimeError("Not connected to an MCP server.")
        return await self.session.list_tools()

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict] = None):
        """Retrieves a prompt from the appropriate MCP server."""
        prompt_info = next(
            (
                prompt
                for prompt in self.available_prompts
                if prompt["name"] == prompt_name
            ),
            None,
        )

        if not prompt_info:
            logger.error(f"Prompt '{prompt_name}' not found on any server.")
            return None

        server_name = prompt_info["server"]
        if server_name not in self.sessions:
            logger.error(
                f"Server '{server_name}' for prompt '{prompt_name}' not connected."
            )
            return None

        try:
            return await self.sessions[server_name].get_prompt(prompt_name, arguments)
        except Exception as e:
            logger.error(
                f"Error getting prompt '{prompt_name}' from server '{server_name}': {e}"
            )
            return None

    async def process_tool_calls(self, tool_calls: List[dict], messages: List[dict]):
        """Process tool calls and append results to messages."""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            await self.pipe.emit_status("tool", f"Calling {tool_name}...", False)
            tool_result = await self.call_tool(tool_name, tool_args)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.get("id"),
                }
            )

    async def chat_completion(self, request_data: Dict) -> Dict:
        """Make a chat completion request to OpenAI API"""
        async with aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.pipe.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
        ) as session:
            url = f"{self.pipe.valves.OPENAI_API_BASE}/chat/completions"
            try:
                async with session.post(url, json=request_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: {error_text}")
                        raise Exception(f"API error: {error_text}")

                    return await response.json()
            except Exception as e:
                logger.error(f"Error in chat completion: {str(e)}")
                raise

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI API directly instead of generate_chat_completions"""
        logger.debug(f"Processing query: {query}")

        messages = self.pipe.build_messages_with_tools_and_prompts(
            query, self.available_tools, self.available_prompts
        )

        final_text = []
        iteration_count = 0

        while True:
            iteration_count += 1
            logger.debug(f"\n=== Starting iteration {iteration_count} ===")

            request = self.pipe.build_llm_request(messages)
            tools_for_request = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["id"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in self.available_tools
            ]
            request["tools"] = tools_for_request

            logger.debug(f"Making LLM request with tools:")
            logger.debug(f"Full request: {json.dumps(request, indent=1)}")

            try:
                response = await self.chat_completion(request)
                content = response["choices"][0]["message"]

                message_content = content.get("content", "")
                if message_content:
                    final_text.append(message_content)
                    await self.pipe.emit_message(message_content)

                if tool_calls := content.get("tool_calls"):
                    logger.debug(f"Processing {len(tool_calls)} tool calls")

                    for tool_idx, tool_call in enumerate(tool_calls):
                        logger.debug(
                            f"\n--- Processing tool call {tool_idx + 1}/{len(tool_calls)} ---"
                        )
                        try:
                            tool_name = tool_call["function"]["name"]
                            logger.debug(f"Tool name: {tool_name}")

                            args = tool_call["function"].get("arguments", "{}")
                            logger.debug(f"Raw arguments: {args}")

                            tool_args = (
                                json.loads(args) if isinstance(args, str) else args
                            )
                            logger.debug(
                                f"Parsed tool arguments: {json.dumps(tool_args, indent=2)}"
                            )

                            # Verify tool exists
                            available_tool_names = [
                                t["id"] for t in self.available_tools
                            ]
                            if tool_name not in available_tool_names:
                                logger.error(
                                    f"Tool {tool_name} not found in available tools: {available_tool_names}"
                                )
                                continue

                            logger.debug(
                                f"Calling tool {tool_name} with args: {json.dumps(tool_args, indent=2)}"
                            )
                            tool_result = await self.call_tool(tool_name, tool_args)
                            logger.debug(f"Tool result: {tool_result}")

                            tool_message = {
                                "role": "tool",
                                "content": tool_result,
                                "tool_call_id": tool_call.get("id"),
                                "name": tool_name,
                            }
                            messages.append(tool_message)

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse tool arguments: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing tool call: {str(e)}")
                            logger.error(
                                f"Full tool call was: {json.dumps(tool_call, indent=2)}"
                            )
                            continue
                else:
                    logger.debug("No tool calls in response, breaking loop")
                    break

            except Exception as e:
                logger.error(f"Error in main processing loop: {str(e)}", exc_info=True)
                break

        return "\n".join(final_text)

    async def cleanup(self):
        """Clean up resources, closing all sessions"""
        logger.debug("Starting cleanup of all MCP sessions")
        try:
            for server_name, session in self.sessions.items():
                try:
                    logger.debug(f"Closing session for server: {server_name}")
                    await session.close()
                except Exception as e:
                    logger.exception(f"Error during {server_name} cleanup: {str(e)}")
            await self.exit_stack.close()
            self.sessions.clear()
            logger.info("All MCP sessions cleaned up successfully")
        except Exception as e:
            logger.exception(f"Error during cleanup: {str(e)}")
            raise


class Pipe:
    class Valves(BaseModel):
        MODEL: str = Field(default="", description="Model to use")
        SYSTEM_PROMPT: str = Field(
            default="""You are an AI assistant designed to answer user queries accurately and efficiently. You have access to the following tools:

{tools_desc}

To use a tool, you must use the exact name and parameters as specified. For example:
`get_current_time(timezone="UTC")`

When a user asks a question:
1. If it requires a tool, use ONLY the exact tool name and parameters available
2. If you can answer directly, do so
3. If you cannot answer and no appropriate tool exists, say "I do not have the information to answer this question."

Do not fabricate tool names or parameters. Only use the exact tools and parameters listed above.""",
            description="MCP client system prompt",
        )
        OPENAI_API_KEY: str = Field(
            default="1111", description="OpenAI API key",
            json_schema_extra={"input": {"type": "password"}},
        )
        OPENAI_API_BASE: str = Field(
            default="http://host.docker.internal:11434/v1",
            description="OpenAI API base URL",
        )
        TEMPERATURE: float = Field(default=0.5, description="Model temperature")
        MAX_TOKENS: int = Field(default=1500, description="Maximum tokens to generate")
        TOP_P: float = Field(default=0.8, description="Top p sampling parameter")
        PRESENCE_PENALTY: float = Field(default=0.8, description="Presence penalty")
        FREQUENCY_PENALTY: float = Field(default=0.8, description="Frequency penalty")
        DEBUG: bool = Field(
            default=False, description="Enable debug logging for LLM interactions"
        )

    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __current_node__: Optional[dict] = None
    __user__: User
    __model__: str
    __request__: None

    def __init__(self):
        self.valves = self.Valves()
        self.mcp_client = None

    def pipes(self) -> list[dict[str, str]]:
        """Return available pipes"""
        return [{"id": f"{name}", "name": name}]

    async def emit_message(self, message: str):
        """Emit message event"""
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        """Emit status event"""
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

    def build_messages_with_tools_and_prompts(
        self, query: str, tools: List[Dict], prompts: List[Dict]
    ) -> List[Dict]:
        # Create tools and prompts descriptions grouped by server
        server_capabilities = {}

        # Group tools by server
        for tool in tools:
            server = tool["server"]
            if server not in server_capabilities:
                server_capabilities[server] = {"tools": [], "prompts": []}
            tool_desc = f"- {tool['id']}: {tool['description']}\n  Parameters: {json.dumps(tool['input_schema'], indent=2)}"
            server_capabilities[server]["tools"].append(tool_desc)

        # Group prompts by server
        for prompt in prompts:
            server = prompt["server"]
            if server not in server_capabilities:
                server_capabilities[server] = {"tools": [], "prompts": []}
            prompt_desc = f"- {prompt['name']}: {prompt['description']}"
            if prompt.get("arguments"):
                # Use the serialized arguments that are now dicts
                args_desc = json.dumps(
                    [
                        {
                            "name": arg["name"],
                            "description": arg["description"],
                            "required": arg["required"],
                        }
                        for arg in prompt["arguments"]
                    ],
                    indent=2,
                )
                prompt_desc += f"\n  Arguments: {args_desc}"
            server_capabilities[server]["prompts"].append(prompt_desc)

        # Build system message with server-grouped capabilities
        capabilities_desc = []
        for server, caps in server_capabilities.items():
            server_desc = f"\nServer: {server}"
            if caps["tools"]:
                server_desc += f"\nTools:\n{chr(10).join(caps['tools'])}"
            if caps["prompts"]:
                server_desc += f"\nPrompts:\n{chr(10).join(caps['prompts'])}"
            capabilities_desc.append(server_desc)

        system_message = self.valves.SYSTEM_PROMPT.format(
            tools_desc="\n".join(capabilities_desc)
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
        return messages

    def build_llm_request(self, messages: List[Dict]) -> Dict:
        """Builds the request data for the LLM API call."""
        return {
            "model": self.__model__,
            "messages": messages,
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "top_p": self.valves.TOP_P,
            "presence_penalty": self.valves.PRESENCE_PENALTY,
            "frequency_penalty": self.valves.FREQUENCY_PENALTY,
            "stream": False,
        }

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        """Main pipe function"""
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__)
        self.__model__ = self.valves.MODEL
        self.__request__ = __request__

        # Handle special tasks
        if __task__ in (TASKS.TITLE_GENERATION, TASKS.TAGS_GENERATION):
            try:
                response = await generate_chat_completions(
                    self.__request__,
                    {
                        "model": self.__model__,
                        "messages": body.get("messages"),
                        "stream": False,
                    },
                    user=self.__user__,
                )
                return f"{name}: {response['choices'][0]['message']['content']}"
            except Exception as e:
                logger.error(f"Error during {__task__}: {e}")
                await self.emit_status(
                    "error", f"Failed to generate {__task__}: {e}", True
                )
                return ""

        try:
            # Initialize MCPClient here
            if self.mcp_client is None:
                config_path = os.path.join(
                    os.getenv("APP_DIR", "."), "data/mcp_config.json"
                )
                logger.info(f"Initializing MCP client with config from: {config_path}")
                self.mcp_client = MCPClient(mcp_config_path=config_path, pipe=self)

                await self.mcp_client.connect_to_servers()
                await self.__current_event_emitter__(
                    {
                        "type": "tools",
                        "data": {"tools": self.mcp_client.available_tools},
                    }
                )

            messages = body["messages"]
            query = get_last_user_message(messages)
            await self.emit_status("info", "Processing query...", False)
            logger.debug(f"Processing query...{query}")
            response = await self.mcp_client.process_query(query)

            await self.emit_status("info", "Query processed successfully", True)
            return response

        except Exception as e:
            logger.error(f"Pipe error: {e}")
            await self.emit_status("error", f"Error: {str(e)}", True)
            return ""

        finally:
            if self.mcp_client:
                await self.mcp_client.cleanup()
