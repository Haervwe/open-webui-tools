import logging
import json
import asyncio
from typing import Optional, Callable, Awaitable, Dict, List, Any, Union, Generator, Iterator
from dataclasses import dataclass
from pydantic import BaseModel, Field
from contextlib import AsyncExitStack
from mcp import ClientSession as Cls
from mcp import StdioServerParameters
from mcp.types import ListToolsResult, CallToolResult, GetPromptResult
from mcp.client.stdio import stdio_client
import inspect
import aiohttp
import os
import uuid
import time
import requests

name = "MCP Pipeline"


def setup_logger():
    """
    Initialize and configure a logger for the MCP pipeline.
    Sets up logging with proper formatting and avoids duplicate handlers.
    Returns a configured logger instance.
    """
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
    """
    Main client class for handling Model Context Protocol (MCP) server connections.
    Manages multiple server sessions, tools, prompts, and resources.
    
    Attributes:
        mcp_config_path (str): Path to MCP configuration file
        sessions (Dict[str, ClientSession]): Active server sessions keyed by server name
        exit_stack (AsyncExitStack): Stack for managing async context managers
        pipe (Pipeline): Reference to parent pipeline instance
        available_tools (List): List of all available tools across servers
        available_prompts (List): List of all available prompts across servers
        available_resources (List): List of all available resources across servers
        is_initialized (bool): Flag indicating if client is initialized
    """
    def __init__(
        self,
        mcp_config_path: str,  # Path to the mcp_config.json file
        pipe,
    ):
        self.mcp_config_path = mcp_config_path
        self.sessions: Dict[str, Cls] = {}  # Store sessions by server name
        self.exit_stack = AsyncExitStack()
        self.pipe = pipe
        self.available_tools = []
        self.available_prompts = []  # Change from dict to list to track server info
        self.available_resources = []  # Add a list to track available resources
        self.is_initialized = False

    async def initialize_servers(self):
         """
        Initialize connections to all configured MCP servers.
        
        Process:
        1. Reads server configurations from mcp_config.json
        2. Establishes connections to each server
        3. Aggregates available tools, prompts, and resources
        4. Sets up stdio transport and sessions
        5. Handles initialization errors gracefully
        
        Raises:
            Exception: If server initialization fails
        """
         if self.is_initialized:
             logger.warning("MCPClient already initialized, skipping initialization.")
             return
         
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

                     # Add resources handling similar to tools
                     try:
                         resources_response = await session.list_resources()
                         for resource in resources_response.resources:
                             self.available_resources.append(
                                 {
                                     "uri": resource.uri,
                                     "name": resource.name,
                                     "description": resource.description,
                                     "mimeType": resource.mimeType,
                                     "server": server_name,  # Add server information
                                 }
                             )
                         resource_names = [
                             resource["name"]
                             for resource in self.available_resources
                             if resource["server"] == server_name
                         ]
                         logger.info(f"Added resources from {server_name}: {resource_names}")

                         # Log the contents of the first resource
                         server_resources = [
                             resource for resource in self.available_resources
                             if resource["server"] == server_name
                         ]
                         if server_resources:
                             first_resource_uri = server_resources[0]["uri"]
                             logger.info(f"Content of the first resource ({first_resource_uri}) from server {server_name}:")
                             resource_content = await session.read_resource(first_resource_uri)
                             logger.info(f"{resource_content}")

                     except Exception as e:
                         logger.warning(
                             f"Server {server_name} doesn't support resources: {str(e)}"
                         )
                    

                 except Exception as e:
                     logger.error(f"Failed to connect to server {server_name}: {str(e)}")
             self.is_initialized = True

         except Exception as e:
            error_msg = f"Failed to connect to MCP servers: {str(e)}"
            logger.exception(error_msg)
            raise

    async def call_tool(self, tool_name: str, tool_args: Dict) -> str:
        """
        Execute a tool on the appropriate MCP server.
        
        Args:
            tool_name (str): Name of the tool to call
            tool_args (Dict): Arguments to pass to the tool
            
        Returns:
            str: Tool execution result or error message
            
        Special handling:
        - Handles load_resource tool differently
        - Finds correct server for tool execution
        - Processes tool response format
        """
        # Special handling for load_resource
        if tool_name == "load_resource":
            return await self.load_resource(tool_args.get("uri", ""))

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

    async def load_resource(self, uri: str) -> str:
        """Load a resource from the appropriate server"""
        try:
            logger.debug(f"Attempting to load resource: {uri}")
            logger.debug(f"Available resources: {[r['uri'] for r in self.available_resources]}")
            
            # Convert string URIs to their string representation for comparison
            request_uri = str(uri)
            
            # Find which server has the resource
            for resource in self.available_resources:
                # Convert AnyUrl to string for comparison
                resource_uri = str(resource["uri"])
                if resource_uri == request_uri:
                    server_name = resource["server"]
                    logger.debug(f"Found resource on server {server_name}")
                    
                    if server_name not in self.sessions:
                        logger.error(f"Server {server_name} not connected")
                        return f"Server {server_name} not connected"
                    
                    session = self.sessions[server_name]
                    try:
                        content = await session.read_resource(uri)
                        if hasattr(content, 'contents'):
                            # Handle structured resource content
                            text_contents = [c.text for c in content.contents if hasattr(c, 'text')]
                            return f"Resource content from {server_name} ({uri}):\n" + "\n".join(text_contents)
                        return f"Resource content from {server_name} ({uri}):\n{content}"
                    except Exception as e:
                        logger.error(f"Error reading resource from {server_name}: {str(e)}")
                        return f"Failed to read resource from {server_name}: {str(e)}"
                        
            logger.error(f"Resource {uri} not found in available resources")
            return f"Resource {uri} not found. Available resources: {[str(r['uri']) for r in self.available_resources]}"
            
        except Exception as e:
            logger.error(f"Error loading resource {uri}: {str(e)}")
            return f"Failed to load resource {uri}: {str(e)}"

    async def process_tool_calls(self, tool_calls: List[dict], messages: List[dict]):
        """Process tool calls and append results to messages."""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            logger.debug(f"Processing tool call: {tool_name} with args: {tool_args}")

            try:
                tool_result = await self.call_tool(tool_name, tool_args)
                logger.debug(f"Tool result for {tool_name}: {tool_result}")

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.get("id"),
                    }
                )
            except Exception as e:
                logger.error(f"Error processing tool call {tool_name}: {str(e)}")
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Error processing tool call {tool_name}: {str(e)}",
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
    
    async def cleanup(self):
      """Clean up resources, closing all sessions"""
      logger.debug("Starting cleanup of all MCP sessions")
      try:
             
        await self.exit_stack.aclose()
        self.sessions.clear()
        logger.info("All MCP sessions cleaned up successfully")
        
      except Exception as e:
        logger.exception(f"Error during cleanup: {str(e)}")
        raise



class Pipeline:
    """
    Main pipeline class implementing the MCP integration with Open WebUI.
    Handles message processing, server selection, and LLM interactions.
    
    Components:
    - Valves: Configuration parameters for the pipeline
    - Server selection: Intelligent routing of queries to appropriate servers
    - Message processing: Handling of conversation context and tool results
    - LLM integration: Interface with language models via API
    """
    class Valves(BaseModel):
        MODEL: str = Field(default="Qwen2_5_16k:latest", description="Model to use")
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
        OPENAI_API_KEY: str = Field(default="1111", description="OpenAI API key")
        OPENAI_API_BASE: str = Field(
            default="http://0.0.0.0:11434/v1",
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
        ORCHESTRATOR_PROMPT: str = Field(
            default="""You are a server orchestration agent. Your job is to analyze the user query and select the appropriate MCP servers to handle it.
            You have access to these servers with their capabilities:
            {server_descriptions}
            
            Use the select_servers tool to choose which servers should handle this query. You must justify your selection.
            If the query doesn't require any specific server capabilities, don't select any servers and just respond to the query as a helpful assistant.""",
            description="Server orchestrator system prompt"
        )
        TOOL_AGENT_PROMPT: str = Field(
            default="""You are a tool orchestration agent with access to multiple servers and their resources. Available servers: {server_names}

            You can use these tools:
            {tools_desc}

            Available resources that can be loaded:
            {resources_desc}

            To use a resource, first load it with the load_resource tool, then use the server-specific tools.
            
            Follow these rules:
            1. If a resource might help, load it first
            2. Use tools from selected servers only
            3. Chain multiple tools if needed
            4. If no tools are needed or no servers are selected, respond directly to the query as a helpful assistant
            5. Keep the conversation context in mind when responding""",
            description="Tool agent system prompt"
        )
        BYPASS_TASKS: List[str] = Field(
            default=["### Task:"],
            description="Message prefixes that bypass MCP pipeline and go directly to LLM"
        )
    def __init__(self):
        self.valves = self.Valves()
        self.mcp_client = None
        
    async def select_servers(self, query: str, messages: List[Dict] = None) -> tuple[bool, List[str]]:
        """
        Determine which MCP servers should handle a given query.
        
        Process:
        1. Gathers server descriptions and capabilities
        2. Creates orchestrator messages preserving conversation context
        3. Sends request to LLM for server selection
        4. Processes selection response
        
        Args:
            query (str): User query to analyze
            messages (List[Dict], optional): Existing conversation context
            
        Returns:
            tuple[bool, List[str]]: (success flag, list of selected servers or error message)
        """
        server_descriptions = []
        for server_name, session in self.mcp_client.sessions.items():
            server_config = json.loads(open(self.mcp_client.mcp_config_path).read())["mcpServers"][server_name]
            desc = server_config.get("description", "No description available")
            server_descriptions.append({"name": server_name, "description": desc})

        select_server_tool = {
            "type": "function",
            "function": {
                "name": "select_servers",
                "description": "Select which MCP servers should handle this query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_servers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of server names to use"
                        },
                        "justification": {
                            "type": "string",
                            "description": "Explanation for server selection"
                        }
                    },
                    "required": ["selected_servers", "justification"]
                }
            }
        }

        # Create orchestrator messages preserving conversation context
        orchestrator_messages = []
        if messages:
            # Copy all messages except system prompt
            orchestrator_messages.extend(messages[1:])
        
        # Add system prompt at the beginning
        orchestrator_messages.insert(0, {
            "role": "system", 
            "content": self.valves.ORCHESTRATOR_PROMPT.format(
                server_descriptions=json.dumps(server_descriptions, indent=2)
            )
        })
        
        # Add the current task explanation
        orchestrator_messages.append({
            "role": "user",
            "content": "Based on the conversation above and this new query, select the appropriate servers to handle it: " + query
        })

        request_data = self.build_llm_request(orchestrator_messages)
        request_data["tools"] = [select_server_tool]
        
        response = await self.mcp_client.chat_completion(request_data)
        
        if "choices" in response and "tool_calls" in response["choices"][0]["message"]:
            tool_call = response["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(tool_call["function"]["arguments"])
            return True, args["selected_servers"]
        else:
            return False, response["choices"][0]["message"]["content"]

    def pipes(self) -> list[dict[str, str]]:
        """Return available pipes"""
        return [{"id": f"{name}", "name": name}]



    def build_messages_with_tools_and_prompts(
        self, query: str, tools: List[Dict], prompts: List[Dict], selected_servers: List[str], messages: List[Dict] = None
    ) -> List[Dict]:
        """
        Construct or update message context with available tools and prompts.
        
        Process:
        1. Filters tools and prompts for selected servers
        2. Groups capabilities by server
        3. Builds system message with detailed capability descriptions
        4. Maintains conversation context while updating system message
        
        Args:
            query (str): Current user query
            tools (List[Dict]): Available tools
            prompts (List[Dict]): Available prompts
            selected_servers (List[str]): Servers to include
            messages (List[Dict], optional): Existing conversation context
            
        Returns:
            List[Dict]: Updated message list with proper context and system prompt
        """
        # Filter tools and prompts for selected servers
        filtered_tools = [tool for tool in tools if tool["server"] in selected_servers] if selected_servers else []
        filtered_prompts = [prompt for prompt in prompts if prompt["server"] in selected_servers] if selected_servers else []
        
        server_capabilities = {}

        # Group tools by server (now using filtered tools)
        for tool in filtered_tools:
            server = tool["server"]
            if server not in server_capabilities:
                server_capabilities[server] = {"tools": [], "prompts": []}
            tool_desc = f"- {tool['id']}: {tool['description']}\n  Parameters: {json.dumps(tool['input_schema'], indent=2)}"
            server_capabilities[server]["tools"].append(tool_desc)

        # Group prompts by server (now using filtered prompts)
        for prompt in filtered_prompts:
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

        # Add resources section
        resources_desc = []
        for resource in self.mcp_client.available_resources:
            if resource["server"] in selected_servers:
                resources_desc.append(
                    f"- {resource['name']} (URI: {resource['uri']}): {resource['description']}"
                )


        system_message = self.valves.TOOL_AGENT_PROMPT.format(
            server_names=", ".join(selected_servers) if selected_servers else "None",
            tools_desc=capabilities_desc if selected_servers else "No tools available",
            resources_desc="\n".join(resources_desc) if selected_servers else "No resources available"
        )

        if messages:
            # Bug fix: Check if first message is system message before overwriting
            if messages[0]["role"] != "system":
                logger.debug("Messages missing system message, creating new message list")
                return [
                    {"role": "system", "content": system_message},
                    *messages,  # Preserve existing messages after system message
                    {"role": "user", "content": query}
                ]
            else:
                # Only update existing system message content
                messages[0]["content"] = system_message
                return messages
        else:
            # Create new conversation with system and user messages
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]

    def build_llm_request(self, messages: List[Dict]) -> Dict:
        """Builds the request data for the LLM API call."""
        return {
            "model": self.valves.MODEL,
            "messages": messages,
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "top_p": self.valves.TOP_P,
            "presence_penalty": self.valves.PRESENCE_PENALTY,
            "frequency_penalty": self.valves.FREQUENCY_PENALTY,
            "stream": False,
        }
    
    async def on_startup(self, **kwargs) -> None:
        pass
        

    async def on_shutdown(self) -> None:
        pass

    def has_resources_for_servers(self, selected_servers: List[str]) -> bool:
        """Check if any of the selected servers have resources"""
        return any(
            resource["server"] in selected_servers 
            for resource in self.mcp_client.available_resources
        )

    async def process_query(self, query: str, messages: List[dict], body: dict, selected_servers) -> dict:
        """Process a query using MCP tools and return OpenAI-compatible response"""
        try:
            # Build or update messages with tools and prompts
            if not messages:
                messages = self.build_messages_with_tools_and_prompts(
                    query, 
                    self.mcp_client.available_tools,
                    self.mcp_client.available_prompts,
                    selected_servers
                )
                logger.debug("Created new message context")
            else:
                logger.debug("Updating existing message context")
                # Ensure user message is included in subsequent requests
                messages = self.build_messages_with_tools_and_prompts(
                    query,
                    self.mcp_client.available_tools,
                    self.mcp_client.available_prompts,
                    selected_servers,
                    messages
                )
                # Add the current query as a user message if it's not the last message
                if messages[-1]["role"] != "user":
                    messages.append({"role": "user", "content": query})
            
            # Detailed message context logging
            logger.debug("Message context structure:")
            for i, msg in enumerate(messages):
                logger.debug(f"Message {i}: role={msg['role']}, content_preview={msg['content'][:100]}...")

            if not messages or messages[0]["role"] != "system":
                logger.error("Invalid message context: Missing system message")
                raise ValueError("Message context must start with a system message")

            if len(messages) < 2 or messages[-1]["role"] != "user":
                logger.error("Invalid message context: Missing or misplaced user message")
                logger.debug(f"Current messages: {[m['role'] for m in messages]}")
                raise ValueError("Message context must include a user message")

            # Create tools list once and reuse
            tools_for_request = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["id"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in self.mcp_client.available_tools
                if tool["server"] in selected_servers
            ]

            # Log available tools for debugging
            logger.debug(f"Available tools for servers {selected_servers}:")
            for tool in tools_for_request:
                logger.debug(f"- {tool['function']['name']}: {tool['function']['description'][:100]}...")

            # Add load_resource tool if needed
            if self.has_resources_for_servers(selected_servers):
                tools_for_request.append({
                    "type": "function",
                    "function": {
                        "name": "load_resource",
                        "description": "Load content from a resource",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "uri": {
                                    "type": "string",
                                    "description": "URI of the resource to load"
                                }
                            },
                            "required": ["uri"]
                        }
                    }
                })

            request_data = self.build_llm_request(messages)
            request_data["tools"] = tools_for_request

            logger.debug(f"Final request structure:")
            logger.debug(f"- Model: {request_data['model']}")
            logger.debug(f"- Message count: {len(request_data['messages'])}")
            logger.debug(f"- Tools count: {len(request_data['tools'])}")
            logger.debug(f"- First and last messages: {[{'role': m['role'], 'content': m['content'][:50]} for m in [request_data['messages'][0], request_data['messages'][-1]]]}")

            # ...rest of existing code...

            logger.debug(f"Request data: {[tool['function']['name'] for tool in request_data['tools']]}")
            response = await self.mcp_client.chat_completion(request_data)

            # If the response includes tool calls, process them
            while "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "tool_calls" in choice["message"]:
                    await self.mcp_client.process_tool_calls(
                        choice["message"]["tool_calls"], messages
                    )
                    # Make another request with tool results AND tools list
                    request_data = self.build_llm_request(messages)
                    request_data["tools"] = tools_for_request  # Reuse the same tools list
                    logger.debug(f"Follow-up request with tool results: {[tool['function']['name'] for tool in request_data['tools']]}")
                    response = await self.mcp_client.chat_completion(request_data)
                else:
                    break
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        """
        Main entry point for processing queries through the MCP pipeline.
        
        Flow:
        1. Checks for bypass conditions
        2. Initializes MCP client if needed
        3. Selects appropriate servers
        4. Updates message context
        5. Processes query through selected servers
        6. Handles responses and errors
        
        Args:
            user_message (str): User input to process
            model_id (str): Requested model identifier
            messages (List[dict]): Conversation history
            body (dict): Additional request parameters
            
        Returns:
            Union[str, Generator, Iterator]: Response content or error message
        """
        self.__model__ = self.valves.MODEL

        async def _async_pipe():
            try:
                # Check if this is a special task that should bypass MCP
                if messages and any(
                    messages[-1].get("content", "").startswith(prefix) 
                    for prefix in self.valves.BYPASS_TASKS
                ):
                    logger.debug("Detected bypass task, sending directly to LLM")
                    request_data = self.build_llm_request(messages)
                    response = await self.mcp_client.chat_completion(request_data)
                    if "choices" in response:
                        return response["choices"][0]["message"]["content"]
                    return "No valid response found"

                # Regular MCP pipeline flow
                if self.valves.DEBUG:
                    logger.debug(f"Request details: message={user_message}, model={model_id}")
                    logger.debug(f"Messages: {[msg['content'] for msg in messages]}")

                
                config_path = os.path.join(os.getenv("APP_DIR", "."), "data/mcp_config.json")
                logger.warning(f"MCP client not initialized, re-initializing. Config: {config_path}")
                self.mcp_client = MCPClient(mcp_config_path=config_path, pipe=self)
                await self.mcp_client.initialize_servers()

                # Select servers with conversation context
                selected, selected_servers = await self.select_servers(user_message, messages)
                
                if not selected:
                    return selected_servers
                # Update messages with new system prompt and tools
                updated_messages = self.build_messages_with_tools_and_prompts(
                    user_message,
                    self.mcp_client.available_tools,
                    self.mcp_client.available_prompts,
                    selected_servers,
                    messages
                )

                # Ensure selected_servers is a list
                if isinstance(selected_servers, str):
                    selected_servers = [server.strip() for server in selected_servers.split(",")]

                # Process query
                response = await self.process_query(
                    user_message, 
                    updated_messages, 
                    body, 
                    selected_servers if selected_servers else []
                )
                
                if "choices" in response:
                    return response["choices"][0]["message"]["content"]
                return "No valid response found"

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Pipe error: {error_msg}")
                logger.info(f"Shutting down {name} pipeline.")
                if self.mcp_client:
                    await self.mcp_client.cleanup()
                return json.dumps({
                    "error": {
                        "message": error_msg,
                        "type": "internal_error",
                        "param": None,
                        "code": "500"
                    }
                })

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no running loop exists, create and manage a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_async_pipe())
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        else:
            # If we're already in an event loop, create and await the task
            return loop.run_until_complete(_async_pipe())
