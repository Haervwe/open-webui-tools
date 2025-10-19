# MCP Pipeline for Open WebUI

This repository provides Model Context Protocol (MCP) integrations for Open WebUI pipelines compatible with most MCP servers in python and NPM.

## Prerequisites

- [Open WebUI Pipelines](https://github.com/open-webui/pipelines) installed and configured
- Python 3.8+
- Node.js and NPM


## Installation



1. Set up Open WebUI Pipelines:
```bash
git clone https://github.com/open-webui/pipelines.git
cd pipelines
```

2. create a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate
```

3. Install required Python packages inside the virtual environment:
```bash
pip install mcp mcp-tavily mcp_server_time
pip install -r requirements-minimum.txt
```

4. Create the MCP configuration file config.json inside the pipelines directory as data/config.json:

```bash
mkdir -p data
touch config.json
```

5. Add the contents to config.json the only different part is the addition to the description field for each server meant to be passed to the orchestrator to select the proper server to use:

```bash
nano config.json
```

Example
```json
{
    "mcpServers": {
      "time_server": {
        "command": "python",
        "args": ["-m", "mcp_server_time" , "--local-timezone=America/New_York"],
        "description":"Provides Time and Timezone conversion tools."
      },
      "tavily_server": { 
        "command": "python",
        "args": ["-m", "mcp_server_tavily", "--api-key=tvly-xxx"],
        "description":"Provides web search capabilites tools."
      },
      "puppeteer": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "description":"Provides tools for using puppeteer in a web browser"
      },
      "postgres": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-postgres",
          "postgresql://postgres:URL"
        ],
        "description":"Provides tools for querying Postgress databases"
      }
    }
  }
```

5. Launch Pipeline Server:
   - Run the pipeline server using the command:
```bash
   start.sh
```

5. Configure Pipeline Server:
   - Navigate to your Open WebUI settings
   - Add your pipeline server URL
   - Add the MCP_pipeline.py in the UI

6. Set up Pipeline Valves:
   

## Configuration Options

## Pipeline Valve Configuration

The MCP pipeline uses several configurable valves to control its behavior. Here are the key valve settings:

### Model Settings
- `MODEL`: Default "Qwen2_5_16k:latest" - The LLM model to use
- `OPENAI_API_KEY`: Your OpenAI API key for API access
- `OPENAI_API_BASE`: Default "http://0.0.0.0:11434/v1" - Base URL for API requests

### Generation Parameters
- `TEMPERATURE`: Default 0.5 - Controls randomness in responses (0.0-1.0)
- `MAX_TOKENS`: Default 1500 - Maximum tokens to generate
- `TOP_P`: Default 0.8 - Top-p sampling parameter
- `PRESENCE_PENALTY`: Default 0.8 - Penalty for repeating topics
- `FREQUENCY_PENALTY`: Default 0.8 - Penalty for repeating tokens

### System Prompts
The pipeline uses three main system prompts that control different aspects:

1. **Main System Prompt**
   Controls how the AI uses tools:
   ```
   You are an AI assistant with access to tools.
   To use a tool, use exact name and parameters:
   get_current_time(timezone="UTC")
   
   Follow these rules:
   1. Use ONLY exact tool names/parameters available
   2. Answer directly if no tool needed
   3. Admit when you can't answer
   ```

2. **Orchestrator Prompt**
   Controls server selection:
   ```
   You are a server orchestrator that:
   - Analyzes queries
   - Selects appropriate MCP servers
   - Justifies server selections
   - Responds directly if no servers needed
   ```

3. **Tool Agent Prompt**
   Controls tool usage across servers:
   ```
   You are a tool agent that:
   - Manages multiple server tools/resources
   - Loads resources when needed
   - Uses tools from selected servers
   - Chains multiple tools if needed
   - Maintains conversation context
   ```

### Debug Options
- `DEBUG`: Default False - Enable detailed logging
- `BYPASS_TASKS`: Default ["### Task:"] - Messages that skip MCP and go directly to LLM

### Configuring Valves in Open WebUI

1. Navigate to Pipelines > Valves
2. Configure the above settings
3. Save and test the configuration

## Example Usage

select the pipeline as normal model in open webui, then send a query.

## Understanding the Pipeline Flow

![MCP Pipeline Flow](assets/mcp_pipeline_flow.png)

The MCP Pipeline works in the following way:

1. **User Query Stage**
   - User sends a query through Open WebUI
   - Pipeline receives the query and initializes MCP servers

2. **Server Selection Stage**
   - Orchestrator analyzes query and server capabilities
   - Selects appropriate servers based on query needs
   - Uses server descriptions from config.json

3. **Tool Orchestration Stage**
   - Tool agent loads necessary resources
   - Selects and chains appropriate tools
   - Manages multiple server interactions

4. **Execution Stage**
   - Tools are called in sequence
   - Results are gathered and processed
   - Final response is formatted

5. **Response Stage**
   - Combined results returned to user
   - Conversation context maintained
   - Debug information logged if enabled








