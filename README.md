# Open WebUI Tools Collection

[![Open WebUI](https://img.shields.io/badge/Open%20WebUI-Compatible-blue?style=flat-square&logo=github)](https://github.com/open-webui/open-webui)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)

> **🚀 A modular collection of tools, function pipes, and filters to supercharge your Open WebUI experience.**

Transform your Open WebUI instance into a powerful AI workstation with this comprehensive toolkit. From academic research and image generation to music creation and autonomous agents, this collection provides everything you need to extend your AI capabilities.

## ✨ What's Inside

This repository contains **15+ specialized tools and functions** designed to enhance your Open WebUI experience:

### 🛠️ **Tools**

- **arXiv Search** - Academic paper discovery (no API key required!)
- **Perplexica Search** - Web search using Perplexica API with citations
- **Pexels Media Search** - High-quality photos and videos from Pexels API
- **Native Image Generator** - Direct Open WebUI image generation with Ollama model management
- **Hugging Face Image Generator** - AI-powered image creation
- **ComfyUI ACE Step Audio** - Advanced music generation
- **Flux Kontext ComfyUI** - Professional image editing

### 🔄 **Function Pipes**

- **Planner Agent v2** - Advanced autonomous agent with specialized models, interactive guidance, and comprehensive execution management
- **arXiv Research MCTS** - Advanced research with Monte Carlo Tree Search
- **Multi Model Conversations** - Multi-agent discussions
- **Resume Analyzer** - Professional resume analysis
- **Mopidy Music Controller** - Music server management
- **Letta Agent** - Autonomous agent integration
- **MCP Pipe** - Model Context Protocol integration

### 🔧 **Filters**

- **Prompt Enhancer** - Automatic prompt improvement
- **Semantic Router** - Intelligent model selection
- **Full Document** - File processing capabilities
- **Clean Thinking Tags** - Conversation cleanup

## 🚀 Quick Start

### Option 1: Open WebUI Hub (Recommended)

1. Visit [https://openwebui.com/u/haervwe](https://openwebui.com/u/haervwe)
2. Browse the collection and click "Get" for desired tools
3. Follow the installation prompts in your Open WebUI instance

### Option 2: Manual Installation

1. Copy `.py` files from `tools/`, `functions/`, or `filters/` directories
2. Navigate to Open WebUI Workspace > Tools/Functions/Filters
3. Paste the code, provide a name and description, then save

## 🎯 Key Features

- **🔌 Plug-and-Play**: Most tools work out of the box with minimal configuration
- **🎨 Visual Integration**: Seamless integration with ComfyUI workflows
- **🤖 AI-Powered**: Advanced features like MCTS research and autonomous planning
- **📚 Academic Focus**: arXiv integration for research and academic work
- **🎵 Creative Tools**: Music generation and image editing capabilities
- **🔍 Smart Routing**: Intelligent model selection and conversation management
- **📄 Document Processing**: Full document analysis and resume processing


## 📋 Prerequisites

- **Open WebUI**: Version 0.6.0+ recommended
- **Python**: 3.8 or higher
- **Optional Dependencies**:
  - ComfyUI (for image/music generation tools)
  - Mopidy (for music controller)
  - Various API keys (Hugging Face, Tavily, etc.)

## 🔧 Configuration

Most tools are designed to work with minimal configuration. Key configuration areas:

- **API Keys**: Required for some tools (Hugging Face, Tavily, etc.)
- **ComfyUI Integration**: For image and music generation tools
- **Model Selection**: Choose appropriate models for your use case
- **Filter Setup**: Enable filters in your model configuration

---

## 📖 Detailed Documentation

### Table of Contents

1. [arXiv Search Tool](#arxiv-search-tool)
2. [Perplexica Search Tool](#perplexica-search-tool)
3. [Pexels Media Search Tool](#pexels-media-search-tool)
4. [Native Image Generator](#native-image-generator)
5. [Hugging Face Image Generator](#hugging-face-image-generator)
6. [Cloudflare Workers AI Image Generator](#cloudflare-workers-ai-image-generator)
7. [SearxNG Image Search Tool](#searxng-image-search-tool)
8. [ComfyUI ACE Step Audio Tool](#comfyui-ace-step-audio-tool)
9. [Flux Kontext ComfyUI Pipe](#flux-kontext-comfyui-pipe)
10. [Planner Agent v2](#planner-agent-v2)
11. [arXiv Research MCTS Pipe](#arxiv-research-mcts-pipe)
12. [Multi Model Conversations Pipe](#multi-model-conversations-pipe)
13. [Resume Analyzer Pipe](#resume-analyzer-pipe)
14. [Mopidy Music Controller](#mopidy-music-controller)
15. [Letta Agent Pipe](#letta-agent-pipe)
16. [MCP Pipe](#mcp-pipe)
17. [Prompt Enhancer Filter](#prompt-enhancer-filter)
18. [Semantic Router Filter](#semantic-router-filter)
19. [Full Document Filter](#full-document-filter)
20. [Clean Thinking Tags Filter](#clean-thinking-tags-filter)
21. [Using the Provided ComfyUI Workflows](#using-the-provided-comfyui-workflows)
22. [Installation](#installation)
23. [Contributing](#contributing)
24. [License](#license)
25. [Credits](#credits)
26. [Support](#support)

---

## 🧪 Tools

### arXiv Search Tool

### Description

Search arXiv.org for relevant academic papers on any topic. No API key required!

### Configuration

- No configuration required. Works out of the box.

### Usage

- **Example:**

  ```python
  Search for recent papers about "tree of thought"
  ```

- Returns up to 5 most relevant papers, sorted by most recent.

![arXiv Search Example](img/arxiv_search.png)
*Example arXiv search result in Open WebUI*

---

### Perplexica Search Tool

### Description

Search the web for factual information, current events, or specific topics using the Perplexica API. This tool provides comprehensive search results with citations and sources, making it ideal for research and information gathering. [Perplexica](https://github.com/ItzCrazyKns/Perplexica) is an open-source AI-powered search engine and alternative to Perplexity AI that must be self-hosted locally. It uses advanced language models to provide accurate, contextual answers with proper source attribution.

### Configuration

- `BASE_URL` (str): Base URL for the Perplexica API (default: `http://host.docker.internal:3001`)
- `OPTIMIZATION_MODE` (str): Search optimization mode - "speed" or "balanced" (default: `balanced`)
- `CHAT_MODEL` (str): Default chat model for search processing (default: `llama3.1:latest`)
- `EMBEDDING_MODEL` (str): Default embedding model for search (default: `bge-m3:latest`)
- `OLLAMA_BASE_URL` (str): Base URL for Ollama API (default: `http://host.docker.internal:11434`)

**Prerequisites**: You must have [Perplexica](https://github.com/ItzCrazyKns/Perplexica) installed and running locally at the configured URL. Perplexica is a self-hosted open-source search engine that requires Ollama with the specified chat and embedding models. Follow the installation instructions in the Perplexica repository to set up your local instance.

### Usage

- **Example:**

  ```python
  Search for "latest developments in AI safety research 2024"
  ```

- Returns comprehensive search results with proper citations

- Automatically emits citations for source tracking in Open WebUI

- Provides both summary and individual source links

### Features

- **Web Search Integration**: Direct access to current web information
- **Citation Support**: Automatic citation generation for Open WebUI
- **Model Flexibility**: Configurable chat and embedding models
- **Real-time Status**: Progress updates during search execution
- **Source Tracking**: Individual source citations with metadata

---

### Pexels Media Search Tool

### Description

Search and retrieve high-quality photos and videos from the Pexels API. This tool provides access to Pexels' extensive collection of free stock photos and videos, with comprehensive search capabilities, automatic citation generation, and direct image display in chat. Perfect for finding professional-quality media for presentations, content creation, or creative projects.

### Configuration

- `PEXELS_API_KEY` (str): Free Pexels API key from https://www.pexels.com/api/ (required)
- `DEFAULT_PER_PAGE` (int): Default number of results per search (default: 5, recommended for LLMs)
- `MAX_RESULTS_PER_PAGE` (int): Maximum allowed results per page (default: 15, prevents overwhelming LLMs)
- `DEFAULT_ORIENTATION` (str): Default photo orientation - "all", "landscape", "portrait", or "square" (default: "all")
- `DEFAULT_SIZE` (str): Default minimum photo size - "all", "large" (24MP), "medium" (12MP), or "small" (4MP) (default: "all")

**Prerequisites**: Get a free API key from [Pexels API](https://www.pexels.com/api/) and configure it in the tool's Valves settings.

### Usage

- **Photo Search Example:**

  ```python
  Search for photos of "modern office workspace"
  ```

- **Video Search Example:**

  ```python
  Search for videos of "ocean waves at sunset"
  ```

- **Curated Photos Example:**

  ```python
  Get curated photos from Pexels
  ```

### Features

- **Three Search Functions**: `search_photos`, `search_videos`, and `get_curated_photos`
- **Direct Image Display**: Images are automatically formatted with markdown for immediate display in chat
- **Advanced Filtering**: Filter by orientation, size, color, and quality
- **Attribution Support**: Automatic citation generation with photographer credits
- **Rate Limit Handling**: Built-in error handling for API limits and invalid keys
- **LLM Optimized**: Results are limited and formatted to prevent overwhelming language models
- **Real-time Status**: Progress updates during search execution

---

### Native Image Generator

### Description

Generate images using Open WebUI's native image generation middleware configured in admin settings. This tool leverages whatever image generation backend you have configured (such as AUTOMATIC1111, ComfyUI, or OpenAI DALL-E) through Open WebUI's built-in image generation system, with optional Ollama model management to free up VRAM when needed.

### Configuration

- `unload_ollama_models` (bool): Whether to unload all Ollama models from VRAM before generating images (default: `False`)
- `ollama_url` (str): Ollama API URL for model management (default: `http://host.docker.internal:11434`)

**Prerequisites**: You must have image generation configured in Open WebUI's admin settings under Settings > Images. This tool works with any image generation backend you have set up (AUTOMATIC1111, ComfyUI, OpenAI, etc.).

### Usage

- **Example:**

  ```python
  Generate an image of "a serene mountain landscape at sunset"
  ```

- Uses whatever image generation backend is configured in Open WebUI admin settings

- Automatically manages model resources if Ollama unloading is enabled

- Returns markdown-formatted image links for immediate display

### Features

- **Native Integration**: Uses Open WebUI's native image generation middleware without external dependencies
- **Backend Agnostic**: Works with any image generation backend configured in admin settings (AUTOMATIC1111, ComfyUI, OpenAI, etc.)
- **Memory Management**: Optional Ollama model unloading to optimize VRAM usage
- **Flexible Model Support**: You can prompt de agent to change the image generation model, providing the name is given to it.
- **Real-time Status**: Provides generation progress updates via event emitter
- **Error Handling**: Comprehensive error reporting and recovery

---

## Hugging Face Image Generator

### Description

Generate high-quality images from text descriptions using Hugging Face's Stable Diffusion models.

### Configuration

- **API Key** (Required): Obtain a Hugging Face API key from your HuggingFace account and set it in the tool's configuration in Open WebUI.
- **API URL** (Optional): Uses Stability AI's SD 3.5 Turbo model as default. Can be customized to use other HF text-to-image model endpoints.

### Usage

- **Example:**

  ```python
  Create an image of "beautiful horse running free"
  ```

- Multiple image format options: Square, Landscape, Portrait, etc.

![Image Generation Example](img/generate_image_hf.png)
*Example image generated with Hugging Face tool*

---

### Cloudflare Workers AI Image Generator

### Description

Generate images using Cloudflare Workers AI text-to-image models, including FLUX, Stable Diffusion XL, SDXL Lightning, and DreamShaper LCM. This tool provides model-specific prompt preprocessing, parameter optimization, and direct image display in chat. It supports fast and high-quality image generation with minimal configuration.

### Configuration

- `cloudflare_api_token` (str): Your Cloudflare API Token (required)
- `cloudflare_account_id` (str): Your Cloudflare Account ID (required)
- `default_model` (str): Default model to use (e.g., `@cf/black-forest-labs/flux-1-schnell`)

**Prerequisites**: Obtain a Cloudflare API Token and Account ID from your Cloudflare dashboard. No additional dependencies beyond `requests`.

### Usage

- **Example:**

  ```python
  # Generate an image with a prompt
  await tools.generate_image(prompt="A futuristic cityscape at sunset, vibrant colors")
  ```

- Returns a markdown-formatted image link for immediate display in chat.

### Features

- **Multiple Models:** Supports FLUX, SDXL, SDXL Lightning, DreamShaper LCM
- **Prompt Optimization:** Automatic prompt enhancement for best results per model
- **Parameter Handling:** Smart handling of steps, guidance, negative prompts, and size
- **Direct Image Display:** Returns markdown image links for chat
- **Error Handling:** Comprehensive error and status reporting
- **Real-time Status:** Progress updates via event emitter

---

### SearxNG Image Search Tool

### Description

Search and retrieve images from the web using a self-hosted [SearxNG](https://searxng.org/) instance. This tool provides privacy-respecting, multi-engine image search with direct image display in chat. Ideal for finding diverse images from multiple sources without tracking or ads.

### Configuration

- `SEARXNG_ENGINE_API_BASE_URL` (str): The base URL for the SearxNG search engine API (default: `http://searxng:4000/search`)
- `MAX_RESULTS` (int): Maximum number of images to return per search (default: 5)

**Prerequisites**: You must have a running SearxNG instance. See [SearxNG documentation](https://docs.searxng.org/) for setup instructions.

### Usage

- **Example:**

  ```python
  # Search for images of cats
  await tools.search_images(query="cats", max_results=3)
  ```

- Returns a list of markdown-formatted image links for immediate display in chat.

### Features

- **Privacy-Respecting:** No tracking, ads, or profiling
- **Multi-Engine:** Aggregates results from multiple search engines
- **Direct Image Display:** Images are formatted for chat display
- **Customizable:** Choose engines, result count, and more
- **Error Handling:** Handles connection and search errors gracefully

---

### ComfyUI ACE Step Audio Tool

### Description

Generate music using the ACE Step AI model via ComfyUI. This tool lets you create songs from tags and lyrics, with full control over the workflow JSON and node numbers. Designed for advanced music generation and can be customized for different genres and moods.

### Configuration

- `comfyui_api_url` (str): ComfyUI API endpoint (e.g., `http://localhost:8188`)
- `model_name` (str): Model checkpoint to use (default: `ACE_STEP/ace_step_v1_3.5b.safetensors`)
- `workflow_json` (str): Full ACE Step workflow JSON as a string. Use `{tags}`, `{lyrics}`, and `{model_name}` as placeholders.
- `tags_node` (str): Node number for the tags input (default: `"14"`)
- `lyrics_node` (str): Node number for the lyrics input (default: `"14"`)
- `model_node` (str): Node number for the model checkpoint input (default: `"40"`)

### Usage

1. **Import the ACE Step workflow:**
   - In ComfyUI, go to the workflow import section and load `extras/ace_step_api.json`.
   - Adjust nodes as needed for your setup.
2. **Configure the tool in Open WebUI:**
   - Set the `comfyui_api_url` to your ComfyUI backend.
   - Paste the workflow JSON (from the file or your own) into `workflow_json`.
   - Set the correct node numbers if you modified the workflow.
3. **Generate music:**
   - Provide tags and (optionally) lyrics.
   - The tool will return a link to the generated audio file.

- **Example:**

  ```python
  Generate a song in the style of "funk, pop, soul" with the following lyrics: "In the shadows where secrets hide..."
  ```

*Returns a link to the generated audio or a status message. Advanced users can fully customize the workflow for different genres, moods, or creative experiments.*

---

## 🔄 Function Pipes

### Flux Kontext ComfyUI Pipe

### Description

A pipe that connects Open WebUI to the **Flux Kontext** image-to-image editing model through ComfyUI. This integration allows for advanced image editing, style transfers, and other creative transformations using the Flux Kontext workflow. Features an interactive `/setup` command system for easy configuration by administrators.

### Configuration

The pipe includes an interactive setup system that allows administrators to configure all settings through chat commands. Most configuration can be done using the `/setup` command, which provides an interactive form for easy adjustment of parameters.

**Key Configuration Options:**

- **COMFYUI_ADDRESS**: Address of the running ComfyUI server (default: `http://127.0.0.1:8188`)
- **COMFYUI_WORKFLOW_JSON**: The entire ComfyUI workflow in JSON format
- **PROMPT_NODE_ID**: Node ID for text prompt input (default: `"6"`)
- **IMAGE_NODE_ID**: Node ID for Base64 image input (default: `"196"`)
- **KSAMPLER_NODE_ID**: Node ID for the sampler node (default: `"194"`)
- **ENHANCE_PROMPT**: Enable vision model-based prompt enhancement (default: `False`)
- **VISION_MODEL_ID**: Vision model to use for prompt enhancement
- **UNLOAD_OLLAMA_MODELS**: Free RAM by unloading Ollama models before generation (default: `False`)
- **MAX_WAIT_TIME**: Maximum wait time for generation in seconds (default: `1200`)
- **AUTO_CHECK_MODEL_LOADER**: Auto-detect model loader type for .safetensors or .gguf (default: `False`)

### Usage

#### Initial Setup

1. **Import the workflow:**
   - In ComfyUI, import `extras/flux_context_owui_api_v1.json` as a workflow
   - Adjust node IDs if you modify the workflow

2. **Configure using /setup command (Admin only):**
   - Type `/setup` in the chat to launch the interactive configuration form
   - The form will display all current settings with input fields
   - Adjust any settings you need to change
   - Submit the form to apply and optionally save the configuration
   - Settings can be persisted to a backend config file for permanent storage

3. **Alternative: Manual configuration:**
   - Access the pipe's Valves in Open WebUI's admin panel
   - Set `COMFYUI_ADDRESS` to your ComfyUI backend
   - Paste the workflow JSON into `COMFYUI_WORKFLOW_JSON`
   - Configure node IDs and other parameters as needed

#### Using the Pipe

1. **Basic image editing:**
   - Upload an image to the chat
   - Provide a text prompt describing the desired changes
   - The pipe processes the image through ComfyUI and returns the edited result

2. **Enhanced prompts (optional):**
   - Enable `ENHANCE_PROMPT` in settings
   - Set a `VISION_MODEL_ID` (e.g., a multimodal model like LLaVA or GPT-4V)
   - The vision model will analyze the input image and automatically refine your prompt for better results

3. **Memory management:**
   - Enable `UNLOAD_OLLAMA_MODELS` to free RAM before generation
   - The default workflow includes a `Clean VRAM` node for VRAM management in ComfyUI

**Example - Image editing:**

```
Prompt: "Edit this image to look like a medieval fantasy king, preserving facial features."
[Upload image]
```

![Flux Kontext Setup](img/flux_kontext_setup.png)
*Example of Flux Kontext /setup command interface*

![Flux Kontext Example](img/flux_kontext_without_parameters.png)
*Example of Flux Kontext image editing output*


---

### Planner Agent v2

**Advanced autonomous agent with specialized model support, interactive user guidance, and comprehensive execution management.**

This powerful agent autonomously generates and executes multi-step plans to achieve complex goals. It's a generalist agent capable of handling any text-based task, making it ideal for complex requests that would typically require multiple prompts and manual intervention.

### 🚀 Key Features

* **🧠 Intelligent Planning:** Automatically breaks down goals into actionable steps with dependency mapping
* **🎨 Specialized Models:** Dedicated models for writing (WRITER_MODEL), coding (CODER_MODEL), and tool usage (ACTION_MODEL) with automatic routing
* **🔍 Quality Control:** Real-time output analysis with quality scoring (0.0-1.0) and iterative improvement
* **🎭 Interactive Error Handling:** When actions fail or produce low-quality outputs, the system pauses and prompts you with options: retry with custom guidance/instructions, retry as-is, approve current output despite warnings, or abort the entire plan execution
* **📊 Live Progress:** Real-time Mermaid diagrams with color-coded status indicators
* **🧩 Template System:** Final synthesis using `{{action_id}}` placeholders for seamless content assembly
* **🔧 Native Tool Integration:** Automatically discovers and uses all available Open WebUI tools
* **⚡ Advanced Features:** Lightweight context mode, concurrent execution, cross-action references (`@action_id`), and comprehensive validation
* **🔮 MCP(OpenAPI servers) Support:** Model Context Protocol integration coming soon for extended tool capabilities

### ⚙️ Configuration

**Core Models:**

- `MODEL`: Main planning LLM
- `ACTION_MODEL`: Tool-based actions and general tasks  
- `WRITER_MODEL`: Creative writing and documentation
- `CODER_MODEL`: Code generation and development

**Temperature Controls:**

- `PLANNING_TEMPERATURE` (0.8): Planning creativity
- `ACTION_TEMPERATURE` (0.7): Tool execution precision
- `WRITER_TEMPERATURE` (0.9): Creative writing freedom
- `CODER_TEMPERATURE` (0.3): Code generation accuracy
- `ANALYSIS_TEMPERATURE` (0.4): Output analysis precision

**Execution Settings:**

- `MAX_RETRIES` (3): Retry attempts per action
- `CONCURRENT_ACTIONS` (1): Parallel processing limit
- `ACTION_TIMEOUT` (300): Individual action timeout
- `SHOW_ACTION_SUMMARIES` (true): Detailed execution summaries
- `AUTOMATIC_TAKS_REQUIREMENT_ENHANCEMENT` (false): AI-enhanced requirements

### 💡 Usage Examples


**Multi-Media Content:**

```
search the latest AI news and create a song based on that, with that , search for stock images to use a “album cover” and create a mockup of the spotify in a plain html file with vanilla js layout using those assets embeded for interactivity
```

![Planner Agent Example](img/planner_2.png)
*Example of Planner Agent in action Using Gemini 2.5 flash and local music generation*


**Creative Writing:**

```
create an epic sci fi Adult novel based on the current trends on academia news and social media about AI and other trending topics, with at least 10 chapters, well crafter world with rich characters , save each chapter in a folter named as the novel in obsidian with an illustration
```

![Planner Agent Example](img/planner_3.png)
*Example of Planner Agent in action Using Gemini 2.5 flash and local image generation, local saving to obsidian and websearch*


**Interactive Error Recovery:**
The Planner Agent features intelligent error handling that engages with users when actions fail or produce suboptimal results. When issues occur, the system pauses execution and presents you with interactive options:

- **Retry with Guidance:** Provide custom instructions to help the agent understand what went wrong and how to improve
- **Retry As-Is:** Attempt the action again without modifications
- **Approve Output:** Accept warning-level outputs despite quality concerns
- **Abort Execution:** Stop the entire plan if the issue is critical

```
Example scenario: If an action fails to generate proper code or retrieve expected data, 
you'll be prompted to either provide specific guidance ("try using a different approach") 
or decide whether to continue with the current output.
```

![Planner Agent Example](img/planner_error.png)
*Interactive error recovery dialog showing user options when an action encounters issues during plan execution*



**Technical Development:**

```
Create a fully-featured Conway's Game of Life SPA with responsive UI, game controls, and pattern presets using vanilla HTML/CSS/JS
```

![Planner Agent Example](img/planner.png)
*Example of Planner Agent in action Using local Hermes 8b (previous verision of the script)*

---

### arXiv Research MCTS Pipe

### Description

Search arXiv.org for relevant academic papers and iteratively refine a research summary using a Monte Carlo Tree Search (MCTS) approach.

### Configuration

- `model`: The model ID from your LLM provider
- `tavily_api_key`: Required. Obtain your API key from tavily.com
- `max_web_search_results`: Number of web search results to fetch per query
- `max_arxiv_results`: Number of results to fetch from the arXiv API per query
- `tree_breadth`: Number of child nodes explored per MCTS iteration
- `tree_depth`: Number of MCTS iterations
- `exploration_weight`: Controls balance between exploration and exploitation
- `temperature_decay`: Exponentially decreases LLM temperature with tree depth
- `dynamic_temperature_adjustment`: Adjusts temperature based on parent node scores
- `maximum_temperature`: Initial LLM temperature (default 1.4)
- `minimum_temperature`: Final LLM temperature at max tree depth (default 0.5)

### Usage

- **Example:**

  ```python
  Do a research summary on "DPO laser LLM training"
  ```

![arXiv MCTS Example](img/Research_mcts.png)
*Example of arXiv Research MCTS Pipe output*

---

### Multi Model Conversations Pipe

### Description

Simulate conversations between multiple language models, each acting as a distinct character. Configure up to 5 participants with unique personas, models, and behaviors. This pipe enables engaging multi-agent discussions, debates, and collaborative problem-solving scenarios.

### Configuration

This pipe features **User Valves** - configurable settings that are unique per user and can be customized for each individual chat session. This means each user can have their own conversation setups, and different chats can have different participant configurations.

**Core Settings:**

- `number_of_participants`: Set the number of participants (1-5)
- `rounds_per_user_message`: How many rounds of replies before the user can send another message

**Per-Participant Configuration (for each of the 5 participants):**

- `participant_[1-5]_model`: Model for each participant (e.g., qwen3:14b, gpt-4, claude-3)
- `participant_[1-5]_alias`: Display name for each participant (e.g., "Optimist", "Skeptic")
- `participant_[1-5]_system_message`: Persona and instructions for each participant
- `all_participants_appended_message`: Global instruction appended to each prompt

**Model Parameters:**

- `temperature`, `top_k`, `top_p`: Standard model parameters for generation control

### Accessing User Valves

To configure the conversation participants and settings:

1. **In the new chat interface:** Click the settings icon (gear icon) in the chat input area
2. A "Valves" panel will open showing all configurable parameters
3. Adjust the number of participants, models, aliases, and system messages
4. Each chat can have its own unique configuration
5. Settings are saved per-chat, allowing you to maintain different conversation setups

![Multi Model Conversation Valves](img/conversations.png)
*Example of Multi Model Conversations User Valves configuration panel*

### Usage

- **Example:**

  ```python
  Start a conversation between three AI agents about climate change.
  ```

**Use Cases:**

- **Debates:** Set up opposing viewpoints (optimist vs. skeptic)
- **Brainstorming:** Multiple creative perspectives on a problem
- **Role-playing:** Interactive storytelling with multiple characters
- **Analysis:** Different analytical approaches to the same topic
- **Expert Panels:** Simulate domain experts discussing a complex issue

---

### Resume Analyzer Pipe

### Description

Analyze resumes and provide tags, first impressions, adversarial analysis, potential interview questions, and career advice.

### Configuration

- `model`: The model ID from your LLM provider
- `dataset_path`: Local path to the resume dataset CSV file
- `rapidapi_key` (optional): For job search functionality
- `web_search`: Enable/disable web search for relevant job postings
- `prompt_templates`: Customizable templates for all steps

### Usage

1. **Requires the Full Document Filter** (see below) to work with attached files.
2. **Example:**

  ```python
Analyze this resume:
[Attach resume file]
  ```

![Resume Analyzer Example 1](img/resume_1.png)
![Resume Analyzer Example 2](img/resume_2.png)
![Resume Analyzer Example 3](img/resume_3.png)
*Screenshots of Resume Analyzer Pipe output*

---

### Mopidy Music Controller

### Description

Control your Mopidy music server to play songs from the local library or YouTube, manage playlists, and handle various music commands. This pipe provides an intuitive interface for music playback, search, and playlist management through natural language commands.

**⚠️ Requirements**: This pipe requires [Mopidy-Iris](https://github.com/jaedb/Iris) to be installed for the player interface. Iris provides a beautiful, feature-rich web interface for controlling Mopidy.

### Configuration

- `model`: The model ID from your LLM provider
- `mopidy_url`: URL for the Mopidy JSON-RPC API endpoint (default: `http://localhost:6680/mopidy/rpc`) - Iris UI must be installed
- `youtube_api_key`: YouTube Data API key for search functionality
- `temperature`: Model temperature (default: 0.7)
- `max_search_results`: Maximum number of search results to return (default: 5)
- `system_prompt`: System prompt for request analysis

### Prerequisites

1. **Mopidy Server**: Install and configure [Mopidy](https://mopidy.com/)
2. **Mopidy-Iris**: Install the Iris web interface:
   ```bash
   pip install Mopidy-Iris
   ```
3. **Optional Extensions**:
   - Mopidy-Local (for local library)
   - Mopidy-YouTube (for YouTube playback)

### Usage

- **Example:**

  ```python
  Play the song "Imagine" by John Lennon
  ```

- **Quick text commands:** stop, halt, play, start, resume, continue, next, skip, pause

### Features

- **Natural Language Control**: Use conversational commands to control playback
- **YouTube Integration**: Search and play songs directly from YouTube
- **Local Library Support**: Access and play songs from your local Mopidy library
- **Playlist Management**: Create, modify, and manage playlists
- **Iris UI Integration**: Beautiful, professional web interface with full playback controls
- **Seamless Embedding**: Iris player embedded directly in Open WebUI chat interface

![Mopidy Example](img/mopidy_0_7.png)
*Example of Mopidy Music Controller Pipe with Iris UI (v0.7.0)*

---

### Letta Agent Pipe

### Description

Connect with Letta agents, enabling seamless integration of autonomous agents into Open WebUI conversations. Supports task-specific processing and maintains conversation context while communicating with the agent API.

### Configuration

- `agent_id`: The ID of the Letta agent to communicate with
- `api_url`: Base URL for the Letta agent API (default: `http://localhost:8283`)
- `api_token`: Bearer token for API authentication
- `task_model`: Model to use for title/tags generation tasks
- `custom_name`: Name of the agent to be displayed
- `timeout`: Timeout to wait for Letta agent response in seconds (default: 400)

### Usage

- **Example:**

  ```python
  Chat with the built in Long Term memory Letta MemGPT agent.
  ```

![Letta Example](img/Letta.png)
*Example of Letta Agent Pipe*

---

### MCP Pipe

### Description

The MCP Pipe integrates the Model Context Protocol (MCP) into Open WebUI, enabling seamless connections between AI assistants and various data sources, tools, and development environments. **Note: This implementation only works with Python-based MCP servers. NPX or other server types are not supported by default.**

MCP is a universal, open standard that replaces fragmented integrations with a single protocol for connecting AI systems with data sources. This allows you to:

- Connect to multiple MCP servers simultaneously (Python servers only)
- Access tools and prompts from connected servers
- Process queries using context-aware tools
- Support data repositories, business tools, and development environments
- Automatically discover tools and prompts
- Stream responses from tools
- Maintain conversation context across different data sources

### Prerequisites

- **Open WebUI**: Make sure you are running a compatible version (0.5.0+ recommended)
- **Python MCP servers**: You must have one or more MCP-compatible servers installed and accessible (see [open-webui/openapi-servers](https://github.com/open-webui/openapi-servers) for examples)
- **MCP configuration file**: A `config.json` file must be placed in the `/data/` folder inside your Open WebUI installation
- **Python environment**: Any additional MCP servers you add must be installed in the Open WebUI Python environment

### Step-by-Step Setup

1. **Install or set up your MCP servers**

   - Example: [mcp_server_time](https://github.com/open-webui/openapi-servers) for time and timezone conversion, [mcp_server_tavily](https://github.com/open-webui/openapi-servers) for web search
   - Install via pip or clone and install as needed

2. **Create the MCP configuration file**

   - Place a `config.json` file in the `/data/` directory of your Open WebUI installation

   - Example `config.json`:

     ```json
     {
         "mcpServers": {
             "time_server": {
                 "command": "python",
                 "args": ["-m", "mcp_server_time", "--local-timezone=America/New_York"],
                 "description": "Provides Time and Timezone conversion tools."
             },
             "tavily_server": {
                 "command": "python",
                 "args": ["-m", "mcp_server_tavily", "--api-key=tvly-xxx"],
                 "description": "Provides web search capabilities tools."
             }
         }
     }
     ```

   - Replace `tvly-xxx` with your actual Tavily API key

   - Add additional servers as needed, following the same structure

3. **Install any required MCP servers**

   - For each server listed in your config, ensure it is installed in the Open WebUI Python environment
   - Example: `pip install mcp_server_time` or clone and install from source

4. **Restart Open WebUI**

   - This ensures the new configuration and servers are loaded

5. **Configure the MCP Pipe in Open WebUI**

   - Set the valves as needed (see below)

### Configuration Valves

- `MODEL`: (default: "Qwen2_5_16k:latest") The LLM model to use for MCP queries
- `OPENAI_API_KEY`: Your OpenAI API key for API access (if using OpenAI-compatible models)
- `OPENAI_API_BASE`: (default: "http://0.0.0.0:11434/v1") Base URL for API requests
- `TEMPERATURE`: (default: 0.5) Controls randomness in responses (0.0-1.0)
- `MAX_TOKENS`: (default: 1500) Maximum tokens to generate
- `TOP_P`: (default: 0.8) Top-p sampling parameter
- `PRESENCE_PENALTY`: (default: 0.8) Penalty for repeating topics
- `FREQUENCY_PENALTY`: (default: 0.8) Penalty for repeating tokens

### Example Usage

```python
# Example usage in your prompt
Use the time_server to get the current time in New York.
```

- You can also use the Tavily server for web search, or any other MCP server you have configured.
- The MCP Pipe will automatically discover available tools and prompts from all configured servers.

### Troubleshooting & Tips

- **Python servers only**: This pipe does not support NPX or non-Python MCP servers. For NPX support, see the advanced MCP Pipeline below.
- **Server not found**: Make sure the MCP server is installed and accessible in the Python environment used by Open WebUI
- **Config file not loaded**: Double-check the location (`/data/config.json`) and syntax of your config file
- **API key issues**: Ensure all required API keys (e.g., Tavily, OpenAI) are set correctly in the config and valves
- **Advanced features**: For more advanced MCP features (including NPX server support), see the [MCP Pipeline Documentation](Pipelines/MCP_Pipeline/README_MCP_Pipeline.md)
- **Logs**: Check Open WebUI logs for errors related to MCP server startup or communication

### Reference: Advanced MCP Pipeline

If you need more advanced features, such as NPX server support, see the documentation in `Pipelines/MCP_Pipeline/README_MCP_Pipeline.md` in this repository.

---

## 🔧 Filters

### Prompt Enhancer Filter

### Description

Uses an LLM to automatically improve the quality of your prompts before they are sent to the main language model.

### Configuration

- `user_customizable_template`: Tailor the instructions given to the prompt-enhancing LLM
- `show_status`: Displays status updates during the enhancement process
- `show_enhanced_prompt`: Outputs the enhanced prompt to the chat window
- `model_id`: Select the specific model to use for prompt enhancement

### Usage

- Enable in your model configuration's filters section.
- The filter will automatically process each user message before it's sent to the main LLM.

![Prompt Enhancer Example](img/prompt_enhancer.png)

---


### Semantic Router Filter

### Description

Acts as an intelligent model router that analyzes the user's message and available models, then automatically selects the most appropriate model, pipe, or preset for the task. Features vision model filtering, **dynamic vision re-routing**, conversation persistence, knowledge base integration, and robust file handling with Open WebUI's RAG system.

The filter uses an innovative invisible text marker system to persist routing decisions across conversation turns. When a model is selected, the filter emits zero-width unicode characters in the first assistant message. These markers are **invisible to the LLM** (stripped before processing) but persist in the chat database, ensuring the same model, tools, and knowledge bases are used throughout the entire conversation without requiring metadata or system message manipulation.

The filter automatically detects when images are added to an existing conversation and intelligently re-routes to a vision-capable model if the current model lacks vision support. This enables seamless transitions from text-only conversations to image-based interactions without manual model switching.

![Semantic Router Example](img/semantic_router_with_images.png)

### Configuration Valves

- **vision_fallback_model_id**: Fallback model for image queries when no vision-capable models are available
- **banned_models**: List of model IDs to exclude from routing selection
- **allowed_models**: List of model IDs to whitelist (when set, only these models will be considered)
- **router_model_id**: Specific model to use for routing decisions (leave empty to use current model)
- **system_prompt**: System prompt for the router model (customizable)
- **disable_qwen_thinking**: Append `/no_think` to router prompt for Qwen models
- **show_reasoning**: Display routing reasoning in chat
- **status**: Show status updates in chat
- **debug**: Enable debug logging

### Features

- **Conversation Persistence**: Routes only on first user message, then automatically maintains the selected model throughout the conversation using invisible text markers
- **Dynamic Vision Re-Routing**: Automatically detects when images are added mid-conversation and re-routes to a vision-capable model if the current model lacks vision support
- **Vision Model Filtering**: Automatically filters model selection to only vision-capable models when images are detected in the conversation (checks `meta.capabilities.vision` flag)
- **Smart Fallback**: Uses `vision_fallback_model_id` only when no vision models are available in the filtered list
- **Knowledge Base Integration**: Properly handles files from knowledge collections with full RAG retrieval support
- **Tool Preservation**: Maintains model-specific tools across conversation turns
- **File Structure Compliance**: Passes files in correct INPUT format to Open WebUI's `get_sources_from_items()` for proper RAG processing
- **Whitelist Support**: Use `allowed_models` to restrict selection to specific models only
- **Cross-Backend Compatibility**: Automatically converts payloads between OpenAI and Ollama formats when routing between different backend types
- **Automatic Fallback**: Gracefully handles errors by falling back to the original model

### Usage

1. Enable in your model configuration's filters section
2. Configure `vision_fallback_model_id` to specify a fallback model for image queries
3. Optionally set `allowed_models` to create a whitelist of preferred models, or use `banned_models` to exclude specific ones
4. The filter will automatically:
   - Route on the first user message only (analyzes task requirements and available models)
   - Emit an invisible marker that persists the routing decision in chat history
   - Detect and restore routing on subsequent messages in the conversation
   - **Re-route dynamically when images are added to a conversation if the current model lacks vision capability**
   - Detect images in conversations and filter to vision-capable models when present
   - Preserve the selected model's tools and knowledge bases throughout the conversation
   - Attach relevant files from knowledge collections with proper RAG retrieval
   - Convert payloads between OpenAI and Ollama formats as needed

### How It Works

**First Message (Routing):**

1. Analyzes user message and available models
2. Filters to vision-capable models if images are detected
3. Routes to the best model for the task
4. Emits invisible unicode marker (e.g., `​‌‍⁠model-id​‌‍⁠`) in first assistant message
5. Preserves model's tools, knowledge bases, and configuration

**Subsequent Messages (Persistence):**

1. Detects invisible marker in conversation history
2. Extracts persisted model ID
3. **Checks if images are present but current model lacks vision capability** ⭐ NEW
4. If vision mismatch detected, triggers fresh routing with vision filter
5. Otherwise, reconstructs full routing (model + tools + knowledge + metadata)
6. Strips marker from message content (invisible to LLM)
7. Continues conversation with same model and configuration

**Dynamic Vision Re-Routing Example:**

```
User: "Explain quantum physics"
→ Router selects text model (e.g., llama3.2:latest)

User: "Thanks! Now what's in this picture?" [attaches image]
→ Filter detects: images present + current model lacks vision
→ Automatically triggers re-routing with vision filter
→ Router selects vision model (e.g., llama3.2-vision:latest)
→ Vision model processes image and responds
```

### How Vision Filtering Works

When images are detected in the conversation:

1. Filter checks all available models for `meta.capabilities.vision` flag
2. Only vision-capable models are included in the routing selection
3. If no vision models are found, uses `vision_fallback_model_id` as fallback
4. Router model receives images for contextual routing decisions
5. If router model doesn't support vision, automatically switches to vision fallback for routing

![Semantic Router](img/semantic_router.png)


---

### Full Document Filter

### Description

Allows Open WebUI to process entire attached files (such as resumes or documents) as part of the conversation. Cleans and prepends the file content to the first user message, ensuring the LLM receives the full context.

### Configuration

- `priority` (int): Priority level for the filter operations (default: `0`)
- `max_turns` (int): Maximum allowable conversation turns for a user (default: `8`)

#### User Valves

- `max_turns` (int): Maximum allowable conversation turns for a user (default: `4`)

### Usage

- Enable the filter in your model configuration.

- When you attach a file in Open WebUI, the filter will automatically clean and inject the file content into your message.

- No manual configuration is needed for most users.

- **Example:**

  ```python
  Analyze this resume:
  [Attach resume file]
  ```

---


## Clean Thinking Tags Filter

### Description

Checks if an assistant's message ends with an unclosed or incomplete "thinking" tag. If so, it extracts the unfinished thought and presents it as a user-visible message.

### Configuration

- No configuration required.

### Usage

- Works automatically when enabled.

---


## 🎨 Using the Provided ComfyUI Workflows

### Importing a Workflow

1. Open ComfyUI.
2. Click the "Load Workflow" or "Import" button.
3. Select the provided JSON file (e.g., `ace_step_api.json` or `flux_context_owui_api_v1.json`).
4. Save or modify as needed.
5. Use the node numbers in your Open WebUI tool configuration.

### Best Practices

- Always check node numbers after importing, as they may change if you modify the workflow.
- You can create and share your own workflows by exporting them from ComfyUI.


### Why this matters

This approach allows you to leverage state-of-the-art image and music generation/editing models with full control and customization, directly from Open WebUI.

---

## 📦 Installation

### From Open WebUI Hub (Recommended)

- Visit [https://openwebui.com/u/haervwe](https://openwebui.com/u/haervwe)
- Click "Get" for desired tool/pipe/filter.
- Follow prompts in your Open WebUI instance.

### Manual Installation

- Copy `.py` files from `tools/`, `functions/`, or `filters/` into Open WebUI via the Workspace > Tools/Functions/Filters section.
- Provide a name and description, then save.

---

## 🤝 Contributing

Feel free to contribute to this project by:

1. Forking the repository
2. Creating your feature branch
3. Committing your changes
4. Opening a pull request

---

## 📄 License

MIT License

---

## 🙏 Credits

- Developed by Haervwe
- Credit to the amazing teams behind:
  - https://github.com/ollama/ollama
  - https://github.com/open-webui/open-webui
- And all model trainers out there providing these amazing tools.

---

## 🎯 Usage Examples

### Academic Research

```python
# Search for recent papers on a topic
Search for recent papers about "large language model training"

# Conduct comprehensive research
Do a research summary on "DPO laser LLM training"
```

### Creative Projects

```python
# Generate images
Create an image of "beautiful horse running free"

# Create music
Generate a song in the style of "funk, pop, soul" with lyrics: "In the shadows where secrets hide..."

# Edit images
Edit this image to look like a medieval fantasy king, preserving facial features
```

### Productivity

```python
# Analyze documents
Analyze this resume: [Attach resume file]

# Plan complex tasks
Create a fully-featured Single Page Application (SPA) for Conway's Game of Life
```

### Multi-Agent Conversations

```python
# Start group discussions
Start a conversation between three AI agents about climate change
```

---

## 🌟 Community & Ecosystem

This collection is part of the broader Open WebUI ecosystem. Here's how you can get involved:

- **🔗 Open WebUI Hub**: Discover more tools at [openwebui.com](https://openwebui.com)
- **📚 Documentation**: Learn more about Open WebUI at [docs.openwebui.com](https://docs.openwebui.com)
- **💡 Ideas**: Share your ideas and feature requests
- **🐛 Bug Reports**: Help improve the tools by reporting issues
- **🌟 Star the Repository**: Show your support by starring this repo

---

## 💬 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.