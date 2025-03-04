# Open WebUI Tools Collection

A collection of tools for Open WebUI that provides structured planning and execution capability, arXiv paper search capabilities, Hugging Face text-to-image generation functionality, prompt enhancement, and multi-model conversations. Perfect for enhancing your LLM interactions with academic research, image generation, and advanced conversation management!

## Tools Included

### 1. arXiv Search Tool
Search arXiv.org for relevant academic papers on any topic. No API key required!

Features:
- Search across paper titles, abstracts, and full text
- Returns detailed paper information including:
  - Title
  - Authors
  - Publication date
  - URL
  - Abstract
- Automatically sorts by most recent submissions
- Returns up to 5 most relevant papers

### 2. Hugging Face Image Generator
Generate high-quality images from text descriptions using Hugging Face's Stable Diffusion models.

Features:
- Multiple image format options:
  - Default/Square (1024x1024)
  - Landscape (1024x768)
  - Landscape Large (1440x1024)
  - Portrait (768x1024)
  - Portrait Large (1024x1440)
- Customizable model endpoint
- High-resolution output

## Function Pipes Included

### 1. Planner Agent

This powerful agent allows you to define a goal, and it will autonomously generate and execute a plan to achieve it. The Planner is a generalist agent, capable of handling any text-based task, making it ideal for complex, multi-step requests that would typically require multiple prompts and manual intervention. 

It features advanced capabilities like:

* **Automatic Plan Generation:**  Breaks down your goal into a sequence of actionable steps with defined dependencies.
* **Adaptive Execution:**  Executes each step, dynamically adjusting to the results of previous actions.
* **LLM-Powered Consolidation:**  Intelligently merges the outputs of different steps into a coherent final result.
* **Reflection and Refinement:** Analyzes the output of each step, identifies potential issues, and iteratively refines the output through multiple attempts.
* **Robust Error Handling:** Includes retries and fallback mechanisms to ensure successful execution even with occasional API errors.
* **Detailed Execution Summary:** Provides a comprehensive report of the plan execution, including timings and potential issues.

**Features:**

* **General Purpose:** Can handle a wide range of text-based tasks, from creative writing and code generation to research summarization and problem-solving.
* **Multi-Step Task Management:** Excels at managing complex tasks that require multiple steps and dependencies.
* **Context Awareness:**  Maintains context throughout the execution process, ensuring that each step builds upon the previous ones.
* **Output Optimization:**  Employs a reflection mechanism to analyze and improve the output of each step through multiple iterations.


### 2. arXiv Research MCTS Pipe

Search arXiv.org for relevant academic papers on any topic. No API key required!

**Features:**

* **Comprehensive Search:** Searches across paper titles, abstracts, and full text content from both arXiv and the web using Tavily.
* **MCTS-Driven Refinement:** Employs a Monte Carlo Tree Search (MCTS) approach to iteratively refine a research summary on a given topic.
* **Adaptive Temperature Control:** Offers both static and dynamic temperature decay settings.  Static decay progressively reduces the LLM's temperature with each level of the search tree. Dynamic decay adjusts the temperature based on both depth and parent node scores, allowing the LLM to explore more diverse options when previous results are less promising. This fine-grained control balances exploration and exploitation for optimal refinement.
* **Visual Tree Representation:** Provides a visual representation of the search tree, offering intuitive feedback on the exploration process and the relationships between different research directions.
* **Transparent Intermediate Steps:** Shows intermediate steps of the search, allowing users to track the evolution of the research summary and understand the reasoning behind the refinements.
* **Configurable Search Scope:** Allows users to configure the breadth and depth of the search (tree width and depth) to control the exploration scope and computational resources used.



### 3. Multi Model Conversations Pipe

This pipe allows you to simulate conversations between multiple language models, each acting as a distinct character.  You can configure up to 5 participants, each with their own model, alias, and character description (system message). This enables complex and dynamic interactions, perfect for storytelling, roleplaying, or exploring different perspectives on a topic.

**Features:**

* **Multiple Participants:** Simulate conversations with up to 5 different language models.
* **Character Definition:**  Craft unique personas for each participant using system messages.
* **Round-Robin Turns:** Control the flow of conversation with configurable rounds per user message.
* **Group-Chat-Manager:** Use an LLM model to select the next participant in the conversation. (toggleable in valves)
* **Streaming Support:**  See the conversation unfold in real-time with streaming output.

### 4. Resume Analyzer Pipe
Analyze resumes and provide tags, first impressions, adversarial analysis, potential interview questions, and career advice.

**Features:**
- **Resume Analysis:** Breaks down a resume into relevant categories, highlighting strengths and weaknesses.
- **Tags Generation:** Identifies key skills and experience from the resume and assigns relevant tags.
- **First Impression:** Provides an initial assessment of the resume's effectiveness in showcasing the candidate's qualifications for a target role.
- **Adversarial Analysis:** Compares the analyzed resume to similar ones, offering actionable feedback on areas for improvement.
- **Interview Questions:** Suggests insightful questions tailored to the candidate's experience and the target role.
- **Career Advisor Response:** Offers personalized career advice based on the resume analysis and conversation history.

### 5. Mopidy Music Controller

Control your Mopidy music server to play songs from the local library or YouTube, manage playlists, and handle various music commands.

**Features:**
- **Music Playback** Play songs from local library or YouTube
- **Manage playlists** Can reatrieve or create playlist from local or YouTube
- **Quick chat Commands** Handle various music commands (play, pause, resume, skip, etc.)

### 6. Letta Agent Pipe

Connect with Letta agents, enabling seamless integration of autonomous agents into Open WebUI conversations. Supports task-specific processing and maintains conversation context while communicating with the agent API.

**Features:**
- Communicate with Letta agents
- Task-specific processing
- Maintain conversation context


### 7. MCP Pipe

Integrate the Model Context Protocol (MCP) into Open WebUI, enabling seamless connections between AI assistants and various data sources, tools, and development environments. MCP is a universal, open standard that replaces fragmented integrations with a single protocol for connecting AI systems with data sources.

**Features:**
- Connect to multiple MCP servers simultaneously
- Access tools and prompts from connected servers
- Process queries using context-aware tools
- Support for data repositories, business tools, and development environments
- Automatic tool and prompt discovery
- Stream responses from tools
- Maintain conversation context across different data sources


## Filters Included

### 1. Prompt Enhancer Filter

This filter uses an LLM to automatically improve the quality of your prompts before they are sent to the main language model. It analyzes your prompt and the conversation history to create a more detailed, specific, and effective prompt, leading to better responses.

**Features:**

* **Context-Aware Enhancement:** Considers the entire conversation history when refining the prompt.
* **Customizable Template:**  Control the behavior of the prompt enhancer with a customizable template.
* **Improved Response Quality:**  Get more relevant and insightful responses from the main LLM.




## Installation

**1. Installing from Haervwe's Open WebUI Hub (Recommended):**

* Visit [https://openwebui.com/u/haervwe](https://openwebui.com/u/haervwe) to access the collection of tools.

* **For Tools (arXiv Search Tool, Hugging Face Image Generator):**
    * Locate the desired tool on the hub page.
    * Click the "Get" button next to the tool. This will redirect you to your Open WebUI instance and automatically populate the installation code.
    * (Optional) Review the code, provide a name and description (if needed),
    * Save the tool.


* **For Function Pipes (Planner Agent, arXiv Research MCTS Pipe, Multi Model Conversations) and Filters (Prompt Enhancer):**
    * Locate the desired function pipe or filter on the hub page.
    * Click the "Get" button. This will, again, redirect you to your Open WebUI instance with the installation code.
    * (Optional) Review the code, provide a different name and description,
    * Save the function.


**2. Manual Installation from the Open WebUI Interface:**

* **For Tools (arXiv Search Tool, Hugging Face Image Generator):**
    * In your Open WebUI instance, navigate to the "Workspace" tab, then the "Tools" section.
    * Click the "+" button.
    * Copy the entire code of the respective `.py` file from this repository.
    * Paste the code into the text area in the Open WebUI interface.
    * Provide a name and description , and save the tool.


* **For Function Pipes (Planner Agent, arXiv Research MCTS Pipe, Multi Model Conversations) and Filters (Prompt Enhancer):**
    * Navigate to the "Workspace" tab, then the "Functions" section.
    * Click the "+" button.
    * Copy and paste the code from the corresponding `.py` file.
    * Provide a name and description, and save.

**Important Note for the Prompt Enhancer Filter:**
    * To use the Prompt Enhancer, you **must** create a new model configuration in Open WebUI.
    * Go to "Workspace" -> "Models" -> "+".
    * Select a base model.
    * In the "Filters" section of the model configuration, enable the "Prompt Enhancer" filter.
 

## Configuration

### Planner Agent

* **Model:** the model id from your llm provider conected to Open-WebUI
* **Action-Model:** the model to be used in the task execution , leave as default to use the same in all the process.
* **Concurrency:** ("Concurrency support is currently experimental. Due to resource limitations, comprehensive testing of concurrent LLM operations has not been possible. Users may experience unexpected behavior when running multiple LLM processes simultaneously. Further testing and optimization are planned.")
* **Max retries:** Number of times the refelction step and subsequent refinement can happen per step. 

### arXiv Search Tool

No configuration required! The tool works out of the box.

### arXiv Research MCTS Pipeline

  
* **Model:** The model ID from your LLM provider connected to Open WebUI.
* **Tavily API Key:** Required. Obtain your API key from tavily.com.  This is used for web searches.
* **Max Web Search Results:**  The number of web search results to fetch per query.
* **Max arXiv Results:** The number of results to fetch from the arXiv API per query.
* **Tree Breadth:** The number of child nodes explored during each iteration of the MCTS algorithm.  This controls the width of the search tree.
* **Tree Depth:** The number of iterations of the MCTS algorithm. This controls the depth of the search tree.
* **Exploration Weight:** A constant (recommended range 0-2) controlling the balance between exploration and exploitation. Higher values encourage exploration of new branches, while lower values favor exploitation of promising paths.
* **Temperature Decay:** Exponentially decreases the LLM's temperature parameter with increasing tree depth. This focuses the LLM's output from creative exploration to refinement as the search progresses.
* **Dynamic Temperature Adjustment:** Provides finer-grained control over temperature decay based on parent node scores.  If a parent node has a low score, the temperature is increased for its children, encouraging more diverse outputs and potentially uncovering better paths.
* **Maximum Temperature:** The initial temperature of the LLM (0-2, default 1.4). Higher temperatures encourage more diverse and creative outputs at the beginning of the search.
* **Minimum Temperature:** The final temperature of the LLM at maximum tree depth (0-2, default 0.5). Lower temperatures promote focused refinement of promising branches.

### Multi Model Conversations Pipe

* **Number of Participants:** Set the number of participants (1-5).
* **Rounds per User Message:** Configure how many rounds of replies occur before the user can send another message.
* **Participant [1-5] Model:** Select the model for each participant.
* **Participant [1-5] Alias:** Set a display name for each participant.
* **Participant [1-5] System Message:** Define the persona and instructions for each participant.
* **All Participants Appended Message:** A global instruction appended to each participant's prompt.
* **Temperature, Top_k, Top_p:** Standard model parameters.

* **(note, the valves for the characters that wont be used must be setted to default or have correct paramenters)

### Resume Analyzer Pipe

* **Model:** The model ID from your LLM provider connected to Open WebUI.
* **Dataset Path:** Local path to the resume dataset CSV file. Includes "Category" and "Resume" columns.
* **RapidAPI Key (optional):** Required for job search functionality. Obtain an API key from RapidAPI Jobs API.
* **Web Search:** Enable/disable web search for relevant job postings.
* **Prompt templates:** Customizable templates for all the steps


### Hugging Face Image Generator
Required configuration in Open WebUI:

* **API Key** (Required): Obtain a Hugging Face API key from your HuggingFace account and set it in the tool's configuration in Open WebUI
*  **API URL** (Optional): Uses Stability AI's SD 3.5 Turbo model as Default,Can be customized to use other HF text-to-image model endpoints such as flux
  
### Mopidy Music Controller

**Requirements:**
- YouTube API Key
- Mopidy server installed and configured for local and YouTube playback (https://mopidy.com/)

**Valves:**
* **Model:** The model ID from your LLM provider connected to Open WebUI.
* **Mopidy_URL:** URL for the Mopidy JSON-RPC API endpoint (default: `http://localhost:6680/mopidy/rpc`).
* **YouTube_API_Key:** YouTube Data API key for search.
* **Temperature:** Model temperature (default: 0.7).
* **Max_Search_Results:** Maximum number of search results to return (default: 5).
* **Use_Iris:** Toggle to use Iris interface or custom HTML UI (default: True).
* **system_prompt:** System prompt for request analysis.

### Letta Agent Pipe

**Requirements:**
- Letta instance and a configured Letta agent (https://www.letta.com/)

**Valves:**
* **Agent_ID:** The ID of the Letta agent to communicate with , Not the Name the ID(is a long string of numbers and letters beneath the name in ADER).
* **API_URL:** Base URL for the Letta agent API (default: `http://localhost:8283`).
* **API_Token:** Bearer token for API authentication.
* **Task_Model:** Model to use for title/tags generation tasks. If empty, uses the default model.
* **Custom_Name:** Name of the agent to be displayed.
* **Timeout:** Timeout to wait for Letta agent response in seconds (default: 400).

### MCP Pipe

**Requirements:**
- MCP configuration file (`config.json`) placed in the `/data/` folder inside the Open WebUI installation
- Python MCP servers (for this implementation, if you need npx support check out the MCP pipeline in this repo)

**Note:** For a more comprehensive implementation that includes NPX server support and advanced features, check out the [MCP Pipeline Documentation](Pipelines/MCP_Pipeline/README_MCP_Pipeline.md).

**Configuration:**

**Config File:**
1. Create the MCP configuration file `config.json` inside the `/data/` folder:
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
2. 1f you add another python server make sure to pip install them in the open webui environment and config.json file.

**Valves:**
* **MODEL:** Default "Qwen2_5_16k:latest" - The LLM model to use
* **OPENAI_API_KEY:** Your OpenAI API key for API access
* **OPENAI_API_BASE:** Default "http://0.0.0.0:11434/v1" - Base URL for API requests
* **TEMPERATURE:** Default 0.5 - Controls randomness in responses (0.0-1.0)
* **MAX_TOKENS:** Default 1500 - Maximum tokens to generate
* **TOP_P:** Default 0.8 - Top-p sampling parameter
* **PRESENCE_PENALTY:** Default 0.8 - Penalty for repeating topics
* **FREQUENCY_PENALTY:** Default 0.8 - Penalty for repeating tokens

### Prompt Enhancer Filter

* **User Customizable Template:**  Allows you to tailor the instructions given to the prompt-enhancing LLM.
* **Show Status:** Displays status updates during the enhancement process.
* **Show Enhanced Prompt:**  Outputs the enhanced prompt to the chat window for visibility.
* **Model ID:** Select the specific model to use for prompt enhancement.

  
## Usage for Pipes

### Planner Agent


Select the pipe with the corresponding model, it show as this:

![image](https://github.com/user-attachments/assets/32180607-8dd4-42f2-af43-231f82b4d603)

```python
# Example usage in your prompt
"Create a fully-featured Single Page Application (SPA) for the conways game of life, including a responsive UI. No frameworks No preprocessor, No minifing, No back end, ONLY Clean and CORRECT HTML JS AND CSS PLAIN""
```
![image](https://github.com/user-attachments/assets/979260c8-eb68-4838-b9f1-26068146dea4)



### arXiv Research MCTS Pipe

Select the pipe with the corresponding model, it show as this:

![Screenshot from 2024-11-10 17-36-16](https://github.com/user-attachments/assets/e55f96b6-a893-463e-aa6b-4d2ef24012f2)

```python
# Example usage in your prompt
"Do a research summary on "DPO laser LLM training"
```

![Screenshot from 2024-11-10 17-53-51](https://github.com/user-attachments/assets/28116b58-79ef-4d4a-bd51-bfa2f2102cc3)



### Multi Model Conversations Pipe

1.Select the pipe in the Open WebUI interface.  
2.Configure the valves (settings) for the desired conversation setup in the admin panel. 
3 Start the conversation by sending a user message to the conversation pipe.

![Screenshot from 2024-11-30 20-54-28](https://github.com/user-attachments/assets/dc489a5b-f3a1-42a8-a7d5-b0ed1e570af8)


### Resume Analyzer Pipe

**Requirements:**
1. This script requires the full_document filter added in open web ui to work with attached files, you can find it here : https://openwebui.com/f/haervwe/full_document_filter

**Usage:**
1. Select the Resume Analyzer Pipe in the Open WebUI interface.
2. Configure the valves with the desired model, dataset path (optional), and other settings.
3. Send a resume text as an attachment (make sure to user whle document setting) and a message to start the analysis process.
4. Review the first impression, adversarial analysis, interview questions, and then ask for career advice.

**Example Usage:**
```python
# Example usage in your prompt
Analyze this resume:
[Insert resume or resume text here]
```

![Screenshot from 2024-12-13 03-31-54](https://github.com/user-attachments/assets/2b7b16f8-38b0-40eb-8dac-9cae29d917cb)
![Screenshot from 2024-12-13 02-50-15](https://github.com/user-attachments/assets/3520cec6-01fa-4d88-bc61-7d77e7c8ecc9)
![Screenshot from 2024-12-13 03-23-11](https://github.com/user-attachments/assets/5d0a048a-24ba-4726-ad5e-a36daae2b607)


The Resume Analyzer Pipe offers a comprehensive analysis of resumes, providing valuable insights and actionable feedback to help candidates improve their job prospects.

### Mopidy Music Controller

1. Select the Mopidy Music Controller Pipe in the Open WebUI interface.
2. Configure the valves with the desired settings.
3. Send a music-related command to start the process.
4. Use quick text commands for faster response: 
            "stop": "pause",
            "halt": "pause",
            "play": "play",
            "start": "play",
            "resume": "resume",
            "continue": "resume",
            "next": "skip",
            "skip": "skip",
            "pause": "pause",
5. you can also interact with an artifact showing IRIS in the chat.

**Example Usage:**
```python
# Example usage in your prompt
Play the song "Imagine" by John Lennon
```
![Mopidy Image](img/Mopidy.png)



### Letta Agent Pipe

1. Select the Letta Agent Pipe in the Open WebUI interface.
2. Configure the valves with the desired settings.
3. Send a message to start the interaction with the Letta agent.

**Example Usage:**
```python
# Example usage in your prompt
Chat with the built in Long Term memory Letta MemGPT agent.
```
![Letta Image](img/Letta.png)


### MCP Pipe

1. Select the MCP Pipe in the Open WebUI interface.
2. Configure the valves with the desired settings.
3. Send a query to start using the MCP tools and prompts.

**Example Usage:**
```python
# Example usage in your prompt
Use the time_server to get the current time in New York.
```

## Usage for tools

(Make sure to turn on the tool in chat before requesting it)


![Screenshot from 2024-11-09 15-55-58](https://github.com/user-attachments/assets/2956997f-b14f-4087-99d8-48d11a79234b)


### arXiv Search
```python
# Example usage in your prompt
Search for recent papers about "tree of thought"
```

![Screenshot from 2024-11-09 15-56-51](https://github.com/user-attachments/assets/f3997b8f-e0e8-4db6-bb13-d4a41d1dda13)


### Image Generation
```python
# Example usage in your prompt
Create an image of "beutiful horse running free"

# Specify format
Create a landscape image of "a futuristic cityscape"
```

![Screenshot from 2024-11-09 15-58-24](https://github.com/user-attachments/assets/11a2447e-06f3-4456-ab81-d2bcc6d981f3)



## Usage for Filters

### Prompt Enhancer Filter

Use the custom Model template in the model selector.  The filter will automatically process each user message before it's sent to the main LLM.  Configure the valves to customize the enhancement process.

![Screenshot from 2024-12-01 21-16-01](https://github.com/user-attachments/assets/6c335b42-8fde-4aff-924a-308c4ddf28c0)



## Error Handling

Both tools include comprehensive error handling for:
- Network issues
- API timeouts
- Invalid parameters
- Authentication errors (HF Image Generator)

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating your feature branch
3. Committing your changes
4. Opening a pull request

## License

MIT License

## Credits

- Developed by Haervwe

- Credit to the amazing teams behind:
  - https://github.com/ollama/ollama
  - https://github.com/open-webui/open-webui

And all model trainers out there providing these amazing tools.


## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
