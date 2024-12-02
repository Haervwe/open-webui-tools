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
- Search across paper titles, abstracts, and full text from Arxiv and Web with tavily
- uses a MCTS aproach to make progressive refinements on a Research Summary on a given topic
- uses a visual representantion of the tree nodes to provide some visual feedback
- shows intermediate steps
- Configurable Width and Breadth of the search on valves.


### 3. Multi Model Conversations Pipe

This pipe allows you to simulate conversations between multiple language models, each acting as a distinct character.  You can configure up to 5 participants, each with their own model, alias, and character description (system message). This enables complex and dynamic interactions, perfect for storytelling, roleplaying, or exploring different perspectives on a topic.

**Features:**

* **Multiple Participants:** Simulate conversations with up to 5 different language models.
* **Character Definition:**  Craft unique personas for each participant using system messages.
* **Round-Robin Turns:** Control the flow of conversation with configurable rounds per user message.
* **Streaming Support:**  See the conversation unfold in real-time with streaming output.



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
  - **Admin**:
    - **Model:** the model id from your llm provider conected to Open-WebUI
    - **Concurrency:** (do not use this if you dont whant to debug it, its higly experimental and nor correctly implemented leave it as 1)
    - **Max retries:** Number of times the refelction step and subsequent refinement can happen per step. 

### arXiv Search Tool
No configuration required! The tool works out of the box.

### arXiv Research MCTS Pipe

  - **Admin:**

    - **Tavily API KEY:** required, create it in tavily.com
    - **Max Search Results:** Amount of results fo fech form web search
    - **Arxiv Max Results:** Amount of results to fetch for arxiv API

  - **User:**

    - **Tree Breadth:** how many nodes per round are searched-
    - **Tree Depth:** how many rounds are made.
    - **Exploration Weight:** constant to control exploration vs exploitation, (a higher value means more exploration of new paths, while a low values makes the system stick with one option, range from 0 to 2 recommended)

### Hugging Face Image Generator
Required configuration in Open WebUI:

1. **API Key** (Required):
   - Obtain a Hugging Face API key from your HuggingFace account
   - Set it in the tool's configuration in Open WebUI

2. **API URL** (Optional):
   - Default: Uses Stability AI's SD 3.5 Turbo model
   - Can be customized to use other HF text-to-image model endpoints such as flux
  
### Prompt Enhancer Filter

* **User Customizable Template:**  Allows you to tailor the instructions given to the prompt-enhancing LLM.
* **Show Status:** Displays status updates during the enhancement process.
* **Show Enhanced Prompt:**  Outputs the enhanced prompt to the chat window for visibility.
* **Model ID:** Select the specific model to use for prompt enhancement.


### Multi Model Conversations Pipe

* **Number of Participants:** Set the number of participants (1-5).
* **Rounds per User Message:** Configure how many rounds of replies occur before the user can send another message.
* **Participant [1-5] Model:** Select the model for each participant.
* **Participant [1-5] Alias:** Set a display name for each participant.
* **Participant [1-5] System Message:** Define the persona and instructions for each participant.
* **All Participants Appended Message:** A global instruction appended to each participant's prompt.
* **Temperature, Top_k, Top_p:** Standard model parameters.

* **(note, the valves for the characters that wont be used must be setted to default or have correct paramenters) 


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

Select the pipe in the Open WebUI interface.  Configure the valves (settings) for the desired conversation setup. Start the conversation by sending a user message.

![Screenshot from 2024-11-30 20-54-28](https://github.com/user-attachments/assets/dc489a5b-f3a1-42a8-a7d5-b0ed1e570af8)


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
