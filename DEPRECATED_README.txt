# Deprecated Documentation: Planner Agent v2

This file contains the legacy documentation for Planner Agent v2, which has been replaced by Planner Agent v3 in the main README.

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

\`\`\`
search the latest AI news and create a song based on that, with that , search for stock images to use a “album cover” and create a mockup of the spotify in a plain html file with vanilla js layout using those assets embeded for interactivity
\`\`\`

![Planner Agent Example](img/planner_2.png)
*Example of Planner Agent in action Using Gemini 2.5 flash and local music generation*


**Creative Writing:**

\`\`\`
create an epic sci fi Adult novel based on the current trends on academia news and social media about AI and other trending topics, with at least 10 chapters, well crafter world with rich characters , save each chapter in a folter named as the novel in obsidian with an illustration
\`\`\`

![Planner Agent Example](img/planner_3.png)
*Example of Planner Agent in action Using Gemini 2.5 flash and local image generation, local saving to obsidian and websearch*


**Interactive Error Recovery:**
The Planner Agent features intelligent error handling that engages with users when actions fail or produce suboptimal results. When issues occur, the system pauses execution and presents you with interactive options:

- **Retry with Guidance:** Provide custom instructions to help the agent understand what went wrong and how to improve
- **Retry As-Is:** Attempt the action again without modifications
- **Approve Output:** Accept warning-level outputs despite quality concerns
- **Abort Execution:** Stop the entire plan if the issue is critical

\`\`\`
Example scenario: If an action fails to generate proper code or retrieve expected data, 
you'll be prompted to either provide specific guidance ("try using a different approach") 
decide whether to continue with the current output.
\`\`\`

![Planner Agent Example](img/planner_error.png)
*Interactive error recovery dialog showing user options when an action encounters issues during plan execution*

**Technical Development:**

\`\`\`
Create a fully-featured Conway's Game of Life SPA with responsive UI, game controls, and pattern presets using vanilla HTML/CSS/JS
\`\`\`

![Planner Agent Example](img/planner.png)
*Example of Planner Agent in action Using local Hermes 8b (previous verision of the script)*
