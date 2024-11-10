# Open WebUI Tools Collection

A collection of tools for Open WebUI that provides arXiv paper search capabilities and Hugging Face text-to-image generation functionality. Perfect for enhancing your LLM interactions with academic research and image generation capabilities!

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

### 1. arXiv Reseach MCTS Pipe
Search arXiv.org for relevant academic papers on any topic. No API key required!

Features:
- Search across paper titles, abstracts, and full text from Arxiv and Web with tavily
- uses a MCTS aproach to make progressive refinements on a Reseach Summary on a given topic
- uses a visual representantion of the tree nodes to provide some visual feedback
- shows intermediate steps
- Configurable Width and Breadth of the search on valves.

## Installation

1. Navigate to your Open WebUI Workspace's Tools directory
2. Copy the contents of `arxiv_search_tool.py` and `create_image_hf.py` into your Tools or the `arXiv Reseach MCTS Pipe` into your Tools` 
3. For the Hugging Face Image Generator, configure the required API settings in the UI, for the `arXiv Reseach MCTS Pipe` configure your Tavily API- KEy (its free to here https://tavily.com)

## Configuration

### arXiv Search Tool
No configuration required! The tool works out of the box.

### arXiv Reseach MCTS Pipe
    Admin:
    -Tavily API KEY: required, create it in tavily.com
    -Max Search Results: Amount of results fo fech form web search
    -Arxiv Max Results: Amount of results to fetch for arxiv API

    User:
    -Tree Breadth: how many nodes per round are searched-
    -Tree Depth: how many rounds are made.
    -Exploration Weight: constant to control exploration vs exploitation, (a higher value means more exploration of new paths, while a low values makes the system stick with one option, range from 0 to 2 recommended)

### Hugging Face Image Generator
Required configuration in Open WebUI:

1. **API Key** (Required):
   - Obtain a Hugging Face API key from your HuggingFace account
   - Set it in the tool's configuration in Open WebUI

2. **API URL** (Optional):
   - Default: Uses Stability AI's SD 3.5 Turbo model
   - Can be customized to use other HF text-to-image model endpoints

## Usage

### arXiv Search
```python
# Example usage in your prompt
Search for recent papers about "quantum computing"
```

### Image Generation
```python
# Example usage in your prompt
Create an image of "a serene mountain landscape at sunset"

# Specify format
Create a landscape image of "a futuristic cityscape"
```

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

Developed by Haervwe
- GitHub: https://github.com/Haervwe/open-webui-tools

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
