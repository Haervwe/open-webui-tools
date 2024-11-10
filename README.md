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
2. Copy the contents of `arxiv_search_tool.py` and `create_image_hf.py` into your Tools or the `arXiv Reseach MCTS Pipe` into your Functions` 
3. For the Hugging Face Image Generator, configure the required API settings in the UI, for the `arXiv Reseach MCTS Pipe` configure your Tavily API- KEy (its free to here https://tavily.com)

for those without an HF api key its free, (up to a certain point but very loose) create an account here https://huggingface.co and generate an API key in your profile settings, make sure to add permissions to call the serverless endpoints.

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
   - Can be customized to use other HF text-to-image model endpoints such as flux

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


## Usage for Pipes

### arXiv Reseach MCTS Pipe

Select the pipe with the corresponding model, it show as this:

![Screenshot from 2024-11-10 17-36-16](https://github.com/user-attachments/assets/e55f96b6-a893-463e-aa6b-4d2ef24012f2)

```python
# Example usage in your prompt
Do a reseach summary on "DPO laser LLM training"
```

![Screenshot from 2024-11-10 17-53-51](https://github.com/user-attachments/assets/28116b58-79ef-4d4a-bd51-bfa2f2102cc3)



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

Credit to the amazing teams behind 
https://github.com/ollama/ollama
https://github.com/open-webui/open-webui
And all model trainers out there providing these amazing tools.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
