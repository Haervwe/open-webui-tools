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

## Installation

1. Navigate to your Open WebUI Workspace's Tools directory
2. Copy the contents of `arxiv_search_tool.py` and `create_image_hf.py` into your Tools
3. For the Hugging Face Image Generator, configure the required API settings in the UI

for those without an HF api key its free, (up to a certain point but very loose) create an account here https://huggingface.co and generate an API key in your profile settings, make sure to add permissions to call the serverless endpoints.

## Configuration

### arXiv Search Tool
No configuration required! The tool works out of the box.

### Hugging Face Image Generator
Required configuration in Open WebUI:

1. **API Key** (Required):
   - Obtain a Hugging Face API key from your HuggingFace account
   - Set it in the tool's configuration in Open WebUI

2. **API URL** (Optional):
   - Default: Uses Stability AI's SD 3.5 Turbo model
   - Can be customized to use other HF text-to-image model endpoints

## Usage

(Make sure to turno on the tool in chat before requesting it)

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

Credit to the amazing teams behind 
https://github.com/ollama/ollama
https://github.com/open-webui/open-webui
And all model trainers out there providing these amazing tools.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
