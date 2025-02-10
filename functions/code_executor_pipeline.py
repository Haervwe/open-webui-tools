import logging
from typing import List, Union, Dict
import subprocess
import sys
import os
import tempfile
import re
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import requests
import json

# Enhance logger setup with more detailed formatting
def setup_logger(name: str = "Code Executor Pipeline"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = setup_logger()

class Pipeline:
    class Valves(BaseModel):
        OLLAMA_API_URL: str = "http://localhost:11434"
        OLLAMA_MODEL: str = "Qwen2_5_16k:latest"
        MAX_ITERATIONS: int = 5  # Maximum number of iterations to prevent infinite loops
        SYSTEM_MESSAGE: str = (
            "You are a Python code execution assistant. You can:\n"
            "1. Execute Python code and handle dependencies\n"
            "2. Install required packages automatically\n"
            "3. Support GUI applications with proper display handling\n"
            "4. Manage system resources and security\n\n"
            "When handling code:\n"
            "1. Always check for required packages first\n"
            "2. Use proper error handling\n"
            "3. Support both CLI and GUI outputs\n"
            "4. Provide clear execution results\n\n"
            "For GUI applications:\n"
            "1. Ensure proper display setup\n"
            "2. Handle window management\n"
            "3. Support various UI frameworks"
        )
        DISPLAY_ENABLED: bool = True  # Toggle for GUI support
        SANDBOX_MODE: bool = True  # Enable security restrictions
        REQUIRED_PACKAGES: List[str] = ["pyvirtualdisplay", "xvfbwrapper", "python-xlib"]  # Added python-xlib
        DISPLAY_BACKEND: str = "xvfb"  # Lowercase for compatibility
        HEADLESS: bool = True  # Force headless mode for server environments
        X11_SETUP_COMMANDS: List[str] = [
            "xhost +local:",  # Allow local connections
            "xhost +SI:localuser:$USER"  # Allow current user
        ]
        DEFAULT_XDG_RUNTIME_DIR: str = "/run/user/1000"  # Default XDG runtime directory
        SETUP_ENV_VARS: Dict[str, str] = {
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "MPLBACKEND": "Agg",  # For matplotlib
            "QT_QPA_PLATFORM": "offscreen",  # For Qt applications
            "SDL_VIDEODRIVER": "dummy"  # For SDL/pygame
        }

    def _setup_environment(self) -> None:
        """Setup required environment variables"""
        logger.info("Setting up environment variables...")
        try:
            # Get user ID for XDG_RUNTIME_DIR
            uid = os.getuid()
            runtime_dir = f"/run/user/{uid}"

            # Create XDG_RUNTIME_DIR if it doesn't exist
            if not os.path.exists(runtime_dir):
                try:
                    os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
                    os.chmod(runtime_dir, 0o700)  # Ensure correct permissions
                except PermissionError:
                    runtime_dir = os.path.join(tempfile.gettempdir(), f"runtime-{uid}")
                    os.makedirs(runtime_dir, mode=0o700, exist_ok=True)

            # Set environment variables
            env_vars = self.valves.SETUP_ENV_VARS.copy()
            env_vars["XDG_RUNTIME_DIR"] = runtime_dir

            for key, value in env_vars.items():
                os.environ[key] = value
                logger.debug(f"Set environment variable: {key}={value}")

        except Exception as e:
            logger.error(f"Failed to setup environment variables: {e}")
            # Fallback to temp directory
            fallback_dir = os.path.join(tempfile.gettempdir(), f"runtime-{os.getuid()}")
            os.environ["XDG_RUNTIME_DIR"] = fallback_dir
            logger.info(f"Using fallback XDG_RUNTIME_DIR: {fallback_dir}")

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Code Execution Pipeline"
        logger.info(f"Initializing {self.name}")
        try:
            self._setup_environment()  # Setup environment first
            self._ensure_system_dependencies()
            self._ensure_core_dependencies()
            # Initialize display once for reuse
            self._init_display()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.display = None

    def _ensure_core_dependencies(self) -> None:
        """Ensure core dependencies are installed for the pipeline"""
        logger.info("Checking core dependencies...")
        for package in self.valves.REQUIRED_PACKAGES:
            try:
                __import__(package)
            except ImportError:
                logger.info(f"Installing required package: {package}")
                self.install_dependencies([package])

    def _ensure_system_dependencies(self) -> None:
        """Ensure system-level dependencies are installed"""
        required_system_packages = ['xvfb', 'x11-xserver-utils', 'xserver-xephyr']
        logger.info("Checking system dependencies...")
        
        missing_packages = []
        for package in required_system_packages:
            try:
                subprocess.run(['which', package], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing system packages: {missing_packages}")
            logger.error("Please install using: sudo apt-get install " + " ".join(missing_packages))
            logger.error("Continuing with limited functionality...")

    def _setup_x11_permissions(self) -> None:
        """Setup X11 permissions for the current user"""
        logger.info("Setting up X11 permissions...")
        try:
            # First check if we're running as root (not recommended)
            if os.geteuid() == 0:
                logger.warning("Running as root is not recommended for X11 applications")
                
            # Try to set up X11 permissions
            for cmd in self.valves.X11_SETUP_COMMANDS:
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        logger.info(f"X11 permission command successful: {cmd}")
                    else:
                        logger.warning(f"X11 permission command failed: {cmd}")
                        logger.warning(f"Error: {result.stderr}")
                except Exception as e:
                    logger.warning(f"Failed to set X11 permission with {cmd}: {e}")

        except Exception as e:
            logger.error(f"Failed to setup X11 permissions: {e}")
            logger.info("You may need to manually run: xhost +local:")

    def _init_display(self) -> None:
        """Initialize virtual display for GUI applications"""
        try:
            # Set up X11 permissions first
            self._setup_x11_permissions()
            
            if self.valves.HEADLESS:
                from xvfbwrapper import Xvfb
                self.display = Xvfb(width=1024, height=768, colordepth=24)
                self.display.start()
                display_num = str(self.display.new_display).replace(":", "")  # Convert to string and remove colon
                os.environ['DISPLAY'] = f":{display_num}"
                logger.info(f"Initialized headless display: {os.environ['DISPLAY']}")
            else:
                from pyvirtualdisplay.display import Display
                self.display = Display(
                    backend=self.valves.DISPLAY_BACKEND,
                    size=(1024, 768),
                    color_depth=24,
                    use_xauth=True,
                    retries=3
                )
                self.display.start()
                logger.info(f"Initialized virtual display using {self.valves.DISPLAY_BACKEND}")
        except Exception as e:
            logger.error(f"Display initialization failed: {e}")
            self.display = None

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        if hasattr(self, 'display') and self.display:
            try:
                self.display.stop()
                logger.info("Display stopped")
            except Exception as e:
                logger.error(f"Error stopping display: {e}")
        print(f"on_shutdown: {__name__}")

    def install_dependencies(self, packages: List[str]) -> str:
        """
        Installs the specified Python packages.
        """
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                return f"Failed to install package '{package}': {str(e)}"
        return "Packages installed successfully."

    def execute_python_code(self, code: str) -> str:
        """
        Executes Python code with enhanced GUI and security support.
        """
        logger.debug(f"Executing code:\n{code}")
        try:
            # Create a temporary directory for code execution
            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)

                # Ensure environment is setup
                self._setup_environment()

                # Write code to a file instead of direct execution
                script_path = os.path.join(tmpdir, 'script.py')
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(code)

                # Install dependencies
                install_msg = self.install_dependencies_from_code(code)
                if "Failed" in install_msg:
                    logger.error(f"Dependency installation failed: {install_msg}")
                    return install_msg

                # Use the pre-initialized display
                if any(gui_lib in code for gui_lib in ['pygame', 'tkinter', 'PyQt', 'wx']):
                    if not self.display or not hasattr(self.display, 'is_alive') or not self.display.is_alive():
                        logger.warning("Display not available, reinitializing...")
                        self._init_display()
                        if not self.display:
                            return "Failed to initialize display for GUI application"

                try:
                    # Set up environment
                    env = os.environ.copy()
                    if self.display:
                        if hasattr(self.display, 'new_display'):  # xvfbwrapper
                            display_num = str(self.display.new_display).replace(":", "")
                            env['DISPLAY'] = f":{display_num}"
                        else:  # pyvirtualdisplay
                            env['DISPLAY'] = str(self.display.display)
                            if hasattr(self.display, 'xauth'):
                                env['XAUTHORITY'] = self.display.xauth
                        
                        logger.debug(f"Using DISPLAY={env['DISPLAY']}")

                    # Execute the script file instead of -c
                    result = subprocess.run(
                        [sys.executable, script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=60,
                        env=env,
                        encoding='utf-8'
                    )

                    if result.returncode == 0:
                        output = result.stdout
                        logger.info("Code execution successful")
                    else:
                        output = f"Execution Error:\n{result.stderr}"
                        logger.error(f"Code execution failed: {output}")

                    return output

                finally:
                    os.chdir(old_cwd)

        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            return f"Exception during execution: {str(e)}"

    def install_dependencies_from_code(self, code: str) -> str:
        """
        Parses the code to find import statements and installs the packages.
        """
        # Extract package names from import statements
        pattern = r'^(?:from\s+([^\s]+)\s+import|import\s+([^\s]+))'
        matches = re.findall(pattern, code, re.MULTILINE)
        packages = set(sum(matches, ()))

        # Remove empty strings and aliases
        packages = {pkg.split('.')[0] for pkg in packages if pkg and not pkg.startswith(('sys', 'os'))}

        # List of standard library modules (Python 3.x)
        standard_libs = sys.stdlib_module_names

        # Filter out standard library modules
        packages_to_install = list(packages - standard_libs)

        # Install packages
        if packages_to_install:
            return self.install_dependencies(packages_to_install)
        else:
            return "No external packages to install."

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, List[OpenAIChatMessage]]:
        """Enhanced pipe with better tool handling and logging."""
        logger.info(f"Processing message: '{user_message[:100]}...'")
        
        # Extract potential tool call from message if it's malformed
        if "<tool_call>" in user_message:
            try:
                tool_call_text = user_message.split("<tool_call>")[1].strip()
                tool_call_data = json.loads(tool_call_text)
                logger.info(f"Extracted tool call from message: {tool_call_data}")
                
                # Execute the tool directly
                if tool_call_data.get("name") == "execute_python_code":
                    result = self.execute_python_code(tool_call_data["arguments"]["code"])
                    return result
                elif tool_call_data.get("name") == "install_dependencies":
                    result = self.install_dependencies(tool_call_data["arguments"]["packages"])
                    return result
            except Exception as e:
                logger.error(f"Failed to parse tool call: {e}")

        # Prepare conversation and tools as before
        conversation = messages.copy()
        
        # Define tools in OpenAI-compatible format
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Executes the provided Python code and returns the output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The complete Python code to execute as a single string"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "install_dependencies",
                    "description": "Installs Python packages needed for code execution",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "packages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of package names to install"
                            }
                        },
                        "required": ["packages"]
                    }
                }
            }
            
        ]

        # Add system message with better tool usage instructions
        if not messages or messages[0].get("role") != "system":
            system_content = f"""{self.valves.SYSTEM_MESSAGE}

Available tools:
1. execute_python_code: Use this to run Python code
   Example usage: 
   ```json
   {{"function": {{"name": "execute_python_code", "arguments": {{"code": "print('Hello World')}}}}}}
   ```

2. install_dependencies: Use this to install Python packages
   Example usage:
   ```json
   {{"function": {{"name": "install_dependencies", "arguments": {{"packages": ["pygame", "numpy"]}}}}}}
   ```

Important: Always use proper JSON formatting for tool calls."""

            conversation.insert(0, {
                "role": "system",
                "content": system_content
            })
            logger.debug("Added system message with tools description and examples")

        # Process the conversation
        iteration = 0
        while iteration < self.valves.MAX_ITERATIONS:
            try:
                # Prepare payload for Ollama
                payload = {
                    "model": self.valves.OLLAMA_MODEL,
                    "messages": conversation,
                    "tools": tools,
                    "tool_choice": "auto",
                }

                # Log request details
                logger.debug(f"Request payload (iteration {iteration + 1}):")
                logger.debug(f"Messages: {json.dumps(payload['messages'][-2:], indent=2)}")  # Last 2 messages
                logger.debug(f"Tools available: {[t['function']['name'] for t in tools]}")

                # Make API request
                response = requests.post(
                    url=f"{self.valves.OLLAMA_API_URL.rstrip('/')}/v1/chat/completions",
                    headers={"Accept": "application/json"},
                    json=payload
                )
                response.raise_for_status()
                response_json = response.json()
                
                assistant_message = response_json["choices"][0]["message"]
                logger.debug(f"Assistant response: {json.dumps(assistant_message, indent=2)}")

                # Check for tool calls in multiple formats
                tool_calls = []
                if "tool_calls" in assistant_message:
                    tool_calls = assistant_message["tool_calls"]
                elif "function_call" in assistant_message:
                    tool_calls = [{
                        "id": "0",
                        "function": assistant_message["function_call"]
                    }]
                
                if tool_calls:
                    logger.info(f"Found {len(tool_calls)} tool calls")
                    conversation.append(assistant_message)
                    
                    for tool_call in tool_calls:
                        try:
                            function_data = tool_call.get("function", {})
                            function_name = function_data.get("name")
                            arguments = json.loads(function_data.get("arguments", "{}"))
                            
                            logger.info(f"Executing {function_name} with args: {arguments}")
                            
                            if function_name == "execute_python_code":
                                result = self.execute_python_code(arguments["code"])
                            elif function_name == "install_dependencies":
                                result = self.install_dependencies(arguments["packages"])
                            else:
                                result = f"Unknown function: {function_name}"
                            
                            conversation.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", "0"),
                                "name": function_name,
                                "content": result
                            })
                            
                        except Exception as e:
                            error_msg = f"Tool execution error: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            return error_msg
                else:
                    # Handle regular responses and code blocks
                    content = assistant_message.get("content", "")
                    if "```python" in content:
                        code_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
                        if code_blocks:
                            code = code_blocks[0].strip()
                            result = self.execute_python_code(code)
                            return f"{content}\n\nExecution result:\n{result}"
                    return content

                iteration += 1

            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}", exc_info=True)
                return f"Error in pipeline: {str(e)}"

        return "Failed to complete the task within the allowed iterations."