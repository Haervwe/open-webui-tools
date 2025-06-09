"""
title: arXiv Research MCTS Pipe
description: Funtion Pipe made to create summary of searches uning arXiv.org for relevant papers on a topic and web scrape for more contextual information in a MCTS fashion.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
original MCTS implementation i based this project of: https://github.com/av // https://openwebui.com/f/everlier/mcts/
git: https://github.com/Haervwe/open-webui-tools  
version: 0.4.4
"""

import logging
import random
import math
import json
import aiohttp
import asyncio
from typing import List, Dict, Union, Optional, AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from bs4 import BeautifulSoup
from dataclasses import dataclass
import re
from open_webui.main import generate_chat_completions
from open_webui.models.users import User


name = "Research Pipe"





def setup_logger():
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


# Node class for MCTS
class Node:
    def __init__(self, **kwargs):
        self.id = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        self.content = kwargs.get("content")
        self.parent = kwargs.get("parent")
        self.research = kwargs.get("research", [])
        self.exploration_weight = kwargs.get("exploration_weight", 1.414)
        self.max_children = kwargs.get("max_children", 3)
        self.children = []
        self.visits = 0
        self.value = 0
        self.score = 0
        self.temperature = kwargs.get("temperature", 1)
        self.depth = kwargs.get("depth", 1)

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)
        return child

    def fully_expanded(self):
        return len(self.children) >= self.max_children

    def uct_value(self):
        epsilon = 1e-6
        if not self.parent:
            return float("inf")
        return self.value / (
            self.visits + epsilon
        ) + self.exploration_weight * math.sqrt(
            math.log(self.parent.visits) / (self.visits + epsilon)
        )

    def mermaid(self, offset=0, selected=None):
        padding = " " * offset
        
        # Sanitize content for Mermaid compatibility
        def sanitize_content(text):
            if not text:
                return "root"
            # Remove problematic characters and limit length
            sanitized = text[:25].replace("\n", " ")
            # Replace special characters that could break Mermaid syntax
            sanitized = re.sub(r'[(){}<>:"[\]]', '', sanitized)
            # Replace multiple spaces with single space
            sanitized = re.sub(r'\s+', ' ', sanitized)
            # Ensure the text is not empty after sanitization
            return sanitized.strip() or "node"

        # Create node content
        content_preview = sanitize_content(self.content)
        
        # Create node ID and label
        node_label = f"{self.id}:{self.visits} - {content_preview}"
        # Escape any remaining special characters in the label
        node_label = node_label.replace('"', '&quot;')
        
        # Generate node definition
        msg = f"{padding}{self.id}[\"{node_label}\"]\n"

        # Add styling if node is selected
        if selected == self.id:
            msg += f"{padding}style {self.id} stroke:#0ff,stroke-width:4px\n"

        # Generate children connections
        for child in self.children:
            msg += child.mermaid(offset + 4, selected)
            msg += f"{padding}{self.id} --> {child.id}\n"

        return msg


class MCTS:
    def __init__(self, **kwargs):
        self.topic = kwargs.get("topic")
        self.root = kwargs.get("root")
        self.pipe = kwargs.get("pipe")
        self.selected = None
        self.max_depth = kwargs.get("max_depth", 3)
        self.breadth = kwargs.get("breadth", 2)

    async def select(self):
        node = self.root
        while node.children:
            node = max(node.children, key=lambda child: child.uct_value())
        return node

    async def expand(self, node: Node, depth):
        await self.pipe.progress(f"Exploring research paths from {node.id}...")
        await self.pipe.emit_replace(self.mermaid(node))
        temperature = self.define_temperature(
            depth,
            node.score,
            self.max_depth,
            self.pipe.valves.TEMPERATURE_MAX,
            self.pipe.valves.TEMPERATURE_MIN,
            self.pipe.valves.DINAMYC_TEMPERATURE_DECAY,
        )
        for i in range(self.breadth):
            await self.pipe.emit_replace(self.mermaid())
            improvement = await self.pipe.get_improvement(node.content, self.topic)
            await self.pipe.emit_message(
                f"\nResearch direction {i+1}: {improvement}\n\n"
            )
            logger.debug(f"temperature:{temperature}")
            research = await self.pipe.gather_research(
                f"""Generate a new arXiv search query based on the improvement suggestion:
            Topic: {self.topic}
            Improvement: {improvement}"""
            )

            synthesis = await self.pipe.synthesize_research(
                research, self.topic, temperature
            )

            child = Node(
                content=synthesis,
                research=research,
                max_children=self.breadth,
                temperature=temperature,
            )
            node.add_child(child)

            await self.pipe.emit_replace(self.mermaid())

        return random.choice(node.children)

    def define_temperature(
        self,
        current_depth: int,
        parent_score: float,
        max_depth: int,
        temperature_max: float,
        temperature_min: float,
        dynamic: bool,
    ):
        if not self.pipe.valves.TEMPERATURE_DECAY:
            return 1

        if dynamic and parent_score > 0:
            # Inversely proportional to parent_score (higher temperature (creativity) for lower scores)
            score_normalized = parent_score / 10.0  # Normalize to 0-1 range
            scaling_factor = 1.0 + (1.0 - score_normalized) * (
                temperature_max - temperature_min
            )  # Scales with difference from ideal score
            temperature = (
                ((temperature_max - temperature_min) * (current_depth / max_depth))
                + temperature_min
            ) * scaling_factor
            # Clamp within bounds
            temperature_clamped = max(
                temperature_min, min(temperature, temperature_max)
            )

            return temperature_clamped

        else:  # Standard decay, not influenced by parent score
            temperature = temperature_max - (temperature_max - temperature_min) * (
                current_depth / max_depth
            )
            return temperature

    async def simulate(self, node):
        await self.pipe.progress(f"Evaluating research path {node.id}...")
        return await self.pipe.evaluate_content(node.content, self.topic)

    def backpropagate(self, node, score):
        while node:
            node.visits += 1
            node.value += score
            node.score = score
            node = node.parent

    def mermaid(self, selected=None):
        return f"""
```mermaid
graph LR
{self.root.mermaid(0, selected.id if selected else None)}
```
"""

    def best_child(self):
        return max(self.root.children, key=lambda child: child.visits)


EventEmitter = Callable[[dict], Awaitable[None]]


class Pipe:
    __current_event_emitter__: EventEmitter
    __current_node__: Node
    __question__: str
    __model__: str

    class Valves(BaseModel):
        MODEL: str = Field(
            default=None, description="Model to use (model id from ollama)"
        )
        TAVILY_API_KEY: str = Field(
            default="", description="API key for Tavily search service"
        )
        MAX_SEARCH_RESULTS: int = Field(
            default=3, description="Maximum number of search results to fetch per query"
        )
        ARXIV_MAX_RESULTS: int = Field(
            default=3, description="Maximum number of arXiv papers to fetch"
        )
        TREE_DEPTH: int = Field(
            default=4, description="Maximum depth of the research tree"
        )
        TREE_BREADTH: int = Field(
            default=3, description="Number of research paths to explore at each node"
        )
        EXPLORATION_WEIGHT: float = Field(
            default=1.414, description="Controls exploration vs exploitation"
        )
        TEMPERATURE_DECAY: bool = Field(
            default=True,
            description="Activates Temperature , lowers the Temperature in each subsequent step",
        )
        DINAMYC_TEMPERATURE_DECAY: bool = Field(
            default=True,
            description="Activates Temperature  Dynamic mapping, giving higher creativity for lower scored parent nodes",
        )
        TEMPERATURE_MAX: float = Field(
            default=1.4,
            description="Temperature for starting the MCTS process with Temperature decay ONLY if active",
        )
        TEMPERATURE_MIN: float = Field(
            default=0.5,
            description="Temperature the MCTS process will attempt to converge to with Temperature decay, if set to dinamic this value is not fixed",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:

        out = [{"id": f"{name}-{self.valves.MODEL}", "name": f"{name}"}]
        return out

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    def resolve_question(self, body: dict) -> str:
        return body.get("messages")[-1].get("content").strip()

    async def search_arxiv(self, query: str) -> List[Dict]:
        """Gather research from arXiv"""
        await self.emit_status("tool", f"Fetching arXiv papers for: {query}...", False)
        try:
            arxiv_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"{query}",
                "max_results": self.valves.ARXIV_MAX_RESULTS,
                "sortBy": "relevance",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(arxiv_url, params=params) as response:
                    logger.debug(f"arXiv API response status: {response.status}")
                    if response.status == 200:
                        data = await response.text()
                        soup = BeautifulSoup(data, "xml")
                        entries = soup.find_all("entry")
                        return [
                            {
                                "title": entry.find("title").text,
                                "url": entry.find("link")["href"],
                                "content": entry.find("summary").text,
                            }
                            for entry in entries
                        ]

        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        return []

    async def search_web(self, query: str) -> List[Dict]:
        """Simplified web search using Tavily API"""
        if not self.valves.TAVILY_API_KEY:
            return []

        async with aiohttp.ClientSession() as session:
            try:
                url = "https://api.tavily.com/search"
                headers = {"Content-Type": "application/json"}
                data = {
                    "api_key": self.valves.TAVILY_API_KEY,
                    "query": query,
                    "max_results": self.valves.MAX_SEARCH_RESULTS,
                    "search_depth": "advanced",
                }
                async with session.post(url, headers=headers, json=data) as response:
                    logger.debug(f"Tavily API response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        results = result.get("results", [])
                        return [
                            {
                                "title": result["title"],
                                "url": result["url"],
                                "content": result["content"],
                                "score": result["score"],
                            }
                            for result in results
                        ]
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

    async def gather_research(self, topic: str) -> List[Dict]:
        """Gather initial research for the given topic"""
        await self.emit_status("tool", f"Researching...", False)

        # Preprocess the initial user query
        web_query, arxiv_query = await self.preprocess_query(topic)

        # Perform web search and arXiv search using the preprocessed queries
        web_research = await self.search_web(web_query)
        await self.emit_status(
            "tool", f"Web sources found:: {len(web_research)}", False
        )
        arxiv_research = await self.search_arxiv(arxiv_query)

        await self.emit_status(
            "tool", f"ArXiv papers found:: {len(arxiv_research)}", False
        )
        research = web_research + arxiv_research
        logger.debug(
            f"Research Result Created : ArXiv papers found: {len(arxiv_research)}, Web sources found: {len(web_research)}"
        )
        await self.emit_status(
            "user",
            f"Research gathered: ArXiv papers found: {len(arxiv_research)}, Web sources found: {len(web_research)}",
            True,
        )
        return research

    async def preprocess_query(self, query: str) -> tuple[str, str]:
        """Preprocess and enhance the initial user query for optimized web and arXiv searches."""

        # Prompt for web search query enhancement
        prompt_web = f"""
        Enhance the following query to improve the relevance of web search results:
        - Focus on adding relevant keywords, synonyms, or contextual phrases
        - The input query may be an initial vague request or an essay with proposed improvements
        - Only output the enhanced query, ready for an API call, without explanations or titles

        Initial query: "{query}"

        Enhanced web search query:
        """
        web_query = await self.get_completion(prompt_web)

        # Prompt for arXiv API query enhancement
        prompt_arxiv = f"""
        Format an optimized query for the arXiv API based on the following input:
        - Use arXiv's query syntax (AND, OR, NOT) and search fields (ti, au, abs, cat)
        - Select appropriate categories from the provided list
        - The input may be an initial vague request or an essay with proposed improvements
        - Only output the formatted arXiv API query, without explanations or titles

        Initial query: "{query}"

        arXiv categories:
        - cs.AI: Artificial Intelligence
        - cs.LG: Machine Learning 
        - cs.CV: Computer Vision
        - cs.CL: Computation and Language (NLP)
        - cs.RO: Robotics
        - stat.ML: Machine Learning (Statistics)
        - math.OC: Optimization and Control
        - physics: Physics
        - q-bio: Quantitative Biology
        - q-fin: Quantitative Finance
        - econ: Economics

        Enhanced arXiv search query (API format):
        """
        arxiv_query = await self.get_completion(prompt_arxiv)

        return web_query, arxiv_query

    async def get_streaming_completion(
        self,
        messages,
        temperature: float = 1,
    ) -> AsyncGenerator[str, None]:
        try:
            form_data = {
                "model": self.__model__,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
            }
            response = await generate_chat_completions(
                self.__request__,
                form_data,
                user=self.__user__,
            )

            # Ensure the response has body_iterator
            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")

            async for chunk in response.body_iterator:
                # Use the updated chunk content method
                for part in self.get_chunk_content(chunk):
                    yield part

        except Exception as e:
            raise RuntimeError(f"Streaming completion failed: {e}")

    async def get_completion(self, messages) -> str:
        response = await generate_chat_completions(
            self.__request__,
            {
                "model": self.__model__,
                "messages": [{"role": "user", "content": messages}],
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]

    async def get_improvement(self, content: str, topic: str) -> str:
        """Get improvement suggestion"""
        prompt = f"""
    How can this research synthesis be improved?
    Topic: {topic}

    Current synthesis:
    {content}

    Suggest ONE specific improvement in a single sentence.
    """
        return await self.get_completion(prompt)

    async def synthesize_research(
        self, research: List[Dict], topic: str, temperature
    ) -> str:
        """Synthesize research content with streaming"""
        research_text = "\n\n".join(
            f"Title: {r['title']}\nContent: {r['content']}\nURL: {r['url']}"
            for r in research
        )

        prompt = f"""
    Create a research synthesis on the topic: {topic}

    Available research:
    {research_text}

    Create a comprehensive synthesis that:
    1. Integrates the sources
    2. Highlights key findings
    3. Maintains academic rigor while being accessible
    """
        complete = ""
        async for chunk in self.get_streaming_completion(
            [{"role": "user", "content": prompt}], temperature
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def evaluate_content(self, content: str, topic: str) -> float:
        """Evaluate research content quality based on topic and content."""
        logger.debug(f"Evaluating content for topic: {topic[:50]}...")
        # Improved and detailed prompt
        prompt = f"""
        Evaluate the quality of the research synthesis provided below:

        Content: "{content}"
        Topic: "{topic}"

        Consider the following criteria:
        1. Integration of sources.
        2. Depth of analysis.
        3. Clarity and coherence.
        4. Relevance to the topic.

        Provide a single numeric score between 1 and 10, inclusive. 
        Do not include any explanation or additional text in your response—just the number.
        """
        score = 0.0
        try:
            result = await self.get_completion(prompt)
            match = re.search(r"\b(10|\d(\.\d+)?)\b", result.strip())
            if match:
                score = float(match.group())
                if 1.0 <= score <= 10.0:
                    return score
                else:
                    logger.debug(f"Score out of range: {score}. Result was: {result}")
                    return 0.0
            else:
                logger.debug(f"No valid number in response: {result}")
                return score 
        except Exception as e:

            logger.debug(f"Error during evaluation: {e}")
            return score

    def get_chunk_content(self, chunk):

        chunk_str = chunk
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:]

        chunk_str = chunk_str.strip()

        if chunk_str == "[DONE]" or not chunk_str:
            return

        try:
            chunk_data = json.loads(chunk_str)
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except json.JSONDecodeError:
            logger.error(f'ChunkDecodeError: unable to parse "{chunk_str[:100]}"')

    async def get_message_completion(self, model: str, content):
        async for chunk in self.get_streaming_completion(
            [{"role": "user", "content": content}]
        ):
            yield chunk

    async def stream_prompt_completion(self, prompt, **format_args):
        complete = ""
        async for chunk in self.get_message_completion(
            self.__model__,
            prompt.format(**format_args),
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        model = self.valves.MODEL
        logger.debug(f"Model {model}")
        logger.debug(f"User: {__user__}")
        self.__user__ = User(**__user__)
        self.__request__=__request__
        if __task__ and __task__ != TASKS.DEFAULT:
            logger.debug(f"Model {TASKS}")
            response = await generate_chat_completions(
                self.__request__,
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=self.__user__,
            )
            content = response["choices"][0]["message"]["content"]
            return f"{name}: {content}"
        logger.debug(f"Pipe {name} received: {body}"[:70])
        self.__current_event_emitter__ = __event_emitter__
        self.__model__ = model  # Assign after title check

        topic = body.get("messages", [])[-1].get("content", "").strip()

        await self.progress("Initializing research process...")
        initial_temperature = (
            self.valves.TEMPERATURE_MAX if self.valves.TEMPERATURE_DECAY else 1
        )
        # Initial research
        initial_research = await self.gather_research(topic)
        initial_content = await self.synthesize_research(
            initial_research, topic, initial_temperature
        )

        root = Node(
            content=initial_content,
            research=initial_research,
            max_children=self.valves.TREE_BREADTH,
        )

        mcts = MCTS(
            root=root,
            pipe=self,
            topic=topic,
            max_depth=self.valves.TREE_DEPTH,
            breadth=self.valves.TREE_BREADTH,
        )

        best_content = initial_content
        best_score = -float("inf")

        for i in range(self.valves.TREE_DEPTH):
            await self.progress(f"Research iteration {i+1}/{self.valves.TREE_DEPTH}...")

            leaf = await mcts.select()
            child = await mcts.expand(leaf, i + 1)
            score = await mcts.simulate(child)
            mcts.backpropagate(child, score)

            if score > best_score:
                best_score = score
                best_content = child.content

        await self.emit_replace(mcts.mermaid())
        await self.emit_message(best_content)
        await self.done()
        return ""

    async def progress(self, message: str):
        await self.emit_status("info", message, False)

    async def done(self):
        await self.emit_status("info", "Research complete", True)

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_replace(self, message: str):
        await self.__current_event_emitter__(
            {"type": "replace", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
        )
