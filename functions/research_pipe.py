"""
title: arXiv Reseach MCTS Pipe
description: Funtion Pipe made to create summary of searches uning arXiv.org for relevant papers on a topic and web scrape for more contextual information in a MCTS fashion.
author: Haervwe
git: https://github.com/Haervwe/open-webui-tools  
version: 0.1.3
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
from open_webui.apps.ollama import main as ollama
from bs4 import BeautifulSoup

# Constants and Setup
name = "Research"


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
        content_preview = (
            self.content[:25].replace("\n", " ") if self.content else "root"
        )
        msg = f"{padding}{self.id}({self.id}:{self.visits} - {content_preview})\n"

        if selected == self.id:
            msg += f"{padding}style {self.id} stroke:#0ff\n"

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

    async def expand(self, node):
        await self.pipe.progress(f"Exploring research paths from {node.id}...")
        await self.pipe.emit_replace(self.mermaid(node))

        for i in range(self.breadth):
            improvement = await self.pipe.get_improvement(node.content, self.topic)
            await self.pipe.emit_message(
                f"\nResearch direction {i+1}: {improvement}\n\n"
            )

            research = await self.pipe.gather_research(self.topic)
            synthesis = await self.pipe.synthesize_research(research, self.topic)

            child = Node(
                content=synthesis, research=research, max_children=self.breadth
            )
            node.add_child(child)

            await self.pipe.emit_replace(self.mermaid())

        return random.choice(node.children)

    async def simulate(self, node):
        await self.pipe.progress(f"Evaluating research path {node.id}...")
        return await self.pipe.evaluate_content(node.content, self.topic)

    def backpropagate(self, node, score):
        while node:
            node.visits += 1
            node.value += score
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
        TAVILY_API_KEY: str = Field(
            default="", description="API key for Tavily search service"
        )
        MAX_SEARCH_RESULTS: int = Field(
            default=3, description="Maximum number of search results to fetch per query"
        )
        ARXIV_MAX_RESULTS: int = Field(
            default=3, description="Maximum number of arXiv papers to fetch"
        )

    class UserValves(BaseModel):
        TREE_DEPTH: int = Field(
            default=4, description="Maximum depth of the research tree"
        )
        TREE_BREADTH: int = Field(
            default=3, description="Number of research paths to explore at each node"
        )
        EXPLORATION_WEIGHT: float = Field(
            default=1.414, description="Controls exploration vs exploitation"
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def pipes(self) -> list[dict[str, str]]:
        ollama.get_all_models()
        models = ollama.app.state.MODELS

        out = [
            {"id": f"{name}-{key}", "name": f"{name} {models[key]['name']}"}
            for key in models
        ]
        logger.debug(f"Available models: {out}")

        return out

    def resolve_model(self, body: dict) -> str:
        model_id = body.get("model")
        without_pipe = ".".join(model_id.split(".")[1:])
        return without_pipe.replace(f"{name}-", "")

    def resolve_question(self, body: dict) -> str:
        return body.get("messages")[-1].get("content").strip()

    async def search_arxiv(self, query: str) -> List[Dict]:
        """Gather research from arXiv"""
        await self.emit_status("tool", "Fetching arXiv papers...", False)
        try:
            arxiv_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "max_results": self.valves.ARXIV_MAX_RESULTS,
                "sortBy": "relevance",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(arxiv_url, params=params) as response:
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
                        logger.debug(f"Tavily API response data: {result}")
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
        """Gather research from multiple sources"""
        await self.emit_status("tool", "Starting research gathering...", False)

        # Gather from arXiv
        arxiv_results = await self.search_arxiv(topic)
        await self.emit_status(
            "tool", f"ArXiv papers found: {len(arxiv_results)}", False
        )

        # Gather from web
        web_results = await self.search_web(topic)
        await self.emit_status("tool", f"Web sources found: {len(web_results)}", False)

        return arxiv_results + web_results

    async def get_streaming_completion(
        self,
        model: str,
        messages,
    ) -> AsyncGenerator[str, None]:
        response = await ollama.generate_openai_chat_completion(
            {"model": model, "messages": messages, "stream": True}
        )

        async for chunk in response.body_iterator:
            for part in self.get_chunk_content(chunk):
                yield part

    async def get_completion(self, model: str, messages) -> str:
        """Updated to match MCTS signature"""
        response = await ollama.generate_openai_chat_completion(
            {
                "model": model,
                "messages": (
                    messages
                    if isinstance(messages, list)
                    else [{"role": "user", "content": messages}]
                ),
            }
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

    async def synthesize_research(self, research: List[Dict], topic: str) -> str:
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
            self.__model__, [{"role": "user", "content": prompt}]
        ):
            complete += chunk
            await self.emit_message(chunk)
        return complete

    async def evaluate_content(self, content: str, topic: str) -> float:
        """Evaluate research content quality"""
        prompt = f"""
    Rate this research synthesis from 1-10:
    "{content}"
    For topic: "{topic}"

    Consider:
    1. Integration of sources
    2. Depth of analysis
    3. Clarity and coherence
    4. Relevance to topic

    Reply with a single number only.
    """
        result = await self.get_completion(prompt)
        try:
            return float(result.strip())
        except ValueError:
            return 0.0

    def get_chunk_content(self, chunk):
        chunk_str = chunk.decode("utf-8")
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
            model, [{"role": "user", "content": content}]
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

    # Event emission methods unchanged...

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
    ) -> str:
        model = self.resolve_model(body)  # Get model first like MCTS
        logger.debug(f"Model {model}")
        if __task__ == TASKS.TITLE_GENERATION:
            logger.debug(f"Model {TASKS}")
            response = await ollama.generate_openai_chat_completion(
                {"model": model, "messages": body.get("messages"), "stream": False},
                user=__user__,
            )
            content = response["choices"][0]["message"]["content"]
            return f"{name}: {content}"
        logger.debug(f"Pipe {name} received: {body}")
        self.__user__ = __user__
        self.__current_event_emitter__ = __event_emitter__
        self.__model__ = model  # Assign after title check

        topic = body.get("messages", [])[-1].get("content", "").strip()

        await self.progress("Initializing research process...")

        # Initial research
        initial_research = await self.gather_research(topic)
        initial_content = await self.synthesize_research(initial_research, topic)

        root = Node(
            content=initial_content,
            research=initial_research,
            max_children=self.user_valves.TREE_BREADTH,
        )

        mcts = MCTS(
            root=root,
            pipe=self,
            topic=topic,
            max_depth=self.user_valves.TREE_DEPTH,
        )

        best_content = initial_content
        best_score = -float("inf")

        for i in range(self.user_valves.TREE_DEPTH):
            await self.progress(
                f"Research iteration {i+1}/{self.user_valves.TREE_DEPTH}..."
            )

            leaf = await mcts.select()
            child = await mcts.expand(leaf)
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

    async def get_completion(self, prompt: str) -> str:
        response = await ollama.generate_openai_chat_completion(
            {
                "model": self.__model__,
                "messages": [{"role": "user", "content": prompt}],
            },
            user=self.__user__,
        )
        return response["choices"][0]["message"]["content"]