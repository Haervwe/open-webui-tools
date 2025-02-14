"""
title: Mopidy_Music_Controller
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.3.8
description: A pipe to control Mopidy music server to play songs from local library or YouTube, manage playlists, and handle various music commands 
needs a Local and/or a Youtube API endpoint configured in mopidy.
mopidy repo: https://github.com/mopidy
"""

import logging
import asyncio
import json
from typing import Dict, List, Callable, Awaitable, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import aiohttp
import re
import traceback
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions

name = "MopidyController"


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


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str


EventEmitter = Callable[[dict], Awaitable[None]]


class Pipe:
    __current_event_emitter__: EventEmitter
    __user__: User
    __model__: str

    class Valves(BaseModel):
        Model: str = Field(default="", description="Model tag")
        Mopidy_URL: str = Field(
            default="http://localhost:6680/mopidy/rpc",
            description="URL for the Mopidy JSON-RPC API endpoint",
        )
        YouTube_API_Key: str = Field(
            default="", description="YouTube Data API key for search"
        )
        Temperature: float = Field(default=0.7, description="Model temperature")
        Max_Search_Results: int = Field(
            default=5, description="Maximum number of search results to return"
        )
        Use_Iris: bool = Field(
            default=True,
            description="Toggle to use Iris interface or custom HTML UI",
        )
        system_prompt: str = Field(
            default=(
                "You are a helpful assistant for controlling a music server. "
                "Users will ask you to perform various music-related commands like playing songs, adding to playlists, etc. "
                "Your job is to parse the user's request and extract the following information in JSON format without any extra text:\n"
                "{\n"
                '  "action": "action_name",\n'
                '  "parameters": {\n'
                '    "title": "song or playlist title",\n'
                '    "artist": "artist name",\n'
                '    "playlist_name": "playlist name"\n'
                "  }\n"
                "}\n"
                "Possible actions are: play_song, play_playlist, add_to_playlist, create_playlist, show_current_song, pause, resume, skip.\n"
                "If the user mentions an album, treat it as a playlist and set 'action' to 'play_playlist'.\n"
                "If the user asks for an action directly (e.g., 'pause', 'stop', 'play', 'resume', 'skip'), set 'action' to the corresponding action and do not include any parameters.\n"
                "Do not attempt to search for songs with titles like 'pause' or 'stop'.\n"
                "If you cannot determine the action, default to 'play_song' with the user's input as the 'title' parameter.\n"
                "Ensure that the JSON is correctly formatted and no additional text is included in your response."
            ),
            description="System prompt for request analysis",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.playlists = {}  # In-memory storage for playlists

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": ("complete" if done else "in_progress"),
                    "level": level,
                    "description": message,
                    "done": done,
                },
            },
        )

    async def search_local_playlists(self, query: str) -> Optional[List[Dict]]:
        """Search for playlists in the local Mopidy library."""
        logger.debug(f"Searching local playlists for query: {query}")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playlists.as_list",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    playlists = result.get("result", [])
                    # Filter playlists based on the query
                    matching_playlists = [
                        pl for pl in playlists if query.lower() in pl["name"].lower()
                    ]
                    if matching_playlists:
                        logger.debug(f"Found matching playlists: {matching_playlists}")
                        return matching_playlists
            logger.debug("No matching playlists found.")
            return None
        except Exception as e:
            logger.error(f"Error searching local playlists: {e}")
            return None

    async def search_local(self, query: str) -> Optional[List[Dict]]:
        """Search for songs in the local Mopidy library excluding TuneIn radio stations."""
        logger.debug(f"Searching local library for query: {query}")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.library.search",
                "params": {
                    "query": {"any": [query]},
                    "uris": ["local:", "file:"],  # Only search local music files
                },
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    tracks = result.get("result", [])
                    track_info_list = []
                    for res in tracks:
                        for track in res.get("tracks", []):
                            # Exclude TuneIn radio stations
                            if track["uri"].startswith("tunein:"):
                                continue
                            track_info = {
                                "uri": track["uri"],
                                "name": track.get("name", ""),
                                "artists": [
                                    artist.get("name", "")
                                    for artist in track.get("artists", [])
                                ],
                            }
                            track_info_list.append(track_info)
                    if track_info_list:
                        logger.debug(f"Found local tracks: {track_info_list}")
                        return track_info_list
            logger.debug("No local tracks found.")
            return None
        except Exception as e:
            logger.error(f"Error searching local library: {e}")
            return None

    async def is_iris_installed(self) -> bool:
        """Check if Mopidy Iris is installed by attempting to access its URL."""
        iris_url = self.valves.Mopidy_URL.replace("/mopidy/rpc", "/iris/")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(iris_url) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Error checking Iris installation: {e}")
            return False

    async def select_best_playlist(
        self, playlists: List[Dict], query: str
    ) -> Optional[Dict]:
        """Use LLM to select the best matching playlist."""
        logger.debug(f"Selecting best playlist for query: {query}")
        playlist_names = [pl["name"] for pl in playlists]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that selects the best matching playlist name from a given list, "
                    "based on the user's query. Respond with only the exact playlist name from the list, and no additional text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User query: '{query}'.\n"
                    f"Playlists: {playlist_names}.\n"
                    f"Select the best matching playlist name from the list and respond with only that name."
                ),
            },
        ]
        try:
            response = await generate_chat_completions(
                self.__request__,
                {
                    "model": self.valves.Model or self.__model__,
                    "messages": messages,
                    "temperature": self.valves.Temperature,
                    "stream": False,
                },
                user=self.__user__,
            )
            content = response["choices"][0]["message"]["content"].strip()
            logger.debug(f"LLM selected playlist: {content}")
            # Clean the response
            cleaned_content = content.replace('"', "").replace("'", "").strip().lower()
            selected_playlist = None
            for pl in playlists:
                if pl["name"].lower() == cleaned_content:
                    selected_playlist = pl
                    break
            if not selected_playlist:
                # Try partial match
                for pl in playlists:
                    if pl["name"].lower() in cleaned_content:
                        selected_playlist = pl
                        break
            if selected_playlist:
                logger.debug(f"Found matching playlist: {selected_playlist['name']}")
                return selected_playlist
            else:
                logger.debug("LLM selection did not match any playlist names.")
                return None
        except Exception as e:
            logger.error(f"Error selecting best playlist: {e}")
            return None

    async def generate_player_html(self) -> str:
        """Generate HTML code for the music player UI with all logic included in the output."""
        if await self.is_iris_installed() and self.valves.Use_Iris:
            # Use Iris interface
            iris_url = self.valves.Mopidy_URL.replace("/mopidy/rpc", "/iris/")
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Mopidy Iris</title>
                <style>
                    body, html {{
                        margin: 0;
                        padding: 0;
                        height: 100%;
                        overflow: hidden;
                    }}
                    iframe {{
                        width: 100%;
                        height: 100%;
                        border: none;
                    }}
                </style>
            </head>
            <body>
                <iframe src="{iris_url}">
                    Your browser doesn't support iframes
                </iframe>
            </body>
            </html>
            """
        else:
            # Use custom HTML UI with WebSocket implementation
            # Fetch current track info
            current_track = await self.get_current_track_info()

            # Set default values
            track_name = current_track.get("name", "No track playing")
            artists = (
                ", ".join(
                    artist.get("name", "Unknown Artist")
                    for artist in current_track.get("artists", [])
                )
                if current_track.get("artists")
                else "Unknown Artist"
            )
            album = current_track.get("album", {}).get("name", "Unknown Album")
            track_uri = current_track.get("uri", "")

            # Get WebSocket URL from Mopidy RPC URL
            ws_url = self.valves.Mopidy_URL.replace("http://", "ws://").replace(
                "/mopidy/rpc", "/mopidy/ws"
            )

            html = ""
        return html

    async def search_youtube_with_api(
        self, query: str, playlist=False
    ) -> Optional[List[Dict]]:
        """Search YouTube using the YouTube Data API."""
        logger.debug(f"Searching YouTube Data API for query: {query}")
        try:
            if not self.valves.YouTube_API_Key:
                logger.error("YouTube API Key not provided.")
                return None

            if playlist:
                search_type = "playlist"
            else:
                search_type = "video"
            api_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "maxResults": self.valves.Max_Search_Results,
                "key": self.valves.YouTube_API_Key,
                "type": search_type,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"YouTube API error: {data}")
                        return None
                    items = data.get("items", [])
                    tracks = []
                    for item in items:
                        snippet = item.get("snippet", {})
                        if playlist:
                            playlist_id = item["id"]["playlistId"]
                            # Now, fetch all videos in the playlist
                            playlist_videos = await self.get_playlist_videos(
                                playlist_id
                            )
                            tracks.extend(playlist_videos)
                        else:
                            video_id = item["id"]["videoId"]
                            uri = f"yt:https://www.youtube.com/watch?v={video_id}"
                            track_info = {
                                "uri": uri,
                                "name": snippet.get("title", ""),
                                "artists": [snippet.get("channelTitle", "")],
                            }
                            tracks.append(track_info)
                    if tracks:
                        logger.debug(f"Found YouTube tracks: {tracks}")
                        return tracks
            logger.debug("No YouTube content found via API.")
            return None
        except Exception as e:
            logger.error(f"Error searching YouTube API: {e}")
            logger.error(traceback.format_exc())
            return None

    async def search_youtube_playlists(self, query: str) -> Optional[List[Dict]]:
        """Search YouTube for playlists."""
        logger.debug(f"Searching YouTube for playlists with query: {query}")
        try:
            if not self.valves.YouTube_API_Key:
                logger.error("YouTube API Key not provided.")
                return None

            api_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "maxResults": self.valves.Max_Search_Results,
                "key": self.valves.YouTube_API_Key,
                "type": "playlist",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"YouTube API error: {data}")
                        return None
                    items = data.get("items", [])
                    playlists = []
                    for item in items:
                        snippet = item.get("snippet", {})
                        playlist_info = {
                            "id": item["id"]["playlistId"],
                            "name": snippet.get("title", ""),
                            "description": snippet.get("description", ""),
                        }
                        playlists.append(playlist_info)
                    if playlists:
                        logger.debug(f"Found YouTube playlists: {playlists}")
                        return playlists
            logger.debug("No YouTube playlists found.")
            return None
        except Exception as e:
            logger.error(f"Error searching YouTube playlists: {e}")
            logger.error(traceback.format_exc())
            return None

    async def get_playlist_tracks(self, uri: str) -> Optional[List[Dict]]:
        """Get tracks from the specified playlist URI."""
        logger.debug(f"Fetching tracks from playlist URI: {uri}")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playlists.get_items",
                "params": {"uri": uri},
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    tracks = result.get("result", [])
                    if tracks:
                        track_info_list = []
                        for item in tracks:
                            track_info = {
                                "uri": item.get("uri"),
                                "name": item.get("name", ""),
                                "artists": [],  # Artist info might not be available here
                            }
                            track_info_list.append(track_info)
                        logger.debug(f"Tracks in playlist: {track_info_list}")
                        return track_info_list
            logger.debug("No tracks found in playlist.")
            return None
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")
            return None

    async def get_playlist_videos(self, playlist_id: str) -> List[Dict]:
        """Retrieve all videos from a YouTube playlist using the YouTube Data API."""
        logger.debug(f"Fetching videos from playlist ID: {playlist_id}")
        try:
            api_url = "https://www.googleapis.com/youtube/v3/playlistItems"
            params = {
                "part": "snippet",
                "playlistId": playlist_id,
                "maxResults": 50,  # Maximum allowed by the API per request
                "key": self.valves.YouTube_API_Key,
            }
            tracks = []
            async with aiohttp.ClientSession() as session:
                while True:
                    async with session.get(api_url, params=params) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            logger.error(f"YouTube API error: {data}")
                            break
                        items = data.get("items", [])
                        for item in items:
                            snippet = item.get("snippet", {})
                            video_id = snippet["resourceId"]["videoId"]
                            uri = f"yt:https://www.youtube.com/watch?v={video_id}"
                            track_info = {
                                "uri": uri,
                                "name": snippet.get("title", ""),
                                "artists": [snippet.get("channelTitle", "")],
                            }
                            tracks.append(track_info)
                        if "nextPageToken" in data:
                            params["pageToken"] = data["nextPageToken"]
                        else:
                            break  # No more pages
            logger.debug(f"Total videos fetched from playlist: {len(tracks)}")
            return tracks
        except Exception as e:
            logger.error(f"Error fetching playlist videos: {e}")
            logger.error(traceback.format_exc())
            return []

    async def search_youtube(self, query: str, playlist=False) -> Optional[List[Dict]]:
        """Search YouTube for the song or playlist."""
        return await self.search_youtube_with_api(query, playlist)

    async def play_uris(self, tracks: List[Dict]):
        """Play a list of tracks in Mopidy."""
        uris = [track["uri"] for track in tracks]
        logger.debug(f"Playing URIs: {uris}")
        try:
            payloads = [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "core.tracklist.clear",
                },
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "core.tracklist.add",
                    "params": {"uris": uris},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "core.playback.play",
                },
            ]
            async with aiohttp.ClientSession() as session:
                for payload in payloads:
                    async with session.post(
                        self.valves.Mopidy_URL, json=payload
                    ) as response:
                        result = await response.json()
                        logger.debug(f"Response for {payload['method']}: {result}")
            return True
        except Exception as e:
            logger.error(f"Error playing URIs: {e}")
            return False

    async def analyze_request(self, user_input: str) -> Dict:
        """
        Extract the command and parameters from the user's request.
        """
        logger.debug(f"Analyzing user input: {user_input}")
        command_mapping = {
            "stop": "pause",
            "halt": "pause",
            "play": "play",
            "start": "play",
            "resume": "resume",
            "continue": "resume",
            "next": "skip",
            "skip": "skip",
            "pause": "pause",
        }
        user_command = user_input.lower().strip()
        if user_command in command_mapping:
            action = command_mapping[user_command]
            analysis = {"action": action, "parameters": {}}
            logger.debug(f"Directly parsed simple command: {analysis}")
            return analysis
        else:
            # If the input is longer or doesn't match simple commands, call the LLM
            try:
                messages = [
                    {"role": "system", "content": self.valves.system_prompt},
                    {"role": "user", "content": user_input},
                ]

                response = await generate_chat_completions(
                    self.__request__,
                    {
                        "model": self.valves.Model or self.__model__,
                        "messages": messages,
                        "temperature": self.valves.Temperature,
                        "stream": False,
                    },
                    user=self.__user__,
                )

                # Extract and parse the JSON response
                content = response["choices"][0]["message"]["content"]
                logger.debug(f"LLM response: {content}")
                try:
                    # Use regex to extract JSON in case of extra text
                    match = re.search(r"\{[\s\S]*\}", content)
                    if match:
                        content = match.group(0)
                    else:
                        raise ValueError(
                            "No JSON object found in the assistant's response."
                        )
                    analysis = json.loads(content)

                    # Map possible alternative keys from LLM response
                    if "type" in analysis:
                        analysis["action"] = analysis.pop("type")
                    if "query" in analysis:
                        analysis.setdefault("parameters", {})["title"] = analysis.pop(
                            "query"
                        )
                    if "mood" in analysis:
                        analysis.setdefault("parameters", {})["mood"] = analysis.pop(
                            "mood"
                        )
                    if "genre" in analysis:
                        analysis.setdefault("parameters", {})["genre"] = analysis.pop(
                            "genre"
                        )
                    # Map action aliases
                    action_aliases = {
                        "playlist": "play_playlist",
                        "song": "play_song",
                        "album": "play_playlist",  # Treat albums as playlists
                        "stop": "pause",
                        "play": "play",
                    }
                    if analysis.get("action") in action_aliases:
                        analysis["action"] = action_aliases[analysis["action"]]

                    # Ensure required fields are present
                    if "action" not in analysis:
                        analysis["action"] = "play_song"
                        analysis["parameters"] = {"title": user_input}
                    elif "parameters" not in analysis:
                        analysis["parameters"] = {}

                    logger.debug(f"Request analysis: {analysis}")
                    return analysis

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse LLM response as JSON: {content}. Error: {e}"
                    )
                    # Fallback to default action 'play_song' with user input as title
                    logger.debug(
                        "Defaulting to 'play_song' action with the entire input as title."
                    )
                    return {"action": "play_song", "parameters": {"title": user_input}}

            except Exception as e:
                logger.error(f"Error in analyze_request: {e}")
                # Fallback to default action 'play_song' with user input as title
                logger.debug(
                    "Defaulting to 'play_song' action with the entire input as title due to exception."
                )
                return {"action": "play_song", "parameters": {"title": user_input}}

    async def handle_command(self, analysis: Dict):
        """Handle the command extracted from the analysis."""
        action = analysis.get("action")
        parameters = analysis.get("parameters", {})
        title = parameters.get("title", "")
        artist = parameters.get("artist", "")
        playlist_name = parameters.get("playlist_name", "default")

        if action == "play_song":
            query = f"{title} {artist}".strip()
            if not query:
                await self.emit_message("Please specify a song to play.")
                await self.emit_status("error", "No song specified", True)
                return

            # First, try to find the song in the local library
            await self.emit_status(
                "info", f"Searching for '{query}' in local library...", False
            )
            tracks = await self.search_local(query)
            if tracks:
                # Song found locally
                play_success = await self.play_uris(tracks)
                if play_success:
                    track_names = ", ".join(
                        [f"{t['name']} by {t['artists'][0]}" for t in tracks[:3]]
                    )
                    await self.emit_message(
                        f"Now playing from local library: {track_names}..."
                    )
                    # Generate HTML code for the player UI
                    html_code = await self.generate_player_html()
                    # Wrap HTML code in a code block with language specifier
                    html_code_block = (
                        f"""\n ```html \n{html_code}""" if html_code else ""
                    )
                    # Emit the HTML code
                    await self.emit_message(html_code_block)
                    await self.emit_status("success", "Playback started", True)
                else:
                    await self.emit_message("Failed to start playback.")
                    await self.emit_status("error", "Playback failed", True)
                return

            # If not found locally, search YouTube
            await self.emit_status(
                "info", f"Not found locally. Searching YouTube for '{query}'...", False
            )
            tracks = await self.search_youtube(query)
            if tracks:
                # Song found on YouTube
                # Choose the most relevant track
                track = tracks[0]
                play_success = await self.play_uris([track])
                if play_success:
                    await self.emit_message(
                        f"Now playing '{track['name']}' by {track['artists'][0]} from YouTube."
                    )
                    # Generate HTML code for the player UI
                    html_code = await self.generate_player_html()
                    # Wrap HTML code in a code block with language specifier
                    html_code_block = (
                        f"""\n ```html \n{html_code}""" if html_code else ""
                    )
                    # Emit the HTML code
                    await self.emit_message(html_code_block)
                    await self.emit_status("success", "Playback started", True)
                else:
                    await self.emit_message("Failed to start playback.")
                    await self.emit_status("error", "Playback failed", True)
                return
            else:
                await self.emit_message(f"No matching content found for '{query}'.")
                await self.emit_status("error", "No results found", True)
            return

        elif action == "play_playlist":
            query = title or playlist_name
            if not query:
                await self.emit_message("Please specify a playlist to play.")
                await self.emit_status("error", "No playlist specified", True)
                return

            # Search for playlists in the local library
            await self.emit_status(
                "info", f"Searching for playlist '{query}' in local library...", False
            )
            playlists = await self.search_local_playlists(query)
            if playlists:
                # Use LLM to select the best matching playlist
                best_playlist = await self.select_best_playlist(playlists, query)
                if best_playlist:
                    # Get tracks from the selected playlist
                    tracks = await self.get_playlist_tracks(best_playlist["uri"])
                    if tracks:
                        play_success = await self.play_uris(tracks)
                        if play_success:
                            await self.emit_message(
                                f"Now playing playlist '{best_playlist['name']}' from local library."
                            )
                            # Generate and emit HTML code
                            html_code = await self.generate_player_html()
                            html_code_block = (
                                f"""\n```html\n{html_code}\n```""" if html_code else ""
                            )
                            await self.emit_message(html_code_block)
                            await self.emit_status("success", "Playback started", True)
                        else:
                            await self.emit_message("Failed to play playlist.")
                            await self.emit_status("error", "Playback failed", True)
                    else:
                        await self.emit_message(
                            f"No tracks found in playlist '{best_playlist['name']}'."
                        )
                        await self.emit_status("error", "No tracks in playlist", True)
                else:
                    await self.emit_message(
                        "Could not determine the best playlist to play."
                    )
                    await self.emit_status("error", "Playlist selection failed", True)
                return

            # If not found locally, search YouTube for a playlist
            await self.emit_status(
                "info",
                f"Not found locally. Searching YouTube for playlist '{query}'...",
                False,
            )
            playlists = await self.search_youtube_playlists(query)
            if playlists:
                # Use LLM to select the best matching playlist
                best_playlist = await self.select_best_playlist(playlists, query)
                if best_playlist:
                    # Get tracks from the selected YouTube playlist
                    tracks = await self.get_playlist_videos(best_playlist["id"])
                    if tracks:
                        play_success = await self.play_uris(tracks)
                        if play_success:
                            await self.emit_message(
                                f"Now playing YouTube playlist '{best_playlist['name']}'."
                            )
                            html_code = await self.generate_player_html()
                            html_code_block = (
                                f"""\n```html\n{html_code}\n```""" if html_code else ""
                            )
                            await self.emit_message(html_code_block)
                            await self.emit_status("success", "Playback started", True)
                        else:
                            await self.emit_message("Failed to play YouTube playlist.")
                            await self.emit_status("error", "Playback failed", True)
                    else:
                        await self.emit_message(
                            f"No tracks found in YouTube playlist '{best_playlist['name']}'."
                        )
                        await self.emit_status("error", "No tracks in playlist", True)
                else:
                    await self.emit_message(
                        "Could not determine the best playlist to play."
                    )
                    await self.emit_status("error", "Playlist selection failed", True)
            else:
                await self.emit_message(f"No matching playlist found for '{query}'.")
                await self.emit_status("error", "No playlist found", True)
            return

        elif action == "show_current_song":
            # Generate HTML code for the player UI
            html_code = await self.generate_player_html()
            # Wrap HTML code in a code block with language specifier
            html_code_block = f"""\n ```html \n{html_code}""" if html_code else ""
            await self.emit_message(html_code_block)
            await self.emit_status("success", "Displayed current song", True)
            return

        elif action == "pause":
            pause_success = await self.pause()
            if pause_success:
                await self.emit_message("Playback paused.")
                await self.emit_status("success", "Playback paused", True)
            else:
                await self.emit_message("Failed to pause playback.")
                await self.emit_status("error", "Failed to pause", True)
            return

        elif action == "resume" or action == "play":
            play_success = await self.play()
            if play_success:
                await self.emit_message("Playback resumed.")
                await self.emit_status("success", "Playback resumed", True)
            else:
                await self.emit_message("Failed to resume playback.")
                await self.emit_status("error", "Failed to resume", True)
            return

        elif action == "skip":
            skip_success = await self.skip()
            if skip_success:
                await self.emit_message("Skipped to the next track.")
                await self.emit_status("success", "Skipped track", True)
            else:
                await self.emit_message("Failed to skip track.")
                await self.emit_status("error", "Failed to skip", True)
            return

        else:
            # Default action is to try playing the user's input as a song
            await self.emit_message(
                "Command not recognized. Attempting to play as a song."
            )
            new_analysis = {
                "action": "play_song",
                "parameters": {"title": title or action},
            }
            await self.handle_command(new_analysis)
            return

    async def get_current_track_info(self) -> Dict:
        """Get the current track playing."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playback.get_current_track",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    track = result.get("result", {})
                    return track if track else {}
        except Exception as e:
            logger.error(f"Error getting current track: {e}")
            return {}

    async def play(self):
        """Resume playback."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playback.play",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    logger.debug(f"Response for play: {result}")
            return True
        except Exception as e:
            logger.error(f"Error resuming playback: {e}")
            return False

    async def pause(self):
        """Pause playback."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playback.pause",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    logger.debug(f"Response for pause: {result}")
            return True
        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            return False

    async def skip(self):
        """Skip to the next track."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playback.next",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    logger.debug(f"Response for skip: {result}")
            return True
        except Exception as e:
            logger.error(f"Error skipping track: {e}")
            return False

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        """Main pipe function to process music requests."""
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = User(**__user__)
        self.__model__ = self.valves.Model or __model__
        self.__request__ = __request__
        logger.debug(__task__)
        if __task__ and __task__ != TASKS.DEFAULT:
            response = await generate_chat_completions(
                self.__request__,
                {
                    "model": self.__model__,
                    "messages": body.get("messages"),
                    "stream": False,
                },
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        user_input = body.get("messages", [])[-1].get("content", "").strip()
        logger.debug(f"User input: {user_input}")

        try:
            await self.emit_status("info", "Analyzing your request...", False)
            analysis = await self.analyze_request(user_input)
            logger.debug(f"Analysis result: {analysis}")
            await self.handle_command(analysis)

        except Exception as e:
            logger.error(f"Error processing music request: {e}")
            await self.emit_message(f"An error occurred: {str(e)}")
            await self.emit_status("error", f"Error: {str(e)}", True)

        return ""
