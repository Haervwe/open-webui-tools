"""
title: Mopidy_Music_Controller
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools
funding_url: https://github.com/Haervwe/open-webui-tools
version: 0.4.0
description: A pipe to control Mopidy music server to play songs from local library or YouTube, manage playlists, and handle various music commands 
needs a Local and/or a Youtube API endpoint configured in mopidy.
mopidy repo: https://github.com/mopidy
"""

import logging
import json
from typing import Dict, List, Callable, Awaitable, Optional
from pydantic import BaseModel, Field
import aiohttp
import asyncio
import re
import traceback
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import User ,Users

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


def clean_thinking_tags(message: str) -> str:
    """Remove various thinking/reasoning tags that LLMs might include in their responses."""
    pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought|analysis|Analysis)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|",
        re.DOTALL,
    )
    return re.sub(pattern, "", message).strip()


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
        Request_Timeout: float = Field(
            default=10.0,
            description="Timeout in seconds for HTTP requests (YouTube API, Mopidy RPC, etc.)",
        )
        Debug_Logging: bool = Field(
            default=False,
            description="Enable detailed debug logging for troubleshooting",
        )
        system_prompt: str = Field(
            default=(
                "Extract music commands as JSON. Output ONLY this format:\n"
                '{"action": "ACTION", "parameters": {"query": "SEARCH_TERMS"}}\n\n'
                "Valid actions: play_song, play_playlist, pause, resume, skip, show_current_song\n\n"
                "MANDATORY RULES:\n"
                "- Parameters MUST have 'query' field ONLY\n"
                "- NEVER use 'title', 'artist', or 'playlist_name' fields\n"
                "- Remove filler words from query: play, some, songs, music, tracks, by, the, a, an\n"
                "- Keep only essential terms: artist names, song titles, album names\n\n"
                "EXAMPLES (input → output):\n"
                'play some Parov Stelar songs → {"action": "play_playlist", "parameters": {"query": "Parov Stelar"}}\n'
                'play Booty Swing by Parov Stelar → {"action": "play_song", "parameters": {"query": "Booty Swing Parov Stelar"}}\n'
                'play The Princess album → {"action": "play_playlist", "parameters": {"query": "Princess"}}\n'
                'pause → {"action": "pause", "parameters": {}}\n\n'
                "Output JSON only. No explanations, no thinking, no extra fields."
            ),
            description="System prompt for request analysis",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.playlists = {}

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
        if self.valves.Debug_Logging:
            logger.debug(f"Searching local playlists for query: {query}")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playlists.as_list",
            }
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    playlists = result.get("result", [])
                    matching_playlists = [
                        pl for pl in playlists if query.lower() in pl["name"].lower()
                    ]
                    if matching_playlists:
                        if self.valves.Debug_Logging:
                            logger.debug(f"Found matching playlists: {matching_playlists}")
                        return matching_playlists
            if self.valves.Debug_Logging:
                logger.debug("No matching playlists found.")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout searching local playlists after {self.valves.Request_Timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error searching local playlists: {e}")
            return None

    async def search_local(self, query: str) -> Optional[List[Dict]]:
        """Search for songs in the local Mopidy library excluding TuneIn radio stations."""
        if self.valves.Debug_Logging:
            logger.debug(f"Searching local library for query: {query}")
        try:
            # Try multiple search strategies
            search_strategies = [
                # Strategy 1: Search with individual words across all fields
                {
                    "any": query.split(),
                    "artist": query.split(),
                },
                # Strategy 2: Search with full query string
                {
                    "any": [query],
                },
                # Strategy 3: Artist-specific search with full string
                {
                    "artist": [query],
                },
            ]
            
            all_tracks = []
            seen_uris = set()
            
            for strategy_num, search_params in enumerate(search_strategies, 1):
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "core.library.search",
                    "params": {
                        "query": search_params,
                    },
                }
                if self.valves.Debug_Logging:
                    logger.debug(f"Search strategy {strategy_num} payload: {payload}")
                
                timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.valves.Mopidy_URL, json=payload
                    ) as response:
                        result = await response.json()
                        if self.valves.Debug_Logging:
                            logger.debug(f"Strategy {strategy_num} result: {result}")
                        
                        tracks = result.get("result", [])
                        for res in tracks:
                            for track in res.get("tracks", []):
                                uri = track["uri"]
                                if uri.startswith("tunein:") or uri in seen_uris:
                                    continue
                                seen_uris.add(uri)
                                track_info = {
                                    "uri": uri,
                                    "name": track.get("name", ""),
                                    "artists": [
                                        artist.get("name", "")
                                        for artist in track.get("artists", [])
                                    ],
                                }
                                all_tracks.append(track_info)
                
                # If we found tracks, don't try more strategies
                if all_tracks:
                    break
            
            if all_tracks:
                if self.valves.Debug_Logging:
                    logger.debug(f"Found {len(all_tracks)} local tracks: {all_tracks[:5]}")
                return all_tracks
            
            if self.valves.Debug_Logging:
                logger.debug("No local tracks found after trying all strategies.")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout searching local library after {self.valves.Request_Timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error searching local library: {e}")
            return None

    async def select_best_playlist(
        self, playlists: List[Dict], query: str
    ) -> Optional[Dict]:
        """Use LLM to select the best matching playlist."""
        if self.valves.Debug_Logging:
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
            if self.valves.Debug_Logging:
                logger.debug(f"LLM selected playlist: {content}")
            cleaned_content = content.replace('"', "").replace("'", "").strip().lower()
            selected_playlist = None
            for pl in playlists:
                if pl["name"].lower() == cleaned_content:
                    selected_playlist = pl
                    break
            if not selected_playlist:
                for pl in playlists:
                    if pl["name"].lower() in cleaned_content:
                        selected_playlist = pl
                        break
            if selected_playlist:
                if self.valves.Debug_Logging:
                    logger.debug(f"Found matching playlist: {selected_playlist['name']}")
                return selected_playlist
            else:
                if self.valves.Debug_Logging:
                    logger.debug("LLM selection did not match any playlist names.")
                return None
        except Exception as e:
            logger.error(f"Error selecting best playlist: {e}")
            return None

    async def generate_player_html(self) -> str:
        """Generate HTML code for the music player UI with all logic included in the output."""
        current_track = await self.get_current_track_info()
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
        ws_url = self.valves.Mopidy_URL.replace("http://", "ws://").replace(
            "/mopidy/rpc", "/mopidy/ws"
        )
        rpc_url = self.valves.Mopidy_URL
        
        html = f"""<!DOCTYPE html>
<head>
    <title>Mopidy Player</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .player {{
            background: #2a2a2a;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
            max-width: 400px;
            width: 100%;
            border: 1px solid #3a3a3a;
        }}
        
        .album-art {{
            width: 100%;
            height: 300px;
            background: #1a1a1a;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 80px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
            background-size: cover;
            background-position: center;
            border: 1px solid #3a3a3a;
        }}
        
        .album-art-placeholder {{
            font-size: 80px;
            color: #555;
            z-index: 1;
        }}
        
        .album-art img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        
        .album-art.playing::after {{
            content: '';
            position: absolute;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent);
            animation: shine 3s infinite;
            z-index: 2;
        }}
        
        @keyframes shine {{
            0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
            100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
        }}
        
        .track-info {{
            text-align: center;
            margin-bottom: 25px;
        }}
        
        .track-name {{
            font-size: 20px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .track-artist {{
            font-size: 16px;
            color: #b0b0b0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .track-album {{
            font-size: 14px;
            color: #808080;
            margin-top: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .progress-container {{
            margin-bottom: 25px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 6px;
            background: #1a1a1a;
            border-radius: 3px;
            cursor: pointer;
            position: relative;
            margin-bottom: 8px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #555 0%, #888 100%);
            border-radius: 3px;
            width: 0%;
            transition: width 0.1s linear;
        }}
        
        .time-info {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #808080;
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .control-btn {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            padding: 10px;
            border-radius: 50%;
            transition: all 0.3s;
            color: #b0b0b0;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
        }}
        
        .control-btn:hover {{
            background: #3a3a3a;
            color: #ffffff;
            transform: scale(1.1);
        }}
        
        .control-btn.play-pause {{
            background: linear-gradient(135deg, #444 0%, #666 100%);
            color: white;
            font-size: 30px;
            width: 60px;
            height: 60px;
        }}
        
        .control-btn.play-pause:hover {{
            background: linear-gradient(135deg, #555 0%, #777 100%);
            transform: scale(1.15);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
        }}
        
        .volume-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .volume-icon {{
            font-size: 20px;
            color: #b0b0b0;
        }}
        
        .volume-slider {{
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: #1a1a1a;
            border-radius: 3px;
            outline: none;
        }}
        
        .volume-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: linear-gradient(135deg, #555 0%, #888 100%);
            border-radius: 50%;
            cursor: pointer;
        }}
        
        .volume-slider::-moz-range-thumb {{
            width: 16px;
            height: 16px;
            background: linear-gradient(135deg, #555 0%, #888 100%);
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        
        .status {{
            text-align: center;
            font-size: 12px;
            color: #808080;
            padding: 8px;
            background: #1a1a1a;
            border-radius: 8px;
        }}
        
        .status.connected {{
            color: #90ee90;
        }}
        
        .status.error {{
            color: #ff6b6b;
        }}
    </style>
</head>
<body>
    <div class="player">
        <div class="album-art" id="albumArt">
            <span class="album-art-placeholder">🎵</span>
        </div>
        
        <div class="track-info">
            <div class="track-name" id="trackName">{track_name}</div>
            <div class="track-artist" id="trackArtist">{artists}</div>
            <div class="track-album" id="trackAlbum">{album}</div>
        </div>
        
        <div class="progress-container">
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="time-info">
                <span id="currentTime">0:00</span>
                <span id="duration">0:00</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="control-btn" onclick="previousTrack()" title="Previous">⏮</button>
            <button class="control-btn play-pause" onclick="togglePlayPause()" id="playPauseBtn" title="Play/Pause">▶</button>
            <button class="control-btn" onclick="nextTrack()" title="Next">⏭</button>
        </div>
        
        <div class="volume-container">
            <span class="volume-icon">♪</span>
            <input type="range" min="0" max="100" value="50" class="volume-slider" id="volumeSlider">
        </div>
        
        <div class="status" id="status">Connecting...</div>
    </div>
    
    <script>
        const RPC_URL = '{rpc_url}';
        const WS_URL = '{ws_url}';
        let ws = null;
        let isPlaying = false;
        let currentPosition = 0;
        let trackLength = 0;
        let positionInterval = null;
        
        // Format time in seconds to MM:SS
        function formatTime(seconds) {{
            if (!seconds || isNaN(seconds)) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Make RPC call to Mopidy
        async function rpcCall(method, params = {{}}) {{
            try {{
                const response = await fetch(RPC_URL, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        jsonrpc: '2.0',
                        id: Date.now(),
                        method: method,
                        params: params
                    }})
                }});
                const data = await response.json();
                return data.result;
            }} catch (error) {{
                console.error('RPC call failed:', error);
                updateStatus('Error: ' + error.message, 'error');
                return null;
            }}
        }}
        
        // Update album art
        async function updateAlbumArt(track) {{
            const albumArt = document.getElementById('albumArt');
            
            if (track && track.uri) {{
                try {{
                    // Get album art images from Mopidy
                    const images = await rpcCall('core.library.get_images', {{ uris: [track.uri] }});
                    
                    if (images && images[track.uri] && images[track.uri].length > 0) {{
                        // Sort by size to get the best quality image
                        const sortedImages = images[track.uri].sort((a, b) => {{
                            const aSize = (a.width || 0) * (a.height || 0);
                            const bSize = (b.width || 0) * (b.height || 0);
                            return bSize - aSize;
                        }});
                        
                        let imageUrl = sortedImages[0].uri;
                        
                        // Fix the URL if it's relative or has wrong base
                        // Parse the RPC URL to get the Mopidy server base
                        const rpcUrl = new URL(RPC_URL);
                        const mopidyBase = `${{rpcUrl.protocol}}//${{rpcUrl.host}}`;
                        
                        // If the image URL starts with /images, /local, or other Mopidy paths
                        // prepend the Mopidy server base
                        if (imageUrl.startsWith('/')) {{
                            imageUrl = mopidyBase + imageUrl;
                        }} else if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) {{
                            // If it's not an absolute URL, make it one
                            imageUrl = mopidyBase + '/' + imageUrl;
                        }}
                        
                        console.log('Loading album art from:', imageUrl);
                        albumArt.innerHTML = `<img src="${{imageUrl}}" alt="Album Art" onerror="this.style.display='none'; this.parentElement.innerHTML='<span class=\\'album-art-placeholder\\'>🎵</span>'; console.error('Failed to load image:', '${{imageUrl}}');">`;
                    }} else {{
                        albumArt.innerHTML = '<span class="album-art-placeholder">🎵</span>';
                    }}
                }} catch (error) {{
                    console.error('Failed to load album art:', error);
                    albumArt.innerHTML = '<span class="album-art-placeholder">🎵</span>';
                }}
            }} else {{
                albumArt.innerHTML = '<span class="album-art-placeholder">🎵</span>';
            }}
        }}
        
        // Update UI with track info
        function updateTrackInfo(track) {{
            if (!track) return;
            
            document.getElementById('trackName').textContent = track.name || 'Unknown Track';
            
            const artists = track.artists?.map(a => a.name).join(', ') || 'Unknown Artist';
            document.getElementById('trackArtist').textContent = artists;
            
            const album = track.album?.name || 'Unknown Album';
            document.getElementById('trackAlbum').textContent = album;
            
            trackLength = track.length ? track.length / 1000 : 0;
            document.getElementById('duration').textContent = formatTime(trackLength);
            
            // Update album art
            updateAlbumArt(track);
        }}
        
        // Update play/pause button
        function updatePlayPauseButton(playing) {{
            isPlaying = playing;
            const btn = document.getElementById('playPauseBtn');
            btn.textContent = playing ? '⏸' : '▶';
            
            const albumArt = document.getElementById('albumArt');
            if (playing) {{
                albumArt.classList.add('playing');
                startPositionTracking();
            }} else {{
                albumArt.classList.remove('playing');
                stopPositionTracking();
            }}
        }}
        
        // Update progress bar
        function updateProgress(position) {{
            if (!trackLength) return;
            currentPosition = position / 1000;
            const percentage = (currentPosition / trackLength) * 100;
            document.getElementById('progressFill').style.width = percentage + '%';
            document.getElementById('currentTime').textContent = formatTime(currentPosition);
        }}
        
        // Start tracking position
        function startPositionTracking() {{
            stopPositionTracking();
            positionInterval = setInterval(async () => {{
                const position = await rpcCall('core.playback.get_time_position');
                if (position !== null) {{
                    updateProgress(position);
                }}
            }}, 1000);
        }}
        
        function stopPositionTracking() {{
            if (positionInterval) {{
                clearInterval(positionInterval);
                positionInterval = null;
            }}
        }}
        
        // Update status message
        function updateStatus(message, type = '') {{
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status' + (type ? ' ' + type : '');
        }}
        
        // Control functions
        async function togglePlayPause() {{
            const state = await rpcCall('core.playback.get_state');
            if (state === 'playing') {{
                await rpcCall('core.playback.pause');
            }} else {{
                await rpcCall('core.playback.play');
            }}
        }}
        
        async function previousTrack() {{
            await rpcCall('core.playback.previous');
        }}
        
        async function nextTrack() {{
            await rpcCall('core.playback.next');
        }}
        
        // Volume control
        document.getElementById('volumeSlider').addEventListener('input', async (e) => {{
            const volume = parseInt(e.target.value);
            await rpcCall('core.mixer.set_volume', {{ volume: volume }});
        }});
        
        // Progress bar seeking
        document.getElementById('progressBar').addEventListener('click', async (e) => {{
            if (!trackLength) return;
            const bar = e.currentTarget;
            const rect = bar.getBoundingClientRect();
            const percentage = (e.clientX - rect.left) / rect.width;
            const position = Math.floor(percentage * trackLength * 1000);
            await rpcCall('core.playback.seek', {{ time_position: position }});
        }});
        
        // Initialize WebSocket connection
        function connectWebSocket() {{
            try {{
                ws = new WebSocket(WS_URL);
                
                ws.onopen = () => {{
                    console.log('WebSocket connected');
                    updateStatus('Connected', 'connected');
                    initializePlayer();
                }};
                
                ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    
                    if (data.event === 'track_playback_started') {{
                        updateTrackInfo(data.tl_track?.track);
                        updatePlayPauseButton(true);
                    }} else if (data.event === 'playback_state_changed') {{
                        updatePlayPauseButton(data.new_state === 'playing');
                    }} else if (data.event === 'track_playback_paused') {{
                        updatePlayPauseButton(false);
                    }} else if (data.event === 'track_playback_resumed') {{
                        updatePlayPauseButton(true);
                    }} else if (data.event === 'seeked') {{
                        updateProgress(data.time_position);
                    }}
                }};
                
                ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    updateStatus('Connection error', 'error');
                }};
                
                ws.onclose = () => {{
                    console.log('WebSocket disconnected');
                    updateStatus('Disconnected - Retrying...', 'error');
                    setTimeout(connectWebSocket, 3000);
                }};
            }} catch (error) {{
                console.error('Failed to connect WebSocket:', error);
                updateStatus('Failed to connect', 'error');
            }}
        }}
        
        // Initialize player state
        async function initializePlayer() {{
            try {{
                // Get current track
                const track = await rpcCall('core.playback.get_current_track');
                if (track) {{
                    updateTrackInfo(track);
                }}
                
                // Get playback state
                const state = await rpcCall('core.playback.get_state');
                updatePlayPauseButton(state === 'playing');
                
                // Get current position
                const position = await rpcCall('core.playback.get_time_position');
                if (position !== null) {{
                    updateProgress(position);
                }}
                
                // Get volume
                const volume = await rpcCall('core.mixer.get_volume');
                if (volume !== null) {{
                    document.getElementById('volumeSlider').value = volume;
                }}
                
                updateStatus('Ready', 'connected');
            }} catch (error) {{
                console.error('Failed to initialize player:', error);
                updateStatus('Initialization error', 'error');
            }}
        }}
        
        // Start everything
        connectWebSocket();
    </script>
</body>
"""
        return html

    async def search_youtube_with_api(
        self, query: str, playlist=False
    ) -> Optional[List[Dict]]:
        """Search YouTube using the YouTube Data API."""
        if self.valves.Debug_Logging:
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                        if self.valves.Debug_Logging:
                            logger.debug(f"Found YouTube tracks: {tracks}")
                        return tracks
            if self.valves.Debug_Logging:
                logger.debug("No YouTube content found via API.")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout searching YouTube API after {self.valves.Request_Timeout}s - check internet connection")
            return None
        except Exception as e:
            logger.error(f"Error searching YouTube API: {e}")
            logger.error(traceback.format_exc())
            return None

    async def search_youtube_playlists(self, query: str) -> Optional[List[Dict]]:
        """Search YouTube for playlists."""
        if self.valves.Debug_Logging:
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                        if self.valves.Debug_Logging:
                            logger.debug(f"Found YouTube playlists: {playlists}")
                        return playlists
            if self.valves.Debug_Logging:
                logger.debug("No YouTube playlists found.")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout searching YouTube playlists after {self.valves.Request_Timeout}s - check internet connection")
            return None
        except Exception as e:
            logger.error(f"Error searching YouTube playlists: {e}")
            logger.error(traceback.format_exc())
            return None

    async def get_playlist_tracks(self, uri: str) -> Optional[List[Dict]]:
        """Get tracks from the specified playlist URI."""
        if self.valves.Debug_Logging:
            logger.debug(f"Fetching tracks from playlist URI: {uri}")
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "core.playlists.get_items",
                "params": {"uri": uri},
            }
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                                "artists": [],
                            }
                            track_info_list.append(track_info)
                        if self.valves.Debug_Logging:
                            logger.debug(f"Tracks in playlist: {track_info_list}")
                        return track_info_list
            if self.valves.Debug_Logging:
                logger.debug("No tracks found in playlist.")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting playlist tracks after {self.valves.Request_Timeout}s")
            return None
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")
            return None

    async def get_playlist_videos(self, playlist_id: str) -> List[Dict]:
        """Retrieve all videos from a YouTube playlist using the YouTube Data API."""
        if self.valves.Debug_Logging:
            logger.debug(f"Fetching videos from playlist ID: {playlist_id}")
        try:
            api_url = "https://www.googleapis.com/youtube/v3/playlistItems"
            params = {
                "part": "snippet",
                "playlistId": playlist_id,
                "maxResults": 50,
                "key": self.valves.YouTube_API_Key,
            }
            tracks = []
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                            break
            if self.valves.Debug_Logging:
                logger.debug(f"Total videos fetched from playlist: {len(tracks)}")
            return tracks
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching playlist videos after {self.valves.Request_Timeout}s - check internet connection")
            return []
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
        if self.valves.Debug_Logging:
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for payload in payloads:
                    async with session.post(
                        self.valves.Mopidy_URL, json=payload
                    ) as response:
                        result = await response.json()
                        if self.valves.Debug_Logging:
                            logger.debug(f"Response for {payload['method']}: {result}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout playing URIs after {self.valves.Request_Timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error playing URIs: {e}")
            return False

    async def analyze_request(self, user_input: str) -> Dict:
        """
        Extract the command and parameters from the user's request.
        """
        if self.valves.Debug_Logging:
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
            if self.valves.Debug_Logging:
                logger.debug(f"Directly parsed simple command: {analysis}")
            return analysis
        else:
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

                content = response["choices"][0]["message"]["content"]
                if self.valves.Debug_Logging:
                    logger.debug(f"LLM response (raw): {content}")
                content = clean_thinking_tags(content)
                if self.valves.Debug_Logging:
                    logger.debug(f"LLM response (cleaned): {content}")
                try:
                    match = re.search(r"\{[\s\S]*\}", content)
                    if match:
                        content = match.group(0)
                    else:
                        raise ValueError(
                            "No JSON object found in the assistant's response."
                        )
                    analysis = json.loads(content)
                    if "type" in analysis:
                        analysis["action"] = analysis.pop("type")
                    
                    if "title" in analysis.get("parameters", {}):
                        title = analysis["parameters"].pop("title", "")
                        artist = analysis["parameters"].pop("artist", "")
                        analysis["parameters"]["query"] = f"{title} {artist}".strip()
                    elif "artist" in analysis.get("parameters", {}):
                        analysis["parameters"]["query"] = analysis["parameters"].pop("artist")
                    elif "playlist_name" in analysis.get("parameters", {}):
                        analysis["parameters"]["query"] = analysis["parameters"].pop("playlist_name")
                    
                    action_aliases = {
                        "playlist": "play_playlist",
                        "song": "play_song",
                        "album": "play_playlist",
                        "stop": "pause",
                        "play": "play",
                    }
                    if analysis.get("action") in action_aliases:
                        analysis["action"] = action_aliases[analysis["action"]]

                    if "action" not in analysis:
                        analysis["action"] = "play_song"
                        analysis["parameters"] = {"query": user_input}
                    elif "parameters" not in analysis:
                        analysis["parameters"] = {}

                    if self.valves.Debug_Logging:
                        logger.debug(f"Request analysis: {analysis}")
                    return analysis

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse LLM response as JSON: {content}. Error: {e}"
                    )
                    logger.debug(
                        "Defaulting to 'play_song' action with the entire input as query."
                    )
                    return {"action": "play_song", "parameters": {"query": user_input}}

            except Exception as e:
                logger.error(f"Error in analyze_request: {e}")
                logger.debug(
                    "Defaulting to 'play_song' action with the entire input as query due to exception."
                )
                return {"action": "play_song", "parameters": {"query": user_input}}

    async def handle_command(self, analysis: Dict):
        """Handle the command extracted from the analysis."""
        action = analysis.get("action")
        parameters = analysis.get("parameters", {})
        query = parameters.get("query", "").strip()

        if action == "play_song":
            if not query:
                await self.emit_message("Please specify a song to play.")
                await self.emit_status("error", "No song specified", True)
                return
            
            await self.emit_status(
                "info", f"Searching for '{query}' in local library...", False
            )
            tracks = await self.search_local(query)
            if tracks:
                play_success = await self.play_uris(tracks)
                if play_success:
                    track_names = ", ".join(
                        [f"{t['name']} by {t['artists'][0]}" for t in tracks[:3]]
                    )
                    await self.emit_message(
                        f"Now playing from local library: {track_names}..."
                    )
                    html_code = await self.generate_player_html()
                    html_code_block = (
                        f"""\n ```html \n{html_code}""" if html_code else ""
                    )
                    await self.emit_message(html_code_block)
                    await self.emit_status("success", "Playback started", True)
                else:
                    await self.emit_message("Failed to start playback.")
                    await self.emit_status("error", "Playback failed", True)
                return

            await self.emit_status(
                "info", f"Not found locally. Searching YouTube for '{query}'...", False
            )
            tracks = await self.search_youtube(query)
            if tracks:
                track = tracks[0]
                play_success = await self.play_uris([track])
                if play_success:
                    await self.emit_message(
                        f"Now playing '{track['name']}' by {track['artists'][0]} from YouTube."
                    )
                    html_code = await self.generate_player_html()
                    html_code_block = (
                        f"""\n ```html \n{html_code}""" if html_code else ""
                    )
                    await self.emit_message(html_code_block)
                    await self.emit_status("success", "Playback started", True)
                else:
                    await self.emit_message("Failed to start playback.")
                    await self.emit_status("error", "Playback failed", True)
                return
            else:
                await self.emit_message(
                    f"No matching content found for '{query}'. "
                    "This could be due to no internet connection or YouTube API issues."
                )
                await self.emit_status("error", "No results found", True)
            return

        elif action == "play_playlist":
            if not query:
                await self.emit_message("Please specify a playlist to play.")
                await self.emit_status("error", "No playlist specified", True)
                return

            await self.emit_status(
                "info", f"Searching for playlist '{query}' in local library...", False
            )
            playlists = await self.search_local_playlists(query)
            if playlists:
                best_playlist = await self.select_best_playlist(playlists, query)
                if best_playlist:
                    tracks = await self.get_playlist_tracks(best_playlist["uri"])
                    if tracks:
                        play_success = await self.play_uris(tracks)
                        if play_success:
                            await self.emit_message(
                                f"Now playing playlist '{best_playlist['name']}' from local library."
                            )
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

            await self.emit_status(
                "info", f"No playlists found. Searching for local tracks matching '{query}'...", False
            )
            tracks = await self.search_local(query)
            if tracks:
                play_success = await self.play_uris(tracks)
                if play_success:
                    track_names = ", ".join(
                        [f"{t['name']} by {', '.join(t['artists'])}" for t in tracks[:3]]
                    )
                    await self.emit_message(
                        f"Now playing from local library: {track_names}{'...' if len(tracks) > 3 else ''}"
                    )
                    html_code = await self.generate_player_html()
                    html_code_block = (
                        f"""\n```html\n{html_code}\n```""" if html_code else ""
                    )
                    await self.emit_message(html_code_block)
                    await self.emit_status("success", "Playback started", True)
                else:
                    await self.emit_message("Failed to play tracks.")
                    await self.emit_status("error", "Playback failed", True)
                return

            await self.emit_status(
                "info",
                f"Not found locally. Searching YouTube for playlist '{query}'...",
                False,
            )
            playlists = await self.search_youtube_playlists(query)
            if playlists:
                best_playlist = await self.select_best_playlist(playlists, query)
                if best_playlist:
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
                await self.emit_message(
                    f"No matching playlist found for '{query}'. "
                    "This could be due to no internet connection or YouTube API issues."
                )
                await self.emit_status("error", "No playlist found", True)
            return

        elif action == "show_current_song":
            html_code = await self.generate_player_html()
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
            await self.emit_message(
                "Command not recognized. Attempting to play as a song."
            )
            new_analysis = {
                "action": "play_song",
                "parameters": {"query": query or action},
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    track = result.get("result", {})
                    return track if track else {}
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting current track info after {self.valves.Request_Timeout}s")
            return {}
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    if self.valves.Debug_Logging:
                        logger.debug(f"Response for play: {result}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout resuming playback after {self.valves.Request_Timeout}s")
            return False
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    if self.valves.Debug_Logging:
                        logger.debug(f"Response for pause: {result}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout pausing playback after {self.valves.Request_Timeout}s")
            return False
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
            timeout = aiohttp.ClientTimeout(total=self.valves.Request_Timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.valves.Mopidy_URL, json=payload
                ) as response:
                    result = await response.json()
                    if self.valves.Debug_Logging:
                        logger.debug(f"Response for skip: {result}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout skipping track after {self.valves.Request_Timeout}s")
            return False
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
        self.__user__ = Users.get_user_by_id(__user__["id"])
        self.__model__ = self.valves.Model or __model__
        self.__request__ = __request__
        if self.valves.Debug_Logging:
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
        if self.valves.Debug_Logging:
            logger.debug(f"User input: {user_input}")

        try:
            await self.emit_status("info", "Analyzing your request...", False)
            analysis = await self.analyze_request(user_input)
            if self.valves.Debug_Logging:
                logger.debug(f"Analysis result: {analysis}")
            await self.handle_command(analysis)

        except Exception as e:
            logger.error(f"Error processing music request: {e}")
            await self.emit_message(f"An error occurred: {str(e)}")
            await self.emit_status("error", f"Error: {str(e)}", True)

        return ""
