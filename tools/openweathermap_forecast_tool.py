"""
title: OpenWeatherMap Forecast
description: Tool that fetches weather forecasts using the OpenWeatherMap API and displays an interactive HTML weather widget with current conditions, hourly, and daily forecasts. Supports both the free 2.5 API and the premium One Call 3.0 API.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.1.0
license: MIT
"""

import uuid
from typing import Optional, Dict, Any, Callable, Awaitable, Literal
from datetime import datetime, timezone, timedelta
import aiohttp
from pydantic import BaseModel, Field


def _generate_weather_embed(
    current: Dict[str, Any],
    hourly_items: list,
    daily_items: list,
    location_name: str,
    country: str,
    units: str,
    tz: timezone,
) -> str:
    """Generate a sleek weather forecast HTML embed with glassmorphism design."""

    widget_id = uuid.uuid4().hex[:8]

    # Unit labels
    temp_unit = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
    wind_unit = "m/s" if units != "imperial" else "mph"

    # Current weather
    current_temp = round(current.get("temp", 0))
    current_feels = round(current.get("feels_like", 0))
    current_humidity = current.get("humidity", 0)
    current_pressure = current.get("pressure", 0)
    current_wind = current.get("wind_speed", 0)
    current_wind_deg = current.get("wind_deg", 0)
    current_uvi = current.get("uvi", "N/A")
    current_clouds = current.get("clouds", 0)
    current_visibility = current.get("visibility", 10000)
    current_weather = current.get("weather", [{}])[0] if current.get("weather") else {}
    current_desc = current_weather.get("description", "N/A").title()
    current_icon = current_weather.get("icon", "01d")
    current_main = current_weather.get("main", "Clear")

    # Sunrise/sunset
    sunrise_ts = current.get("sunrise")
    sunset_ts = current.get("sunset")
    sunrise_str = (
        datetime.fromtimestamp(sunrise_ts, tz=tz).strftime("%H:%M")
        if sunrise_ts
        else "--:--"
    )
    sunset_str = (
        datetime.fromtimestamp(sunset_ts, tz=tz).strftime("%H:%M")
        if sunset_ts
        else "--:--"
    )

    # Current local time
    current_dt = datetime.fromtimestamp(current.get("dt", 0), tz=tz)
    date_str = current_dt.strftime("%A, %B %d")
    time_str = current_dt.strftime("%H:%M")

    # Hourly forecast HTML
    hourly_html_temp = ""
    hourly_html_precip = ""
    hourly_html_wind = ""
    for h in hourly_items:
        h_time = h["time"]
        h_temp = h["temp"]
        h_icon = h["icon"]
        h_pop = h["pop"]
        h_wind = h["wind"]
        h_wind_deg = h.get("wind_deg", 0)

        hourly_html_temp += f"""
        <div style="display:flex;flex-direction:column;align-items:center;min-width:60px;flex:1;gap:4px;padding:6px 2px;">
            <span style="font-size:10px;color:#888;font-weight:500;">{h_time}</span>
            <img src="https://openweathermap.org/img/wn/{h_icon}.png" style="width:32px;height:32px;filter:brightness(1.2);" alt="">
            <span style="font-size:11px;color:#999;">{h_pop}%</span>
            <span style="font-size:14px;color:#f0f0f0;font-weight:600;">{h_temp}{temp_unit}</span>
        </div>"""

        hourly_html_precip += f"""
        <div style="display:flex;flex-direction:column;align-items:center;min-width:60px;flex:1;gap:4px;padding:6px 2px;">
            <span style="font-size:10px;color:#888;font-weight:500;">{h_time}</span>
            <img src="https://openweathermap.org/img/wn/{h_icon}.png" style="width:32px;height:32px;filter:brightness(1.2);" alt="">
            <span style="font-size:14px;color:#6cb4ee;font-weight:600;">{h_pop}%</span>
            <span style="font-size:11px;color:#999;">{h_temp}{temp_unit}</span>
        </div>"""

        hourly_html_wind += f"""
        <div style="display:flex;flex-direction:column;align-items:center;min-width:60px;flex:1;gap:4px;padding:6px 2px;">
            <span style="font-size:10px;color:#888;font-weight:500;">{h_time}</span>
            <svg viewBox="0 0 24 24" style="width:20px;height:20px;fill:#aaa;transform:rotate({h_wind_deg}deg);"><path d="M12 2L4.5 20.3l.7.7L12 18l6.8 3 .7-.7z"/></svg>
            <span style="font-size:14px;color:#f0f0f0;font-weight:600;">{h_wind}</span>
            <span style="font-size:10px;color:#888;">{wind_unit}</span>
        </div>"""

    # Daily forecast HTML — grid cards for wide, list for narrow
    daily_cards_html = ""
    daily_list_html = ""
    for d in daily_items:
        daily_cards_html += f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;display:flex;flex-direction:column;align-items:center;gap:4px;min-width:0;">
            <div style="font-size:13px;color:#f0f0f0;font-weight:600;">{d["day"]}</div>
            <div style="font-size:10px;color:#666;">{d["date"]}</div>
            <img src="https://openweathermap.org/img/wn/{d["icon"]}.png" style="width:40px;height:40px;filter:brightness(1.2);" alt="">
            <div style="font-size:11px;color:#999;text-transform:capitalize;text-align:center;line-height:1.3;">{d["desc"]}</div>
            <div style="margin-top:auto;padding-top:4px;">
                <span style="font-size:16px;color:#f0f0f0;font-weight:600;">{d["high"]}°</span>
                <span style="font-size:12px;color:#666;margin-left:2px;">{d["low"]}°</span>
            </div>
            <div style="display:flex;gap:6px;">
                <span style="font-size:10px;color:#6cb4ee;">💧{d["pop"]}%</span>
                <span style="font-size:10px;color:#888;">💦{d["humidity"]}%</span>
            </div>
        </div>"""

        daily_list_html += f"""
        <div style="display:flex;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);gap:8px;">
            <div style="min-width:50px;">
                <div style="font-size:13px;color:#f0f0f0;font-weight:600;">{d["day"]}</div>
                <div style="font-size:10px;color:#666;">{d["date"]}</div>
            </div>
            <img src="https://openweathermap.org/img/wn/{d["icon"]}.png" style="width:32px;height:32px;filter:brightness(1.2);" alt="">
            <div style="flex:1;font-size:11px;color:#999;text-transform:capitalize;">{d["desc"]}</div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="font-size:10px;color:#6cb4ee;">💧{d["pop"]}%</span>
                <span style="font-size:10px;color:#888;">💦{d["humidity"]}%</span>
            </div>
            <div style="min-width:65px;text-align:right;">
                <span style="font-size:14px;color:#f0f0f0;font-weight:600;">{d["high"]}°</span>
                <span style="font-size:12px;color:#666;margin-left:4px;">{d["low"]}°</span>
            </div>
        </div>"""

    # Wind direction
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    wind_dir_idx = round(current_wind_deg / 22.5) % 16
    wind_dir_str = directions[wind_dir_idx]
    vis_km = round(current_visibility / 1000, 1)

    safe_location = (
        location_name.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )
    safe_country = (
        country.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )

    # UVI section (only shown if data available)
    uvi_html = ""
    if current_uvi != "N/A":
        uvi_html = f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">UV Index</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{current_uvi}</div>
            </div>"""
    else:
        uvi_html = f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">Clouds</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{current_clouds}%</div>
            </div>"""

    # Hourly section (only if data available)
    hourly_section = ""
    if hourly_items:
        hourly_section = f"""
        <div style="margin-bottom:16px;">
            <div style="display:flex;gap:8px;margin-bottom:12px;">
                <button id="tabTemp_{widget_id}" class="wtab" style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.15);color:#fff;padding:6px 14px;border-radius:20px;font-size:11px;cursor:pointer;transition:all 0.2s;font-family:inherit;">Temperature</button>
                <button id="tabPrecip_{widget_id}" class="wtab" style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);color:#888;padding:6px 14px;border-radius:20px;font-size:11px;cursor:pointer;transition:all 0.2s;font-family:inherit;">Precipitation</button>
                <button id="tabWind_{widget_id}" class="wtab" style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);color:#888;padding:6px 14px;border-radius:20px;font-size:11px;cursor:pointer;transition:all 0.2s;font-family:inherit;">Wind</button>
            </div>
            <div id="hourlyTemp_{widget_id}" class="hourly-strip" style="display:flex;gap:4px;overflow-x:auto;padding-bottom:6px;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.2) rgba(255,255,255,0.05);">
                {hourly_html_temp}
            </div>
            <div id="hourlyPrecip_{widget_id}" class="hourly-strip" style="display:none;gap:4px;overflow-x:auto;padding-bottom:6px;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.2) rgba(255,255,255,0.05);">
                {hourly_html_precip}
            </div>
            <div id="hourlyWind_{widget_id}" class="hourly-strip" style="display:none;gap:4px;overflow-x:auto;padding-bottom:6px;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.2) rgba(255,255,255,0.05);">
                {hourly_html_wind}
            </div>
        </div>"""

    # Daily section — two views, toggled by container width
    daily_section = ""
    if daily_items:
        daily_section = f"""
        <div style="border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">
            <div style="font-size:8px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Forecast</div>
            <div class="daily-grid-{widget_id}" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;">
                {daily_cards_html}
            </div>
            <div class="daily-list-{widget_id}" style="display:none;">
                {daily_list_html}
            </div>
        </div>"""

    html = f"""
    <div style="display: flex; justify-content: center; width: 100%;">
        <div id="weather_{widget_id}" style="background: rgba(20, 20, 25, 0.4); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); max-width: 800px; width: 100%; margin-bottom: 20px; font-family: system-ui, -apple-system, sans-serif; box-sizing: border-box;">
        <style>
            #weather_{widget_id} .hourly-strip::-webkit-scrollbar {{ height: 4px; }}
            #weather_{widget_id} .hourly-strip::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.05); border-radius: 2px; }}
            #weather_{widget_id} .hourly-strip::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.2); border-radius: 2px; }}
            #weather_{widget_id} .hourly-strip::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.3); }}
        </style>

        <!-- Header -->
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px;">
            <div>
                <div style="font-size:18px;font-weight:600;color:#f0f0f0;letter-spacing:-0.2px;">{safe_location}</div>
                <div style="font-size:10px;color:#888;font-weight:500;text-transform:uppercase;letter-spacing:1px;margin-top:2px;">{safe_country} · {date_str}</div>
            </div>
            <div style="font-size:12px;color:#666;text-align:right;">
                <div style="font-size:20px;color:#aaa;font-weight:300;">{time_str}</div>
            </div>
        </div>

        <!-- Current Weather -->
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
            <img src="https://openweathermap.org/img/wn/{current_icon}@2x.png" style="width:72px;height:72px;filter:brightness(1.2);" alt="{current_main}">
            <div>
                <div style="font-size:42px;font-weight:300;color:#f0f0f0;line-height:1;letter-spacing:-2px;">{current_temp}<span style="font-size:20px;color:#888;font-weight:400;">{temp_unit}</span></div>
                <div style="font-size:14px;color:#ccc;margin-top:2px;text-transform:capitalize;">{current_desc}</div>
            </div>
        </div>

        <!-- Current Details Grid -->
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-bottom:20px;">
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">Feels Like</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{current_feels}{temp_unit}</div>
            </div>
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">Humidity</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{current_humidity}%</div>
            </div>
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">Wind</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{current_wind} <span style="font-size:10px;color:#888;">{wind_unit}</span></div>
                <div style="font-size:9px;color:#666;margin-top:2px;">{wind_dir_str}</div>
            </div>
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">Pressure</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{current_pressure} <span style="font-size:10px;color:#888;">hPa</span></div>
            </div>
            {uvi_html}
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:700;">Visibility</div>
                <div style="font-size:16px;color:#f0f0f0;font-weight:500;">{vis_km} <span style="font-size:10px;color:#888;">km</span></div>
            </div>
        </div>

        <!-- Sunrise/Sunset -->
        <div style="display:flex;gap:12px;margin-bottom:20px;">
            <div style="flex:1;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;display:flex;align-items:center;gap:8px;">
                <svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:#f59e0b;flex-shrink:0;"><path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0-3a1 1 0 0 0 1-1V1a1 1 0 0 0-2 0v2a1 1 0 0 0 1 1zm0 18a1 1 0 0 0-1 1v2a1 1 0 0 0 2 0v-2a1 1 0 0 0-1-1zM5.64 5.64a1 1 0 0 0 0-1.41l-1.42-1.42a1 1 0 1 0-1.41 1.42l1.42 1.41a1 1 0 0 0 1.41 0zM19.78 18.36a1 1 0 1 0-1.41 1.42l1.42 1.41a1 1 0 0 0 1.41-1.41l-1.42-1.42zM4 12a1 1 0 0 0-1-1H1a1 1 0 0 0 0 2h2a1 1 0 0 0 1-1zm19-1h-2a1 1 0 0 0 0 2h2a1 1 0 0 0 0-2zM5.64 18.36l-1.42 1.42a1 1 0 0 0 1.41 1.41l1.42-1.41a1 1 0 0 0-1.41-1.42zM19.78 5.64a1 1 0 0 0 .7-.29l1.42-1.42a1 1 0 1 0-1.41-1.41l-1.42 1.41a1 1 0 0 0 .71 1.71z"/></svg>
                <div>
                    <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;font-weight:700;">Sunrise</div>
                    <div style="font-size:14px;color:#f0f0f0;font-weight:500;">{sunrise_str}</div>
                </div>
            </div>
            <div style="flex:1;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px;display:flex;align-items:center;gap:8px;">
                <svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:#8b5cf6;flex-shrink:0;"><path d="M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0 1a1 1 0 0 0-1 1v2a1 1 0 0 0 2 0v-2a1 1 0 0 0-1-1zm7.78-.64l-1.42-1.42a1 1 0 1 0-1.41 1.42l1.42 1.41a1 1 0 0 0 1.41-1.41zM20 12a1 1 0 0 0 1 1h2a1 1 0 0 0 0-2h-2a1 1 0 0 0-1 1zM5.64 5.64a1 1 0 0 0 0-1.41L4.22 2.81a1 1 0 1 0-1.41 1.42l1.42 1.41a1 1 0 0 0 1.41 0zM4 12a1 1 0 0 0-1-1H1a1 1 0 0 0 0 2h2a1 1 0 0 0 1-1zm1.64 6.36l-1.42 1.42a1 1 0 0 0 1.41 1.41l1.42-1.41a1 1 0 0 0-1.41-1.42zM12 7a1 1 0 0 0 1-1V4a1 1 0 0 0-2 0v2a1 1 0 0 0 1 1zm7.78-4.19l-1.42 1.41a1 1 0 1 0 1.41 1.42l1.42-1.42a1 1 0 0 0-1.41-1.41z"/></svg>
                <div>
                    <div style="font-size:8px;color:#555;text-transform:uppercase;letter-spacing:0.5px;font-weight:700;">Sunset</div>
                    <div style="font-size:14px;color:#f0f0f0;font-weight:500;">{sunset_str}</div>
                </div>
            </div>
        </div>

        {hourly_section}
        {daily_section}

        <!-- Footer -->
        <div style="margin-top:12px;text-align:center;">
            <span style="font-size:9px;color:#444;">OpenWeatherMap</span>
        </div>

        </div>
    </div>

    <script>
    (function() {{
        const wid = '{widget_id}';

        // --- Hourly tab switching ---
        const tabTemp = document.getElementById('tabTemp_' + wid);
        const tabPrecip = document.getElementById('tabPrecip_' + wid);
        const tabWind = document.getElementById('tabWind_' + wid);
        if (tabTemp) {{
            const tabs = [
                {{ btn: tabTemp, panel: document.getElementById('hourlyTemp_' + wid) }},
                {{ btn: tabPrecip, panel: document.getElementById('hourlyPrecip_' + wid) }},
                {{ btn: tabWind, panel: document.getElementById('hourlyWind_' + wid) }},
            ];
            tabs.forEach((tab) => {{
                if (!tab.btn || !tab.panel) return;
                tab.btn.addEventListener('click', () => {{
                    tabs.forEach(t => {{
                        t.btn.style.background = 'rgba(255,255,255,0.04)';
                        t.btn.style.borderColor = 'rgba(255,255,255,0.08)';
                        t.btn.style.color = '#888';
                        t.panel.style.display = 'none';
                    }});
                    tab.btn.style.background = 'rgba(255,255,255,0.12)';
                    tab.btn.style.borderColor = 'rgba(255,255,255,0.15)';
                    tab.btn.style.color = '#fff';
                    tab.panel.style.display = 'flex';
                }});
            }});
        }}

        // --- Responsive daily layout: grid vs list ---
        const container = document.getElementById('weather_' + wid);
        const grid = container ? container.querySelector('.daily-grid-' + wid) : null;
        const list = container ? container.querySelector('.daily-list-' + wid) : null;
        if (grid && list) {{
            function updateDailyLayout() {{
                const w = container.offsetWidth;
                if (w < 420) {{
                    grid.style.display = 'none';
                    list.style.display = 'block';
                }} else {{
                    grid.style.display = 'grid';
                    list.style.display = 'none';
                }}
            }}
            updateDailyLayout();
            const ro = new ResizeObserver(updateDailyLayout);
            ro.observe(container);
        }}
    }})();
    </script>
    """
    return html


async def _fetch_weather_v25(
    location: str, api_key: str, units: str, lang: str
) -> Dict[str, Any]:
    """Fetch weather using the free 2.5 API (current + 5-day/3h forecast)."""

    # Fetch current weather
    current_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&units={units}&lang={lang}&appid={api_key}"
    )
    # Fetch 5-day/3-hour forecast
    forecast_url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?q={location}&units={units}&lang={lang}&cnt=40&appid={api_key}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(current_url) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise Exception(f"Weather API error (HTTP {resp.status}): {err}")
            current_data = await resp.json()

        async with session.get(forecast_url) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise Exception(f"Forecast API error (HTTP {resp.status}): {err}")
            forecast_data = await resp.json()

    # Extract location info
    location_name = current_data.get("name", location)
    country = current_data.get("sys", {}).get("country", "")
    tz_offset = current_data.get("timezone", 0)
    tz = timezone(timedelta(seconds=tz_offset))

    # Normalize current weather to a common format
    main = current_data.get("main", {})
    wind = current_data.get("wind", {})
    current = {
        "dt": current_data.get("dt", 0),
        "temp": main.get("temp", 0),
        "feels_like": main.get("feels_like", 0),
        "humidity": main.get("humidity", 0),
        "pressure": main.get("pressure", 0),
        "wind_speed": wind.get("speed", 0),
        "wind_deg": wind.get("deg", 0),
        "clouds": current_data.get("clouds", {}).get("all", 0),
        "visibility": current_data.get("visibility", 10000),
        "weather": current_data.get("weather", []),
        "sunrise": current_data.get("sys", {}).get("sunrise"),
        "sunset": current_data.get("sys", {}).get("sunset"),
        "uvi": "N/A",  # Not available in 2.5
    }

    # Process hourly forecast (3-hour intervals, take first 8)
    forecast_list = forecast_data.get("list", [])
    hourly_items = []
    for item in forecast_list[:8]:
        item_dt = datetime.fromtimestamp(item.get("dt", 0), tz=tz)
        item_weather = item.get("weather", [{}])[0] if item.get("weather") else {}
        hourly_items.append(
            {
                "time": item_dt.strftime("%H:%M"),
                "temp": round(item.get("main", {}).get("temp", 0)),
                "icon": item_weather.get("icon", "01d"),
                "pop": round(item.get("pop", 0) * 100),
                "wind": item.get("wind", {}).get("speed", 0),
                "wind_deg": item.get("wind", {}).get("deg", 0),
                "desc": item_weather.get("description", ""),
            }
        )

    # Process daily forecast — aggregate 3-hour entries by date
    daily_map: Dict[str, Dict[str, Any]] = {}
    for item in forecast_list:
        item_dt = datetime.fromtimestamp(item.get("dt", 0), tz=tz)
        day_key = item_dt.strftime("%Y-%m-%d")

        # Skip today
        today_key = datetime.now(tz=tz).strftime("%Y-%m-%d")
        if day_key == today_key:
            continue

        temp = item.get("main", {}).get("temp", 0)
        item_weather = item.get("weather", [{}])[0] if item.get("weather") else {}

        if day_key not in daily_map:
            daily_map[day_key] = {
                "dt": item_dt,
                "temps": [],
                "pops": [],
                "humidities": [],
                "icon": item_weather.get("icon", "01d"),
                "desc": item_weather.get("description", ""),
            }

        daily_map[day_key]["temps"].append(temp)
        daily_map[day_key]["pops"].append(item.get("pop", 0))
        daily_map[day_key]["humidities"].append(item.get("main", {}).get("humidity", 0))
        # Use midday icons if available (12:00-15:00 range)
        if 12 <= item_dt.hour <= 15:
            daily_map[day_key]["icon"] = item_weather.get("icon", "01d")
            daily_map[day_key]["desc"] = item_weather.get("description", "")

    daily_items = []
    for day_key in sorted(daily_map.keys())[:5]:
        d = daily_map[day_key]
        daily_items.append(
            {
                "day": d["dt"].strftime("%a"),
                "date": d["dt"].strftime("%d/%m"),
                "high": round(max(d["temps"])),
                "low": round(min(d["temps"])),
                "icon": d["icon"],
                "desc": d["desc"],
                "pop": round(max(d["pops"]) * 100),
                "humidity": round(sum(d["humidities"]) / len(d["humidities"])),
            }
        )

    return {
        "location_name": location_name,
        "country": country,
        "tz": tz,
        "current": current,
        "hourly_items": hourly_items,
        "daily_items": daily_items,
    }


async def _fetch_weather_v30(
    location: str, api_key: str, units: str, lang: str
) -> Dict[str, Any]:
    """Fetch weather using the One Call 3.0 API."""

    # Geocode
    geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(geo_url) as resp:
            if resp.status != 200:
                raise Exception(f"Geocoding error: HTTP {resp.status}")
            geo_data = await resp.json()

    if not geo_data:
        raise Exception(
            f"Could not find location: '{location}'. Try a more specific name."
        )

    lat = geo_data[0].get("lat", 0)
    lon = geo_data[0].get("lon", 0)
    location_name = geo_data[0].get("name", location)
    country = geo_data[0].get("country", "")
    state = geo_data[0].get("state", "")
    if state:
        location_name = f"{location_name}, {state}"

    # One Call 3.0
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall"
        f"?lat={lat}&lon={lon}&units={units}&lang={lang}"
        f"&exclude=minutely&appid={api_key}"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise Exception(f"One Call API error (HTTP {resp.status}): {err}")
            data = await resp.json()

    tz_offset = data.get("timezone_offset", 0)
    tz = timezone(timedelta(seconds=tz_offset))

    current = data.get("current", {})

    # Process hourly
    hourly = data.get("hourly", [])[:8]
    hourly_items = []
    for h in hourly:
        h_dt = datetime.fromtimestamp(h.get("dt", 0), tz=tz)
        h_weather = h.get("weather", [{}])[0] if h.get("weather") else {}
        hourly_items.append(
            {
                "time": h_dt.strftime("%H:%M"),
                "temp": round(h.get("temp", 0)),
                "icon": h_weather.get("icon", "01d"),
                "pop": round(h.get("pop", 0) * 100),
                "wind": h.get("wind_speed", 0),
                "wind_deg": h.get("wind_deg", 0),
                "desc": h_weather.get("description", ""),
            }
        )

    # Process daily (skip today)
    daily = data.get("daily", [])[1:6]
    daily_items = []
    for d in daily:
        d_dt = datetime.fromtimestamp(d.get("dt", 0), tz=tz)
        d_weather = d.get("weather", [{}])[0] if d.get("weather") else {}
        daily_items.append(
            {
                "day": d_dt.strftime("%a"),
                "date": d_dt.strftime("%d/%m"),
                "high": round(d.get("temp", {}).get("max", 0)),
                "low": round(d.get("temp", {}).get("min", 0)),
                "icon": d_weather.get("icon", "01d"),
                "desc": d_weather.get("description", ""),
                "pop": round(d.get("pop", 0) * 100),
                "humidity": d.get("humidity", 0),
            }
        )

    return {
        "location_name": location_name,
        "country": country,
        "tz": tz,
        "current": current,
        "hourly_items": hourly_items,
        "daily_items": daily_items,
    }


class Tools:
    class Valves(BaseModel):
        openweathermap_api_key: str = Field(
            default="",
            description="Your OpenWeatherMap API key.",
            json_schema_extra={"input": {"type": "password"}},
        )
        api_version: Literal["2.5", "3.0"] = Field(
            default="2.5",
            description="API version: '2.5' (free, includes current + 5-day/3h forecast) or '3.0' (One Call API, requires separate subscription).",
        )
        units: Literal["metric", "imperial", "standard"] = Field(
            default="metric",
            description="Units of measurement: 'metric' (°C, m/s), 'imperial' (°F, mph), or 'standard' (K, m/s).",
        )
        language: str = Field(
            default="en",
            description="Language code for weather descriptions (e.g. 'en', 'es', 'fr', 'de', 'pt_br', 'ja', 'zh_cn').",
        )
        show_weather_embed: bool = Field(
            default=True,
            description="Show the embedded weather widget. If false, only returns text data.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def get_weather_forecast(
        self,
        location: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """
        Get the current weather and forecast for a given location.

        Fetches current conditions, hourly forecast, and multi-day daily forecast
        from the OpenWeatherMap API. Displays an interactive weather widget
        and returns a text summary.

        :param location: City name, optionally with country code (e.g. "London", "Tokyo, JP", "New York, US").
        """
        if not self.valves.openweathermap_api_key:
            return "Error: OpenWeatherMap API key is not configured. Please set it in the tool's Valves settings."

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Fetching weather for {location}...",
                        "done": False,
                    },
                }
            )

        try:
            api_key = self.valves.openweathermap_api_key
            units = self.valves.units
            lang = self.valves.language
            if self.valves.api_version == "3.0":
                result = await _fetch_weather_v30(location, api_key, units, lang)
            else:
                result = await _fetch_weather_v25(location, api_key, units, lang)
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )
            return f"Error: {str(e)}"

        location_name = result["location_name"]
        country = result["country"]
        tz = result["tz"]
        current = result["current"]
        hourly_items = result["hourly_items"]
        daily_items = result["daily_items"]

        # Build text summary for the LLM
        temp_unit = (
            "°C"
            if self.valves.units == "metric"
            else ("°F" if self.valves.units == "imperial" else "K")
        )
        wind_unit = "m/s" if self.valves.units != "imperial" else "mph"

        current_temp = round(current.get("temp", 0))
        current_feels = round(current.get("feels_like", 0))
        current_humidity = current.get("humidity", 0)
        current_wind = current.get("wind_speed", 0)
        current_weather = (
            current.get("weather", [{}])[0] if current.get("weather") else {}
        )
        current_desc = current_weather.get("description", "N/A")

        daily_summary_parts = []
        for d in daily_items:
            daily_summary_parts.append(
                f"  - {d['day']} {d['date']}: {d['desc']}, "
                f"High {d['high']}{temp_unit}, Low {d['low']}{temp_unit}, "
                f"Precip {d['pop']}%"
            )
        daily_summary = "\n".join(daily_summary_parts)

        text_summary = (
            f"Weather for {location_name}, {country}:\n"
            f"Currently: {current_desc}, {current_temp}{temp_unit} "
            f"(feels like {current_feels}{temp_unit})\n"
            f"Humidity: {current_humidity}% | Wind: {current_wind} {wind_unit}\n"
            f"\nForecast:\n{daily_summary}"
        )

        # Emit the weather widget
        if self.valves.show_weather_embed and __event_emitter__:
            embed_html = _generate_weather_embed(
                current,
                hourly_items,
                daily_items,
                location_name,
                country,
                self.valves.units,
                tz,
            )
            await __event_emitter__(
                {
                    "type": "embeds",
                    "data": {"embeds": [embed_html]},
                }
            )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Weather data loaded!", "done": True},
                }
            )

        return (
            "The weather widget has been successfully embedded above. "
            "Use the following data to give the user a natural language summary:\n\n"
            + text_summary
        )
