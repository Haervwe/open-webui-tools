"""
title: RPG Tool Set
description: Interactive RPG companion with 3D dice rolling overlays, character creation forms, and rich glassmorphism embeds. Uses CSS 3D transforms for animated dice and event_call for interactive overlays.
author: Haervwe
author_url: https://github.com/Haervwe/open-webui-tools/
funding_url: https://github.com/Haervwe/open-webui-tools
version: 1.0.0
license: MIT
required_open_webui_version: 0.8.11
"""

import random
import re
import json
import logging
from typing import Optional, Dict, Any, Callable, Awaitable, List, Tuple, Union
from pydantic import BaseModel, Field
from open_webui.routers.images import image_generations, GenerateImageForm
from open_webui.models.users import Users
from fastapi import Request
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rpg_tool_set")

# ---------------------------------------------------------------------------
# Dice notation parser
# ---------------------------------------------------------------------------


def parse_dice_notation(notation: str) -> List[Dict[str, Any]]:
    """Parse dice notation like '2d20+5', '4d6-2', '1d100+1d4+3'."""
    notation = notation.strip().lower().replace(" ", "")
    # Match patterns like 2d20, 1d6, d4, +5, -3
    tokens = re.findall(r"[+-]?(?:\d*d\d+|\d+)", notation)
    if not tokens:
        return [{"count": 1, "sides": 20, "modifier": 0}]

    dice_parts = []
    modifier = 0
    for tok in tokens:
        sign = 1
        t = tok
        if t.startswith("+"):
            t = t[1:]
        elif t.startswith("-"):
            sign = -1
            t = t[1:]

        if "d" in t:
            parts = t.split("d")
            count = int(parts[0]) if parts[0] else 1
            sides = int(parts[1])
            dice_parts.append({"count": count * sign, "sides": sides})
        else:
            modifier += int(t) * sign

    if not dice_parts:
        dice_parts = [{"count": 1, "sides": 20}]

    return [{"dice": dice_parts, "modifier": modifier}]


def roll_dice_server(notation: str) -> Dict[str, Any]:
    """Roll dice server-side and return results."""
    parsed = parse_dice_notation(notation)
    group = parsed[0]

    all_rolls = []
    for dp in group["dice"]:
        count = abs(dp["count"])
        sides = dp["sides"]
        rolls = [random.randint(1, sides) for _ in range(count)]
        sign = 1 if dp["count"] > 0 else -1
        all_rolls.append(
            {
                "count": count,
                "sides": sides,
                "rolls": rolls,
                "subtotal": sum(rolls) * sign,
                "sign": sign,
            }
        )

    modifier = group["modifier"]
    total = sum(r["subtotal"] for r in all_rolls) + modifier

    return {
        "notation": notation,
        "groups": all_rolls,
        "modifier": modifier,
        "total": total,
    }


def format_roll_text(result: Dict[str, Any]) -> str:
    """Format roll result as text for the LLM."""
    parts = []
    for g in result["groups"]:
        prefix = "" if g["sign"] > 0 else "-"
        rolls_str = ", ".join(str(r) for r in g["rolls"])
        parts.append(f"{prefix}{g['count']}d{g['sides']}: [{rolls_str}]")

    mod = result["modifier"]
    mod_str = f" + {mod}" if mod > 0 else (f" - {abs(mod)}" if mod < 0 else "")

    return f"Rolled {result['notation']}: {' + '.join(parts)}{mod_str} = **{result['total']}**"


# ---------------------------------------------------------------------------
# HSL color helpers (from audio tool pattern)
# ---------------------------------------------------------------------------


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    r, g, b = int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255)
    return f"#{r:02X}{g:02X}{b:02X}"


def _rpg_palette(seed_val: int) -> list:
    rng = random.Random(seed_val)
    start_hue = rng.randint(0, 359)
    colors = []
    for i in range(5):
        hue = (start_hue + i * 72) % 360
        sat = rng.uniform(0.75, 0.95)
        light = rng.uniform(0.40, 0.55)
        colors.append(_hsl_to_hex(hue, sat, light))
    return colors


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Character creation helpers
# ---------------------------------------------------------------------------

RPG_RACES = [
    "Human",
    "Elf",
    "Dwarf",
    "Halfling",
    "Half-Orc",
    "Gnome",
    "Tiefling",
    "Dragonborn",
    "Half-Elf",
]
RPG_CLASSES = [
    "Fighter",
    "Wizard",
    "Rogue",
    "Cleric",
    "Ranger",
    "Paladin",
    "Barbarian",
    "Bard",
    "Druid",
    "Sorcerer",
    "Warlock",
    "Monk",
]
CLASS_HIT_DIE = {
    "Fighter": 10,
    "Wizard": 6,
    "Rogue": 8,
    "Cleric": 8,
    "Ranger": 10,
    "Paladin": 10,
    "Barbarian": 12,
    "Bard": 8,
    "Druid": 8,
    "Sorcerer": 6,
    "Warlock": 8,
    "Monk": 8,
}
ABILITY_NAMES = ["STR", "DEX", "CON", "INT", "WIS", "CHA"]

PORTRAIT_SVGS = {
    "warrior": '<path d="M12 2C9 2 7 4 7 7c0 2 1 3.5 2.5 4.5L8 13l-3 8h14l-3-8-1.5-1.5C16 10.5 17 9 17 7c0-3-2-5-5-5z" fill="currentColor"/>',
    "mage": '<path d="M12 1l-2 6H6l4 3-1.5 5L12 12l3.5 3L14 10l4-3h-4L12 1zM8 17l-2 6h12l-2-6H8z" fill="currentColor"/>',
    "rogue": '<path d="M12 2c-2 0-4 1.5-4 4 0 1.5.8 3 2 3.8L9 11l-1 2-3 8h14l-3-8-1-2-1-1.2c1.2-.8 2-2.3 2-3.8 0-2.5-2-4-4-4zM10 7l4 0" fill="currentColor"/>',
    "cleric": '<path d="M12 2l-1 4H8v2h3l-1 4-5 9h14l-5-9-1-4h3V6h-3L12 2zm-1 14a1 1 0 102 0 1 1 0 00-2 0z" fill="currentColor"/>',
}


def calc_modifier(score: int) -> int:
    return (score - 10) // 2


def format_modifier(mod: int) -> str:
    return f"+{mod}" if mod >= 0 else str(mod)


# ---------------------------------------------------------------------------
# Roll result embed generator
# ---------------------------------------------------------------------------


def get_die_geometry(sides: int) -> Dict[str, Any]:
    """Helper to return CSS geometry configurations for polyhedral dice."""
    if sides == 4:
        clip = "polygon(50% 0%, 0% 86.6%, 100% 86.6%)"
        origin = "50% 57.7%"
        faces = [
            "rotateY(0deg) rotateX(19.5deg) translateZ(14px)",
            "rotateY(120deg) rotateX(19.5deg) translateZ(14px)",
            "rotateY(240deg) rotateX(19.5deg) translateZ(14px)",
            "rotateX(-90deg) translateZ(14px)",
        ]
        return {"clip": clip, "origin": origin, "cx": -19.5, "cy": 0, "faces": faces}

    elif sides == 8:
        clip = "polygon(50% 0%, 0% 86.6%, 100% 86.6%)"
        origin = "50% 57.7%"
        tz = 28
        faces = []
        for i in range(4):
            faces.append(f"rotateY({i * 90}deg) rotateX(35.3deg) translateZ({tz}px)")
            faces.append(f"rotateY({i * 90}deg) rotateX(-35.3deg) rotateZ(180deg) translateZ({tz}px)")
        return {"clip": clip, "origin": origin, "cx": -35.3, "cy": 0, "faces": faces}

    elif sides == 10 or sides == 100:
        clip = "polygon(50% 0%, 100% 30%, 50% 100%, 0% 30%)"
        origin = "50% 50%"
        tz = 35
        faces = []
        for i in range(5):
            faces.append(f"rotateY({i * 72}deg) rotateX(25deg) translateZ({tz}px)")
            faces.append(f"rotateY({i * 72 + 36}deg) rotateX(-25deg) rotateZ(180deg) translateZ({tz}px)")
        return {"clip": clip, "origin": origin, "cx": -25, "cy": 0, "faces": faces}

    elif sides == 12:
        clip = "polygon(50% 0%, 100% 38%, 81% 100%, 19% 100%, 0% 38%)"
        origin = "50% 50%"
        tz = 46
        faces = ["rotateX(90deg) translateZ(46px)"]
        for i in range(5):
            faces.append(f"rotateY({i * 72}deg) rotateX(26.6deg) translateZ({tz}px)")
            faces.append(f"rotateY({i * 72 + 36}deg) rotateX(-26.6deg) rotateZ(180deg) translateZ({tz}px)")
        faces.append("rotateX(-90deg) translateZ(46px)")
        return {"clip": clip, "origin": origin, "cx": -90, "cy": 0, "faces": faces}

    elif sides == 20:
        clip = "polygon(50% 0%, 0% 86.6%, 100% 86.6%)"
        origin = "50% 57.7%"
        tz = 51
        faces = []
        for i in range(5):
            faces.append(f"rotateY({i * 72}deg) rotateX(52.6deg) translateZ({tz}px)")
            faces.append(f"rotateY({i * 72}deg) rotateX(-52.6deg) rotateZ(180deg) translateZ({tz}px)")
            faces.append(f"rotateY({i * 72 + 36}deg) rotateX(10.8deg) translateZ({tz}px)")
            faces.append(f"rotateY({i * 72 + 36}deg) rotateX(-10.8deg) rotateZ(180deg) translateZ({tz}px)")
        return {"clip": clip, "origin": origin, "cx": -52.6, "cy": 0, "faces": faces}

    faces = [
        "rotateY(0deg) translateZ(34px)",
        "rotateY(180deg) translateZ(34px)",
        "rotateY(90deg) translateZ(34px)",
        "rotateY(-90deg) translateZ(34px)",
        "rotateX(90deg) translateZ(34px)",
        "rotateX(-90deg) translateZ(34px)",
    ]
    return {
        "clip": "none",
        "origin": "50% 50%",
        "cx": 0,
        "cy": 0,
        "faces": faces[:sides] if sides < 6 else faces,
    }

def get_die_silhouette(sides: int) -> str:
    if sides == 4: return "polygon(50% 0%, 0% 100%, 100% 100%)"
    if sides == 8: return "polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)"
    if sides in (10, 100): return "polygon(50% 0%, 100% 40%, 50% 100%, 0% 40%)"
    if sides == 12: return "polygon(50% 0%, 100% 38%, 81% 100%, 19% 100%, 0% 38%)"
    if sides == 20: return "polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)"
    return "none"



def get_die_geometry_dict() -> Dict[int, Any]:
    return {
        4: get_die_geometry(4),
        6: get_die_geometry(6),
        8: get_die_geometry(8),
        10: get_die_geometry(10),
        12: get_die_geometry(12),
        20: get_die_geometry(20),
    }



def generate_roll_embed(
    result: Dict[str, Any], context: str = "", palette_seed: int = None
) -> str:
    if palette_seed is None:
        palette_seed = random.randint(0, 9999999)
    c0, c1, c2, c3, c4 = _rpg_palette(palette_seed)

    pid = f"rpg{random.randint(100000, 999999)}"
    total = result["total"]
    notation = _esc(result["notation"])
    safe_context = _esc(context) if context else ""
    anim_ms = 1800

    # Determine crits
    is_nat20 = any(
        r == 20 and g["sides"] == 20 for g in result["groups"] for r in g["rolls"]
    )
    is_nat1 = any(
        r == 1 and g["sides"] == 20 for g in result["groups"] for r in g["rolls"]
    )

    # Build 2D stylized dice icons HTML
    dice_blocks_html = ""
    die_idx = 0
    for g in result["groups"]:
        prefix = "" if g["sign"] > 0 else "-"
        for r_val in g["rolls"]:
            is_crit = r_val == g["sides"]
            is_fumble = r_val == 1 and g["sides"] >= 4
            delay_ms = die_idx * 100

            face_bg = "#ffd700" if is_crit else ("#c0392b" if is_fumble else c0)
            face_glow = (
                f"box-shadow:0 0 24px rgba(255,215,0,0.7),0 0 48px rgba(255,215,0,0.3);"
                if is_crit
                else (
                    f"box-shadow:0 0 18px rgba(255,68,68,0.5);"
                    if is_fumble
                    else f"box-shadow:0 0 14px {c0}55;"
                )
            )

            sil_clip = get_die_silhouette(g["sides"])
            text_mt = '8px' if g["sides"] == 4 else '0px'
            
            dice_blocks_html += f"""
      <div style="width:52px;height:52px;display:flex;align-items:center;justify-content:center;
        font-size:20px;font-weight:800;color:#fff;background:{face_bg};
        clip-path:{sil_clip};{face_glow};margin-bottom:8px;
        animation:{pid}_pop 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) {delay_ms}ms both;">
        <span style="display:block;margin-top:{text_mt};">{prefix}{r_val}</span>
      </div>"""
            die_idx += 1

    dice_keyframes = f"""
    @keyframes {pid}_pop {{
      0%   {{ transform: scale(0.3); opacity:0; }}
      60%  {{ transform: scale(1.1); opacity:1; }}
      100% {{ transform: scale(1); opacity:1; }}
    }}"""

    # Modifier chip
    mod = result["modifier"]
    mod_html = ""
    if mod != 0:
        mod_sign = "+" if mod > 0 else ""
        mod_html = f"""<div style="display:inline-flex;align-items:center;justify-content:center;
            padding:0 14px;height:44px;border-radius:10px;
            background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);
            font-size:18px;font-weight:700;color:#ccc;
            animation:{pid}_fadeIn 0.4s ease {anim_ms + die_idx * 120 + 100}ms both;">{mod_sign}{mod}</div>"""

    context_html = (
        f"""<div style="font-size:12px;color:#999;margin-top:8px;font-style:italic;">{safe_context}</div>"""
        if safe_context
        else ""
    )

    # Total section styling
    total_glow = ""
    total_label = ""
    total_extra = ""
    if is_nat20:
        total_glow = f"text-shadow:0 0 20px rgba(255,215,0,0.8),0 0 40px rgba(255,215,0,0.4);color:#ffd700;"
        total_label = f'<div style="font-size:10px;font-weight:800;letter-spacing:2.5px;color:#ffd700;text-transform:uppercase;margin-top:4px;animation:{pid}_pulse 0.8s ease infinite alternate;">CRITICAL HIT</div>'
        # Particle burst for crits
        total_extra = "".join(
            f'<div style="position:absolute;width:4px;height:4px;border-radius:50%;background:#ffd700;'
            f'top:50%;left:50%;animation:{pid}_part{i} 1s ease {anim_ms + 200}ms both;"></div>'
            for i in range(12)
        )
        dice_keyframes += "".join(
            f"@keyframes {pid}_part{i} {{ 0%{{transform:translate(0,0) scale(1);opacity:1;}} 100%{{transform:translate({random.randint(-80, 80)}px,{random.randint(-80, 80)}px) scale(0);opacity:0;}} }}"
            for i in range(12)
        )
    elif is_nat1:
        total_glow = "text-shadow:0 0 20px rgba(255,68,68,0.8);color:#ff6b6b;"
        total_label = f'<div style="font-size:10px;font-weight:800;letter-spacing:2.5px;color:#ff6b6b;text-transform:uppercase;margin-top:4px;animation:{pid}_shake 0.3s ease {anim_ms + 200}ms 3;">CRITICAL FAIL</div>'
        dice_keyframes += f"@keyframes {pid}_shake {{ 0%,100%{{transform:translateX(0);}} 25%{{transform:translateX(-6px);}} 75%{{transform:translateX(6px);}} }}"

    total_delay_ms = anim_ms + die_idx * 120 + 300

    html = f"""
<div style="display:flex;justify-content:center;width:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
<div id="{pid}" style="position:relative;overflow:hidden;border-radius:18px;max-width:420px;width:100%;
  box-shadow:0 20px 56px rgba(0,0,0,0.45),0 0 0 1px rgba(255,255,255,0.08);
  color:#fff;box-sizing:border-box;margin-bottom:16px;">

  <div style="position:absolute;inset:0;z-index:0;
    background:linear-gradient(135deg,{c0}18,{c1}18,{c2}18,rgba(15,15,20,0.95));"></div>
  <div style="position:absolute;inset:0;z-index:0;
    background:rgba(15,15,20,0.78);backdrop-filter:blur(14px);"></div>

  <div style="position:relative;z-index:2;padding:22px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <div>
        <div style="font-size:8px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;
          color:{c0};opacity:0.85;">DICE ROLL</div>
        <div style="font-size:17px;font-weight:700;color:#f0f0f0;letter-spacing:-0.3px;margin-top:2px;">{notation}</div>
      </div>
      <div style="width:42px;height:42px;border-radius:11px;background:rgba(255,255,255,0.05);
        border:1px solid rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;">
        <svg viewBox="0 0 24 24" style="width:20px;height:20px;fill:{c0};opacity:0.9;">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM7.5 18A1.5 1.5 0 116 15a1.5 1.5 0 011.5 3zm0-9A1.5 1.5 0 116 6a1.5 1.5 0 011.5 3zM12 13.5A1.5 1.5 0 1110.5 12 1.5 1.5 0 0112 13.5zm4.5 4.5A1.5 1.5 0 1115 16.5a1.5 1.5 0 011.5 1.5zm0-9A1.5 1.5 0 1115 7.5 1.5 1.5 0 0116.5 9z"/>
        </svg>
      </div>
    </div>

    <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:center;
      margin-bottom:18px;padding:16px 12px;border-radius:12px;
      background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.05);min-height:80px;">
      {dice_blocks_html}
      {mod_html}
    </div>

    <div style="text-align:center;padding:14px;border-radius:12px;position:relative;
      background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
      animation:{pid}_fadeIn 0.5s ease {total_delay_ms}ms both;">
      {total_extra}
      <div style="font-size:8px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
        color:#666;margin-bottom:4px;">Total</div>
      <div style="font-size:42px;font-weight:300;letter-spacing:-2px;{total_glow}
        animation:{pid}_scaleIn 0.4s cubic-bezier(.2,1.4,.4,1) {total_delay_ms + 100}ms both;">{total}</div>
      {total_label}
    </div>
    {context_html}
  </div>
</div>
</div>

<style>
  {dice_keyframes}
  @keyframes {pid}_fadeIn {{
    0% {{ opacity:0; transform:translateY(8px); }}
    100% {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes {pid}_scaleIn {{
    0% {{ opacity:0; transform:scale(0.3); }}
    100% {{ opacity:1; transform:scale(1); }}
  }}
  @keyframes {pid}_pulse {{
    0% {{ opacity:0.7; letter-spacing:2.5px; }}
    100% {{ opacity:1; letter-spacing:4px; }}
  }}
  .{pid}_dwrap {{
    display:inline-flex;
    animation:{pid}_fadeIn 0.3s ease both;
  }}
</style>"""
    return html


# ---------------------------------------------------------------------------
# Character card embed generator
# ---------------------------------------------------------------------------


def generate_character_embed(
    char_data: Dict[str, Any], palette_seed: int = None
) -> str:
    if palette_seed is None:
        palette_seed = random.randint(0, 9999999)
    c0, c1, c2, c3, c4 = _rpg_palette(palette_seed)

    name = _esc(char_data.get("name", "Unknown Hero"))
    race = _esc(char_data.get("race", "Human"))
    cls = _esc(char_data.get("class", "Fighter"))
    level = char_data.get("level", 1)
    hp = char_data.get("hp", 10)
    portrait_key = char_data.get("portrait", "warrior")
    portrait_svg = PORTRAIT_SVGS.get(portrait_key, PORTRAIT_SVGS["warrior"])
    image_url = char_data.get("image_url", "")

    if image_url:
        portrait_html = f'<img src="{_esc(image_url)}" style="width:100%;height:100%;object-fit:cover;border-radius:12px;" />'
    else:
        portrait_html = f'<svg viewBox="0 0 24 24" style="width:30px;height:30px;fill:{c0};">{portrait_svg}</svg>'

    abilities = char_data.get("abilities", {})

    stats_html = ""
    for ab in ABILITY_NAMES:
        score = abilities.get(ab, 10)
        mod = calc_modifier(score)
        mod_str = format_modifier(mod)
        stats_html += f"""
        <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.08);border-radius:10px;
          padding:8px 4px;text-align:center;min-width:0;">
          <div style="font-size:8px;font-weight:800;letter-spacing:1px;color:{c0};margin-bottom:4px;">{ab}</div>
          <div style="font-size:22px;font-weight:300;color:#f0f0f0;">{score}</div>
          <div style="font-size:11px;font-weight:600;color:#aaa;margin-top:2px;
            background:rgba(255,255,255,0.06);border-radius:99px;padding:1px 6px;">{mod_str}</div>
        </div>"""

    html = f"""
<div style="display:flex;justify-content:center;width:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
<div style="position:relative;overflow:hidden;border-radius:18px;max-width:420px;width:100%;
  box-shadow:0 20px 56px rgba(0,0,0,0.45),0 0 0 1px rgba(255,255,255,0.08);
  color:#fff;box-sizing:border-box;margin-bottom:16px;">

  <div style="position:absolute;inset:0;z-index:0;
    background:linear-gradient(145deg,{c0}15,{c1}15,{c2}15,rgba(20,20,25,0.95));"></div>
  <div style="position:absolute;inset:0;z-index:0;
    background:rgba(20,20,25,0.8);backdrop-filter:blur(12px);"></div>

  <div style="position:relative;z-index:2;padding:22px;">
    <div style="font-size:8px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;
      color:{c0};opacity:0.8;text-align:center;margin-bottom:12px;">🛡 Character Sheet</div>

    <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
      <div style="width:56px;height:56px;border-radius:14px;background:rgba(255,255,255,0.06);
        border:1px solid rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        {portrait_html}
      </div>
      <div style="min-width:0;">
        <div style="font-size:20px;font-weight:700;color:#f0f0f0;letter-spacing:-0.3px;
          overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</div>
        <div style="font-size:12px;color:#999;">Level {level} {race} {cls}</div>
      </div>
      <div style="margin-left:auto;text-align:center;flex-shrink:0;">
        <div style="font-size:8px;font-weight:800;letter-spacing:1px;color:#e74c3c;">HP</div>
        <div style="font-size:22px;font-weight:600;color:#e74c3c;">{hp}</div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:14px;">
      {stats_html}
    </div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <div style="flex:1;min-width:80px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:8px;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">AC</div>
        <div style="font-size:16px;font-weight:500;color:#f0f0f0;">{10 + calc_modifier(abilities.get("DEX", 10))}</div>
      </div>
      <div style="flex:1;min-width:80px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:8px;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Initiative</div>
        <div style="font-size:16px;font-weight:500;color:#f0f0f0;">{format_modifier(calc_modifier(abilities.get("DEX", 10)))}</div>
      </div>
      <div style="flex:1;min-width:80px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:8px;text-align:center;">
        <div style="font-size:8px;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Prof Bonus</div>
        <div style="font-size:16px;font-weight:500;color:#f0f0f0;">+{max(2, 2 + (level - 1) // 4)}</div>
      </div>
    </div>
  </div>
</div>
</div>"""
    return html


def format_character_text(char_data: Dict[str, Any]) -> str:
    """Format character data as text for the LLM."""
    abilities = char_data.get("abilities", {})
    stats = ", ".join(
        f"{ab}: {abilities.get(ab, 10)} ({format_modifier(calc_modifier(abilities.get(ab, 10)))})"
        for ab in ABILITY_NAMES
    )
    return (
        f"**{char_data.get('name', 'Unknown')}** — Level {char_data.get('level', 1)} "
        f"{char_data.get('race', 'Human')} {char_data.get('class', 'Fighter')}\n"
        f"HP: {char_data.get('hp', 10)} | AC: {10 + calc_modifier(abilities.get('DEX', 10))} | "
        f"Initiative: {format_modifier(calc_modifier(abilities.get('DEX', 10)))}\n"
        f"Abilities: {stats}"
    )


# ---------------------------------------------------------------------------
# 3D Dice Roller JS overlay (called via __event_call__)
# ---------------------------------------------------------------------------


def build_dice_roller_js(notation: str, dice_color: str, anim_speed: float) -> str:
    """Build JavaScript for the 3D dice roller fullscreen overlay."""
    parsed = parse_dice_notation(notation)
    group = parsed[0]
    dice_list = []
    for dp in group["dice"]:
        for _ in range(abs(dp["count"])):
            dice_list.append(
                {"sides": dp["sides"], "sign": 1 if dp["count"] > 0 else -1}
            )
    modifier = group["modifier"]
    dice_json = json.dumps(dice_list)
    anim_ms = int(anim_speed * 1000)
    esc_notation = _esc(notation)

    # Build the JS as a plain string (no f-string) to avoid brace escaping hell
    js = "return (function(){\n"
    js += "return new Promise(function(resolve){\n"
    js += f"var DICE={dice_json};\n"
    js += f"var MODIFIER={modifier};\n"
    js += f"var NOTATION='{esc_notation}';\n"
    js += f"var ANIM_MS={anim_ms};\n"
    js += f"var DICE_COLOR='{dice_color}';\n"
    geo_data_json = json.dumps(get_die_geometry_dict())
    js += f"var GEO_DATA={geo_data_json};\n"
    js += r"""
var overlay=document.createElement('div');
overlay.style.cssText='position:fixed;inset:0;z-index:999999;background:rgba(0,0,0,0.7);backdrop-filter:blur(14px);display:flex;align-items:center;justify-content:center;font-family:system-ui,-apple-system,sans-serif;';

var panel=document.createElement('div');
panel.style.cssText='background:rgba(20,20,25,0.35);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;max-width:500px;width:92vw;max-height:90vh;overflow-y:auto;box-shadow:0 12px 40px rgba(0,0,0,0.4);color:#fff;text-align:center;';
overlay.appendChild(panel);

var hdr=document.createElement('div');
hdr.innerHTML='<div style="font-size:8px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:'+DICE_COLOR+';opacity:0.8;margin-bottom:4px;">Dice Roller</div><div style="font-size:18px;font-weight:700;letter-spacing:-0.3px;">'+NOTATION+'</div>';
hdr.style.marginBottom='20px';
panel.appendChild(hdr);

var diceArea=document.createElement('div');
diceArea.style.cssText='display:flex;flex-wrap:wrap;gap:16px;justify-content:center;margin-bottom:20px;perspective:600px;min-height:100px;align-items:center;';
panel.appendChild(diceArea);

var resultsArea=document.createElement('div');
resultsArea.style.cssText='margin-bottom:20px;';
panel.appendChild(resultsArea);

var btnArea=document.createElement('div');
btnArea.style.cssText='display:flex;justify-content:center;';
panel.appendChild(btnArea);

// Show placeholder dice
DICE.forEach(function(){
  var ph=document.createElement('div');
  ph.style.cssText='width:72px;height:72px;border-radius:12px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;font-size:28px;color:rgba(255,255,255,0.2);';
  ph.textContent='?';
  diceArea.appendChild(ph);
});

function makeBtn(label){
  var b=document.createElement('button');
  b.textContent=label;
  b.style.cssText='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.18);color:#fff;padding:14px 48px;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.2s;font-family:inherit;min-width:160px;box-shadow:0 4px 16px rgba(0,0,0,0.2);';
  b.onmouseenter=function(){b.style.background='rgba(255,255,255,0.2)';};
  b.onmouseleave=function(){b.style.background='rgba(255,255,255,0.12)';};
  return b;
}

var rollBtn=makeBtn('\ud83c\udfb2 Roll');
btnArea.appendChild(rollBtn);

rollBtn.onclick=function(){
  rollBtn.style.display='none';
  diceArea.innerHTML='';
  var results=[];

  DICE.forEach(function(d,idx){
    var val=Math.floor(Math.random()*d.sides)+1;
    results.push({value:val,sides:d.sides,sign:d.sign});

    var wrap=document.createElement('div');
    wrap.style.cssText='width:72px;height:72px;perspective:400px;';

    var cube=document.createElement('div');
    cube.style.cssText='width:72px;height:72px;position:relative;transform-style:preserve-3d;animation:rpgSpin_'+idx+' '+ANIM_MS+'ms cubic-bezier(0.2,0.8,0.3,1) forwards;';

    var geo = GEO_DATA[d.sides] || GEO_DATA[6];
    if (d.sides === 100) geo = GEO_DATA[10];

    geo.faces.forEach(function(tr, fi){
      var isMain = (fi === 0);
      var lbl = isMain ? val : Math.ceil(Math.random()*d.sides);
      var face=document.createElement('div');
      var sz=isMain?'28':'20';
      var wt=isMain?'800':'600';
      var bg=isMain?DICE_COLOR:'rgba(20,20,30,0.85)';
      var bd=isMain?'0.4':'0.25';
      var glow=isMain?'box-shadow:0 0 20px '+DICE_COLOR+'44;':'';
      var mt=geo.clip.startsWith('polygon(50% 0%') ? '12px' : '0px';
      var rt=tr.includes('rotateZ(180deg)') ? 'transform:rotateZ(180deg);' : '';
      face.style.cssText='position:absolute;width:72px;height:72px;display:flex;align-items:center;justify-content:center;font-size:'+sz+'px;font-weight:'+wt+';color:#fff;border-radius:12px;backface-visibility:hidden;background:'+bg+';border:1px solid rgba(255,255,255,'+bd+');'+glow+'clip-path:'+geo.clip+';transform-origin:'+geo.origin+';transform:'+tr+';';
      face.innerHTML='<span style="display:block;margin-top:'+mt+';'+rt+'">'+lbl+'</span>';
      cube.appendChild(face);
    });

    var sx=2+Math.floor(Math.random()*3);
    var sy=2+Math.floor(Math.random()*3);
    var st=document.createElement('style');
    st.textContent='@keyframes rpgSpin_'+idx+'{0%{transform:rotateX(0) rotateY(0) rotateZ(0)}30%{transform:rotateX('+(sx*360+180)+'deg) rotateY('+(sy*180)+'deg) rotateZ(45deg)}70%{transform:rotateX('+(sx*360+90)+'deg) rotateY('+(sy*360-45)+'deg) rotateZ(-20deg)}100%{transform:rotateX('+(720+geo.cx-15)+'deg) rotateY('+(360+geo.cy-20)+'deg) rotateZ(0deg)}}';
    document.head.appendChild(st);

    wrap.appendChild(cube);
    diceArea.appendChild(wrap);
  });

  // Show results + Accept button after animation
  setTimeout(function(){
    var total=MODIFIER;
    var html='<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:12px;">';
    results.forEach(function(r){
      total+=r.value*r.sign;
      var isCrit=r.value===r.sides;
      var isFumble=r.value===1&&r.sides>=4;
      var bc=isCrit?'#ffd700':(isFumble?'#ff4444':'rgba(255,255,255,0.15)');
      var glow=isCrit?'box-shadow:0 0 12px rgba(255,215,0,0.6);':(isFumble?'box-shadow:0 0 12px rgba(255,68,68,0.4);':'');
      var prefix=r.sign<0?'-':'';
      html+='<div style="display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:8px;background:rgba(0,0,0,0.3);border:2px solid '+bc+';font-size:20px;font-weight:700;'+glow+'">'+prefix+r.value+'</div>';
    });
    if(MODIFIER!==0){
      var ms=MODIFIER>0?'+':'';
      html+='<div style="display:inline-flex;align-items:center;justify-content:center;padding:0 12px;height:44px;border-radius:8px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);font-size:16px;font-weight:600;color:#ccc;">'+ms+MODIFIER+'</div>';
    }
    html+='</div>';

    var isNat20=results.some(function(r){return r.value===20&&r.sides===20;});
    var isNat1=results.some(function(r){return r.value===1&&r.sides===20;});
    var tGlow='';
    var tLabel='';
    if(isNat20){tGlow='text-shadow:0 0 20px rgba(255,215,0,0.8);color:#ffd700;';tLabel='<div style="font-size:10px;font-weight:800;letter-spacing:2px;color:#ffd700;margin-top:4px;">CRITICAL HIT</div>';}
    else if(isNat1){tGlow='text-shadow:0 0 20px rgba(255,68,68,0.8);color:#ff6b6b;';tLabel='<div style="font-size:10px;font-weight:800;letter-spacing:2px;color:#ff6b6b;margin-top:4px;">CRITICAL FAIL</div>';}

    html+='<div style="padding:14px;border-radius:10px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);">';
    html+='<div style="font-size:8px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#888;margin-bottom:4px;">Total</div>';
    html+='<div style="font-size:40px;font-weight:300;letter-spacing:-2px;'+tGlow+'">'+total+'</div>';
    html+=tLabel+'</div>';
    resultsArea.innerHTML=html;

    // Show Accept button
    var acceptBtn=makeBtn('\u2714 Accept Roll');
    btnArea.innerHTML='';
    btnArea.appendChild(acceptBtn);
    acceptBtn.onclick=function(){
      if(overlay.parentNode) overlay.parentNode.removeChild(overlay);
      resolve(JSON.stringify({notation:NOTATION,results:results.map(function(r){return{value:r.value,sides:r.sides,sign:r.sign};}),modifier:MODIFIER,total:total}));
    };
  },ANIM_MS+100);
};

document.body.appendChild(overlay);
});
})()
"""
    return js


# ---------------------------------------------------------------------------
# Character Creator JS overlay (called via __event_call__)
# ---------------------------------------------------------------------------


def build_character_creator_js() -> str:
    """Build JavaScript for the character creation fullscreen overlay."""
    races_json = json.dumps(RPG_RACES)
    classes_json = json.dumps(RPG_CLASSES)
    hit_die_json = json.dumps(CLASS_HIT_DIE)
    portraits_json = json.dumps(list(PORTRAIT_SVGS.keys()))

    return f"""
return (function() {{
  return new Promise((resolve) => {{
    const RACES = {races_json};
    const CLASSES = {classes_json};
    const HIT_DIE = {hit_die_json};
    const PORTRAITS = {portraits_json};
    const ABILITIES = ['STR','DEX','CON','INT','WIS','CHA'];

    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;
      background:rgba(0,0,0,0.7);backdrop-filter:blur(14px);
      display:flex;align-items:center;justify-content:center;
      font-family:system-ui,-apple-system,'Segoe UI',sans-serif;`;

    const panel = document.createElement('div');
    panel.style.cssText = `background:rgba(20,20,25,0.35);backdrop-filter:blur(10px);
      border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;
      max-width:480px;width:92vw;max-height:92vh;overflow-y:auto;
      box-shadow:0 12px 40px rgba(0,0,0,0.4);color:#fff;
      scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.2) transparent;`;
    overlay.appendChild(panel);

    const hdr = document.createElement('div');
    hdr.style.cssText = 'text-align:center;margin-bottom:18px;';
    hdr.innerHTML = '<div style="font-size:8px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:#e67e22;opacity:0.8;margin-bottom:4px;">\ud83d\udee1 Character Creator</div><div style="font-size:18px;font-weight:700;">Create Your Hero</div>';
    panel.appendChild(hdr);

    const fieldStyle = 'width:100%;padding:10px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);background:rgba(0,0,0,0.3);color:#fff;font-size:13px;font-family:inherit;outline:none;box-sizing:border-box;';
    const labelStyle = 'font-size:10px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;display:block;';

    function makeField(labelText, inputEl) {{
      const wrap = document.createElement('div');
      wrap.style.marginBottom = '12px';
      const lbl = document.createElement('label');
      lbl.style.cssText = labelStyle;
      lbl.textContent = labelText;
      wrap.appendChild(lbl);
      wrap.appendChild(inputEl);
      return wrap;
    }}

    function makeInput(placeholder) {{
      const inp = document.createElement('input');
      inp.style.cssText = fieldStyle;
      inp.placeholder = placeholder;
      return inp;
    }}

    function makeSelect(options) {{
      const sel = document.createElement('select');
      sel.style.cssText = fieldStyle;
      options.forEach(o => {{
        const opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        sel.appendChild(opt);
      }});
      return sel;
    }}

    const nameInput = makeInput('Enter character name...');
    panel.appendChild(makeField('Name', nameInput));

    const row1 = document.createElement('div');
    row1.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:10px;';
    const raceSelect = makeSelect(RACES);
    const classSelect = makeSelect(CLASSES);
    row1.appendChild(makeField('Race', raceSelect));
    row1.appendChild(makeField('Class', classSelect));
    panel.appendChild(row1);

    const levelInput = makeInput('1');
    levelInput.type = 'number'; levelInput.min = '1'; levelInput.max = '20'; levelInput.value = '1';
    panel.appendChild(makeField('Level', levelInput));

    const abLabel = document.createElement('div');
    abLabel.style.cssText = 'font-size:10px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;';
    abLabel.innerHTML = '<span>Ability Scores</span>';
    const rollAllBtn = document.createElement('button');
    rollAllBtn.textContent = '\ud83c\udfb2 Roll All (4d6 drop lowest)';
    rollAllBtn.style.cssText = 'background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);color:#ccc;padding:4px 10px;border-radius:6px;font-size:10px;cursor:pointer;font-family:inherit;transition:background 0.2s;';
    rollAllBtn.onmouseenter = () => rollAllBtn.style.background = 'rgba(255,255,255,0.15)';
    rollAllBtn.onmouseleave = () => rollAllBtn.style.background = 'rgba(255,255,255,0.08)';
    abLabel.appendChild(rollAllBtn);
    panel.appendChild(abLabel);

    const abGrid = document.createElement('div');
    abGrid.style.cssText = 'display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;';
    const abInputs = {{}};

    function roll4d6dl() {{
      const rolls = [1,2,3,4].map(() => Math.floor(Math.random()*6)+1);
      rolls.sort((a,b) => a-b);
      return rolls[1]+rolls[2]+rolls[3];
    }}

    ABILITIES.forEach(ab => {{
      const cell = document.createElement('div');
      cell.style.cssText = 'background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:8px;text-align:center;';
      const lbl = document.createElement('div');
      lbl.style.cssText = 'font-size:9px;font-weight:800;letter-spacing:1px;color:#e67e22;margin-bottom:4px;';
      lbl.textContent = ab;
      const inp = document.createElement('input');
      inp.type = 'number'; inp.min = '3'; inp.max = '20'; inp.value = '10';
      inp.style.cssText = 'width:100%;text-align:center;padding:6px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);background:rgba(0,0,0,0.3);color:#fff;font-size:18px;font-weight:300;font-family:inherit;outline:none;box-sizing:border-box;';
      const rollBtn = document.createElement('button');
      rollBtn.textContent = '\ud83c\udfb2';
      rollBtn.style.cssText = 'background:none;border:none;cursor:pointer;font-size:14px;margin-top:4px;transition:transform 0.2s;';
      rollBtn.onclick = () => {{
        rollBtn.style.transform = 'rotate(360deg)';
        setTimeout(() => rollBtn.style.transform = '', 300);
        inp.value = roll4d6dl();
      }};
      cell.appendChild(lbl);
      cell.appendChild(inp);
      cell.appendChild(rollBtn);
      abGrid.appendChild(cell);
      abInputs[ab] = inp;
    }});

    rollAllBtn.onclick = () => {{
      ABILITIES.forEach(ab => {{ abInputs[ab].value = roll4d6dl(); }});
    }};
    panel.appendChild(abGrid);

    const pLabel = document.createElement('div');
    pLabel.style.cssText = labelStyle;
    pLabel.textContent = 'Portrait';
    panel.appendChild(pLabel);

    const portraitModeWrap = document.createElement('div');
    portraitModeWrap.style.cssText = 'display:flex;gap:12px;margin-bottom:12px;align-items:center;';
    const portraitModeLabel = document.createElement('label');
    portraitModeLabel.style.cssText = 'color:#ccc;font-size:12px;display:flex;align-items:center;cursor:pointer;';
    const portraitModeCbox = document.createElement('input');
    portraitModeCbox.type = 'checkbox';
    portraitModeCbox.style.marginRight = '6px';
    portraitModeLabel.appendChild(portraitModeCbox);
    portraitModeLabel.appendChild(document.createTextNode('Generate AI Portrait'));
    const customUrlInput = makeInput('Or Custom Image URL...');
    customUrlInput.style.flex = '1';
    portraitModeWrap.appendChild(portraitModeLabel);
    portraitModeWrap.appendChild(customUrlInput);
    panel.appendChild(portraitModeWrap);

    const pRow = document.createElement('div');
    pRow.style.cssText = 'display:none;'; // hide the row entirely
    let selectedPortrait = PORTRAITS[0];
    
    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:8px;';

    function makeActBtn(label, primary) {{
      const b = document.createElement('button');
      b.textContent = label;
      b.style.cssText = primary
        ? 'background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.18);color:#fff;padding:12px 24px;border-radius:10px;font-size:13px;font-weight:600;flex:1;cursor:pointer;transition:all 0.2s;font-family:inherit;'
        : 'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);color:#ccc;padding:12px 24px;border-radius:10px;font-size:13px;flex:1;cursor:pointer;transition:all 0.2s;font-family:inherit;';
      b.onmouseenter = () => b.style.background = primary ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.12)';
      b.onmouseleave = () => b.style.background = primary ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.06)';
      return b;
    }}

    const btnCreate = makeActBtn('\\u2714 Create Character', true);
    const btnCancel = makeActBtn('Cancel', false);

    btnCreate.onclick = () => {{
      const charName = nameInput.value.trim() || 'Unknown Hero';
      const charClass = classSelect.value;
      const level = parseInt(levelInput.value) || 1;
      const abilities = {{}};
      ABILITIES.forEach(ab => {{ abilities[ab] = parseInt(abInputs[ab].value) || 10; }});
      const conMod = Math.floor((abilities.CON - 10) / 2);
      const hitDie = HIT_DIE[charClass] || 8;
      const hp = hitDie + conMod + ((level - 1) * (Math.floor(hitDie/2) + 1 + conMod));

      // Auto-assign fallback portrait
      let autoPortrait = 'warrior';
      const lowerClass = charClass.toLowerCase();
      if (lowerClass.includes('mage') || lowerClass.includes('wizard') || lowerClass.includes('sorcerer')) autoPortrait = 'mage';
      if (lowerClass.includes('rogue') || lowerClass.includes('bard') || lowerClass.includes('ranger')) autoPortrait = 'rogue';
      if (lowerClass.includes('cleric') || lowerClass.includes('paladin')) autoPortrait = 'cleric';

      cleanup();
      resolve(JSON.stringify({{
        name: charName,
        race: raceSelect.value,
        class: charClass,
        level: level,
        hp: Math.max(1, hp),
        abilities: abilities,
        portrait: autoPortrait,
        generate_image: portraitModeCbox.checked,
        image_url: customUrlInput.value.trim()
      }}));
    }};
    btnCancel.onclick = () => {{ cleanup(); resolve(null); }};

    actions.appendChild(btnCreate);
    actions.appendChild(btnCancel);
    panel.appendChild(actions);

    const keyHandler = (e) => {{ if (e.key === 'Escape') {{ cleanup(); resolve(null); }} }};
    document.addEventListener('keydown', keyHandler);
    function cleanup() {{
      document.removeEventListener('keydown', keyHandler);
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }}

    document.body.appendChild(overlay);
    nameInput.focus();
  }});
}})()
"""


# ---------------------------------------------------------------------------
# Choice selector JS overlay (called via __event_call__)
# ---------------------------------------------------------------------------


def build_choice_selector_js(
    prompt_text: str, choices: List[str], context: str = ""
) -> str:
    """Build JavaScript for an interactive choice selector overlay."""
    safe_prompt = json.dumps(prompt_text)
    safe_context = json.dumps(context)
    choices_json = json.dumps(choices)

    return f"""
return (function() {{
  return new Promise((resolve) => {{
    const PROMPT = {safe_prompt};
    const CONTEXT = {safe_context};
    const CHOICES = {choices_json};

    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;
      background:rgba(0,0,0,0.75);backdrop-filter:blur(16px);
      display:flex;align-items:center;justify-content:center;
      font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
      animation:rpgChoiceFadeIn 0.3s ease;`;

    const style = document.createElement('style');
    style.textContent = `
      @keyframes rpgChoiceFadeIn {{ 0%{{opacity:0;}} 100%{{opacity:1;}} }}
      @keyframes rpgChoiceSlideUp {{ 0%{{opacity:0;transform:translateY(20px);}} 100%{{opacity:1;transform:translateY(0);}} }}
      @keyframes rpgChoicePulse {{ 0%{{box-shadow:0 0 0 0 rgba(230,126,34,0.4);}} 70%{{box-shadow:0 0 0 8px rgba(230,126,34,0);}} 100%{{box-shadow:0 0 0 0 rgba(230,126,34,0);}} }}
    `;
    document.head.appendChild(style);

    const panel = document.createElement('div');
    panel.style.cssText = `background:rgba(20,20,25,0.4);backdrop-filter:blur(12px);
      border:1px solid rgba(255,255,255,0.1);border-radius:18px;padding:28px;
      max-width:480px;width:92vw;max-height:90vh;overflow-y:auto;
      box-shadow:0 16px 48px rgba(0,0,0,0.5);color:#fff;
      animation:rpgChoiceSlideUp 0.4s cubic-bezier(0.2,0.8,0.3,1);`;
    overlay.appendChild(panel);

    // Header icon
    const icon = document.createElement('div');
    icon.style.cssText = 'text-align:center;margin-bottom:16px;';
    icon.innerHTML = '<div style="display:inline-flex;width:48px;height:48px;border-radius:14px;background:rgba(230,126,34,0.15);border:1px solid rgba(230,126,34,0.25);align-items:center;justify-content:center;font-size:22px;">\\u2694\\uFE0F</div>';
    panel.appendChild(icon);

    // Prompt text
    const prompt = document.createElement('div');
    prompt.style.cssText = 'text-align:center;font-size:17px;font-weight:600;color:#f0f0f0;line-height:1.5;margin-bottom:8px;';
    prompt.textContent = PROMPT;
    panel.appendChild(prompt);

    // Context text
    if (CONTEXT) {{
      const ctx = document.createElement('div');
      ctx.style.cssText = 'text-align:center;font-size:12px;color:#888;font-style:italic;margin-bottom:16px;line-height:1.4;';
      ctx.textContent = CONTEXT;
      panel.appendChild(ctx);
    }}

    // Divider
    const divider = document.createElement('div');
    divider.style.cssText = 'height:1px;background:rgba(255,255,255,0.08);margin:12px 0 18px;';
    panel.appendChild(divider);

    // Choice buttons
    const choicesWrap = document.createElement('div');
    choicesWrap.style.cssText = 'display:flex;flex-direction:column;gap:10px;';
    panel.appendChild(choicesWrap);

    CHOICES.forEach((choice, idx) => {{
      const btn = document.createElement('button');
      btn.style.cssText = `width:100%;padding:14px 18px;border-radius:12px;
        background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
        color:#f0f0f0;font-size:14px;font-weight:500;cursor:pointer;
        text-align:left;font-family:inherit;transition:all 0.2s;
        display:flex;align-items:center;gap:12px;
        animation:rpgChoiceSlideUp 0.4s cubic-bezier(0.2,0.8,0.3,1) ${{idx * 80 + 200}}ms both;`;

      const num = document.createElement('span');
      num.style.cssText = `display:inline-flex;align-items:center;justify-content:center;
        width:28px;height:28px;border-radius:8px;background:rgba(230,126,34,0.2);
        border:1px solid rgba(230,126,34,0.3);font-size:12px;font-weight:700;
        color:#e67e22;flex-shrink:0;`;
      num.textContent = String.fromCharCode(65 + idx);

      const label = document.createElement('span');
      label.style.cssText = 'flex:1;';
      label.textContent = choice;

      btn.appendChild(num);
      btn.appendChild(label);

      btn.onmouseenter = () => {{
        btn.style.background = 'rgba(230,126,34,0.15)';
        btn.style.borderColor = 'rgba(230,126,34,0.4)';
        btn.style.transform = 'translateX(4px)';
      }};
      btn.onmouseleave = () => {{
        btn.style.background = 'rgba(255,255,255,0.06)';
        btn.style.borderColor = 'rgba(255,255,255,0.1)';
        btn.style.transform = 'translateX(0)';
      }};
      btn.onclick = () => {{
        // Highlight selection before closing
        btn.style.background = 'rgba(230,126,34,0.3)';
        btn.style.borderColor = 'rgba(230,126,34,0.6)';
        btn.style.animation = 'rpgChoicePulse 0.6s ease';
        setTimeout(() => {{
          cleanup();
          resolve(JSON.stringify({{ choice: choice, index: idx, is_custom: false }}));
        }}, 300);
      }};
      choicesWrap.appendChild(btn);
    }});

    // Custom response section
    const customDivider = document.createElement('div');
    customDivider.style.cssText = 'display:flex;align-items:center;gap:12px;margin:16px 0 12px;animation:rpgChoiceSlideUp 0.4s cubic-bezier(0.2,0.8,0.3,1) ' + (CHOICES.length * 80 + 300) + 'ms both;';
    customDivider.innerHTML = '<div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div><div style="font-size:10px;font-weight:700;color:#666;text-transform:uppercase;letter-spacing:1.5px;">or</div><div style="flex:1;height:1px;background:rgba(255,255,255,0.08);"></div>';
    panel.appendChild(customDivider);

    const customWrap = document.createElement('div');
    customWrap.style.cssText = 'display:flex;gap:8px;animation:rpgChoiceSlideUp 0.4s cubic-bezier(0.2,0.8,0.3,1) ' + (CHOICES.length * 80 + 400) + 'ms both;';
    const customInput = document.createElement('input');
    customInput.type = 'text';
    customInput.placeholder = 'Type your own action...';
    customInput.style.cssText = 'flex:1;padding:12px 16px;border-radius:12px;border:1px solid rgba(255,255,255,0.1);background:rgba(0,0,0,0.3);color:#f0f0f0;font-size:14px;font-family:inherit;outline:none;transition:border-color 0.2s;';
    customInput.onfocus = () => {{ customInput.style.borderColor = 'rgba(230,126,34,0.4)'; }};
    customInput.onblur = () => {{ customInput.style.borderColor = 'rgba(255,255,255,0.1)'; }};
    const customBtn = document.createElement('button');
    customBtn.textContent = '\\u27A4';
    customBtn.style.cssText = 'width:44px;height:44px;border-radius:12px;background:rgba(230,126,34,0.2);border:1px solid rgba(230,126,34,0.3);color:#e67e22;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s;flex-shrink:0;';
    customBtn.onmouseenter = () => {{ customBtn.style.background = 'rgba(230,126,34,0.35)'; customBtn.style.borderColor = 'rgba(230,126,34,0.5)'; }};
    customBtn.onmouseleave = () => {{ customBtn.style.background = 'rgba(230,126,34,0.2)'; customBtn.style.borderColor = 'rgba(230,126,34,0.3)'; }};
    function submitCustom() {{
      const val = customInput.value.trim();
      if (!val) return;
      customBtn.style.background = 'rgba(230,126,34,0.5)';
      setTimeout(() => {{
        cleanup();
        resolve(JSON.stringify({{ choice: val, index: -1, is_custom: true }}));
      }}, 200);
    }}
    customBtn.onclick = submitCustom;
    customInput.onkeydown = (e) => {{ if (e.key === 'Enter') {{ e.stopPropagation(); submitCustom(); }} }};
    customWrap.appendChild(customInput);
    customWrap.appendChild(customBtn);
    panel.appendChild(customWrap);

    const keyHandler = (e) => {{
      if (e.key === 'Escape') {{ cleanup(); resolve(null); }}
      if (document.activeElement === customInput) return;
      // Allow keyboard selection with number keys or letter keys
      const keyIdx = e.key.charCodeAt(0) - 65;
      const numIdx = parseInt(e.key) - 1;
      const idx = (keyIdx >= 0 && keyIdx < CHOICES.length) ? keyIdx : ((numIdx >= 0 && numIdx < CHOICES.length) ? numIdx : -1);
      if (idx >= 0 && idx < CHOICES.length) {{
        cleanup();
        resolve(JSON.stringify({{ choice: CHOICES[idx], index: idx, is_custom: false }}));
      }}
    }};
    document.addEventListener('keydown', keyHandler);
    function cleanup() {{
      document.removeEventListener('keydown', keyHandler);
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }}

    document.body.appendChild(overlay);
  }});
}})()
"""


def generate_choice_embed(prompt_text: str, selected: str, accent: str, is_custom: bool = False) -> str:
    """Generate a small embed showing the player's choice."""
    safe_prompt = _esc(prompt_text)
    safe_choice = _esc(selected)
    pid = f"rpgc{random.randint(100000, 999999)}"
    badge = '<span style="font-size:9px;font-weight:700;color:#e67e22;background:rgba(230,126,34,0.15);border:1px solid rgba(230,126,34,0.25);border-radius:4px;padding:1px 6px;margin-left:8px;vertical-align:middle;">CUSTOM</span>' if is_custom else ''
    return f"""
<div style="display:flex;justify-content:center;width:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
<div style="position:relative;overflow:hidden;border-radius:14px;max-width:400px;width:100%;
  box-shadow:0 12px 36px rgba(0,0,0,0.35),0 0 0 1px rgba(255,255,255,0.08);
  color:#fff;box-sizing:border-box;margin-bottom:12px;">
  <div style="position:absolute;inset:0;background:rgba(15,15,20,0.85);backdrop-filter:blur(12px);"></div>
  <div style="position:relative;z-index:2;padding:16px 20px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <span style="font-size:16px;">&#9876;&#65039;</span>
      <div style="font-size:8px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:{accent};opacity:0.8;">Player Choice{badge}</div>
    </div>
    <div style="font-size:12px;color:#999;margin-bottom:8px;font-style:italic;">{safe_prompt}</div>
    <div style="padding:10px 14px;border-radius:10px;background:rgba(230,126,34,0.12);
      border:1px solid rgba(230,126,34,0.25);font-size:14px;font-weight:600;color:#f0f0f0;">
      &#10148; {safe_choice}
    </div>
  </div>
</div>
</div>"""


# ---------------------------------------------------------------------------
# Confirmation prompt JS overlay (called via __event_call__)
# ---------------------------------------------------------------------------


def build_confirm_js(
    prompt_text: str, confirm_label: str, cancel_label: str
) -> str:
    """Build JavaScript for a dramatic yes/no confirmation overlay."""
    safe_prompt = json.dumps(prompt_text)
    safe_confirm = json.dumps(confirm_label)
    safe_cancel = json.dumps(cancel_label)

    return f"""
return (function() {{
  return new Promise((resolve) => {{
    const PROMPT = {safe_prompt};
    const CONFIRM = {safe_confirm};
    const CANCEL = {safe_cancel};

    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;
      background:rgba(0,0,0,0.8);backdrop-filter:blur(18px);
      display:flex;align-items:center;justify-content:center;
      font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
      animation:rpgConfFadeIn 0.3s ease;`;

    const style = document.createElement('style');
    style.textContent = `
      @keyframes rpgConfFadeIn {{ 0%{{opacity:0;}} 100%{{opacity:1;}} }}
      @keyframes rpgConfSlideUp {{ 0%{{opacity:0;transform:translateY(30px) scale(0.95);}} 100%{{opacity:1;transform:translateY(0) scale(1);}} }}
      @keyframes rpgConfGlow {{ 0%,100%{{opacity:0.5;}} 50%{{opacity:1;}} }}
    `;
    document.head.appendChild(style);

    const panel = document.createElement('div');
    panel.style.cssText = `background:rgba(25,15,15,0.5);backdrop-filter:blur(12px);
      border:1px solid rgba(255,80,80,0.15);border-radius:18px;padding:32px;
      max-width:420px;width:90vw;
      box-shadow:0 20px 56px rgba(0,0,0,0.5),0 0 80px rgba(255,50,50,0.08);color:#fff;
      text-align:center;
      animation:rpgConfSlideUp 0.5s cubic-bezier(0.2,0.8,0.3,1);`;
    overlay.appendChild(panel);

    // Warning icon
    const icon = document.createElement('div');
    icon.style.cssText = 'margin-bottom:18px;';
    icon.innerHTML = `<div style="display:inline-flex;width:56px;height:56px;border-radius:50%;
      background:rgba(255,80,80,0.15);border:2px solid rgba(255,80,80,0.3);
      align-items:center;justify-content:center;font-size:26px;
      animation:rpgConfGlow 2s ease infinite;">\\u26A0\\uFE0F</div>`;
    panel.appendChild(icon);

    // Prompt
    const prompt = document.createElement('div');
    prompt.style.cssText = 'font-size:18px;font-weight:600;color:#f0f0f0;line-height:1.5;margin-bottom:24px;';
    prompt.textContent = PROMPT;
    panel.appendChild(prompt);

    // Buttons
    const btnWrap = document.createElement('div');
    btnWrap.style.cssText = 'display:flex;gap:12px;';
    panel.appendChild(btnWrap);

    function makeBtn(label, primary) {{
      const b = document.createElement('button');
      b.textContent = label;
      if (primary) {{
        b.style.cssText = `flex:1;padding:14px 20px;border-radius:12px;
          background:rgba(220,50,50,0.25);border:1px solid rgba(220,50,50,0.4);
          color:#ff8a8a;font-size:14px;font-weight:700;cursor:pointer;
          font-family:inherit;transition:all 0.2s;`;
        b.onmouseenter = () => {{
          b.style.background = 'rgba(220,50,50,0.4)';
          b.style.borderColor = 'rgba(220,50,50,0.7)';
          b.style.color = '#fff';
        }};
        b.onmouseleave = () => {{
          b.style.background = 'rgba(220,50,50,0.25)';
          b.style.borderColor = 'rgba(220,50,50,0.4)';
          b.style.color = '#ff8a8a';
        }};
      }} else {{
        b.style.cssText = `flex:1;padding:14px 20px;border-radius:12px;
          background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
          color:#ccc;font-size:14px;font-weight:500;cursor:pointer;
          font-family:inherit;transition:all 0.2s;`;
        b.onmouseenter = () => {{
          b.style.background = 'rgba(255,255,255,0.12)';
          b.style.borderColor = 'rgba(255,255,255,0.2)';
        }};
        b.onmouseleave = () => {{
          b.style.background = 'rgba(255,255,255,0.06)';
          b.style.borderColor = 'rgba(255,255,255,0.1)';
        }};
      }}
      return b;
    }}

    const cancelBtn = makeBtn(CANCEL, false);
    const confirmBtn = makeBtn(CONFIRM, true);

    cancelBtn.onclick = () => {{
      cleanup();
      resolve(JSON.stringify({{ confirmed: false, label: CANCEL }}));
    }};
    confirmBtn.onclick = () => {{
      confirmBtn.style.background = 'rgba(220,50,50,0.6)';
      confirmBtn.style.color = '#fff';
      setTimeout(() => {{
        cleanup();
        resolve(JSON.stringify({{ confirmed: true, label: CONFIRM }}));
      }}, 200);
    }};

    btnWrap.appendChild(cancelBtn);
    btnWrap.appendChild(confirmBtn);

    const keyHandler = (e) => {{
      if (e.key === 'Escape') {{ cleanup(); resolve(JSON.stringify({{ confirmed: false, label: CANCEL }})); }}
      if (e.key === 'Enter') {{ cleanup(); resolve(JSON.stringify({{ confirmed: true, label: CONFIRM }})); }}
    }};
    document.addEventListener('keydown', keyHandler);
    function cleanup() {{
      document.removeEventListener('keydown', keyHandler);
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }}

    document.body.appendChild(overlay);
  }});
}})()
"""


def generate_confirm_embed(
    prompt_text: str, confirmed: bool, action: str, accent: str
) -> str:
    """Generate a small embed showing the player's decision."""
    safe_prompt = _esc(prompt_text)
    safe_action = _esc(action)
    pid = f"rpgcf{random.randint(100000, 999999)}"
    icon = "&#9989;" if confirmed else "&#10060;"
    color = "#4ade80" if confirmed else "#f87171"
    label = "Confirmed" if confirmed else "Declined"
    return f"""
<div style="display:flex;justify-content:center;width:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
<div style="position:relative;overflow:hidden;border-radius:14px;max-width:400px;width:100%;
  box-shadow:0 12px 36px rgba(0,0,0,0.35),0 0 0 1px rgba(255,255,255,0.08);
  color:#fff;box-sizing:border-box;margin-bottom:12px;">
  <div style="position:absolute;inset:0;background:rgba(15,15,20,0.85);backdrop-filter:blur(12px);"></div>
  <div style="position:relative;z-index:2;padding:16px 20px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
      <span style="font-size:16px;">{icon}</span>
      <div style="font-size:8px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:{color};">{label}</div>
    </div>
    <div style="font-size:12px;color:#999;margin-bottom:8px;font-style:italic;">{safe_prompt}</div>
    <div style="padding:10px 14px;border-radius:10px;background:rgba({('74,222,128' if confirmed else '248,113,113')},0.1);
      border:1px solid {color}33;font-size:14px;font-weight:600;color:#f0f0f0;">
      {safe_action}
    </div>
  </div>
</div>
</div>"""


# ---------------------------------------------------------------------------
# Tools class — only actual tool methods live here
# ---------------------------------------------------------------------------


class Tools:
    class Valves(BaseModel):
        dice_animation_speed: float = Field(
            default=1.5,
            description="Duration of the 3D dice animation in seconds (0.5-3.0).",
            ge=0.5,
            le=3.0,
        )
        default_dice_color: str = Field(
            default="#e74c3c",
            description="Primary color for dice faces (hex code).",
        )
        rpg_system: str = Field(
            default="dnd5e",
            description="RPG system for character creation defaults.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def roll_dice(
        self,
        notation: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Roll dice for the PLAYER with a 3D animated overlay. The player physically clicks to roll and accepts the result.
        ONLY use this when the player character performs an action that requires a dice check (attacks, saves, skill checks, etc.).
        NEVER use this for NPC or GM rolls — use quick_roll instead.
        After narrating the roll outcome, if the player needs to decide what to do next, you MUST call present_choices — NEVER write options as text.

        :param notation: Dice notation (e.g. "2d20+5", "4d6", "1d100", "3d8-2").
        :param context: What the roll is for — always provide this (e.g. "Strength check to break the door", "Attack roll vs Goblin").
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Rolling {notation}...", "done": False},
                }
            )

        result = None

        if __event_call__:
            try:
                js_code = build_dice_roller_js(
                    notation,
                    self.valves.default_dice_color,
                    self.valves.dice_animation_speed,
                )
                raw = await __event_call__(
                    {"type": "execute", "data": {"code": js_code}}
                )
                if raw:
                    data = (
                        raw
                        if isinstance(raw, str)
                        else (
                            raw.get("result") or raw.get("value") or raw.get("data")
                            if isinstance(raw, dict)
                            else None
                        )
                    )
                    if data and isinstance(data, str):
                        parsed = json.loads(data)
                        groups = []
                        for r in parsed.get("results", []):
                            found = False
                            for g in groups:
                                if g["sides"] == r["sides"] and g["sign"] == r["sign"]:
                                    g["rolls"].append(r["value"])
                                    g["count"] += 1
                                    g["subtotal"] += r["value"] * r["sign"]
                                    found = True
                                    break
                            if not found:
                                groups.append(
                                    {
                                        "count": 1,
                                        "sides": r["sides"],
                                        "rolls": [r["value"]],
                                        "subtotal": r["value"] * r["sign"],
                                        "sign": r["sign"],
                                    }
                                )
                        result = {
                            "notation": parsed.get("notation", notation),
                            "groups": groups,
                            "modifier": parsed.get("modifier", 0),
                            "total": parsed.get("total", 0),
                        }
            except Exception as e:
                logger.warning(f"RPG Tool: 3D overlay failed ({e}), falling back")

        if result is None:
            result = roll_dice_server(notation)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Dice rolled!", "done": True},
                }
            )

        text = format_roll_text(result)
        if context:
            text = f"**{context}:** {text}"

        tool_result_message = (
            "[TOOL RESULT — UI RENDERED] A visual dice roll embed is NOW VISIBLE to the player in the chat. "
            "The player can already see the dice, the rolls, and the total. "
            "DO NOT list the roll numbers or total again — just narrate what happens as a result. "
            "NEXT STEP: If the player needs to decide what to do, you MUST call present_choices. NEVER write options as text.\n\n"
            + text
        )

        embed_html = generate_roll_embed(result, context)
        return (
            HTMLResponse(
                content=embed_html,
                headers={"Content-Disposition": "inline"},
            ),
            tool_result_message,
        )

    async def create_character(
        self,
        __request__: Request | None = None,
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Open the interactive character creation form. The player fills in their name, race, class, level, and rolls ability scores.
        Call this at the START of a new campaign or when the player explicitly wants to create/re-create a character.
        Returns a rendered character sheet card and structured text data for you to use throughout the session.
        IMPORTANT: After this tool returns, you MUST call present_choices for the player's first decision. NEVER write options as text.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Opening character creator...",
                        "done": False,
                    },
                }
            )

        char_data = None

        if __event_call__:
            try:
                js_code = build_character_creator_js()
                raw = await __event_call__(
                    {"type": "execute", "data": {"code": js_code}}
                )
                if raw:
                    data = (
                        raw
                        if isinstance(raw, str)
                        else (
                            raw.get("result") or raw.get("value") or raw.get("data")
                            if isinstance(raw, dict)
                            else None
                        )
                    )
                    if data and isinstance(data, str):
                        char_data = json.loads(data)
            except Exception as e:
                logger.warning(f"RPG Tool: character creator failed ({e})")

        if char_data is None:
            return "Character creation was cancelled."

        # Handle Image Generation
        if char_data.get("generate_image") and __request__ and __user__:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Generating character portrait...",
                            "done": False,
                        },
                    }
                )
            try:
                race = char_data.get("race", "Human")
                cls = char_data.get("class", "Fighter")
                prompt = f"A high-fantasy engaging portrait of a {race} {cls}. RPG character art style, detailed, digital painting."
                images = await image_generations(
                    request=__request__,
                    form_data=GenerateImageForm(prompt=prompt),
                    user=Users.get_user_by_id(__user__["id"]),
                )
                if images and len(images) > 0:
                    char_data["image_url"] = images[0]["url"]
            except Exception as e:
                logger.error(f"Image generation failed: {e}")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Character created!", "done": True},
                }
            )

        text = format_character_text(char_data)
        tool_result_message = (
            "[TOOL RESULT — UI RENDERED] A visual character sheet card is NOW VISIBLE to the player in the chat. "
            "The player can already see their name, race, class, stats, HP, and AC. "
            "DO NOT restate the full stat block — briefly acknowledge the character and immediately call present_choices for the next decision.\n\n"
            + text
        )

        embed_html = generate_character_embed(char_data)
        return (
            HTMLResponse(
                content=embed_html,
                headers={"Content-Disposition": "inline"},
            ),
            tool_result_message,
        )

    async def quick_roll(
        self,
        notation: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Instantly roll dice for NPCs, monsters, and GM-driven events — NO player interaction needed.
        Use this for enemy attack rolls, NPC saves, random encounter tables, damage from traps, etc.
        The result is shown as a visual embed but the player does NOT click anything.
        After narrating the roll outcome, if the player needs to decide what to do next, you MUST call present_choices — NEVER write options as text.

        :param notation: Dice notation (e.g. "1d20+5", "2d6", "1d100").
        :param context: What the roll is for — always provide this (e.g. "Goblin attack roll", "Trap damage", "Wandering monster check").
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"Rolling {notation}...", "done": False},
                }
            )

        result = roll_dice_server(notation)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Quick roll done!", "done": True},
                }
            )

        text = format_roll_text(result)
        if context:
            text = f"**{context}:** {text}"

        tool_result_message = (
            "[TOOL RESULT — UI RENDERED] A visual dice roll embed is NOW VISIBLE to the player in the chat. "
            "The player can already see the dice, the rolls, and the total. "
            "DO NOT list the roll numbers or total again — just narrate what happens as a result. "
            "NEXT STEP: If the player needs to decide what to do, you MUST call present_choices. NEVER write options as text.\n\n"
            + text
        )

        embed_html = generate_roll_embed(result, context)
        return (
            HTMLResponse(
                content=embed_html,
                headers={"Content-Disposition": "inline"},
            ),
            tool_result_message,
        )

    async def present_choices(
        self,
        prompt_text: str,
        choices: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Present the player with interactive clickable choices. YOU MUST CALL THIS TOOL every time the player needs to make a decision.
        NEVER write choices, options, bullet points, numbered lists, or "what do you do?" prompts as text in your message.
        Writing options as text instead of calling this tool is FORBIDDEN — it breaks the interactive experience.
        The overlay includes clickable buttons AND a free-text input for custom actions, so the player always has full freedom.

        WHEN TO CALL: After narrating a scene, after combat results, after NPC dialogue, after any event where the player must decide.
        HOW TO USE: Write your narrative, then stop writing and call this tool. Do NOT add text after the tool call.

        :param prompt_text: The question or situation to present (e.g. "The corridor splits into three passages. Which do you take?").
        :param choices: JSON array of choice strings, 2-6 options (e.g. '["Take the left passage", "Take the right passage", "Go back"]').
        :param context: Optional flavor text or additional details shown below the prompt.
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Awaiting player choice...", "done": False},
                }
            )

        try:
            choice_list = json.loads(choices) if isinstance(choices, str) else choices
        except (json.JSONDecodeError, TypeError):
            choice_list = [c.strip() for c in choices.split(",") if c.strip()]

        if not choice_list or len(choice_list) < 2:
            return "Error: provide at least 2 choices as a JSON array."

        selected = None
        is_custom = False

        if __event_call__:
            try:
                js_code = build_choice_selector_js(
                    prompt_text, choice_list, context
                )
                raw = await __event_call__(
                    {"type": "execute", "data": {"code": js_code}}
                )
                if raw:
                    data = (
                        raw
                        if isinstance(raw, str)
                        else (
                            raw.get("result") or raw.get("value") or raw.get("data")
                            if isinstance(raw, dict)
                            else None
                        )
                    )
                    if data and isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                            selected = parsed.get("choice", data)
                            is_custom = parsed.get("is_custom", False)
                        except json.JSONDecodeError:
                            selected = data
                            is_custom = True
            except Exception as e:
                logger.warning(f"RPG Tool: choice selector failed ({e})")

        if selected is None:
            return "The player did not make a choice (overlay was dismissed)."

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Player has chosen!", "done": True},
                }
            )

        if is_custom:
            tool_result_message = (
                f"[TOOL RESULT — UI RENDERED] The choice overlay was shown and the player typed a CUSTOM action: **{selected}**\n\n"
                f"A choice embed showing their selection is NOW VISIBLE in the chat. The player has already chosen.\n"
                f"DO NOT repeat the choices, DO NOT list options, DO NOT ask 'what do you do?' — just narrate what happens next.\n"
                f"This was not one of the suggested options — the player is being creative. "
                f"Honor their intent and adapt the narrative. If another decision point comes up, call present_choices again."
            )
        else:
            tool_result_message = (
                f"[TOOL RESULT — UI RENDERED] The choice overlay was shown and the player selected: **{selected}**\n\n"
                f"A choice embed showing their selection is NOW VISIBLE in the chat. The player has already chosen.\n"
                f"DO NOT repeat the choices, DO NOT list options, DO NOT ask 'what do you do?' — just narrate what happens next.\n"
                f"If another decision point comes up, call present_choices again."
            )

        palette_seed = random.randint(0, 9999999)
        c0, c1, c2, c3, c4 = _rpg_palette(palette_seed)
        embed_html = generate_choice_embed(prompt_text, selected, c0, is_custom)
        return (
            HTMLResponse(
                content=embed_html,
                headers={"Content-Disposition": "inline"},
            ),
            tool_result_message,
        )

    async def confirm_action(
        self,
        prompt_text: str,
        confirm_label: str = "Yes, do it",
        cancel_label: str = "No, back away",
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[Any], Awaitable[Any]]] = None,
    ) -> Union[str, Tuple[HTMLResponse, str]]:
        """
        Ask the player for a dramatic yes/no confirmation before a risky or consequential action.
        Use this for: opening cursed items, entering dangerous areas, making irreversible deals,
        triggering traps, confronting powerful enemies, accepting quests with consequences.
        Builds tension and gives the player a moment to reconsider.
        After narrating the outcome, if the player needs to decide what to do next, you MUST call present_choices — NEVER write options as text.

        :param prompt_text: The dramatic question (e.g. "The chest radiates dark energy. Do you dare open it?").
        :param confirm_label: Text for the confirm button (e.g. "Open the chest", "Accept the deal").
        :param cancel_label: Text for the cancel button (e.g. "Step back", "Refuse").
        """
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Awaiting player decision...", "done": False},
                }
            )

        confirmed = None

        if __event_call__:
            try:
                js_code = build_confirm_js(prompt_text, confirm_label, cancel_label)
                raw = await __event_call__(
                    {"type": "execute", "data": {"code": js_code}}
                )
                if raw:
                    data = (
                        raw
                        if isinstance(raw, str)
                        else (
                            raw.get("result") or raw.get("value") or raw.get("data")
                            if isinstance(raw, dict)
                            else None
                        )
                    )
                    if data and isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                            confirmed = parsed.get("confirmed", False)
                        except json.JSONDecodeError:
                            confirmed = data.lower().strip() in ("true", "yes", "confirmed")
            except Exception as e:
                logger.warning(f"RPG Tool: confirm overlay failed ({e})")

        if confirmed is None:
            return "The player did not respond (overlay was dismissed). Treat as hesitation — describe them pausing uncertainly."

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Player decided!", "done": True},
                }
            )

        if confirmed:
            tool_result_message = (
                f"[TOOL RESULT — UI RENDERED] The confirmation overlay was shown and the player chose to proceed: **{confirm_label}**\n\n"
                f"A confirmation embed is NOW VISIBLE in the chat. The player has already decided.\n"
                f"DO NOT repeat the question or options, DO NOT list choices as text.\n"
                f"Narrate the consequences — they committed to this action. "
                f"If another decision point comes up, call present_choices again."
            )
        else:
            tool_result_message = (
                f"[TOOL RESULT — UI RENDERED] The confirmation overlay was shown and the player chose to back away: **{cancel_label}**\n\n"
                f"A confirmation embed is NOW VISIBLE in the chat. The player has already decided.\n"
                f"DO NOT repeat the question or options, DO NOT list choices as text.\n"
                f"Narrate their hesitation or retreat. The danger still looms but they avoided it for now. "
                f"If another decision point comes up, call present_choices again."
            )

        palette_seed = random.randint(0, 9999999)
        c0, c1, c2, c3, c4 = _rpg_palette(palette_seed)
        action = confirm_label if confirmed else cancel_label
        embed_html = generate_confirm_embed(prompt_text, confirmed, action, c0)
        return (
            HTMLResponse(
                content=embed_html,
                headers={"Content-Disposition": "inline"},
            ),
            tool_result_message,
        )
