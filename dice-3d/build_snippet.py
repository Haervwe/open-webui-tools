#!/usr/bin/env python3
"""
Embeds the built 3D dice roller bundle into rpg_tool_set.py.
Run this after: npm run build in dice-3d/
"""
import gzip
import base64
import os

BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist', 'dice-roller.js')
TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'rpg_tool_set.py')

# Read and compress the bundle
with open(BUNDLE, 'rb') as f:
    raw = f.read()
compressed = gzip.compress(raw, compresslevel=9)
b64 = base64.b64encode(compressed).decode()
print(f"Bundle: {len(raw):,} bytes → gzip {len(compressed):,} → b64 {len(b64):,} chars")

# Read target file
with open(TARGET, 'r', encoding='utf-8') as f:
    src = f.read()

# Find start of the old function
func_start_marker = '\ndef build_dice_roller_js('
start_idx = src.find(func_start_marker)
if start_idx == -1:
    print("ERROR: Could not find build_dice_roller_js in target file!")
    exit(1)
start_idx += 1  # skip the leading newline

# Find the end: the next top-level def at column 0
import re
# After the function signature, find the next `def ` at column 0 or `# ---`
after_start = start_idx + len('def build_dice_roller_js(')
end_match = re.search(r'\ndef |\n# ---', src[after_start:])
if not end_match:
    print("ERROR: Could not find end of function!")
    exit(1)
end_idx = after_start + end_match.start() + 1  # +1 to include the \n

start_line = src[:start_idx].count('\n') + 1
end_line = src[:end_idx].count('\n') + 1
print(f"Replacing function at lines {start_line}–{end_line}")

# Build the new function with the actual b64 value embedded
# Use a raw approach: write each part separately to avoid escaping issues
new_func_lines = [
    'def build_dice_roller_js(notation: str, dice_color: str, anim_speed: float) -> str:',
    '    """Build JavaScript for the 3D dice roller overlay using Three.js + cannon-es physics."""',
    '    parsed = parse_dice_notation(notation)',
    '    group = parsed[0]',
    '    dice_list = []',
    '    for dp in group["dice"]:',
    '        for _ in range(abs(dp["count"])):',
    '            dice_list.append(',
    '                {"sides": dp["sides"], "sign": 1 if dp["count"] > 0 else -1}',
    '            )',
    '    modifier = group["modifier"]',
    '    dice_json = json.dumps(dice_list)',
    '    esc_notation = _esc(notation)',
    '',
    '    # Gzip-compressed, base64-encoded Three.js + cannon-es dice roller bundle',
    f'    bundle_b64 = {repr(b64)}',
    '',
    '    js = "return (function(){\\n"',
    '    js += "return new Promise(function(resolve){\\n"',
    '    js += f"var DICE={dice_json};\\n"',
    '    js += f"var MODIFIER={modifier};\\n"',
    '    js += f"var NOTATION=\'{esc_notation}\';\\n"',
    '    js += f"var DICE_COLOR=\'{dice_color}\';\\n"',
    '    js += f"var BUNDLE_B64={json.dumps(bundle_b64)};\\n"',
    '    js += r"""',
    '// Decompress gzip bundle and launch 3D dice overlay',
    '(function(){',
    '  var b64=BUNDLE_B64;',
    '  var bin=atob(b64);',
    '  var bytes=new Uint8Array(bin.length);',
    '  for(var i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);',
    '  var ds=new DecompressionStream(\'gzip\');',
    '  var writer=ds.writable.getWriter();',
    '  var reader=ds.readable.getReader();',
    '  writer.write(bytes);',
    '  writer.close();',
    '  var chunks=[];',
    '  function readChunk(){',
    '    return reader.read().then(function(res){',
    '      if(!res.done){chunks.push(res.value);return readChunk();}',
    '      var total=chunks.reduce(function(a,c){return a+c.length;},0);',
    '      var buf=new Uint8Array(total);',
    '      var off=0;',
    '      chunks.forEach(function(c){buf.set(c,off);off+=c.length;});',
    '      var text=new TextDecoder().decode(buf);',
    '      var fn=new Function(text);',
    '      fn();',
    '      window.DiceRoller3D.rollDice3D(NOTATION,DICE,MODIFIER,DICE_COLOR).then(resolve);',
    '    });',
    '  }',
    '  readChunk();',
    '})();',
    '"""',
    '    js += "})()}\\n"',
    '    return js',
    '',
]

new_func = '\n'.join(new_func_lines) + '\n'

# Assemble the new file
new_src = src[:start_idx] + new_func + src[end_idx:]

with open(TARGET, 'w', encoding='utf-8') as f:
    f.write(new_src)

# Verify syntax
import ast
try:
    ast.parse(new_src)
    print("Syntax OK!")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    exit(1)

print(f"Done! rpg_tool_set.py updated. New size: {len(new_src):,} bytes")
