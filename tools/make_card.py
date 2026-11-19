#!/usr/bin/env python3
"""Render the README result card from the measured receipts in results/.

    python3 tools/make_card.py        # writes assets/result-card-{light,dark}.svg

Every number on the card is read from results/<arm>/result.json; nothing is typed in by hand.
Colours: v1 -> v2 is an ordered pair, so it is one blue hue as an ordinal ramp (validated light and dark).
"""
import json, pathlib, statistics
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / 'results'

def load(arm):
    return json.loads((RES / arm / 'result.json').read_text())

def temp_means(arm):
    by = {}
    for r in load(arm)['temps']:
        by.setdefault(r['temp'], []).append(r['median_tps'])
    return {t: statistics.mean(v) for t, v in by.items()}

# ---------------------------------------------------------------- numbers (all from receipts)
v1, v2 = load('r1-base'), load('r7-final-cram0')
SIZES = [('2048', '2K'), ('8192', '8K'), ('32768', '32K')]
prefill = [(lab, v1['prefill'][k]['prefill_tps'], v2['prefill'][k]['prefill_tps']) for k, lab in SIZES]
d2, d4 = temp_means('r6-262k-d2'), temp_means('r6-262k-d4')
TEMPS = [(0.0, 'temp 0'), (0.7, 'temp 0.7'), (0.9, 'temp 0.9')]
decode = [(lab, d2[t], d4[t]) for t, lab in TEMPS]
needles = load('r4-262k-depth')['needle']
longest = max((v['prompt_n'] for v in needles.values() if v['passed']))
swap_off = load('r7-final-cram0')['swap']['A2']['ttft_s']
swap_on = load('r7-final-cram8g')['swap']['A2']['ttft_s']
stress = load('r8-cram4g-stress')
floor_gib = stress['min_avail_gib']
swap_growth = max(load(a)['peak_swap_growth'] for a in ('r4-262k-depth', 'r7-final-cram0', 'r7-final-cram8g', 'r8-cram4g-stress'))
pf8_v1, pf8_v2 = prefill[1][1], prefill[1][2]

# ---------------------------------------------------------------- themes
THEMES = {
    'light': dict(surface='#fcfcfb', border='rgba(11,11,11,0.10)', ink='#0b0b0b', ink2='#52514e', muted='#898781',
                  grid='#e1e0d9', base='#c3c2b7', v1='#86b6ef', v2='#256abf', good='#006300', tile='#f6f6f3'),
    'dark':  dict(surface='#1a1a19', border='rgba(255,255,255,0.10)', ink='#f0efec', ink2='#c3c2b7', muted='#898781',
                  grid='#2c2c2a', base='#383835', v1='#1c5cab', v2='#5598e7', good='#0ca30c', tile='#212120'),
}
FONT = "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
W, H, PAD = 1200, 668, 48

def txt(x, y, s, size, fill, weight=400, anchor='start', extra=''):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{escape(s)}</text>')

def bar(x, base, top, w, fill, r=4):
    r = min(r, (base - top) / 2, w / 2)
    return (f'<path d="M{x:.1f},{base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{base:.1f} Z" fill="{fill}"/>')

def tile(x, y, w, h, c, label, value, unit, note, note_fill):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{c["tile"]}" stroke="{c["border"]}"/>',
           txt(x + 18, y + 32, label, 15, c['ink2'])]
    out.append(f'<text x="{x + 18}" y="{y + 80}" fill="{c["ink"]}" font-weight="600">'
               f'<tspan font-size="34">{escape(value)}</tspan><tspan font-size="17" dx="5" fill="{c["ink2"]}" font-weight="500">{escape(unit)}</tspan></text>')
    out.append(txt(x + 18, y + 112, note, 14, note_fill, 500 if note_fill == c['good'] else 400))
    return out

def chart(x, y, w, c, title, unit, rows, legend, vmax, step):
    out = [f'<text x="{x}" y="{y}" font-size="18" font-weight="600" fill="{c["ink"]}">{escape(title)}<tspan dx="8" font-size="14.5" font-weight="400" fill="{c["muted"]}">{escape(unit)}</tspan></text>']
    lx = x
    for name, col in zip(legend, (c['v1'], c['v2'])):
        out.append(f'<rect x="{lx}" y="{y + 16}" width="12" height="12" rx="3" fill="{col}"/>')
        out.append(txt(lx + 18, y + 27, name, 14, c['ink2']))
        lx += 30 + len(name) * 7.6
    top, base = y + 62, y + 232
    left = x + 40
    scale = lambda v: base - (base - top) * v / vmax
    for t in range(0, vmax + 1, step):
        yy = scale(t)
        out.append(f'<line x1="{left}" x2="{x + w}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{c["grid" if t else "base"]}" stroke-width="1"/>')
        out.append(txt(left - 8, yy + 4, f'{t:,}', 12.5, c['muted'], anchor='end', extra='font-variant-numeric="tabular-nums"'))
    gw = (x + w - left) / len(rows); bw = 40
    for i, (lab, a, b) in enumerate(rows):
        cx = left + gw * (i + 0.5)
        xa, xb = cx - bw - 1, cx + 1                                   # 2px surface gap between the pair
        out.append(bar(xa, base, scale(a), bw, c['v1'])); out.append(bar(xb, base, scale(b), bw, c['v2']))
        fmt = (lambda v: f'{v:.0f}') if vmax > 100 else (lambda v: f'{v:.1f}')
        out.append(txt(xa + bw / 2, scale(a) - 7, fmt(a), 13.5, c['ink2'], anchor='middle'))
        out.append(txt(xb + bw / 2, scale(b) - 7, fmt(b), 13.5, c['ink'], 600, anchor='middle'))
        out.append(txt(cx, base + 21, lab, 14, c['ink2'], anchor='middle'))
    return out

def card(theme):
    c = THEMES[theme]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" '
         f'aria-labelledby="t d">',
         f'<title id="t">Hy4 Flash on one DGX Spark: recipe v2 results</title>',
         f'<desc id="d">Decode {d4[0.0]:.1f} tok/s at temp 0. Prefill at 8K {pf8_v2:.0f} tok/s vs {pf8_v1:.0f} in v1. '
         f'Model load {v2["load_s"]:.0f} s vs {v1["load_s"]:.0f} s. Retrieval passed at {round(longest / 1000)}K tokens. '
         f'Returning to a conversation {swap_on:.1f} s vs {swap_off:.0f} s.</desc>',
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="20" fill="{c["surface"]}" stroke="{c["border"]}"/>']
    # header
    s.append(txt(PAD, 78, 'Hy4 Flash on one DGX Spark', 34, c['ink'], 650))
    s.append(txt(PAD, 110, 'AngelSlim SPARK-4Bit-Fast GGUF · llama.cpp + Spark patch · GB10, 128 GB unified memory', 16.5, c['ink2']))
    s.append(txt(W - PAD, 78, 'recipe v2', 15, c['ink2'], 600, anchor='end'))
    s.append(txt(W - PAD, 104, 'single stream · measured 2026-11-19', 14, c['muted'], anchor='end'))
    # hero + tiles
    ty, th = 144, 134
    s.append(f'<rect x="{PAD}" y="{ty}" width="330" height="{th}" rx="12" fill="{c["tile"]}" stroke="{c["border"]}"/>')
    s.append(txt(PAD + 20, ty + 32, 'Decode, temp 0', 15, c['ink2']))
    s.append(f'<text x="{PAD + 20}" y="{ty + 92}" fill="{c["ink"]}" font-weight="650"><tspan font-size="60">{d4[0.0]:.1f}</tspan>'
             f'<tspan font-size="22" dx="8" fill="{c["ink2"]}" font-weight="500">tok/s</tspan></text>')
    s.append(txt(PAD + 20, ty + 120, f'{d4[0.7]:.1f} at temp 0.7 · MTP depth 4 · 256K context', 14, c['ink2']))
    tx, tw, gap = PAD + 330 + 16, 0, 14
    tw = (W - PAD - tx - 3 * gap) / 4
    tiles = [('Prefill, 8K prompt', f'{pf8_v2:.0f}', 'tok/s', f'▲ {100 * (pf8_v2 / pf8_v1 - 1):.0f}% vs v1', c['good']),
             ('Model load', f'{v2["load_s"]:.0f}', 's', f'was {v1["load_s"]:.0f} s', c['ink2']),
             ('Context', '256K', 'tokens', f'retrieval passed at {round(longest / 1000)}K', c['ink2']),
             ('Back to a conversation', f'{swap_on:.1f}', 's', f'was {swap_off:.0f} s (prompt cache)', c['ink2'])]
    for i, t in enumerate(tiles):
        s += tile(tx + i * (tw + gap), ty, tw, th, c, *t)
    # charts
    cy, cw = 330, (W - 2 * PAD - 56) / 2
    s += chart(PAD, cy, cw, c, 'Prefill, cold prompt', 'tok/s', prefill, ('v1 recipe', 'v2 recipe'), 900, 300)
    s += chart(PAD + cw + 56, cy, cw, c, 'Decode at 256K context', 'tok/s', decode, ('MTP depth 2 (v1)', 'MTP depth 4 (v2)'), 45, 15)
    # footer
    s.append(f'<line x1="{PAD}" x2="{W - PAD}" y1="{H - 56}" y2="{H - 56}" stroke="{c["grid"]}"/>')
    s.append(txt(PAD, H - 26, f'Non-repetitive prompts, cache off for prefill · lowest free memory {floor_gib:.1f} GiB (8 cached conversations + 120K prompt), '
                 f'{"no" if swap_growth == 0 else swap_growth} swap growth · receipts in results/', 14, c['muted']))
    s.append('</svg>')
    return '\n'.join(s) + '\n'

if __name__ == '__main__':
    (ROOT / 'assets').mkdir(exist_ok=True)
    for theme in THEMES:
        p = ROOT / 'assets' / f'result-card-{theme}.svg'; p.write_text(card(theme), encoding='utf-8', newline='\n'); print('wrote', p.relative_to(ROOT))
