"""Five slide bullets with one pictogram each (CER-figure palette, flat line style).

  python3 make_slide_bullets.py -> pictograms/*.svg (one per bullet) + slide_bullets.{html,png,pdf}
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pictograms"
PUR, ORA, RED, TEAL, GRN, INK, MUT = "#6f5bc0", "#df7f33", "#c8352b", "#148f88", "#3f8a1f", "#222428", "#8c9099"
S = 96  # pictogram size (px)


def svg(body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 96 96">{body}</svg>'


def multimodal():
    """Four different data types converging into one row of tokens."""
    b = (f'<polyline points="6,22 12,15 18,18 24,9" fill="none" stroke="{PUR}" stroke-width="3" stroke-linecap="round"/>'
         f'<polyline points="6,40 9,33 12,46 15,31 18,44 21,36 24,40" fill="none" stroke="{ORA}" stroke-width="2.5"/>')
    for i in range(4):
        for j in range(3):
            b += f'<rect x="{6 + 5 * i}" y="{54 + 5 * j}" width="4.4" height="4.4" fill="{RED}" fill-opacity="{0.25 + 0.25 * ((i + j) % 4)}"/>'
    b += (f'<rect x="6" y="74" width="19" height="14" rx="2" fill="none" stroke="{TEAL}" stroke-width="2.5"/>'
          f'<circle cx="15.5" cy="81" r="3.5" fill="{TEAL}"/>')
    for y in (15, 38, 61, 81):
        b += f'<path d="M30,{y} C48,{y} 48,48 62,48" fill="none" stroke="{MUT}" stroke-width="2"/>'
    for k, c in enumerate((PUR, ORA, RED, TEAL, PUR, RED)):
        b += f'<rect x="{64 + 5 * (k % 3)}" y="{36 + 9 * (k // 3)}" width="4.5" height="7.5" rx="1" fill="{c}"/>'
    b += f'<path d="M80,40 L90,48 L80,56" fill="none" stroke="{INK}" stroke-width="2.5"/>'
    return svg(b)


def timescales():
    """Same 50 ms frame, three sampling rates: 100 Hz -> kHz -> 500 kHz."""
    b = f'<rect x="12" y="8" width="76" height="80" rx="4" fill="{GRN}" fill-opacity="0.08" stroke="{GRN}" stroke-width="2" stroke-dasharray="5 3"/>'
    for y, n, c in ((24, 5, PUR), (48, 16, ORA), (72, 60, RED)):
        b += f'<line x1="14" y1="{y}" x2="86" y2="{y}" stroke="{MUT}" stroke-width="1.2"/>'
        for k in range(n):
            x = 18 + 64 * k / max(n - 1, 1)
            h = 9 if n < 30 else 7
            b += f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y - h}" stroke="{c}" stroke-width="{2.6 if n < 30 else 1}"/>'
    b += f'<text x="50" y="94" font-family="IBM Plex Sans,Helvetica,sans-serif" font-size="9" fill="{INK}" text-anchor="middle">50 ms</text>'
    return svg(b)


def joint_model():
    """Token grid (space x time) with attention arcs: one model for all signals."""
    b = ""
    cols = (PUR, ORA, RED, TEAL)
    for i in range(4):          # frames (time)
        for j in range(4):      # modalities (space)
            b += f'<rect x="{14 + 18 * i}" y="{20 + 16 * j}" width="12" height="10" rx="2" fill="{cols[j]}" fill-opacity="{0.35 + 0.2 * i}"/>'
    for j in (0, 2):           # spatial attention (within the last frame)
        b += f'<path d="M80,{25 + 16 * j} C90,{25 + 16 * j} 90,{57} 80,{57}" fill="none" stroke="{INK}" stroke-width="1.6"/>'
    b += f'<path d="M20,16 C20,4 74,4 74,16" fill="none" stroke="{INK}" stroke-width="1.6" stroke-dasharray="3 2"/>'  # causal, over time
    b += f'<path d="M70,12 L74,17 L78,12" fill="none" stroke="{INK}" stroke-width="1.6"/>'
    return svg(b)


def actuators():
    """Control knobs (ECH, NBI, gas, coils) feeding the model."""
    b = ""
    for k, (y, v, c) in enumerate(((18, 0.7, RED), (38, 0.35, TEAL), (58, 0.85, PUR), (78, 0.5, ORA))):
        b += (f'<line x1="8" y1="{y}" x2="44" y2="{y}" stroke="{MUT}" stroke-width="3" stroke-linecap="round"/>'
              f'<line x1="8" y1="{y}" x2="{8 + 36 * v}" y2="{y}" stroke="{c}" stroke-width="3" stroke-linecap="round"/>'
              f'<circle cx="{8 + 36 * v}" cy="{y}" r="5" fill="#fff" stroke="{c}" stroke-width="2.5"/>')
    b += (f'<path d="M50,48 H66" stroke="{INK}" stroke-width="2.5"/><path d="M64,42 L72,48 L64,54 Z" fill="{INK}"/>'
          f'<rect x="74" y="30" width="16" height="36" rx="4" fill="{GRN}" fill-opacity="0.15" stroke="{GRN}" stroke-width="2.5"/>')
    return svg(b)


def rollout():
    """1 s measured (solid) -> 4 s predicted (dashed, widening uncertainty)."""
    b = (f'<line x1="6" y1="78" x2="90" y2="78" stroke="{MUT}" stroke-width="1.5"/>'
         f'<path d="M26,58 C40,46 54,52 68,40 C76,34 84,38 90,32 L90,62 C84,62 76,60 68,58 C54,62 40,62 26,58 Z" '
         f'fill="{PUR}" fill-opacity="0.15"/>'
         f'<polyline points="6,60 11,55 16,58 21,52 26,58" fill="none" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>'
         f'<path d="M26,58 C40,52 54,58 68,49 C76,45 84,49 90,47" fill="none" stroke="{PUR}" stroke-width="3" stroke-dasharray="5 3"/>'
         f'<line x1="26" y1="20" x2="26" y2="82" stroke="{INK}" stroke-width="1.5" stroke-dasharray="2 2"/>'
         f'<text x="16" y="92" font-family="IBM Plex Sans,Helvetica,sans-serif" font-size="9" fill="{INK}" text-anchor="middle">1 s</text>'
         f'<text x="58" y="92" font-family="IBM Plex Sans,Helvetica,sans-serif" font-size="9" fill="{PUR}" text-anchor="middle">4 s ahead</text>')
    return svg(b)


BULLETS = [
    ("multimodal", multimodal, "Multi-modal",
     "Profiles, filterscopes, spectrograms and camera video share one token space"),
    ("timescales", timescales, "Multi-timescale",
     "100 Hz profiles up to 500 kHz fast signals (ECE, CO₂) in one 50 ms frame"),
    ("joint_model", joint_model, "One model for the whole plasma",
     "Space–time transformer: cross-modal within a frame, causal over time"),
    ("actuators", actuators, "Actuator-driven",
     "ECH, NBI, gas and coil commands steer the prediction: “what if?” scenarios"),
    ("rollout", rollout, "Long-horizon rollout",
     "1 s of measurements → 4 s of predicted discharge evolution"),
]


def main():
    OUT.mkdir(exist_ok=True)
    rows = ""
    for key, fn, title, text in BULLETS:
        s = fn()
        (OUT / f"{key}.svg").write_text(s)
        rows += (f'<div class="row"><div class="ic">{s}</div><div><div class="t">{title}</div>'
                 f'<div class="x">{text}</div></div></div>')
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>@page{{size:1100px 640px;margin:0}} body{{margin:0;background:#fff;font-family:'IBM Plex Sans',sans-serif;color:{INK}}}
.wrap{{padding:22px 34px;display:flex;flex-direction:column;gap:20px}}
.row{{display:flex;align-items:center;gap:26px}} .ic{{flex:0 0 96px}}
.t{{font-size:30px;font-weight:700}} .x{{font-size:23px;color:#5f636c}}</style></head>
<body><div class="wrap">{rows}</div></body></html>"""
    src = HERE / "slide_bullets.html"
    src.write_text(html)
    base = ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=6000"]
    subprocess.run(base + ["--hide-scrollbars", "--window-size=1100,640", f"--screenshot={HERE / 'slide_bullets.png'}",
                           f"file://{src}"], check=True, capture_output=True)
    subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={HERE / 'slide_bullets.pdf'}", f"file://{src}"],
                   check=True, capture_output=True)
    print("wrote", OUT, "and slide_bullets.{html,png,pdf}")


if __name__ == "__main__":
    main()
