"""Methods figures (tokenizers and dynamics model) in the style of the overview figure.

Same palette, font, numbered step headers, boxes and arrows as
``overview_architecture_restyled`` (``make_figure.py``), one figure per codec family plus
one for the dynamics model. Every number is from the frozen ``ignite_codecs_v4``
checkpoints and the dynamics code (``config.py``, ``fastts_nets.py``, ``dynamics.py``,
``frame_layout.py``, ``maskgit.py``).

  python3 make_figure.py methods   -> methods/<name>.{html,pdf,png}
"""
import re
import subprocess

import numpy as np
from PIL import Image

from make_figure import (ACT, CHROME, DYN, HERE, INK, MUTED, ROWS, attention, codebook,
                         cuboids, mlp, signal, svg)

OUT = HERE / "methods"
PRINT_W_PT = 6.5 * 72        # included at width = \textwidth
COL_W = 292                  # box width in px


def sub(s):
    return f"<sub>{s}</sub>"


# ------------------------------------------------------------------ data sketches (inline SVG)
def sketch_slow(face, dark, light, n_rows=16, n_cols=5, bands=4, gap=0):
    """Channels x samples matrix; with ``gap`` the 4 channel groups are pulled apart."""
    cw, ch = 16, 6.2
    s, y = "", 2
    per = n_rows // bands
    for b in range(bands):
        for r in range(per):
            for c in range(n_cols):
                col = face if b % 2 == 0 else dark
                op = 0.35 + 0.65 * abs(np.sin(0.7 * (b * per + r) + 0.9 * c))
                s += (f'<rect x="{4 + c * cw}" y="{y:.1f}" width="{cw - 3}" height="{ch - 1.4:.1f}" rx="1" '
                      f'fill="{col}" fill-opacity="{op:.2f}"/>')
            y += ch
        if gap and b < bands - 1:
            y += gap
    return svg(4 + n_cols * cw + 4, int(y + 4), s)


def sketch_envelope(face, dark):
    xs = np.linspace(4, 176, 160)
    ys = 30 - 22 * np.exp(-((xs - 52) / 4) ** 2) - 16 * np.exp(-((xs - 128) / 5) ** 2) \
        + 1.6 * np.sin(xs * 2.1)
    trace = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    s = f'<polyline points="{trace}" fill="none" stroke="{face}" stroke-width="1.4"/>'
    for i, h in enumerate([5, 22, 7, 6, 18]):        # 5 RMS bins of 10 ms
        x = 4 + i * 34.4
        s += (f'<rect x="{x + 2:.1f}" y="{62 - h}" width="30" height="{h}" rx="2" fill="{dark}" fill-opacity="0.8"/>')
    s += f'<line x1="4" y1="62.5" x2="176" y2="62.5" stroke="{MUTED}" stroke-width="1"/>'
    return svg(180, 66, s)


def sketch_spec(face, dark, light, nf=8, nt=6):
    """Spectrogram with the time-frequency patch grid."""
    w, h = 26, 8.5
    s = ""
    for i in range(nt):
        for j in range(nf):
            v = 0.15 + 0.85 * abs(np.sin(0.8 * i + 1.3 * j)) * (1.0 if j != 5 else 0.3)
            if j == 3:
                v = 0.95               # a coherent mode line
            s += (f'<rect x="{4 + i * w}" y="{4 + j * h}" width="{w - 2.4}" height="{h - 2.4}" fill="{face}" '
                  f'fill-opacity="{v:.2f}"/>')
    for k in range(3):                 # stacked channels behind
        s = (f'<rect x="{4 + 6 * (k + 1)}" y="{4 - 4 * (k + 1)}" width="{nt * w - 2}" height="{nf * h - 2}" '
             f'fill="#fff" stroke="{light}" stroke-width="1.2"/>') + s
    return svg(4 + nt * w + 24, 4 + nf * h + 6, s)


def sketch_video(face, dark, light):
    s = ""
    for k in range(5)[::-1]:           # 5 frames
        x, y = 4 + 9 * k, 4 + 6 * (4 - k)
        s += (f'<rect x="{x}" y="{y}" width="144" height="48" rx="2" fill="#fff" stroke="{dark}" '
              f'stroke-width="1.2" stroke-opacity="{0.4 + 0.15 * (4 - k):.2f}"/>')
    x, y = 4, 28
    for i in range(1, 18 // 3):
        s += f'<line x1="{x + i * 24}" y1="{y}" x2="{x + i * 24}" y2="{y + 48}" stroke="{light}" stroke-width="1"/>'
    for j in range(1, 2):
        s += f'<line x1="{x}" y1="{y + j * 24}" x2="{x + 144}" y2="{y + j * 24}" stroke="{light}" stroke-width="1"/>'
    s += (f'<ellipse cx="{x + 92}" cy="{y + 34}" rx="34" ry="9" fill="{face}" fill-opacity="0.55"/>'
          f'<rect x="{x}" y="{y}" width="24" height="24" fill="{face}" fill-opacity="0.35" stroke="{dark}" stroke-width="1.6"/>')
    return svg(190, 82, s)


def sketch_tokens(groups):
    """Row of token blocks, one colour per modality group: [(n_blocks, face)]."""
    s, x = "", 2
    for n, col in groups:
        for _ in range(n):
            s += f'<rect x="{x}" y="4" width="9" height="18" rx="1.5" fill="{col}"/>'
            x += 11
        x += 6
    return svg(x, 26, s)


# ------------------------------------------------------------------ boxes and headers
def box(bid, kind, title, lines=(), glyph="", bold=None, solid=False, viz="", note=None):
    face, dark, fill = ROWS.get(kind) or {"dyn": DYN, "act": ACT}[kind]
    style = (f"background:{dark};border-color:{dark};color:#fff" if solid
             else f"background:{fill};border-color:{dark}")
    tcol = "#fff" if solid else dark
    body = "".join(f'<div class="ln">{l}</div>' for l in lines)
    b = f'<div class="ln b" style="color:{tcol}">{bold}</div>' if bold else ""
    v = f'<div class="viz">{viz}</div>' if viz else ""
    gl = f'<div class="gl">{glyph}</div>' if glyph else ""
    return (f'<div id="{bid}" class="box{" solid" if solid else ""}" style="{style}">{gl}'
            f'<div class="tt" style="color:{tcol}">{title}</div>{v}{body}{b}</div>')


def head(n, title, sub_):
    return (f'<div class="hd"><span class="badge">{n}</span><span><span class="ht">{title}</span>'
            f'<span class="hs">{sub_}</span></span></div>')


CSS = f"""
@page{{size:__W__px __H__px;margin:0}}
body{{margin:0;background:#fff}}
#fig{{position:relative;display:inline-grid;padding:12px 12px 14px 8px;column-gap:34px;row-gap:16px;
     grid-template-columns:repeat(__NCOL__,{COL_W}px);font-family:'IBM Plex Sans',sans-serif;color:{INK}}}
.cell{{display:flex}}
.box{{position:relative;border:2.2px solid;border-radius:10px;padding:10px 12px 10px 12px;box-sizing:border-box;
      width:100%;font-size:20px;line-height:1.45;color:{MUTED}}}
.box.solid{{color:#ffffffe6}}
.tt{{font-size:22px;font-weight:700;line-height:1.3;margin-bottom:4px;padding-right:46px}}
.ln.b{{font-weight:700;margin-top:2px}}
.gl{{position:absolute;right:10px;top:10px}}
.viz{{margin:6px 0 8px 0}}
.hd{{display:flex;gap:10px;align-items:flex-start;white-space:nowrap}}
.badge{{display:inline-flex;width:30px;height:30px;border-radius:15px;background:{INK};color:#fff;font-size:17px;
        font-weight:700;align-items:center;justify-content:center;flex-shrink:0}}
.ht{{display:block;font-size:22px;font-weight:700;letter-spacing:0.8px}}
.hs{{display:block;font-size:19px;color:{MUTED}}}
.note{{font-size:19px;line-height:1.4;color:{MUTED}}}
sub,sup{{font-size:72%;line-height:0}}
#arrows{{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}}
"""

# Arrow routing in the browser, from the laid-out boxes (as in the overview figure).
JS = """
document.fonts.ready.then(()=>{
const INK="%(ink)s", MUTED="%(muted)s";
const F=document.getElementById('fig').getBoundingClientRect();
function rect(id){const r=document.getElementById(id).getBoundingClientRect();
  return {l:r.left-F.left,r:r.right-F.left,t:r.top-F.top,b:r.bottom-F.top,cx:(r.left+r.right)/2-F.left,cy:(r.top+r.bottom)/2-F.top};}
function ns(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e;}
const A=document.getElementById('arrows'); A.setAttribute('width',F.width); A.setAttribute('height',F.height);
function head(x,y,dir,col){const s=7,h=13;let p;
  if(dir==='r')p=`${x},${y} ${x-h},${y-s} ${x-h},${y+s}`;
  if(dir==='u')p=`${x},${y} ${x-s},${y+h} ${x+s},${y+h}`;
  if(dir==='d')p=`${x},${y} ${x-s},${y-h} ${x+s},${y-h}`;
  A.appendChild(ns('polygon',{points:p,fill:col||INK}));}
function path(d,col,dash){A.appendChild(ns('path',{d:d,fill:'none',stroke:col||INK,'stroke-width':2.4,'stroke-dasharray':dash||''}));}
const ARROWS=%(arrows)s;
for(const [a,b,kind] of ARROWS){
  const ra=rect(a), rb=rect(b);
  if(kind==='h'){ const y=Math.max(ra.t,rb.t)+Math.min(ra.b-ra.t,rb.b-rb.t)/2;
    path(`M${ra.r},${y} H${rb.l-12}`); head(rb.l-1,y,'r'); }
  if(kind==='elbow'){ const xm=(ra.r+rb.l)/2;
    path(`M${ra.r},${ra.cy} H${xm} V${rb.cy} H${rb.l-12}`); head(rb.l-1,rb.cy,'r'); }
  if(kind==='up'){ const x=rb.cx;
    path(`M${x},${ra.t} V${rb.b+12}`); head(x,rb.b+1,'u'); }
  if(kind==='back'){ const y=Math.max(...[...document.querySelectorAll('.box')].map(e=>e.getBoundingClientRect().bottom-F.top))+24;
    A.setAttribute('height',y+6);
    path(`M${ra.cx},${ra.b} V${y} H${rb.cx} V${rb.b+12}`,MUTED,'9 6'); head(rb.cx,rb.b+1,'u',MUTED); }
}
});
"""


def page(name, ncol, cells, arrows):
    js = JS % {"ink": INK, "muted": MUTED, "arrows": arrows}
    css = CSS.replace("__NCOL__", str(ncol))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{name}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{css}</style></head><body><div id="fig">{''.join(cells)}<svg id="arrows"></svg></div>
<script>{js}</script></body></html>"""


def at(row, col, html, rowspan=1, colspan=1, align="stretch"):
    return (f'<div class="cell" style="grid-row:{row} / span {rowspan};grid-column:{col} / span {colspan};'
            f'align-items:{align}">{html}</div>')


def headers(items):
    return [at(1, i + 1, head(i + 1, t, s)) for i, (t, s) in enumerate(items)]


ENC_HEADS = [("INPUT", "one 50 ms window"), ("PATCHES", "&nbsp;"), ("ENCODER", "frozen after training"),
             ("FSQ", "&nbsp;"), ("DECODER", "frozen · mirrors encoder")]


def fsq_box(bid, kind, d, k):
    face, dark, _ = ROWS[kind]
    return box(bid, kind, "FSQ", [f"Linear {d} → 4", "levels (8, 5, 5, 5)"], codebook(dark, face + "55"),
               bold=f"1,000 codes · K = {k}")


# ------------------------------------------------------------------ figures
def slow_ts():
    k = "slow"
    face, dark, fill = ROWS[k]
    cells = headers(ENC_HEADS) + [
        at(2, 1, box("in", k, "Slow time series",
                     ["C channels × 5 samples", "at 100 Hz"],
                     signal(k, face), viz=sketch_slow(face, face, fill),
                     bold=f"TS n{sub('e')}, T{sub('e')}: 44 core, 10 tang.<br>CER T{sub('i')}, v{sub('tor')}: 48 · MSE: 69")),
        at(2, 2, box("pa", k, "4 channel groups",
                     ["⌈C/4⌉ adjacent channels ×", "5 samples per patch,", "zero-padded"],
                     viz=sketch_slow(face, dark, fill, gap=7), bold="4 patches")),
        at(2, 3, box("en", k, "Linear → 128-d",
                     ["+ group and time", "embeddings", "transformer: 4 layers,", "4 heads, d = 128"],
                     cuboids(face, dark, face + "66"))),
        at(2, 4, fsq_box("fq", k, 128, 4)),
        at(2, 5, box("de", k, "Linear 4 → 128",
                     ["+ group and time", "embeddings", "transformer (4 layers)", "Linear → 4 patches",
                      "→ C channels × 5 samples"],
                     cuboids("#ffffff", "#ffffffaa", "#ffffffcc"), solid=True)),
        at(3, 1, f'<div class="note">One codec per signal (7 codecs, about 2.1 M parameters each). '
                 f'TS is log-standardized, CER and MSE are standardized per channel.</div>', colspan=5),
    ]
    arrows = [["in", "pa", "h"], ["pa", "en", "h"], ["en", "fq", "h"], ["fq", "de", "h"]]
    return page("Slow time-series tokenizer", 5, cells, arrows)


def fast_ts():
    k = "fast"
    face, dark, fill = ROWS[k]
    heads = [("INPUT", "one 50 ms window"), ("ENVELOPE", "fixed, not trained"), ("ENCODER", "frozen after training"),
             ("FSQ", "&nbsp;"), ("DECODER", "frozen · mirrors encoder")]
    cells = headers(heads) + [
        at(2, 1, box("in", k, f"D{sub('α')} filterscopes", ["8 channels × 500 samples", "at 10 kHz"],
                     signal(k, face)), rowspan=2, align="center"),
        at(2, 2, box("ev", k, "RMS envelope",
                     ["− 2 ms moving mean", "RMS in 5 bins of 10 ms", "log(1 + RMS/ε), ε = 10<sup>−3</sup>"],
                     viz=sketch_envelope(face, dark), bold="8 channels × 5 bins"), rowspan=2, align="center"),
        at(2, 3, box("ga", k, "Gain",
                     ["level and log spread", "per channel (16 values)", "MLP 16 → 256 → 256", "→ 3 × 256"],
                     mlp(dark, face + "88"), bold="3 gain tokens")),
        at(3, 3, box("sh", k, "Shape",
                     ["envelope normalized by", "the gain; per 10 ms bin", "Linear 8 → 256,", "transformer: 6 layers,",
                      "8 heads; mix 5 → 2 tokens"],
                     cuboids(face, dark, face + "66"), bold="2 shape tokens")),
        at(2, 4, box("fq", k, "FSQ", ["+ token embeddings", "Linear 256 → 4", "levels (8, 5, 5, 5)"],
                     codebook(dark, face + "55"), bold="1,000 codes · K = 5"), rowspan=2, align="center"),
        at(2, 5, box("de", k, "Gain → MLP",
                     ["→ level, spread", "shape: unmix 2 → 5,", "transformer (6 layers),", "Linear → 8 per bin",
                      "→ level + spread × shape"],
                     cuboids("#ffffff", "#ffffffaa", "#ffffffcc"), solid=True), rowspan=2, align="center"),
        at(4, 1, '<div class="note">The gain tokens bypass the transformer, so the shape path cannot change '
                 'the level code. The codec models the ELM envelope at 10 ms resolution.</div>', colspan=5),
    ]
    arrows = [["in", "ev", "h"], ["ev", "ga", "elbow"], ["ev", "sh", "elbow"], ["ga", "fq", "elbow"],
              ["sh", "fq", "elbow"], ["fq", "de", "h"]]
    return page("Fast time-series tokenizer", 5, cells, arrows)


def spectrogram():
    k = "spec"
    face, dark, fill = ROWS[k]
    cells = headers(ENC_HEADS) + [
        at(2, 1, box("in", k, "Spectrograms",
                     ["C channels at 500 kHz", "STFT 1024, hop 256, Hann", f"log{sub('10')} power, DC dropped"],
                     signal(k, face), bold="C × 512 bins × 96 frames")),
        at(2, 2, box("pa", k, "Time–frequency",
                     ["channels stacked per patch", f"BES, MHR, CO{sub('2')}: 16 × 16", "Mirnov: 8 bins × 32 frames",
                      "ECE: 4 bins × 8 frames"],
                     viz=sketch_spec(face, dark, face + "77"), bold="192 patches")),
        at(2, 3, box("en", k, "Linear → 256-d",
                     ["+ frequency and time", "embeddings", "transformer: 6 layers,", "8 heads, d = 256"],
                     cuboids(face, dark, face + "66"))),
        at(2, 4, fsq_box("fq", k, 256, 192)),
        at(2, 5, box("de", k, "transformer (6)",
                     ["Linear → patches", "→ C × bins × 96 frames", "+ residual refinement:", "6 dilated 3 × 3 convs"],
                     cuboids("#ffffff", "#ffffffaa", "#ffffffcc"), solid=True)),
        at(3, 1, f'<div class="note">One codec per diagnostic: ECE 40 channels (first 64 bins, up to 31 kHz), '
                 f'BES 16, MHR 6, CO{sub("2")} 4, Mirnov 29. Log power standardized per channel and frequency '
                 f'bin, except BES.</div>', colspan=5),
    ]
    arrows = [["in", "pa", "h"], ["pa", "en", "h"], ["en", "fq", "h"], ["fq", "de", "h"]]
    return page("Spectrogram tokenizer", 5, cells, arrows)


def video():
    k = "cam"
    face, dark, fill = ROWS[k]
    heads = list(ENC_HEADS)
    heads[1] = ("TUBES", "&nbsp;")
    cells = headers(heads) + [
        at(2, 1, box("in", k, "Divertor camera", ["TangTV lower or upper", "2 channels × 5 frames", "(100 Hz)"],
                     signal(k, face), bold="120 × 360 pixels")),
        at(2, 2, box("pa", k, "5 × 2 × 20 × 20 tubes",
                     ["all 5 frames and both", "channels in each tube"],
                     viz=sketch_video(face, dark, face + "77"), bold="6 × 18 = 108 tubes")),
        at(2, 3, box("en", k, "Linear 4000 → 256",
                     ["+ time, height and width", "embeddings", "transformer: 6 layers,", "8 heads, d = 256"],
                     cuboids(face, dark, face + "66"))),
        at(2, 4, fsq_box("fq", k, 256, 108)),
        at(2, 5, box("de", k, "transformer (6)",
                     ["Linear → tubes → frames", "+ residual refinement:", "4 convs (3 × 3)",
                      "→ 2 × 5 × 120 × 360"],
                     cuboids("#ffffff", "#ffffffaa", "#ffffffcc"), solid=True)),
        at(3, 1, '<div class="note">One codec per divertor (lower, upper). The camera frames are '
                 'resampled from 60 Hz to 100 Hz and downsampled to 120 × 360 pixels.</div>', colspan=5),
    ]
    arrows = [["in", "pa", "h"], ["pa", "en", "h"], ["en", "fq", "h"], ["fq", "de", "h"]]
    return page("Video tokenizer", 5, cells, arrows)


def dynamics():
    k = "dyn"
    face, dark, fill = DYN
    toks = sketch_tokens([(4, ROWS["slow"][0]), (1, ROWS["fast"][0]), (6, ROWS["spec"][0]), (3, ROWS["cam"][0])])
    heads = [("FRAME CODES", "20 seed + predicted frames"), ("EMBEDDING", "trained"), ("BACKBONE", "× 48 blocks"),
             ("HEADS", "&nbsp;"), ("SAMPLING", "MaskGIT")]
    cells = headers(heads) + [
        at(2, 1, box("co", k, "1,209 tokens",
                     ["per frame: slow TS 7 × 4,", "filterscopes 5,", "spectrograms 5 × 192,", "cameras 2 × 108",
                      "next frame: all [MASK]"],
                     viz=toks, bold="1,000 codes per token")),
        at(2, 2, box("em", k, "Embedding, d = 1024",
                     ["code (per modality,", "1,000 codes + [MASK])", "+ token position (1,209)",
                      "+ modality (15)", "+ frame index", "+ actuators of the frame"])),
        at(2, 3, box("bb", k, "Space–time block",
                     ["spatial attention:", "all 1,209 tokens of a frame", "causal temporal attention:",
                      "same token position,", "earlier frames", "feed-forward (4 × d)",
                      "pre-LayerNorm, 16 heads"],
                     attention(face, face + "77"), bold="838 M parameters")),
        at(2, 4, box("hd", k, "Output heads",
                     ["final LayerNorm", "Linear 1024 → 1,000", "per modality, shared", "over its tokens"],
                     mlp(dark, face + "88"), bold="code distributions")),
        at(2, 5, box("sa", k, "Next frame",
                     ["10 reveal steps,", "cosine schedule", "temperature 1", "Gumbel reveal noise 3",
                      "→ decoders"],
                     codebook(dark, face + "55"), bold="appended to history")),
        at(3, 2, box("ac", "act", "Actuators",
                     ["88 channels, 8 groups", "frame mean, normalized", "Linear 88 → 1024,",
                      "added to every token"])),
    ]
    arrows = [["co", "em", "h"], ["em", "bb", "h"], ["bb", "hd", "h"], ["hd", "sa", "h"], ["ac", "em", "up"],
              ["sa", "co", "back"]]
    return page("Dynamics model", 5, cells, arrows)


FIGS = {"slow_ts_tokenizer": slow_ts, "fast_ts_tokenizer": fast_ts, "spectrogram_tokenizer": spectrogram,
        "video_tokenizer": video, "dynamics_model": dynamics}


# ------------------------------------------------------------------ rendering
def render(html, stem):
    """Two passes as in make_figure.render: measure the content, then the exact page."""
    OUT.mkdir(exist_ok=True)
    src = OUT / f"{stem}.html"
    base = [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=8000"]
    src.write_text(html.replace("__W__", "2400").replace("__H__", "1400"))
    probe = OUT / "_probe.png"
    subprocess.run(base + ["--hide-scrollbars", "--window-size=2400,1400", f"--screenshot={probe}",
                           f"file://{src}"], check=True, capture_output=True)
    im = Image.open(probe).convert("L").point(lambda v: 255 if v < 250 else 0)
    _, _, w, h = im.getbbox()
    w, h = w + 18, h + 18
    probe.unlink()
    src.write_text(html.replace("__W__", str(w)).replace("__H__", str(h)))
    subprocess.run(base + ["--hide-scrollbars", f"--window-size={w},{h}", "--force-device-scale-factor=2",
                           f"--screenshot={OUT / (stem + '.png')}", f"file://{src}"], check=True, capture_output=True)
    subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={OUT / (stem + '.pdf')}", f"file://{src}"],
                   check=True, capture_output=True)
    sizes = [float(x) for x in re.findall(r"font-size:([0-9.]+)px", html)]
    print(f"{stem}: canvas {w} x {h} px; at \\textwidth smallest text {min(sizes) * PRINT_W_PT / w:.1f} pt, "
          f"body {20 * PRINT_W_PT / w:.1f} pt")


def build_all():
    for stem, fn in FIGS.items():
        render(fn(), stem)
