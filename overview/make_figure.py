"""overview_architecture.pdf restyled in the visual style of the CER hero figure, with a DIII-D cutaway render
(left cut face = diagnostics, right cut face = actuators, laid out as in overview_machine.png).

Content (all text, structure, row/column meaning) is taken unchanged from overview_architecture.pdf.
Style from the CER figure: IBM Plex Sans, numbered step headers, light rounded boxes, solid output boxes,
thin dark arrows, small layer glyphs, CER palette (one colour per row).

  python3 make_figure.py   -> overview_architecture_restyled.{html,pdf,png}
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PRINT_W_PT = 1.38 * 6.5 * 72            # included at width = 1.38\textwidth

ROWS = {  # one CER palette colour per row: (face, dark, fill)
    "slow": ("#6f5bc0", "#4d3c98", "#f1eefa"),
    "fast": ("#df7f33", "#b35f1c", "#fcf0e5"),
    "spec": ("#c8352b", "#9e2a22", "#fbebea"),
    "cam":  ("#148f88", "#0e6b66", "#e7f4f3"),
}
DYN = ("#3f8a1f", "#2f6a16", "#eef6ea")   # dynamics model (its own colour, as in the original)
ACT = ("#8c9099", "#3a3d45", "#f3f4f6")   # actuators (neutral, as in the original)
INK, MUTED = "#222428", "#5f636c"


# ------------------------------------------------------------------ render crop + marker positions
def prepare_render():
    im = Image.open(HERE / "d3d_render.png")
    x0, y0, x1, y1 = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    W0, H0 = im.size
    crop = im.crop((x0, y0, x1, y1))
    crop.thumbnail((1100, 1100))
    crop.save(HERE / "d3d_crop.png")
    pts = json.loads((HERE / "d3d_render.points.json").read_text())
    for p in pts:   # image fraction of the crop
        p["x"] = (p["u"] * W0 - x0) / (x1 - x0)
        p["y"] = (p["v"] * H0 - y0) / (y1 - y0)
    return crop.size, {p["name"]: p for p in pts}


# ------------------------------------------------------------------ small glyphs (inline SVG)
def svg(w, h, body):
    return f'<svg class="glyph" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>'


def cuboids(face, dark, top, n=3, w=7, h=24, d=8, gap=11):
    s = ""
    for i in range(n):
        x = 2 + i * gap
        dx, dy = d * 0.8, -d * 0.5
        y0 = 6 - dy
        s += (f'<polygon points="{x},{y0} {x + w},{y0} {x + w + dx},{y0 + dy} {x + dx},{y0 + dy}" fill="{top}"/>'
              f'<polygon points="{x + w},{y0} {x + w + dx},{y0 + dy} {x + w + dx},{y0 + dy + h} {x + w},{y0 + h}" fill="{dark}"/>'
              f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" fill="{face}"/>')
    return svg(2 + (n - 1) * gap + w + d * 0.8 + 2, h + 12, s)


def mlp(col, light):
    nodes = [(5, [6, 15, 24]), (21, [10, 20])]
    s = "".join(f'<line x1="5" y1="{a}" x2="21" y2="{b}" stroke="{light}" stroke-width="1"/>'
                for a in nodes[0][1] for b in nodes[1][1])
    s += "".join(f'<circle cx="{x}" cy="{y}" r="3.2" fill="{col}"/>' for x, ys in nodes for y in ys)
    return svg(26, 30, s)


def codebook(col, light):
    s = ""
    for i in range(3):
        for j in range(3):
            s += f'<rect x="{2 + 8 * j}" y="{2 + 8 * i}" width="6.5" height="6.5" rx="1" fill="{col if (i + j) % 3 == 0 else light}"/>'
    return svg(28, 28, s)


def signal(kind, col):
    if kind == "slow":
        b = f'<polyline points="2,20 8,14 14,16 20,8 26,11 32,5" fill="none" stroke="{col}" stroke-width="2"/>'
    elif kind == "fast":
        xs = np.linspace(2, 32, 40)
        ys = 13 + 8 * np.sin(xs * 1.7) * np.exp(-((xs - 17) / 9) ** 2)
        b = f'<polyline points="{" ".join(f"{a:.1f},{c:.1f}" for a, c in zip(xs, ys))}" fill="none" stroke="{col}" stroke-width="1.5"/>'
    elif kind == "spec":
        b = "".join(f'<rect x="{2 + 5 * i}" y="{2 + 5 * j}" width="4.5" height="4.5" fill="{col}" '
                    f'fill-opacity="{0.2 + 0.8 * abs(np.sin(i * 0.9 + j * 0.6)):.2f}"/>' for i in range(6) for j in range(4))
    else:
        b = (f'<rect x="2" y="3" width="30" height="20" rx="2" fill="none" stroke="{col}" stroke-width="1.6"/>'
             f'<circle cx="17" cy="13" r="5" fill="{col}" fill-opacity="0.5"/>')
    return svg(34, 26, b)


def attention(col, light):
    ys = [6, 14, 22, 30]
    s = "".join(f'<rect x="2" y="{y - 3}" width="7" height="6" fill="{col}"/>' for y in ys)
    for i in range(4):
        for j in range(i + 1, 4):
            k = 4 + 4 * (j - i)
            s += f'<path d="M9,{ys[i]} C{9 + k},{ys[i]} {9 + k},{ys[j]} 9,{ys[j]}" fill="none" stroke="{light}" stroke-width="1.2"/>'
    return svg(28, 36, s)


# ------------------------------------------------------------------ boxes
def box(bid, kind, title, lines, glyph="", bold=None, solid=False, cls=""):
    face, dark, fill = ROWS.get(kind) or {"dyn": DYN, "act": ACT}[kind]
    style = (f"background:{face if not solid else dark};border-color:{dark};color:#fff" if solid
             else f"background:{fill};border-color:{dark}")
    tcol = "#fff" if solid else dark
    body = "".join(f'<div class="ln">{l}</div>' for l in lines)
    b = f'<div class="ln b" style="color:{tcol}">{bold}</div>' if bold else ""
    return (f'<div id="{bid}" class="box {cls}" style="{style}"><div class="gl">{glyph}</div>'
            f'<div class="tt" style="color:{tcol}">{title}</div>{body}{b}</div>')


def head(n, title, sub):
    return (f'<div class="hd"><span class="badge">{n}</span><span><span class="ht">{title}</span>'
            f'<span class="hs">{sub}</span></span></div>')


def main():
    (iw, ih), P = prepare_render()
    sub = lambda s: f"<sub>{s}</sub>"
    R = {k: v for k, v in ROWS.items()}
    rows = [
        ("slow", "Slow time series (7)", ["273 ch × 5 samples (100 Hz)", f"TS n{sub('e')}, T{sub('e')}; CER T{sub('i')}, v{sub('tor')}; MSE"],
         "4 channel-group patches", ["Linear → 128-d, transformer (4)"], "K = 4 per modality", ("transformer (4) → Linear", "→ 273 ch × 5 samples")),
        ("fast", "Fast time series (1)", ["8 ch × 500 samples (10 kHz)", f"D{sub('α')} filterscopes"],
         "RMS envelope, 8 ch × 5 bins", ["gain (3) + shape (2) tokens"], "K = 5", ("gain + shape → envelope", "→ 8 ch × 5 bins")),
        ("spec", "Spectrograms (5)", ["95 ch, 96 frames × ≤512 bins", f"ECE, BES, MHR, CO{sub('2')}, Mirnov"],
         "time–frequency patches", ["Linear → 256-d, transformer (6)"], "K = 192 per modality", ("transformer (6) → Linear", "→ time–frequency patches")),
        ("cam", "Divertor cameras (2)", ["2 ch × 5 frames × 120 × 360", "TangTV lower, upper"],
         "5 × 20 × 20 tubes", ["Linear → 256-d, transformer (6)"], "K = 108 per camera", ("transformer (6) → Linear", "→ 5 × 20 × 20 tubes")),
    ]
    cells = []
    for i, (k, mt, ml, tt, tl, tb, dl) in enumerate(rows):
        face, dark, fill = R[k]
        top = fill
        r = i + 2
        cells.append(f'<div style="grid-row:{r};grid-column:2">{box(f"m{i}", k, mt, ml, signal(k, face))}</div>')
        cells.append(f'<div style="grid-row:{r};grid-column:3">{box(f"t{i}", k, tt, tl, cuboids(face, dark, ROWS[k][0] + "66"), tb)}</div>')
        cells.append(f'<div style="grid-row:{r};grid-column:4">{box(f"f{i}", k, "FSQ", ["1,000 codes"], codebook(dark, fill if False else face + "55"))}</div>')
        cells.append(f'<div style="grid-row:{r};grid-column:6">{box(f"h{i}", k, "logits", ["sample"], mlp(dark, face + "88"))}</div>')
        cells.append(f'<div style="grid-row:{r};grid-column:7">{box(f"d{i}", k, dl[0], [dl[1]], cuboids("#ffffff", "#ffffffaa", "#ffffffcc"), solid=True)}</div>')
    dyn_glyph = attention(DYN[0], DYN[0] + "77") + cuboids(DYN[0], DYN[1], DYN[0] + "77", n=3)
    cells.append('<div style="grid-row:2 / 6;grid-column:5;display:flex">' + box(
        "dyn", "dyn", "MaskGIT space–time<br>transformer",
        ["48 blocks · d = 1024 · 16 heads", "838 M parameters", "&nbsp;",
         f'<b style="color:{DYN[1]}">1,209 tokens per frame</b>', "&nbsp;",
         "spatial attention: all tokens", "of a frame (cross-modal)", "&nbsp;",
         "temporal attention: causal,", "same token position, every", "earlier frame"],
        dyn_glyph, cls="tall") + "</div>")
    cells.append('<div style="grid-row:6;grid-column:3 / 5">' + box(
        "act", "act", "Actuators: 8 groups, 88 channels",
        ["ECH power; NBI power, voltage, torque; gas", "flow (calibrated, raw); C- and I-coil currents.",
         "Frame mean → Linear → added to every token", "of its frame, including the frame predicted"]) + "</div>")
    cells.append('<div id="roll" style="grid-row:6;grid-column:6 / 8;padding:22px 0 0 0;font-size:20px;contain:inline-size;'
                 f'line-height:1.35;color:{INK}"><div id="rolltxt">Rollout: 20 measured seed frames <span style="white-space:nowrap">(1.00–1.95 s)</span>, '
                 'then 80 predicted frames <span style="white-space:nowrap">(2.00–5.95 s)</span>, 10 MaskGIT steps each, appended to the context</div></div>')

    heads = [head(1, "DIII-D", "diagnostics · actuators"), head(2, "MEASURED FRAME", "50 ms"),
             head(3, "TOKENIZER", "frozen"), head(4, "FSQ", "&nbsp;"), head(5, "DYNAMICS MODEL", "trained"),
             head(6, "HEADS", "&nbsp;"), head(7, "DECODER", "frozen · mirrors tokenizer")]
    cells += [f'<div style="grid-row:1;grid-column:{i + 1}">{h}</div>' for i, h in enumerate(heads)]

    # tokamak panel: render + markers (left face diagnostics, right face actuators) + legend
    TW = 470
    IW = 390                       # render width; the rest of the column holds the actuator labels
    TH = IW * ih / iw
    cells.append(f'<div id="tok" style="grid-row:2 / 7;grid-column:1;position:relative;width:{TW}px">'
                 f'<img src="d3d_crop.png" style="position:absolute;left:0;top:30px;width:{IW}px;height:{TH:.0f}px">'
                 f'<svg id="mk" style="position:absolute;left:0;top:30px;overflow:visible" width="{TW}" height="{TH:.0f}"></svg></div>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Overview architecture</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
@page{{size:__W__px __H__px;margin:0}}
body{{margin:0;background:#fff}}
#fig{{position:relative;display:inline-grid;padding:14px 12px 16px 8px;column-gap:30px;row-gap:14px;
     grid-template-columns:{TW}px repeat(6,max-content);font-family:'IBM Plex Sans',sans-serif;color:{INK}}}
.box{{position:relative;border:2.2px solid;border-radius:10px;padding:10px 12px 10px 12px;box-sizing:border-box;
      min-height:124px;height:100%;font-size:20px;line-height:1.5;color:{MUTED}}}
.box.tall{{width:100%}}
.tt{{font-size:22px;font-weight:700;line-height:1.35;margin-bottom:2px;padding-right:50px}}
.tall .tt{{padding-right:78px}}
.ln.b{{font-weight:700}}
.gl{{position:absolute;right:10px;top:10px}}
.hd{{display:flex;gap:10px;align-items:flex-start;white-space:nowrap}}
.badge{{display:inline-flex;width:30px;height:30px;border-radius:15px;background:{INK};color:#fff;font-size:17px;
        font-weight:700;align-items:center;justify-content:center;flex-shrink:0}}
.ht{{display:block;font-size:22px;font-weight:700;letter-spacing:0.8px}}
.hs{{display:block;font-size:19px;color:{MUTED}}}
sub{{font-size:72%}}
#arrows{{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}}
.mlab{{font:600 18px 'IBM Plex Sans',sans-serif}}
</style></head><body><div id="fig">{''.join(cells)}<svg id="arrows"></svg></div>
<script>
document.fonts.ready.then(()=>{{
const P = {json.dumps({k: [v['x'], v['y'], v['face']] for k, v in P.items()})};
const INK = "{INK}";
function rect(id){{const f=document.getElementById('fig').getBoundingClientRect();
  const r=document.getElementById(id).getBoundingClientRect();
  return {{l:r.left-f.left,r:r.right-f.left,t:r.top-f.top,b:r.bottom-f.top,cx:(r.left+r.right)/2-f.left,cy:(r.top+r.bottom)/2-f.top}};}}
function ns(t,a){{const e=document.createElementNS('http://www.w3.org/2000/svg',t);for(const k in a)e.setAttribute(k,a[k]);return e;}}
const A=document.getElementById('arrows');
const f=document.getElementById('fig').getBoundingClientRect();
A.setAttribute('width',f.width);A.setAttribute('height',f.height);
function head(x,y,dir,col){{const s=7,h=13;let p;
  if(dir==='r')p=`${{x}},${{y}} ${{x-h}},${{y-s}} ${{x-h}},${{y+s}}`;
  if(dir==='u')p=`${{x}},${{y}} ${{x-s}},${{y+h}} ${{x+s}},${{y+h}}`;
  A.appendChild(ns('polygon',{{points:p,fill:col||INK}}));}}
function path(d,col,dash){{A.appendChild(ns('path',{{d:d,fill:'none',stroke:col||INK,'stroke-width':2.4,'stroke-dasharray':dash||''}}));}}
function hArrow(a,b){{const ra=rect(a),rb=rect(b);path(`M${{ra.r}},${{ra.cy}} H${{rb.l-12}}`);head(rb.l-1,ra.cy,'r');}}
for(let i=0;i<4;i++){{
  const rm=rect('m'+i), tk=rect('tok');
  path(`M${{tk.r+6}},${{rm.cy}} H${{rm.l-12}}`); head(rm.l-1,rm.cy,'r');
  hArrow('m'+i,'t'+i); hArrow('t'+i,'f'+i); hArrow('f'+i,'dyn');
  const rd=rect('dyn'), rh=rect('h'+i); path(`M${{rd.r}},${{rh.cy}} H${{rh.l-12}}`); head(rh.l-1,rh.cy,'r');
  hArrow('h'+i,'d'+i);}}
{{const tk=rect('tok'), ra=rect('act'); path(`M${{tk.r+6}},${{ra.cy}} H${{ra.l-12}}`); head(ra.l-1,ra.cy,'r');
 const rd=rect('dyn'); const x=rd.cx-60; path(`M${{ra.r}},${{ra.cy}} H${{x}} V${{rd.b+12}}`); head(x,rd.b+1,'u');}}
{{const rh0=rect('h3'), rl=rect('roll'); document.getElementById('rolltxt').style.marginLeft=(rh0.cx-rl.l+22)+'px';}}
{{const rh=rect('h3'), rd=rect('dyn'), rr=rect('roll'); const x=rd.cx+60, y=rr.b+8;
 path(`M${{rh.cx}},${{rh.b}} V${{y}} H${{x}} V${{rd.b+12}}`,'{MUTED}','9 6'); head(x,rd.b+1,'u','{MUTED}');}}
// tokamak markers
const M=document.getElementById('mk'); const W={IW}, H={TH:.1f}, TWID={TW};
const X=n=>P[n][0]*W, Y=n=>P[n][1]*H;
function mk(t,a){{M.appendChild(ns(t,a));}}
function lab(x,y,s){{const e=ns('text',{{x:TWID,y:y+6,fill:INK,'text-anchor':'end',class:'mlab'}});e.textContent=s;M.appendChild(e);
  const w=e.getComputedTextLength(); if(TWID-w-6>x) mk('line',{{x1:x,y1:y,x2:TWID-w-6,y2:y,stroke:'#8c9099','stroke-width':1.2}});}}
function txt(x,y,s,col,anchor){{const e=ns('text',{{x:x,y:y,fill:col,'text-anchor':anchor||'start',class:'mlab'}});e.textContent=s;M.appendChild(e);}}
for(const n in P){{ if(n.startsWith('mag')) mk('rect',{{x:X(n)-3,y:Y(n)-3,width:6,height:6,fill:'#df7f33',stroke:'#7a3e0e','stroke-width':0.8}}); }}
for(let i=0;i<3;i++) mk('line',{{x1:X('co2top'+i),y1:Y('co2top'+i),x2:X('co2bot'+i),y2:Y('co2bot'+i),stroke:'#8c6d4f','stroke-width':2.4}});
for(const n in P){{
  if(n.startsWith('ts')){{const x=X(n),y=Y(n); mk('path',{{d:`M${{x-4}},${{y-4}} L${{x+4}},${{y+4}} M${{x-4}},${{y+4}} L${{x+4}},${{y-4}}`,stroke:'#c8352b','stroke-width':2.2}});}}
  if(n.startsWith('ece')){{const x=X(n),y=Y(n); mk('path',{{d:`M${{x-4}},${{y}} H${{x+4}} M${{x}},${{y-4}} V${{y+4}}`,stroke:'#4d3c98','stroke-width':2.2}});}}
  if(n.startsWith('cer')){{const x=X(n),y=Y(n)+5; mk('path',{{d:`M${{x-4}},${{y}} H${{x+4}} M${{x}},${{y-4}} V${{y+4}}`,stroke:'#3f8a1f','stroke-width':2.2}});}}
}}
// actuators (right face)
const nbiCol=['#148f88','#6f5bc0','#df7f33'];
for(let i=0;i<3;i++){{ mk('line',{{x1:X('nbi_out'+i)+40,y1:Y('nbi_out'+i),x2:X('nbi_in'+i),y2:Y('nbi_in'+i),stroke:nbiCol[i],'stroke-width':9,'stroke-opacity':0.55}}); }}
lab(X('nbi_out0')+40,Y('nbi_out1'),'NBI');
const echCol=['#c8352b','#4d3c98','#3f8a1f','#df7f33'];
for(let i=0;i<4;i++) mk('line',{{x1:X('ech_src')+30,y1:Y('ech_src')-40,x2:X('ech_tgt'+i),y2:Y('ech_tgt'+i),stroke:echCol[i],'stroke-width':4,'stroke-opacity':0.75}});
lab(X('ech_src')+30,Y('ech_src')-40,'ECH');
for(const n of ['icoil_u','icoil_l']){{ mk('circle',{{cx:X(n),cy:Y(n),r:6,fill:'none',stroke:'#b35f1c','stroke-width':2.6}}); }}
lab(X('icoil_u')+6,Y('icoil_u'),'3D coils');
mk('line',{{x1:X('gas_src')+26,y1:Y('gas_src')+22,x2:X('gas_tgt'),y2:Y('gas_tgt'),stroke:'#9b3fb5','stroke-width':6,'stroke-opacity':0.6}});
mk('circle',{{cx:X('gas_tgt'),cy:Y('gas_tgt'),r:11,fill:'#9b3fb5','fill-opacity':0.35}});
lab(X('gas_src')+26,Y('gas_src')+22,'Gas valve');
// face labels
const meanX=f=>{{const xs=Object.keys(P).filter(n=>P[n][2]===f).map(n=>P[n][0]*W);return xs.reduce((a,b)=>a+b,0)/xs.length;}};
{{const e=ns('text',{{x:Math.max(70,meanX(1)),y:-4,'text-anchor':'middle',class:'mlab',fill:INK}});e.setAttribute('style','font-size:22px;font-weight:700');e.textContent='Diagnostics';M.appendChild(e);}}
{{const e=ns('text',{{x:Math.min(W-60,meanX(0)),y:-4,'text-anchor':'middle',class:'mlab',fill:INK}});e.setAttribute('style','font-size:22px;font-weight:700');e.textContent='Actuators';M.appendChild(e);}}
// legend under the render (diagnostics)
const L=[['m','Magnetic probe','#df7f33'],['p','ECE','#4d3c98'],['x','Thomson scattering','#c8352b'],['c','CER spectroscopy','#3f8a1f'],['l','CO₂ interferometer','#8c6d4f']];
L.forEach((it,i)=>{{const x=12+(i%2)*230, y=H+34+Math.floor(i/2)*28;
  if(it[0]==='m')mk('rect',{{x:x-3,y:y-9,width:7,height:7,fill:it[2],stroke:'#7a3e0e','stroke-width':0.8}});
  if(it[0]==='p'||it[0]==='c')mk('path',{{d:`M${{x-5}},${{y-5}} H${{x+5}} M${{x}},${{y-10}} V${{y}}`,stroke:it[2],'stroke-width':2.2}});
  if(it[0]==='x')mk('path',{{d:`M${{x-4}},${{y-9}} L${{x+4}},${{y-1}} M${{x-4}},${{y-1}} L${{x+4}},${{y-9}}`,stroke:it[2],'stroke-width':2.2}});
  if(it[0]==='l')mk('line',{{x1:x-6,y1:y-5,x2:x+6,y2:y-5,stroke:it[2],'stroke-width':2.4}});
  txt(x+14,y,it[1],'{MUTED}');}});
}});
</script></body></html>"""
    return html


def render(html):
    import re
    src = HERE / "overview_architecture_restyled.html"
    base = ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=8000"]
    # pass 1: measure the content size
    src.write_text(html.replace("__W__", "3600").replace("__H__", "2000"))
    probe = HERE / "_probe.png"
    subprocess.run(base + ["--hide-scrollbars", "--window-size=3600,2000", f"--screenshot={probe}", f"file://{src}"],
                   check=True, capture_output=True)
    im = Image.open(probe).convert("L").point(lambda v: 255 if v < 250 else 0)
    _, _, w, h = im.getbbox()
    w, h = w + 18, h + 18
    probe.unlink()
    # pass 2: exact page
    src.write_text(html.replace("__W__", str(w)).replace("__H__", str(h)))
    subprocess.run(base + ["--hide-scrollbars", f"--window-size={w},{h}",
                           f"--screenshot={HERE / 'overview_architecture_restyled.png'}", f"file://{src}"],
                   check=True, capture_output=True)
    subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={HERE / 'overview_architecture_restyled.pdf'}",
                           f"file://{src}"], check=True, capture_output=True)
    sizes = [float(x) for x in re.findall(r"font-size:([0-9.]+)px", html)]
    scale = PRINT_W_PT / w
    print(f"canvas {w} x {h} px; at width=1.38\\textwidth: smallest text {min(sizes) * scale:.1f} pt, body {20 * scale:.1f} pt")


if __name__ == "__main__":
    render(main())
