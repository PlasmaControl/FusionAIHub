"""IGNITE overview figure rendered with Blender (Cycles, flat emission shading).

Same content and layout as make_figure.py (overview_architecture_restyled.html); positions
are in that figure's pixels (x right, y down), 1 px = 0.01 Blender units.

  blender -b -P overview_blender.py -- <out.png> [scale]
"""
import json
import math
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
args = sys.argv[sys.argv.index("--") + 1:]
OUT = args[0]
SCALE = float(args[1]) if len(args) > 1 else 2.0
FONTS = "/scratch/gpfs/EKOLEMEN/ps9551/FusionAIHub/.pixi/envs/stellarai/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/"
W, H = 2264, 836

ROWS = {"slow": ("#6f5bc0", "#4d3c98", "#f1eefa"), "fast": ("#df7f33", "#b35f1c", "#fcf0e5"),
        "spec": ("#c8352b", "#9e2a22", "#fbebea"), "cam": ("#148f88", "#0e6b66", "#e7f4f3")}
DYN = ("#3f8a1f", "#2f6a16", "#eef6ea")
ACT = ("#8c9099", "#3a3d45", "#f3f4f6")
INK, MUTED = "#222428", "#5f636c"

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = False
scene.render.resolution_x, scene.render.resolution_y = int(W * SCALE), int(H * SCALE)
scene.view_settings.view_transform = "Standard"
scene.render.film_transparent = False
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
scene.world = world
FONT = bpy.data.fonts.load(FONTS + "DejaVuSans.ttf")
BOLD = bpy.data.fonts.load(FONTS + "DejaVuSans-Bold.ttf")
_z = [0.0]


def z():
    _z[0] += 0.0005
    return _z[0]


def srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


_mats = {}


def mat(hexcol, alpha=1.0):
    key = (hexcol, round(alpha, 3))
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(f"m{len(_mats)}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*srgb(hexcol), 1)
    if alpha < 1:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = alpha
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs[0])
    else:
        nt.links.new(em.outputs[0], out.inputs[0])
    _mats[key] = m
    return m


def P(x, y):
    return (x * 0.01, -y * 0.01)


def poly(points, hexcol, alpha=1.0):
    mesh = bpy.data.meshes.new("p")
    zz = z()
    mesh.from_pydata([(*P(x, y), zz) for x, y in points], [], [list(range(len(points)))])
    obj = bpy.data.objects.new("p", mesh)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat(hexcol, alpha))
    return obj


def rounded_rect(x0, y0, x1, y1, r, hexcol, alpha=1.0, n=8):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return poly(pts, hexcol, alpha)


def line(x0, y0, x1, y1, width, hexcol, alpha=1.0):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L * width / 2, dx / L * width / 2
    return poly([(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)], hexcol, alpha)


def circle(cx, cy, r, hexcol, alpha=1.0, n=32):
    return poly([(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)],
                hexcol, alpha)


def ring(cx, cy, r, width, hexcol, n=40):
    for i in range(n):
        a, b = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        line(cx + r * math.cos(a), cy + r * math.sin(a), cx + r * math.cos(b), cy + r * math.sin(b), width, hexcol)


def text(s, x, y, size, hexcol, bold=False, align="LEFT"):
    curve = bpy.data.curves.new("t", "FONT")
    curve.body = s
    curve.font = BOLD if bold else FONT
    curve.size = size * 0.01 * 1.32  # Blender sizes the em box; match the HTML glyph height
    curve.align_x = align
    obj = bpy.data.objects.new("t", curve)
    obj.location = (*P(x, y), z())
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat(hexcol))
    return obj


def arrow_head(x, y, direction, hexcol=INK):
    s, h = 7, 13
    if direction == "r":
        poly([(x, y), (x - h, y - s), (x - h, y + s)], hexcol)
    else:  # up
        poly([(x, y), (x - s, y + h), (x + s, y + h)], hexcol)


def harrow(x0, x1, y, hexcol=INK):
    line(x0, y, x1 - 12, y, 2.4, hexcol)
    arrow_head(x1 - 1, y, "r", hexcol)


def dashed(points, hexcol, dash=9, gap=6):
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        t = 0.0
        while t < L:
            t1 = min(L, t + dash)
            line(x0 + (x1 - x0) * t / L, y0 + (y1 - y0) * t / L, x0 + (x1 - x0) * t1 / L, y0 + (y1 - y0) * t1 / L,
                 2.4, hexcol)
            t = t1 + gap


# ------------------------------------------------------------------ glyphs (top-right of boxes)
def g_cuboids(x, y, face, dark, top, n=3, w=7, h=24, d=8, gap=11):
    for i in range(n):
        xi = x + 2 + i * gap
        dx, dy = d * 0.8, -d * 0.5
        y0 = y + 6 - dy
        poly([(xi, y0), (xi + w, y0), (xi + w + dx, y0 + dy), (xi + dx, y0 + dy)], top[0], top[1])
        poly([(xi + w, y0), (xi + w + dx, y0 + dy), (xi + w + dx, y0 + dy + h), (xi + w, y0 + h)], dark[0], dark[1])
        poly([(xi, y0), (xi + w, y0), (xi + w, y0 + h), (xi, y0 + h)], face[0], face[1])


def g_codebook(x, y, col, light):
    for i in range(3):
        for j in range(3):
            c = (col, 1.0) if (i + j) % 3 == 0 else light
            rounded_rect(x + 2 + 8 * j, y + 2 + 8 * i, x + 8.5 + 8 * j, y + 8.5 + 8 * i, 1, c[0], c[1], n=2)


def g_mlp(x, y, col, light):
    a, b = [6, 15, 24], [10, 20]
    for ya in a:
        for yb in b:
            line(x + 5, y + ya, x + 21, y + yb, 1, light[0], light[1])
    for yy in a:
        circle(x + 5, y + yy, 3.2, col)
    for yy in b:
        circle(x + 21, y + yy, 3.2, col)


def g_signal(kind, x, y, col):
    if kind == "slow":
        pts = [(2, 20), (8, 14), (14, 16), (20, 8), (26, 11), (32, 5)]
        for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
            line(x + a, y + b, x + c, y + d, 2, col)
    elif kind == "fast":
        prev = None
        for i in range(41):
            xs = 2 + 30 * i / 40
            ys = 13 + 8 * math.sin(xs * 1.7) * math.exp(-((xs - 17) / 9) ** 2)
            if prev:
                line(x + prev[0], y + prev[1], x + xs, y + ys, 1.5, col)
            prev = (xs, ys)
    elif kind == "spec":
        for i in range(6):
            for j in range(4):
                poly([(x + 2 + 5 * i, y + 2 + 5 * j), (x + 6.5 + 5 * i, y + 2 + 5 * j),
                      (x + 6.5 + 5 * i, y + 6.5 + 5 * j), (x + 2 + 5 * i, y + 6.5 + 5 * j)],
                     col, 0.2 + 0.8 * abs(math.sin(i * 0.9 + j * 0.6)))
    else:
        for (a, b, c, d) in ((2, 3, 32, 3), (32, 3, 32, 23), (32, 23, 2, 23), (2, 23, 2, 3)):
            line(x + a, y + b, x + c, y + d, 1.6, col)
        circle(x + 17, y + 13, 5, col, 0.5)


def g_attention(x, y, col, light):
    ys = [6, 14, 22, 30]
    for yy in ys:
        poly([(x + 2, y + yy - 3), (x + 9, y + yy - 3), (x + 9, y + yy + 3), (x + 2, y + yy + 3)], col)
    for i in range(4):
        for j in range(i + 1, 4):
            k = 4 + 4 * (j - i)
            prev = None
            for s in range(13):
                t = s / 12
                bx = (1 - t) ** 3 * 9 + 3 * (1 - t) ** 2 * t * (9 + k) + 3 * (1 - t) * t ** 2 * (9 + k) + t ** 3 * 9
                by = (1 - t) ** 3 * ys[i] + 3 * (1 - t) ** 2 * t * ys[i] + 3 * (1 - t) * t ** 2 * ys[j] + t ** 3 * ys[j]
                if prev:
                    line(x + prev[0], y + prev[1], x + bx, y + by, 1.2, light[0], light[1])
                prev = (bx, by)


# ------------------------------------------------------------------ boxes
def box(x0, y0, x1, y1, colours, title, lines, bold=None, solid=False, title_lines=None):
    face, dark, fill = colours
    rounded_rect(x0, y0, x1, y1, 10, dark)
    rounded_rect(x0 + 2.2, y0 + 2.2, x1 - 2.2, y1 - 2.2, 8, dark if solid else fill)
    tcol = "#ffffff" if solid else dark
    y = y0 + 31
    for t in (title_lines or [title]):
        text(t, x0 + 13, y, 22, tcol, bold=True)
        y += 30
    y += 0
    for ln in lines:
        if ln:
            text(ln[0] if isinstance(ln, tuple) else ln, x0 + 13, y, 20,
                 ln[1] if isinstance(ln, tuple) else ("#ffffff" if solid else MUTED),
                 bold=isinstance(ln, tuple))
        y += 30
    if bold:
        text(bold, x0 + 13, y, 20, tcol, bold=True)


COL = {"m": (508, 797), "t": (827, 1190), "f": (1222, 1358), "dyn": (1390, 1712), "h": (1744, 1877), "d": (1907, 2241)}
ROWY = [(81, 203), (219, 341), (357, 479), (495, 617)]
rows = [
    ("slow", "Slow time series (7)", ["273 ch × 5 samples (100 Hz)", "TS nₑ, Tₑ; CER Tᵢ, vₜₒᵣ; MSE"],
     "4 channel-group patches", ["Linear → 128-d, transformer (4)"], "K = 4 per modality",
     ("transformer (4) → Linear", "→ 273 ch × 5 samples")),
    ("fast", "Fast time series (1)", ["8 ch × 500 samples (10 kHz)", "Dα filterscopes"],
     "RMS envelope, 8 ch × 5 bins", ["gain (3) + shape (2) tokens"], "K = 5",
     ("gain + shape → envelope", "→ 8 ch × 5 bins")),
    ("spec", "Spectrograms (5)", ["95 ch, 96 frames × ≤512 bins", "ECE, BES, MHR, CO₂, Mirnov"],
     "time–frequency patches", ["Linear → 256-d, transformer (6)"], "K = 192 per modality",
     ("transformer (6) → Linear", "→ time–frequency patches")),
    ("cam", "Divertor cameras (2)", ["2 ch × 5 frames × 120 × 360", "TangTV lower, upper"],
     "5 × 20 × 20 tubes", ["Linear → 256-d, transformer (6)"], "K = 108 per camera",
     ("transformer (6) → Linear", "→ 5 × 20 × 20 tubes")),
]
for i, (k, mt, ml, tt, tl, tb, dl) in enumerate(rows):
    face, dark, fill = ROWS[k]
    y0, y1 = ROWY[i]
    box(*COL["m"][:1], y0, COL["m"][1], y1, ROWS[k], mt, ml) if False else box(COL["m"][0], y0, COL["m"][1], y1, ROWS[k], mt, ml)
    g_signal(k, COL["m"][1] - 44, y0 + 10, face)
    box(COL["t"][0], y0, COL["t"][1], y1, ROWS[k], tt, tl, bold=tb)
    g_cuboids(COL["t"][1] - 44, y0 + 8, (face, 1), (dark, 1), (face, 0.4))
    box(COL["f"][0], y0, COL["f"][1], y1, ROWS[k], "FSQ", ["1,000 codes"])
    g_codebook(COL["f"][1] - 38, y0 + 10, dark, (face, 0.33))
    box(COL["h"][0], y0, COL["h"][1], y1, ROWS[k], "logits", ["sample"])
    g_mlp(COL["h"][1] - 36, y0 + 8, dark, (face, 0.53))
    box(COL["d"][0], y0, COL["d"][1], y1, ROWS[k], dl[0], [dl[1]], solid=True)
    g_cuboids(COL["d"][1] - 44, y0 + 8, ("#ffffff", 1), ("#ffffff", 0.67), ("#ffffff", 0.8))
    cy = (y0 + y1) / 2
    harrow(476, COL["m"][0], cy)
    harrow(COL["m"][1], COL["t"][0], cy)
    harrow(COL["t"][1], COL["f"][0], cy)
    harrow(COL["f"][1], COL["dyn"][0], cy)
    harrow(COL["dyn"][1], COL["h"][0], cy)
    harrow(COL["h"][1], COL["d"][0], cy)

box(COL["dyn"][0], 81, COL["dyn"][1], 617, DYN, "", [
    "48 blocks · d = 1024 · 16 heads", "838 M parameters", "",
    ("1,209 tokens per frame", DYN[1]), "",
    "spatial attention: all tokens", "of a frame (cross-modal)", "",
    "temporal attention: causal,", "same token position, every", "earlier frame"],
    title_lines=["MaskGIT space–time", "transformer"])
g_attention(COL["dyn"][1] - 82, 91, DYN[0], (DYN[0], 0.47))
g_cuboids(COL["dyn"][1] - 50, 89, (DYN[0], 1), (DYN[1], 1), (DYN[0], 0.47))

AX0, AX1, AY0, AY1 = 827, 1358, 633, 807
box(AX0, AY0, AX1, AY1, ACT, "Actuators: 8 groups, 88 channels",
    ["ECH power; NBI power, voltage, torque; gas", "flow (calibrated, raw); C- and I-coil currents.",
     "Frame mean → Linear → added to every token", "of its frame, including the frame predicted"])
acy = (AY0 + AY1) / 2
harrow(476, AX0, acy)
dx = (COL["dyn"][0] + COL["dyn"][1]) / 2 - 60
line(AX1, acy, dx, acy, 2.4, INK)
line(dx, acy + 1.2, dx, 617 + 12, 2.4, INK)
arrow_head(dx, 617 + 1, "u")
rx = (COL["h"][0] + COL["h"][1]) / 2
for k, s in enumerate(["Rollout: 20 measured seed frames", "(1.00–1.95 s), then 80 predicted frames",
                       "(2.00–5.95 s), 10 MaskGIT steps each,", "appended to the context"]):
    text(s, rx + 22, 667 + 27 * k, 20, INK)
dx2 = (COL["dyn"][0] + COL["dyn"][1]) / 2 + 60
dashed([(rx, 617), (rx, 815), (dx2, 815), (dx2, 617 + 12)], MUTED)
arrow_head(dx2, 617 + 1, "u", MUTED)

# headers
for (n, title, sub), x in zip([(1, "DIII-D", "diagnostics · actuators"), (2, "MEASURED FRAME", "50 ms"),
                               (3, "TOKENIZER", "frozen"), (4, "FSQ", ""), (5, "DYNAMICS MODEL", "trained"),
                               (6, "HEADS", ""), (7, "DECODER", "frozen · mirrors tokenizer")],
                              [8, COL["m"][0], COL["t"][0], COL["f"][0], COL["dyn"][0], COL["h"][0], COL["d"][0]]):
    circle(x + 15, 29, 15, INK)
    text(str(n), x + 15, 35, 17, "#ffffff", bold=True, align="CENTER")
    text(title, x + 40, 36, 22, INK, bold=True)
    if sub:
        text(sub, x + 40, 62, 19, MUTED)

# ------------------------------------------------------------------ DIII-D cutaway + markers
layout = json.loads((HERE / "layout.json").read_text())
iw, ih = layout["crop_size"]
IW, TW = 390, 470
TH = IW * ih / iw
IX, IY = 9, 111
img = bpy.data.images.load(str(ROOT / "d3d_crop.png"))
m = bpy.data.materials.new("img")
m.use_nodes = True
nt = m.node_tree
nt.nodes.clear()
o = nt.nodes.new("ShaderNodeOutputMaterial")
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = img
em = nt.nodes.new("ShaderNodeEmission")
tr = nt.nodes.new("ShaderNodeBsdfTransparent")
mix = nt.nodes.new("ShaderNodeMixShader")
nt.links.new(tex.outputs["Color"], em.inputs["Color"])
nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
nt.links.new(tr.outputs[0], mix.inputs[1])
nt.links.new(em.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], o.inputs[0])
mesh = bpy.data.meshes.new("img")
zz = z()
mesh.from_pydata([(*P(IX, IY + TH), zz), (*P(IX + IW, IY + TH), zz), (*P(IX + IW, IY), zz), (*P(IX, IY), zz)],
                 [], [[0, 1, 2, 3]])
mesh.uv_layers.new()
for li, uv in zip(range(4), [(0, 0), (1, 0), (1, 1), (0, 1)]):
    mesh.uv_layers.active.data[li].uv = uv
obj = bpy.data.objects.new("img", mesh)
scene.collection.objects.link(obj)
obj.data.materials.append(m)

PT = layout["points"]
X = lambda n: IX + PT[n][0] * IW  # noqa: E731
Y = lambda n: IY + PT[n][1] * TH  # noqa: E731
for n in PT:
    if n.startswith("mag"):
        poly([(X(n) - 3.4, Y(n) - 3.4), (X(n) + 3.4, Y(n) - 3.4), (X(n) + 3.4, Y(n) + 3.4), (X(n) - 3.4, Y(n) + 3.4)], "#7a3e0e")
        poly([(X(n) - 2.6, Y(n) - 2.6), (X(n) + 2.6, Y(n) - 2.6), (X(n) + 2.6, Y(n) + 2.6), (X(n) - 2.6, Y(n) + 2.6)], "#df7f33")
for i in range(3):
    line(X(f"co2top{i}"), Y(f"co2top{i}"), X(f"co2bot{i}"), Y(f"co2bot{i}"), 2.4, "#8c6d4f")
for n in PT:
    x, y = X(n), Y(n)
    if n.startswith("ts"):
        line(x - 4, y - 4, x + 4, y + 4, 2.2, "#c8352b")
        line(x - 4, y + 4, x + 4, y - 4, 2.2, "#c8352b")
    if n.startswith("ece"):
        line(x - 4, y, x + 4, y, 2.2, "#4d3c98")
        line(x, y - 4, x, y + 4, 2.2, "#4d3c98")
    if n.startswith("cer"):
        line(x - 4, y + 5, x + 4, y + 5, 2.2, "#3f8a1f")
        line(x, y + 1, x, y + 9, 2.2, "#3f8a1f")


def label(x, y, s):
    t = text(s, IX + TW, y + 6, 18, INK, bold=True, align="RIGHT")
    t.data.body = s
    width = len(s) * 14.5
    if IX + TW - width - 6 > x:
        line(x, y, IX + TW - width - 6, y, 1.2, "#8c9099")


for i, c in enumerate(["#148f88", "#6f5bc0", "#df7f33"]):
    line(X(f"nbi_out{i}") + 40, Y(f"nbi_out{i}"), X(f"nbi_in{i}"), Y(f"nbi_in{i}"), 9, c, 0.55)
label(X("nbi_out0") + 40, Y("nbi_out1"), "NBI")
for i, c in enumerate(["#c8352b", "#4d3c98", "#3f8a1f", "#df7f33"]):
    line(X("ech_src") + 30, Y("ech_src") - 40, X(f"ech_tgt{i}"), Y(f"ech_tgt{i}"), 4, c, 0.75)
label(X("ech_src") + 30, Y("ech_src") - 40, "ECH")
for n in ("icoil_u", "icoil_l"):
    ring(X(n), Y(n), 6, 2.6, "#b35f1c")
label(X("icoil_u") + 6, Y("icoil_u"), "3D coils")
line(X("gas_src") + 26, Y("gas_src") + 22, X("gas_tgt"), Y("gas_tgt"), 6, "#9b3fb5", 0.6)
circle(X("gas_tgt"), Y("gas_tgt"), 11, "#9b3fb5", 0.35)
label(X("gas_src") + 26, Y("gas_src") + 22, "Gas valve")


def face_mean(f):
    xs = [IX + v[0] * IW for v in PT.values() if v[2] == f]
    return sum(xs) / len(xs)


text("Diagnostics", max(IX + 70, face_mean(1)), IY - 4, 22, INK, bold=True, align="CENTER")
text("Actuators", min(IX + IW - 60, face_mean(0)), IY - 4, 22, INK, bold=True, align="CENTER")
for i, (kind, name, col) in enumerate([("m", "Magnetic probe", "#df7f33"), ("p", "ECE", "#4d3c98"),
                                       ("x", "Thomson scattering", "#c8352b"), ("c", "CER spectroscopy", "#3f8a1f"),
                                       ("l", "CO₂ interferometer", "#8c6d4f")]):
    x, y = IX + 12 + (i % 2) * 230, IY + TH + 34 + (i // 2) * 28
    if kind == "m":
        poly([(x - 3, y - 9), (x + 4, y - 9), (x + 4, y - 2), (x - 3, y - 2)], col)
    if kind in "pc":
        line(x - 5, y - 5, x + 5, y - 5, 2.2, col)
        line(x, y - 10, x, y, 2.2, col)
    if kind == "x":
        line(x - 4, y - 9, x + 4, y - 1, 2.2, col)
        line(x - 4, y - 1, x + 4, y - 9, 2.2, col)
    if kind == "l":
        line(x - 6, y - 5, x + 6, y - 5, 2.4, col)
    text(name, x + 14, y, 18, MUTED, bold=True)

# ------------------------------------------------------------------ camera
cam = bpy.data.cameras.new("c")
cam.type = "ORTHO"
cam.ortho_scale = W * 0.01
co = bpy.data.objects.new("c", cam)
co.location = (W * 0.005, -H * 0.005, 10)
scene.collection.objects.link(co)
scene.camera = co
scene.render.filepath = OUT
bpy.ops.render.render(write_still=True)
