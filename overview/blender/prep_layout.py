"""Crop of the DIII-D render and marker positions (fractions of the crop) for the Blender overview."""
import json
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parent.parent
im = Image.open(HERE / "d3d_render.png")
x0, y0, x1, y1 = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
W0, H0 = im.size
crop = Image.open(HERE / "d3d_crop.png")
pts = json.loads((HERE / "d3d_render.points.json").read_text())
out = {"crop_size": crop.size, "points": {p["name"]: [(p["u"] * W0 - x0) / (x1 - x0), (p["v"] * H0 - y0) / (y1 - y0), p["face"]] for p in pts}}
(HERE / "blender" / "layout.json").write_text(json.dumps(out))
print(out["crop_size"], len(out["points"]))
