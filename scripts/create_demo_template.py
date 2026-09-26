from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "assets" / "templates" / "demo_two_mugs"
SCENE_DIR.mkdir(parents=True, exist_ok=True)

w, h = 1600, 1200
image = Image.new("RGB", (w, h), (226, 216, 196))
draw = ImageDraw.Draw(image)

# Simple warm tabletop/background placeholder.
draw.rectangle((0, 760, w, h), fill=(156, 112, 76))
draw.ellipse((80, 850, 1520, 1140), fill=(136, 94, 64))

def mug(cx: int, cy: int, mw: int, mh: int):
    x0, y0 = cx - mw // 2, cy - mh // 2
    x1, y1 = cx + mw // 2, cy + mh // 2
    draw.rounded_rectangle((x0, y0, x1, y1), radius=55, fill=(244, 242, 235), outline=(190, 186, 177), width=5)
    draw.ellipse((x1 - 10, cy - 75, x1 + 110, cy + 75), outline=(205, 201, 193), width=34)
    draw.ellipse((x0 + 18, y0 - 8, x1 - 18, y0 + 38), fill=(225, 222, 214), outline=(178, 174, 167), width=3)
    draw.ellipse((x0 + 30, y0 + 4, x1 - 30, y0 + 30), fill=(91, 62, 44))

mug(520, 690, 330, 410)
mug(1060, 710, 320, 400)

image = image.filter(ImageFilter.GaussianBlur(0.25))
image.save(SCENE_DIR / "scene.png")

definition = {
    "id": "demo_two_mugs",
    "name": "Demo — two mugs",
    "image": "scene.png",
    "slots": [
        {
            "id": "mug_01",
            "label": "Left mug",
            "corners": [[405, 585], [630, 592], [620, 805], [414, 800]],
            "curvature": 0.28,
            "opacity": 0.94,
            "orientation": "front",
            "z_order": 10
        },
        {
            "id": "mug_02",
            "label": "Right mug",
            "corners": [[952, 610], [1168, 604], [1162, 812], [960, 818]],
            "curvature": 0.28,
            "opacity": 0.94,
            "orientation": "front",
            "z_order": 20
        }
    ]
}
(SCENE_DIR / "scene.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
print(f"Created {SCENE_DIR}")
