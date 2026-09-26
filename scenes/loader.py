from __future__ import annotations

import json
from pathlib import Path
from .models import MugSlot, Scene


def load_scene(scene_dir: Path) -> Scene:
    data = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    slots = []
    for raw in data["slots"]:
        mask = raw.get("print_mask")
        lighting = raw.get("lighting_map")
        orientation = raw.get("orientation", "front")
        default_angle = 180.0 if orientation == "rear" else 0.0
        slots.append(MugSlot(
            id=raw["id"],
            label=raw.get("label", raw["id"]),
            corners=tuple(tuple(map(float, p)) for p in raw["corners"]),
            curvature=float(raw.get("curvature", 0.20)),
            opacity=float(raw.get("opacity", 1.0)),
            view_angle=float(raw.get("view_angle", default_angle)),
            visible_fraction=float(raw.get("visible_fraction", 0.42)),
            print_mask=(scene_dir / mask) if mask else None,
            lighting_map=(scene_dir / lighting) if lighting else None,
            z_order=int(raw.get("z_order", 0)),
        ))
    return Scene(data["id"], data["name"], scene_dir / data["image"],
                 tuple(sorted(slots, key=lambda s: s.z_order)))
