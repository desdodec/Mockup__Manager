from __future__ import annotations

import json
from pathlib import Path

from .models import MugSlot, Scene


def load_scene(scene_dir: Path) -> Scene:
    definition_path = scene_dir / "scene.json"
    data = json.loads(definition_path.read_text(encoding="utf-8"))

    slots: list[MugSlot] = []
    for raw in data["slots"]:
        mask = raw.get("print_mask")
        lighting = raw.get("lighting_map")
        slots.append(
            MugSlot(
                id=raw["id"],
                label=raw.get("label", raw["id"]),
                corners=tuple(tuple(map(float, point)) for point in raw["corners"]),
                curvature=float(raw.get("curvature", 0.20)),
                opacity=float(raw.get("opacity", 1.0)),
                orientation=raw.get("orientation", "front"),
                print_mask=(scene_dir / mask) if mask else None,
                lighting_map=(scene_dir / lighting) if lighting else None,
                z_order=int(raw.get("z_order", 0)),
            )
        )

    return Scene(
        id=data["id"],
        name=data["name"],
        image_path=scene_dir / data["image"],
        slots=tuple(sorted(slots, key=lambda slot: slot.z_order)),
    )
