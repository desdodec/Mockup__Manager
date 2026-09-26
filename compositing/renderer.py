from __future__ import annotations

from collections.abc import Mapping

from PIL import Image

from scenes.models import Scene

from .artwork import prepare_artwork
from .curvature import cylindrical_warp
from .lighting import apply_scene_lighting
from .perspective import warp_to_canvas


def _slot_aspect(corners: tuple[tuple[float, float], ...]) -> float:
    top = ((corners[1][0] - corners[0][0]) ** 2 + (corners[1][1] - corners[0][1]) ** 2) ** 0.5
    left = ((corners[3][0] - corners[0][0]) ** 2 + (corners[3][1] - corners[0][1]) ** 2) ** 0.5
    return max(top, 1.0) / max(left, 1.0)


def render_scene(
    scene: Scene,
    assignments: Mapping[str, Image.Image],
    *,
    slot_overrides: Mapping[str, dict] | None = None,
) -> Image.Image:
    base = Image.open(scene.image_path).convert("RGBA")
    overrides = slot_overrides or {}

    for slot in scene.slots:
        artwork = assignments.get(slot.id)
        if artwork is None:
            continue

        override = overrides.get(slot.id, {})
        curvature = float(override.get("curvature", slot.curvature))
        opacity = float(override.get("opacity", slot.opacity))
        corners = tuple(override.get("corners", slot.corners))

        prepared = prepare_artwork(artwork, _slot_aspect(corners))
        curved = cylindrical_warp(prepared, curvature)
        layer = warp_to_canvas(curved, corners, base.size)

        if opacity < 1.0:
            alpha = layer.getchannel("A").point(
                lambda value: int(value * max(0.0, min(opacity, 1.0)))
            )
            layer.putalpha(alpha)

        mask = Image.open(slot.print_mask) if slot.print_mask and slot.print_mask.exists() else None
        lighting = (
            Image.open(slot.lighting_map)
            if slot.lighting_map and slot.lighting_map.exists()
            else None
        )
        layer = apply_scene_lighting(base, layer, mask, lighting)
        base = Image.alpha_composite(base, layer)

    return base
