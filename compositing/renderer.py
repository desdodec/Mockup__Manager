from __future__ import annotations

from collections.abc import Mapping

from PIL import Image

from scenes.models import Scene
from .curvature import cylindrical_warp
from .lighting import apply_scene_lighting
from .perspective import warp_to_canvas
from .wrap import extract_visible_wrap


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
        view_angle = float(override.get("view_angle", slot.view_angle))
        visible_fraction = float(override.get("visible_fraction", slot.visible_fraction))
        corners = tuple(override.get("corners", slot.corners))

        # IMPORTANT: artwork is a complete mug-print canvas. Do not trim it or
        # fit the whole sheet onto the visible face. Sample the portion of the
        # circumference that the camera can see.
        visible = extract_visible_wrap(
            artwork,
            view_angle=view_angle,
            visible_fraction=visible_fraction,
        )
        curved = cylindrical_warp(visible, curvature)
        layer = warp_to_canvas(curved, corners, base.size)

        if opacity < 1.0:
            alpha = layer.getchannel("A").point(
                lambda value: int(value * max(0.0, min(opacity, 1.0)))
            )
            layer.putalpha(alpha)

        mask = Image.open(slot.print_mask) if slot.print_mask and slot.print_mask.exists() else None
        lighting = Image.open(slot.lighting_map) if slot.lighting_map and slot.lighting_map.exists() else None
        layer = apply_scene_lighting(base, layer, mask, lighting)
        base = Image.alpha_composite(base, layer)

    return base
