from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def apply_scene_lighting(
    scene: Image.Image,
    print_layer: Image.Image,
    mask: Image.Image | None = None,
    lighting_map: Image.Image | None = None,
) -> Image.Image:
    scene_rgb = np.asarray(scene.convert("RGB"), dtype=np.float32) / 255.0
    layer = np.asarray(print_layer.convert("RGBA"), dtype=np.float32) / 255.0

    alpha = layer[..., 3]
    if mask is not None:
        mask_arr = np.asarray(mask.convert("L").resize(scene.size), dtype=np.float32) / 255.0
        alpha *= mask_arr

    # Preserve local ceramic luminance. The bounded range avoids crushing
    # colours in shadows or blowing out the print under highlights.
    scene_luma = (
        0.2126 * scene_rgb[..., 0]
        + 0.7152 * scene_rgb[..., 1]
        + 0.0722 * scene_rgb[..., 2]
    )
    modulation = np.clip(0.65 + 0.55 * scene_luma, 0.58, 1.18)

    if lighting_map is not None:
        light = np.asarray(
            lighting_map.convert("L").resize(scene.size), dtype=np.float32
        ) / 255.0
        modulation *= np.clip(0.70 + 0.60 * light, 0.55, 1.25)

    rgb = np.clip(layer[..., :3] * modulation[..., None], 0.0, 1.0)

    out = np.zeros_like(layer)
    out[..., :3] = rgb
    out[..., 3] = np.clip(alpha, 0.0, 1.0)

    return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA").filter(
        ImageFilter.GaussianBlur(radius=0.12)
    )
