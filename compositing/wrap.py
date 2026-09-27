from __future__ import annotations

import numpy as np
from PIL import Image


DEFAULT_VISIBLE_FRACTION = 136.0 / 360.0


def extract_visible_wrap(
    artwork: Image.Image,
    *,
    view_angle: float = 0.0,
    visible_fraction: float = DEFAULT_VISIBLE_FRACTION,
    output_width: int = 1200,
) -> Image.Image:
    """Sample the visible angular window from a complete mug-print canvas."""
    source = np.asarray(artwork.convert("RGBA"))
    h, w = source.shape[:2]
    if w < 2 or h < 1:
        return artwork.convert("RGBA")

    fraction = float(np.clip(visible_fraction, 0.05, 1.0))
    output_width = max(32, int(output_width))
    centre = (0.25 + (float(view_angle) / 360.0)) % 1.0

    screen_x = np.linspace(-1.0, 1.0, output_width, dtype=np.float32)
    angular = np.arcsin(np.clip(screen_x, -1.0, 1.0)) / (np.pi / 2.0)
    source_u = (centre + angular * fraction * 0.5) % 1.0
    source_x = np.clip(
        np.rint(source_u * (w - 1)), 0, w - 1
    ).astype(np.int32)

    return Image.fromarray(source[:, source_x, :], "RGBA")
