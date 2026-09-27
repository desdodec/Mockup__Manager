from __future__ import annotations

import numpy as np
from PIL import Image


def extract_visible_wrap(
    artwork: Image.Image,
    *,
    view_angle: float = 0.0,
    visible_fraction: float = 136.0 / 360.0,
    output_width: int = 1200,
) -> Image.Image:
    """Sample a visible cylindrical window from a complete mug print canvas.

    The source canvas is never cropped or rewritten on disk. Horizontal
    coordinates wrap cyclically, so 0 degrees shows the centre of the first
    half of the supplied print and 180 degrees shows the opposite side.
    """
    source = np.asarray(artwork.convert("RGBA"))
    h, w = source.shape[:2]
    if w < 2 or h < 1:
        return artwork.convert("RGBA")

    visible_fraction = float(np.clip(visible_fraction, 0.05, 1.0))
    output_width = max(32, int(output_width))

    # Treat the supplied image width as one complete circumference.
    # Front centre = 25% of the canvas; rear centre = 75%.
    centre = (0.25 + (view_angle / 360.0)) % 1.0
    # The visible fraction must describe the same angular span as the surface\n    # mesh. The standard mesh is +/-68 degrees => 136/360 circumference.\n    span = visible_fraction

    # Cylindrical sampling: equal screen-space steps correspond to increasingly
    # large source-angle changes toward the silhouette.
    screen_x = np.linspace(-1.0, 1.0, output_width, dtype=np.float32)
    angular = np.arcsin(np.clip(screen_x, -1.0, 1.0)) / (np.pi / 2.0)
    source_u = (centre + angular * span * 0.5) % 1.0
    source_x = np.clip(np.rint(source_u * (w - 1)), 0, w - 1).astype(np.int32)

    sampled = source[:, source_x, :]
    return Image.fromarray(sampled, "RGBA")
