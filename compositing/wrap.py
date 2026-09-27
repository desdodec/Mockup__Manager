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
    centre_u: float | None = None,
) -> Image.Image:
    """Extract an angular window from a complete mug-print circumference.

    Important: sampling here is linear in CYLINDER ANGLE. The surface mesh
    performs the angle-to-screen sin() projection. Applying arcsin() here as
    well would curve the artwork twice.
    """
    source = np.asarray(artwork.convert("RGBA"))
    h, w = source.shape[:2]
    if w < 2 or h < 1:
        return artwork.convert("RGBA")

    fraction = float(np.clip(visible_fraction, 0.05, 1.0))
    output_width = max(32, int(output_width))

    # Production canvases: front centre=25%, rear centre=75%.
    # centre_u is available for diagnostic/calibration sheets whose physical
    # centre is elsewhere (the supplied calibration grid uses 50%).
    centre = (
        float(centre_u) % 1.0
        if centre_u is not None
        else (0.25 + (float(view_angle) / 360.0)) % 1.0
    )

    angular = np.linspace(-1.0, 1.0, output_width, dtype=np.float32)
    source_u = (centre + angular * fraction * 0.5) % 1.0
    source_x = np.mod(
        np.rint(source_u * w).astype(np.int64), w
    ).astype(np.int32)

    return Image.fromarray(source[:, source_x, :], "RGBA")
