from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def cylindrical_warp(image: Image.Image, amount: float) -> Image.Image:
    amount = float(np.clip(amount, -0.95, 0.95))
    rgba = np.array(image.convert("RGBA"))
    height, width = rgba.shape[:2]

    if width < 2 or abs(amount) < 1e-5:
        return Image.fromarray(rgba, "RGBA")

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    strength = abs(amount)

    # Compress the artwork toward the visible side edges, approximating
    # cylindrical wrapping without altering the source artwork content itself.
    curved = np.sin(x * np.pi / 2.0) / (np.sin(np.pi / 2.0) or 1.0)
    source_x = (1.0 - strength) * x + strength * curved
    if amount < 0:
        source_x = -source_x[::-1]

    map_x = ((source_x + 1.0) * 0.5 * (width - 1)).astype(np.float32)
    map_x = np.tile(map_x, (height, 1))
    map_y = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))

    warped = cv2.remap(
        rgba,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, "RGBA")
