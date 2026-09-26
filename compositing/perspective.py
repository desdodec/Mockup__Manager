from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def warp_to_canvas(
    image: Image.Image,
    corners: tuple[tuple[float, float], ...],
    canvas_size: tuple[int, int],
) -> Image.Image:
    rgba = np.array(image.convert("RGBA"))
    h, w = rgba.shape[:2]

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(corners)
    matrix = cv2.getPerspectiveTransform(src, dst)

    canvas_w, canvas_h = canvas_size
    warped = cv2.warpPerspective(
        rgba,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, "RGBA")
