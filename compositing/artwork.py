from __future__ import annotations

from PIL import Image


def prepare_artwork(image: Image.Image, target_aspect: float) -> Image.Image:
    image = image.convert("RGBA")

    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        image = image.crop(bbox)

    width, height = image.size
    current_aspect = width / max(height, 1)

    if current_aspect > target_aspect:
        new_height = max(1, round(width / target_aspect))
        canvas = Image.new("RGBA", (width, new_height), (255, 255, 255, 0))
        canvas.alpha_composite(image, (0, (new_height - height) // 2))
        return canvas

    new_width = max(1, round(height * target_aspect))
    canvas = Image.new("RGBA", (new_width, height), (255, 255, 255, 0))
    canvas.alpha_composite(image, ((new_width - width) // 2, 0))
    return canvas
