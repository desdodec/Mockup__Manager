from __future__ import annotations

from PIL import Image, ImageDraw


def rectangle_corners(cx: int, cy: int, width: int, height: int):
    hw, hh = width / 2, height / 2
    return (
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    )


def draw_calibration_overlay(image: Image.Image, slots: list[dict]) -> Image.Image:
    out = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, slot in enumerate(slots, 1):
        corners = slot["corners"]
        polygon = [tuple(p) for p in corners]
        draw.polygon(polygon, fill=(30, 144, 255, 45), outline=(30, 144, 255, 255), width=4)
        for x, y in polygon:
            r = 8
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 255, 255, 255), outline=(30, 144, 255, 255), width=3)
        x, y = polygon[0]
        draw.rounded_rectangle((x, y-34, x+92, y-6), radius=6, fill=(0, 0, 0, 190))
        draw.text((x+8, y-29), f"Mug {index}", fill="white")
    return Image.alpha_composite(out, overlay)


def build_scene_definition(scene_id: str, name: str, image_name: str, slots: list[dict]) -> dict:
    return {
        "id": scene_id,
        "name": name,
        "image": image_name,
        "slots": [
            {
                "id": f"mug_{i:02d}",
                "label": f"Mug {i}",
                "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in s["corners"]],
                "curvature": float(s.get("curvature", 0.28)),
                "opacity": float(s.get("opacity", 0.94)),
                "view_angle": float(s.get("view_angle", 0.0)),
                "visible_fraction": float(s.get("visible_fraction", 0.42)),
                "z_order": i * 10,
            }
            for i, s in enumerate(slots, 1)
        ],
    }
