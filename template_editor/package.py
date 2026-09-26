from __future__ import annotations

import io
import json
import re
import zipfile
from PIL import Image
from .calibration import build_scene_definition


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return value or "custom_scene"


def create_template_zip(name: str, scene_image: Image.Image, slots: list[dict]) -> tuple[str, bytes]:
    scene_id = slugify(name)
    definition = build_scene_definition(scene_id, name, "scene.png", slots)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        img = io.BytesIO()
        scene_image.convert("RGB").save(img, "PNG")
        archive.writestr(f"{scene_id}/scene.png", img.getvalue())
        archive.writestr(f"{scene_id}/scene.json", json.dumps(definition, indent=2))
    return f"{scene_id}.zip", buffer.getvalue()
