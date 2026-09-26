from __future__ import annotations

from pathlib import Path

from .loader import load_scene
from .models import Scene


def discover_scenes(templates_dir: Path) -> dict[str, Scene]:
    scenes: dict[str, Scene] = {}
    if not templates_dir.exists():
        return scenes

    for definition in templates_dir.glob("*/scene.json"):
        scene = load_scene(definition.parent)
        scenes[scene.id] = scene
    return scenes
