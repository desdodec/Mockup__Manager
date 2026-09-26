from __future__ import annotations

from PIL import Image, ImageDraw

from compositing.curvature import cylindrical_warp
from compositing.renderer import render_scene
from config import TEMPLATES_DIR
from scenes.registry import discover_scenes
from scripts import create_demo_template  # noqa: F401


def test_cylindrical_warp_preserves_size_and_alpha():
    image = Image.new("RGBA", (400, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 25, 350, 175), fill=(20, 80, 180, 255))

    warped = cylindrical_warp(image, 0.3)

    assert warped.size == image.size
    assert warped.mode == "RGBA"
    assert warped.getchannel("A").getbbox() is not None


def test_demo_scene_renders_two_assignments():
    scenes = discover_scenes(TEMPLATES_DIR)
    scene = scenes["demo_two_mugs"]

    artwork = Image.new("RGBA", (800, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(artwork)
    draw.rectangle((20, 20, 780, 380), fill=(30, 90, 190, 255))

    result = render_scene(
        scene,
        {
            "mug_01": artwork,
            "mug_02": artwork,
        },
    )

    assert result.mode == "RGBA"
    assert result.size == (1600, 1200)
