from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class GeneratedScene:
    image: Image.Image
    provider_metadata: dict


class SceneGenerator:
    """Provider-neutral adapter for later image-generation API integration."""

    def generate(self, prompt: str, width: int, height: int) -> GeneratedScene:
        raise NotImplementedError(
            "No image-generation provider configured. "
            "The MVP keeps AI scene generation behind this adapter so product "
            "artwork is never sent through the scene generator."
        )
