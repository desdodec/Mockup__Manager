from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from PIL import Image
from openai import OpenAI

from .prompts import build_scene_prompt


@dataclass
class GeneratedScene:
    image: Image.Image
    provider_metadata: dict


class OpenAISceneGenerator:
    """Generates only blank-mug scene photography. Product artwork never enters this class."""

    def __init__(self, api_key: str, model: str = "gpt-image-2"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, mug_count: int, aspect_ratio: str) -> GeneratedScene:
        size_map = {
            "1:1": "1024x1024",
            "4:5": "1024x1536",
            "landscape": "1536x1024",
        }
        size = size_map[aspect_ratio]
        full_prompt = build_scene_prompt(prompt, mug_count)
        response = self.client.images.generate(
            model=self.model,
            prompt=full_prompt,
            size=size,
            quality="high",
            n=1,
        )
        raw = base64.b64decode(response.data[0].b64_json)
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
        return GeneratedScene(image=image, provider_metadata={"model": self.model, "size": size})
