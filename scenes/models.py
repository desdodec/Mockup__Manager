from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Point = tuple[float, float]


@dataclass(frozen=True)
class MugSlot:
    id: str
    label: str
    corners: tuple[Point, Point, Point, Point]
    curvature: float = 0.20
    opacity: float = 1.0
    orientation: Literal["front", "rear"] = "front"
    print_mask: Path | None = None
    lighting_map: Path | None = None
    z_order: int = 0


@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    image_path: Path
    slots: tuple[MugSlot, ...] = field(default_factory=tuple)
