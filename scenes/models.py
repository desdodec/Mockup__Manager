from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

Point = tuple[float, float]


@dataclass(frozen=True)
class MugSlot:
    id: str
    label: str
    corners: tuple[Point, Point, Point, Point]
    curvature: float = 0.20
    opacity: float = 1.0
    view_angle: float = 0.0
    visible_fraction: float = 0.42
    print_mask: Path | None = None
    lighting_map: Path | None = None
    z_order: int = 0


@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    image_path: Path
    slots: tuple[MugSlot, ...] = field(default_factory=tuple)
