from __future__ import annotations
from pathlib import Path
from PIL import Image
from .models import MugSlot,Scene
def make_runtime_scene(image:Image.Image,slots:list[dict],directory:Path)->Scene:
    directory.mkdir(parents=True,exist_ok=True); path=directory/"generated_scene.png"; image.convert("RGB").save(path,"PNG")
    ms=[]
    for i,s in enumerate(slots):
        mesh=s.get("mesh")
        ms.append(MugSlot(id=f"mug_{i+1:02d}",label=f"Mug {i+1}",corners=tuple(tuple(map(float,p)) for p in s["corners"]),
          curvature=float(s.get("curvature",.28)),opacity=float(s.get("opacity",.94)),view_angle=float(s.get("view_angle",0)),
          visible_fraction=float(s.get("visible_fraction",.42)),z_order=(i+1)*10,
          mesh=tuple(tuple(tuple(map(float,p)) for p in row) for row in mesh) if mesh else None,
          handle_side=s.get("handle_side","unknown")))
    return Scene("generated_scene","Generated scene",path,tuple(ms))
