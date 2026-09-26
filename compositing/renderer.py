from __future__ import annotations
from collections.abc import Mapping
from PIL import Image
from scenes.models import Scene
from .curvature import cylindrical_warp
from .lighting import apply_scene_lighting
from .perspective import warp_to_canvas
from .mesh import warp_to_surface_mesh
from .wrap import extract_visible_wrap

def render_scene(scene:Scene,assignments:Mapping[str,Image.Image],*,slot_overrides:Mapping[str,dict]|None=None)->Image.Image:
    base=Image.open(scene.image_path).convert("RGBA"); overrides=slot_overrides or {}
    for slot in scene.slots:
        artwork=assignments.get(slot.id)
        if artwork is None:continue
        o=overrides.get(slot.id,{})
        opacity=float(o.get("opacity",slot.opacity)); angle=float(o.get("view_angle",slot.view_angle))
        visible_fraction=float(o.get("visible_fraction",slot.visible_fraction))
        visible=extract_visible_wrap(artwork,view_angle=angle,visible_fraction=visible_fraction)
        if slot.mesh:
            # Mesh already models photographed body geometry. Do not apply the
            # old second curvature transform (which double-warped the artwork).
            layer=warp_to_surface_mesh(visible,[list(r) for r in slot.mesh],base.size)
        else:
            curved=cylindrical_warp(visible,float(o.get("curvature",slot.curvature)))
            layer=warp_to_canvas(curved,tuple(o.get("corners",slot.corners)),base.size)
        if opacity<1:
            a=layer.getchannel("A").point(lambda v:int(v*max(0,min(opacity,1)))); layer.putalpha(a)
        mask=Image.open(slot.print_mask) if slot.print_mask and slot.print_mask.exists() else None
        lighting=Image.open(slot.lighting_map) if slot.lighting_map and slot.lighting_map.exists() else None
        layer=apply_scene_lighting(base,layer,mask,lighting); base=Image.alpha_composite(base,layer)
    return base
