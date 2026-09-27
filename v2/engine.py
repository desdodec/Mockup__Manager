from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance

@dataclass(frozen=True)
class Slot:
    id: str
    box: tuple[float,float,float,float]  # normalized x0,y0,x1,y1
    yaw_deg: float = 0.0
    print_top: float = 0.07
    print_bottom: float = 0.91
    visible_deg: float = 136.0
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    ink_strength: float = 0.94

@dataclass(frozen=True)
class Template:
    id: str
    name: str
    scene_path: Path
    slots: tuple[Slot,...]

def _sample_wrap(art:Image.Image, centre_u:float, visible_deg:float, width:int)->np.ndarray:
    a=np.asarray(art.convert("RGBA"))
    h,w=a.shape[:2]
    theta=np.linspace(-visible_deg/2,visible_deg/2,width,dtype=np.float32)
    u=(centre_u+theta/360.0)%1.0
    xs=np.mod(np.rint(u*w).astype(np.int32),w)
    return a[:,xs]

def _project(art:Image.Image, size:tuple[int,int], slot:Slot, face:str)->Image.Image:
    W,H=size
    x0,y0,x1,y1=slot.box
    left,right=x0*W,x1*W
    top=(y0+(y1-y0)*slot.print_top)*H
    bottom=(y0+(y1-y0)*slot.print_bottom)*H
    cx=(left+right)/2
    radius=max((right-left)/2,1)
    ix0=max(0,int(left)); ix1=min(W,int(np.ceil(right)))
    iy0=max(0,int(top)); iy1=min(H,int(np.ceil(bottom)))
    if ix1<=ix0 or iy1<=iy0:return Image.new("RGBA",size,(0,0,0,0))
    centre=.25 if face.lower()=="front" else .75
    centre=(centre+slot.yaw_deg/360.0+slot.offset_x)%1.0
    strip=_sample_wrap(art,centre,slot.visible_deg,1400)
    sh,sw=strip.shape[:2]
    yy,xx=np.mgrid[iy0:iy1,ix0:ix1].astype(np.float32)
    xn=np.clip((xx-cx)/radius,-.9999,.9999)
    theta=np.arcsin(xn)
    limit=np.deg2rad(slot.visible_deg/2)
    mapx=((theta+limit)/(2*limit)*(sw-1)).astype(np.float32)
    v=((yy-top)/max(bottom-top,1)-.5)/max(slot.scale,0.05)+.5-slot.offset_y
    mapy=(v*(sh-1)).astype(np.float32)
    valid=(np.abs(theta)<=limit)&(yy>=top)&(yy<=bottom)
    mapx=np.where(valid,mapx,-1).astype(np.float32)
    mapy=np.where(valid,mapy,-1).astype(np.float32)
    warped=cv2.remap(strip,mapx,mapy,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    out=np.zeros((H,W,4),np.uint8); out[iy0:iy1,ix0:ix1]=warped
    return Image.fromarray(out,"RGBA")

def _ceramic_blend(scene:Image.Image, layer:Image.Image)->Image.Image:
    base=np.asarray(scene.convert("RGB")).astype(np.float32)/255
    lay=np.asarray(layer.convert("RGBA")).astype(np.float32)/255
    alpha=lay[...,3:4]
    lum=(.2126*base[...,0]+.7152*base[...,1]+.0722*base[...,2])[...,None]
    # Retain scene shading/highlights without altering artwork geometry.
    shade=np.clip(.72+.38*lum,.72,1.08)
    # Sublimation ink inherits ceramic illumination; retain a controlled amount
    # of the original mug highlight instead of painting an opaque sticker.
    rgb=np.clip(lay[...,:3]*shade,0,1)
    spec=np.clip((lum-.72)/.28,0,1)*0.10
    rgb=np.clip(rgb*(1-spec)+base*spec,0,1)
    strength=.94
    effective_alpha=alpha*strength
    comp=base*(1-effective_alpha)+rgb*effective_alpha
    return Image.fromarray((comp*255).astype(np.uint8),"RGB").convert("RGBA")

def render(template:Template, assignments:dict[str,tuple[Image.Image,str]])->Image.Image:
    scene=Image.open(template.scene_path).convert("RGBA")
    out=scene
    for slot in template.slots:
        item=assignments.get(slot.id)
        if not item:continue
        art,face=item
        layer=_project(art,out.size,slot,face)
        out=_ceramic_blend(out,layer)
    return out

def render_batch(template:Template, artworks:list[tuple[str,Image.Image]], face:str="front")->list[tuple[str,Image.Image]]:
    results=[]
    n=len(template.slots)
    for i in range(0,len(artworks),n):
        group=artworks[i:i+n]
        assignments={template.slots[j].id:(img,face) for j,(_,img) in enumerate(group)}
        label="__".join(name.rsplit(".",1)[0] for name,_ in group)
        results.append((label,render(template,assignments)))
    return results
