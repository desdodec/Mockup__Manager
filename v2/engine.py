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
    marker_mask: bool = False
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
    if slot.marker_mask:
        top=y0*H
        bottom=y1*H
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

def _magenta_mask(scene:Image.Image, slot:Slot)->np.ndarray:
    """Clean anti-aliased printable silhouette for one marker mug."""
    rgb=np.asarray(scene.convert("RGB"))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    H,W=hsv.shape[:2]
    x0,y0,x1,y1=slot.box
    ix0=max(0,int(x0*W)-6); ix1=min(W,int(np.ceil(x1*W))+6)
    iy0=max(0,int(y0*H)-6); iy1=min(H,int(np.ceil(y1*H))+6)

    # Include pale/anti-aliased magenta edge pixels as well as saturated marker pixels.
    h=hsv[...,0]; sat=hsv[...,1]; val=hsv[...,2]
    hue_magenta=(h>=135)|(h<=2)
    chroma=hue_magenta & (sat>=28) & (val>=70)
    roi=np.zeros((H,W),np.uint8)
    roi[iy0:iy1,ix0:ix1]=(chroma[iy0:iy1,ix0:ix1].astype(np.uint8)*255)

    # Keep only the dominant connected marker component inside this slot.
    n,labels,stats,_=cv2.connectedComponentsWithStats(roi,8)
    if n<=1: return roi
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>20]
    if not candidates: return roi
    best=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    mask=np.where(labels==best,255,0).astype(np.uint8)

    # Fill tiny highlight holes, cover marker fringe, then feather only the outer edge.
    k=max(3,int(round(min(W,H)*0.004)))
    if k%2==0:k+=1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8),iterations=1)
    mask=cv2.GaussianBlur(mask,(0,0),sigmaX=0.8,sigmaY=0.8)
    return mask

def _apply_mask(layer:Image.Image, mask:np.ndarray)->Image.Image:
    a=np.asarray(layer.convert("RGBA")).copy()
    a[...,3]=np.minimum(a[...,3],mask)
    return Image.fromarray(a,"RGBA")

def _ceramic_blend(scene:Image.Image, layer:Image.Image, marker_mask:np.ndarray|None=None)->Image.Image:
    base=np.asarray(scene.convert("RGB")).astype(np.float32)/255
    lay=np.asarray(layer.convert("RGBA")).astype(np.float32)/255
    alpha=lay[...,3:4]
    lum=(.2126*base[...,0]+.7152*base[...,1]+.0722*base[...,2])[...,None]

    # Marker magenta is geometry metadata, never part of the finished ceramic.
    # Neutralise it first while retaining its luminance as surface lighting.
    clean_base=base.copy()
    if marker_mask is not None:
        m=(marker_mask.astype(np.float32)/255.0)[...,None]
        neutral=np.repeat(np.clip(.82+.18*lum,0,1),3,axis=2)
        clean_base=base*(1-m)+neutral*m

    shade=np.clip(.72+.38*lum,.72,1.08)
    rgb=np.clip(lay[...,:3]*shade,0,1)
    spec=np.clip((lum-.72)/.28,0,1)*0.10
    rgb=np.clip(rgb*(1-spec)+clean_base*spec,0,1)

    # Within marker regions the artwork replaces the marker rather than
    # translucently blending with it. Artwork alpha is still respected.
    strength=.94
    effective_alpha=alpha*strength
    if marker_mask is not None:
        m=(marker_mask.astype(np.float32)/255.0)[...,None]
        effective_alpha=np.where(m>0,alpha,effective_alpha)
    comp=clean_base*(1-effective_alpha)+rgb*effective_alpha
    return Image.fromarray((np.clip(comp,0,1)*255).astype(np.uint8),"RGB").convert("RGBA")

def render(template:Template, assignments:dict[str,tuple[Image.Image,str]])->Image.Image:
    scene=Image.open(template.scene_path).convert("RGBA")
    out=scene
    for slot in template.slots:
        item=assignments.get(slot.id)
        if not item:continue
        art,face=item
        layer=_project(art,out.size,slot,face)
        marker=None
        if slot.marker_mask:
            marker=_magenta_mask(scene,slot)
            layer=_apply_mask(layer,marker)
        out=_ceramic_blend(out,layer,marker)
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
