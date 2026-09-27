from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def warp_to_surface_mesh(image: Image.Image, mesh: list[list[tuple[float,float]]], canvas_size: tuple[int,int]) -> Image.Image:
    """Piecewise-project artwork through a surface mesh without repeating tiles."""
    src=np.asarray(image.convert("RGBA"))
    h,w=src.shape[:2]; cw,ch=canvas_size
    out=np.zeros((ch,cw,4),np.uint8)
    rows=len(mesh); cols=len(mesh[0])

    for iy in range(rows-1):
        sy0=int(round(iy*h/(rows-1)))
        sy1=int(round((iy+1)*h/(rows-1)))
        if iy==rows-2: sy1=h
        for ix in range(cols-1):
            sx0=int(round(ix*w/(cols-1)))
            sx1=int(round((ix+1)*w/(cols-1)))
            if ix==cols-2: sx1=w
            if sx1<=sx0 or sy1<=sy0: continue

            # Critical: warp only this source cell. Warping the full source
            # with cell-local corner coordinates repeats the artwork per cell.
            crop=src[sy0:sy1,sx0:sx1]
            th,tw=crop.shape[:2]
            sp=np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]])
            dp=np.float32([mesh[iy][ix],mesh[iy][ix+1],mesh[iy+1][ix+1],mesh[iy+1][ix]])
            M=cv2.getPerspectiveTransform(sp,dp)
            tile=cv2.warpPerspective(crop,M,(cw,ch),flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))

            a=tile[...,3:4].astype(np.float32)/255.0
            oa=out[...,3:4].astype(np.float32)/255.0
            combined=a+oa*(1-a)
            rgb=np.where(combined>1e-6,
                (tile[...,:3].astype(np.float32)*a+
                 out[...,:3].astype(np.float32)*oa*(1-a))/np.maximum(combined,1e-6),
                0)
            out[...,:3]=np.clip(rgb,0,255).astype(np.uint8)
            out[...,3]=np.clip(combined[...,0]*255,0,255).astype(np.uint8)

    return Image.fromarray(out,"RGBA")


def warp_to_cylinder(image: Image.Image, mesh: list[list[tuple[float,float]]], canvas_size: tuple[int,int]) -> Image.Image:
    """Continuously map an angular artwork strip onto a straight cylindrical mug.

    The mesh is used only to recover the photographed cylinder envelope. The
    actual resampling is one continuous inverse map, avoiding piecewise seams.
    """
    src=np.asarray(image.convert("RGBA"))
    sh,sw=src.shape[:2]; cw,ch=canvas_size
    pts=np.asarray(mesh,dtype=np.float32)
    left=float(np.median(pts[:,:,0].min(axis=1)))
    right=float(np.median(pts[:,:,0].max(axis=1)))
    top=float(np.median(pts[0,:,1]))
    bottom=float(np.median(pts[-1,:,1]))
    cx=(left+right)*.5; radius=max((right-left)*.5,1.0)
    x0=max(0,int(np.floor(left))); x1=min(cw,int(np.ceil(right))+1)
    y0=max(0,int(np.floor(top))); y1=min(ch,int(np.ceil(bottom))+1)
    if x1<=x0 or y1<=y0:return Image.new("RGBA",(cw,ch),(0,0,0,0))

    yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
    xn=np.clip((xx-cx)/radius,-1.0,1.0)
    # Destination screen x = sin(theta). Source strip is linear in theta.
    theta=np.arcsin(xn)
    theta_max=np.arcsin(np.clip((right-cx)/radius,-1.0,1.0))
    theta_max=max(float(abs(theta_max)),1e-6)
    map_x=(theta/(2*theta_max)+.5)*(sw-1)
    map_y=((yy-top)/max(bottom-top,1.0))*(sh-1)
    valid=(xx>=left)&(xx<=right)&(yy>=top)&(yy<=bottom)
    map_x=np.where(valid,map_x,-1).astype(np.float32)
    map_y=np.where(valid,map_y,-1).astype(np.float32)
    warped=cv2.remap(src,map_x,map_y,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    out=np.zeros((ch,cw,4),np.uint8); out[y0:y1,x0:x1]=warped
    return Image.fromarray(out,"RGBA")
