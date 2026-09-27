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
