from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def warp_to_surface_mesh(image: Image.Image, mesh: list[list[tuple[float,float]]], canvas_size: tuple[int,int]) -> Image.Image:
    """Piecewise-project artwork through a row/column surface mesh."""
    src=np.asarray(image.convert("RGBA"))
    h,w=src.shape[:2]; cw,ch=canvas_size
    out=np.zeros((ch,cw,4),np.uint8)
    rows=len(mesh); cols=len(mesh[0])
    for iy in range(rows-1):
        sy0=iy*(h-1)/(rows-1); sy1=(iy+1)*(h-1)/(rows-1)
        for ix in range(cols-1):
            sx0=ix*(w-1)/(cols-1); sx1=(ix+1)*(w-1)/(cols-1)
            sp=np.float32([[sx0,sy0],[sx1,sy0],[sx1,sy1],[sx0,sy1]])
            dp=np.float32([mesh[iy][ix],mesh[iy][ix+1],mesh[iy+1][ix+1],mesh[iy+1][ix]])
            M=cv2.getPerspectiveTransform(sp,dp)
            tile=cv2.warpPerspective(src,M,(cw,ch),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
            alpha=tile[...,3:4].astype(np.float32)/255
            out[...,:3]=(tile[...,:3]*alpha+out[...,:3]*(1-alpha)).astype(np.uint8)
            out[...,3]=np.maximum(out[...,3],tile[...,3])
    return Image.fromarray(out,"RGBA")
