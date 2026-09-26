from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _white_mask(rgb: np.ndarray) -> np.ndarray:
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # White ceramic: low saturation and reasonably bright. Morphology joins
    # highlight/shadow regions while rejecting small background details.
    mask=cv2.inRange(hsv,np.array([0,0,105],np.uint8),np.array([179,92,255],np.uint8))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1)
    return mask


def detect_mug_surfaces(image: Image.Image, expected_count: int | None=None) -> list[dict]:
    """Conservative CV proposal detector for blank white mug bodies.

    Returns print-surface proposals sorted left-to-right. It intentionally
    proposes geometry rather than altering product artwork.
    """
    rgb=np.asarray(image.convert("RGB"))
    h,w=rgb.shape[:2]
    mask=_white_mask(rgb)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    image_area=w*h
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        area=cv2.contourArea(c)
        if area < image_area*0.003 or area > image_area*0.28: continue
        aspect=bw/max(bh,1)
        if not 0.45 <= aspect <= 2.4: continue
        fill=area/max(bw*bh,1)
        if fill < 0.38: continue
        # Use the central body, excluding rim/base and much of the handle.
        px=x+bw*0.16; py=y+bh*0.20
        pw=bw*0.68; ph=bh*0.62
        corners=((px,py),(px+pw,py),(px+pw,py+ph),(px,py+ph))
        confidence=min(0.95,0.42+0.35*fill+0.18*min(area/(image_area*0.03),1))
        candidates.append({"corners":corners,"curvature":0.30,"visible_fraction":0.42,"confidence":confidence,"bbox":(x,y,bw,bh)})
    candidates.sort(key=lambda s:s["bbox"][0])
    if expected_count:
        candidates=sorted(candidates,key=lambda s:s["confidence"],reverse=True)[:expected_count]
        candidates.sort(key=lambda s:s["bbox"][0])
    return candidates
