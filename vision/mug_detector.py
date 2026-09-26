from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _longest_run(row: np.ndarray) -> tuple[int, int] | None:
    xs=np.flatnonzero(row)
    if len(xs)==0: return None
    cuts=np.where(np.diff(xs)>1)[0]
    starts=np.r_[0,cuts+1]; ends=np.r_[cuts,len(xs)-1]
    lengths=xs[ends]-xs[starts]+1
    j=int(np.argmax(lengths))
    return int(xs[starts[j]]),int(xs[ends[j]])


def _body_surface(mask: np.ndarray, box: tuple[float,float,float,float], confidence: float) -> dict:
    """Estimate the cylindrical ceramic body from an instance mask.

    Mug handles create side protrusions. The body is the vertically persistent
    central run of mask pixels; row medians/quantiles reject those protrusions.
    """
    H,W=mask.shape
    x1,y1,x2,y2=box
    bx1=max(0,int(x1)); bx2=min(W-1,int(x2))
    by1=max(0,int(y1)); by2=min(H-1,int(y2))
    crop=(mask[by1:by2+1,bx1:bx2+1]>0)
    ch,cw=crop.shape
    rows=[]
    # Ignore rim and very bottom while estimating stable body width.
    for yy in range(max(0,int(ch*.16)),max(1,int(ch*.90))):
        run=_longest_run(crop[yy])
        if run and run[1]-run[0] >= cw*.28:
            rows.append((yy,run[0],run[1]))
    if len(rows)<8:
        # conservative fallback inside semantic object, never outside its box
        l=x1+(x2-x1)*.25; r=x2-(x2-x1)*.25
        t=y1+(y2-y1)*.24; b=y2-(y2-y1)*.14
    else:
        arr=np.asarray(rows,float)
        widths=arr[:,2]-arr[:,1]
        # Body rows tend to share a broad, stable width. Handle-only excursions
        # are removed by quantiles rather than expanding the print region.
        stable=arr[widths>=np.quantile(widths,.45)]
        left=float(np.quantile(stable[:,1],.62))+bx1
        right=float(np.quantile(stable[:,2],.38))+bx1
        if right-left < (x2-x1)*.30:
            centre=(x1+x2)/2; half=(x2-x1)*.22
            left,right=centre-half,centre+half
        ys=stable[:,0]+by1
        t=float(np.quantile(ys,.10)); b=float(np.quantile(ys,.92))
        # printable safety inset: stay clear of lip/base where ceramic curvature
        # becomes strong and sublimation printing normally stops.
        l=left+(right-left)*.06; r=right-(right-left)*.06
        t=t+(b-t)*.06; b=b-(b-t)*.04

    # Mild trapezoid derived from body width near top/bottom can be added later;
    # for now mask-derived bounds are materially safer than the object bbox.
    corners=((l,t),(r,t),(r,b),(l,b))
    return {
        "corners":corners,
        "curvature":0.34,
        "visible_fraction":0.40,
        "confidence":float(confidence),
        "bbox":(float(x1),float(y1),float(x2-x1),float(y2-y1)),
        "detector":"semantic-mask",
    }


def _semantic_detect(image: Image.Image, expected_count: int | None) -> list[dict]:
    from ultralytics import YOLO
    model=YOLO("yolo11n-seg.pt")
    rgb=np.asarray(image.convert("RGB"))
    result=model.predict(source=rgb,conf=0.20,imgsz=960,verbose=False)[0]
    if result.boxes is None or result.masks is None: return []

    names=result.names
    # masks.data is model-resolution; resize each instance back to source pixels.
    mask_data=result.masks.data.cpu().numpy()
    found=[]
    for i,box in enumerate(result.boxes):
        cls_id=int(box.cls[0].item())
        if str(names[cls_id]).lower() not in {"cup","mug"}: continue
        conf=float(box.conf[0].item())
        x1,y1,x2,y2=[float(v) for v in box.xyxy[0].tolist()]
        m=cv2.resize(mask_data[i],(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)
        found.append(_body_surface(m,(x1,y1,x2,y2),conf))

    found.sort(key=lambda s:s["confidence"],reverse=True)
    if expected_count: found=found[:expected_count]
    found.sort(key=lambda s:s["bbox"][0])
    return found


def detect_mug_surfaces(image: Image.Image, expected_count: int | None=None) -> list[dict]:
    """Detect semantic mugs and derive printable bodies from instance masks."""
    try:
        detect_mug_surfaces.last_error=""
        return _semantic_detect(image,expected_count)
    except Exception as exc:
        detect_mug_surfaces.last_error=f"{type(exc).__name__}: {exc}"
        return []


detect_mug_surfaces.last_error=""
