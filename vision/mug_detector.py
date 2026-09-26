from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _mesh_from_mask(mask: np.ndarray, box, confidence: float) -> dict:
    x1,y1,x2,y2=map(float,box); H,W=mask.shape
    binary=(mask>.5).astype(np.uint8)
    # Distance transform favours the thick cylindrical body over the thinner
    # handle. Its peak gives a robust body centre even when the handle is large.
    dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    ys,xs=np.where(dist>0)
    if not len(xs): raise ValueError("empty mug mask")
    peak_y,peak_x=np.unravel_index(np.argmax(dist),dist.shape)
    body_half=max(12.0,float(dist[peak_y,peak_x])*.92)

    # Body vertical limits are derived around the centre column, not silhouette.
    band=binary[:,max(0,int(peak_x-body_half*.35)):min(W,int(peak_x+body_half*.35)+1)]
    occupancy=band.mean(axis=1) if band.shape[1] else np.zeros(H)
    valid=np.flatnonzero(occupancy>.45)
    if len(valid)<10:
        top=int(y1+(y2-y1)*.20); bottom=int(y2-(y2-y1)*.12)
    else:
        top=int(np.quantile(valid,.12)); bottom=int(np.quantile(valid,.90))
    top=max(top,int(y1+(y2-y1)*.14))
    bottom=min(bottom,int(y2-(y2-y1)*.06))
    if bottom<=top: bottom=max(top+10,int(y2-(y2-y1)*.1))

    # Detect handle side from mask mass outside the inferred central cylinder.
    yy0=max(0,int(y1)); yy1=min(H,int(y2)+1)
    left_mass=binary[yy0:yy1,:max(0,int(peak_x-body_half))].sum()
    right_mass=binary[yy0:yy1,min(W,int(peak_x+body_half)):].sum()
    handle_side="left" if left_mass>right_mass*1.15 else "right" if right_mass>left_mass*1.15 else "unknown"

    row_count=7; col_count=9
    mesh=[]
    boundaries=[]
    for iy in range(row_count):
        y=top+(bottom-top)*iy/(row_count-1)
        yi=int(np.clip(round(y),0,H-1))
        # Search contiguous mask run containing body centre. This excludes a
        # detached/necked handle instead of using the full object silhouette.
        row=binary[yi]
        cx=int(np.clip(peak_x,0,W-1))
        l=cx
        while l>0 and row[l-1]: l-=1
        r=cx
        while r<W-1 and row[r+1]: r+=1
        # Clamp pathological handle-connected runs around the body radius.
        l=max(float(l),peak_x-body_half*1.08)
        r=min(float(r),peak_x+body_half*1.08)
        inset=(r-l)*.055
        l+=inset; r-=inset
        boundaries.append((l,y,r))
        mesh.append([(l+(r-l)*ix/(col_count-1),y) for ix in range(col_count)])

    corners=(mesh[0][0],mesh[0][-1],mesh[-1][-1],mesh[-1][0])
    return {"corners":corners,"mesh":mesh,"axis":((float(peak_x),float(top)),(float(peak_x),float(bottom))),
            "handle_side":handle_side,"curvature":0.0,"visible_fraction":0.40,
            "confidence":float(confidence),"bbox":(x1,y1,x2-x1,y2-y1),"detector":"semantic-geometry"}


def _semantic_detect(image: Image.Image, expected_count):
    from ultralytics import YOLO
    model=YOLO("yolo11n-seg.pt")
    rgb=np.asarray(image.convert("RGB"))
    result=model.predict(source=rgb,conf=.20,imgsz=960,verbose=False)[0]
    if result.boxes is None or result.masks is None:return []
    masks=result.masks.data.cpu().numpy(); found=[]
    for i,box in enumerate(result.boxes):
        label=str(result.names[int(box.cls[0].item())]).lower()
        if label not in {"cup","mug"}:continue
        x1,y1,x2,y2=map(float,box.xyxy[0].tolist())
        m=cv2.resize(masks[i],(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        found.append(_mesh_from_mask(m,(x1,y1,x2,y2),float(box.conf[0].item())))
    found.sort(key=lambda s:s["confidence"],reverse=True)
    if expected_count:found=found[:expected_count]
    found.sort(key=lambda s:s["bbox"][0])
    return found


def detect_mug_surfaces(image: Image.Image,expected_count=None):
    try:
        detect_mug_surfaces.last_error=""
        return _semantic_detect(image,expected_count)
    except Exception as exc:
        detect_mug_surfaces.last_error=f"{type(exc).__name__}: {exc}"; return []
detect_mug_surfaces.last_error=""
