from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _canonical_surface(mask: np.ndarray, box, confidence: float) -> dict:
    """Fit a constrained upright 11oz mug model to a semantic mug detection.

    Generated scenes are deliberately constrained to upright straight-sided
    mugs. We therefore estimate scale/centre from the mask but never infer a
    free 3D axis from noisy silhouette pixels.
    """
    binary=(mask>.5).astype(np.uint8)
    H,W=binary.shape
    x1,y1,x2,y2=map(float,box); bw=x2-x1; bh=y2-y1

    # Distance-transform maximum locates the thick ceramic body and is much
    # less sensitive to the handle than the full bounding-box centre.
    dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    roi=dist[max(0,int(y1+bh*.15)):min(H,int(y2-bh*.10)),
             max(0,int(x1)):min(W,int(x2))]
    if not roi.size or roi.max()<=0: raise ValueError("empty mug body")
    ry,rx=np.unravel_index(np.argmax(roi),roi.shape)
    cx=float(max(0,int(x1))+rx)

    # Standard 11oz body: infer diameter from the thick central portion, but
    # clamp it to plausible proportions so a handle can never skew geometry.
    radius=float(roi[ry,rx])
    body_w=np.clip(radius*2.0,bw*.46,bw*.72)

    # Canonical print-safe vertical zone. These margins intentionally avoid the
    # rim and base curvature and remain stable across clean generated scenes.
    top=y1+bh*.25
    bottom=y2-bh*.14
    if bottom<=top: raise ValueError("invalid mug proportions")

    # Handle side is the only pose cue needed for constrained scenes.
    mid0=max(0,int(top)); mid1=min(H,int(bottom)+1)
    left=binary[mid0:mid1,:max(0,int(cx-body_w/2))].sum()
    right=binary[mid0:mid1,min(W,int(cx+body_w/2)):].sum()
    handle="left" if left>right*1.10 else "right" if right>left*1.10 else "unknown"

    # Keep the print zone safely inside the physical cylinder.
    radius=body_w*.46
    rows,cols=7,13
    theta_max=np.deg2rad(68.0)
    angles=np.linspace(-theta_max,theta_max,cols)
    xs=cx+radius*np.sin(angles)
    mesh=[]
    for iy in range(rows):
        y=top+(bottom-top)*iy/(rows-1)
        mesh.append([(float(x),float(y)) for x in xs])

    corners=(mesh[0][0],mesh[0][-1],mesh[-1][-1],mesh[-1][0])
    return {
        "corners":corners,"mesh":mesh,
        "axis":((cx,top),(cx,bottom)),
        "handle_side":handle,"curvature":0.0,"visible_fraction":0.40,
        "confidence":float(confidence),"bbox":(x1,y1,bw,bh),
        "detector":"canonical-11oz",
        "cylinder":{"diameter":body_w,"top":top,"bottom":bottom},
    }


def _semantic_detect(image:Image.Image,expected_count):
    from ultralytics import YOLO
    model=YOLO("yolo11n-seg.pt")
    rgb=np.asarray(image.convert("RGB"))
    result=model.predict(source=rgb,conf=.08,imgsz=1280,verbose=False)[0]
    if result.boxes is None or result.masks is None:return []
    masks=result.masks.data.cpu().numpy(); found=[]
    for i,box in enumerate(result.boxes):
        if str(result.names[int(box.cls[0].item())]).lower() not in {"cup","mug"}:continue
        xy=tuple(map(float,box.xyxy[0].tolist()))
        m=cv2.resize(masks[i],(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        found.append(_canonical_surface(m,xy,float(box.conf[0].item())))
    # Count-aware recovery: a clean generated scene may contain a mug the
    # full-frame pass misses. Retry overlapping crops where each mug occupies
    # substantially more detector pixels, then merge non-overlapping results.
    if expected_count and len(found)<expected_count:
        H,W=rgb.shape[:2]
        for xa,xb in [(0,int(W*.62)),(int(W*.38),W)]:
            crop=rgb[:,xa:xb]
            retry=model.predict(source=crop,conf=.06,imgsz=1280,verbose=False)[0]
            if retry.boxes is None or retry.masks is None: continue
            rm=retry.masks.data.cpu().numpy()
            for j,bx in enumerate(retry.boxes):
                if str(retry.names[int(bx.cls[0].item())]).lower() not in {"cup","mug"}: continue
                a,b,c,d=map(float,bx.xyxy[0].tolist()); global_box=(a+xa,b,c+xa,d)
                candidate_bbox=(a+xa,b,c-a,d-b)
                duplicate=False
                for old in found:
                    ox,oy,ow,oh=old["bbox"]
                    ix=max(0,min(c+xa,ox+ow)-max(a+xa,ox))
                    iy=max(0,min(d,oy+oh)-max(b,oy))
                    inter=ix*iy
                    union=(c-a)*(d-b)+ow*oh-inter
                    if inter/max(union,1)>.40: duplicate=True; break
                if duplicate: continue
                local=cv2.resize(rm[j],(xb-xa,H),interpolation=cv2.INTER_LINEAR)
                gm=np.zeros((H,W),np.float32); gm[:,xa:xb]=local
                try: found.append(_canonical_surface(gm,global_box,float(bx.conf[0].item())))
                except ValueError: pass

    found.sort(key=lambda s:s["confidence"],reverse=True)
    if expected_count:found=found[:expected_count]
    found.sort(key=lambda s:s["bbox"][0])
    return found


def detect_mug_surfaces(image:Image.Image,expected_count=None):
    try:
        detect_mug_surfaces.last_error=""
        return _semantic_detect(image,expected_count)
    except Exception as exc:
        detect_mug_surfaces.last_error=f"{type(exc).__name__}: {exc}"
        return []
detect_mug_surfaces.last_error=""
