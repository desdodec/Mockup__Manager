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



def _iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    x1=max(ax,bx); y1=max(ay,by); x2=min(ax+aw,bx+bw); y2=min(ay+ah,by+bh)
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(aw*ah+bw*bh-inter,1)

def _white_body_candidates(rgb, expected_count):
    """Detect constrained blank mug bodies directly, without COCO semantics."""
    H,W=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([0,0,145],np.uint8),np.array([179,62,255],np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(21,9)),iterations=2)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if h < H*.22 or w < W*.08 or h <= w*.65: continue
        if area/(w*h) < .48: continue
        # Handles may join the component. Estimate the solid cylindrical core
        # from horizontal occupancy around the middle of the mug.
        sub=(labels[y:y+h,x:x+w]==k).astype(np.uint8)
        widths=[]
        centres=[]
        for yy in range(int(h*.20),int(h*.82)):
            xs=np.flatnonzero(sub[yy])
            if len(xs)<w*.25: continue
            # Longest dense run is the body; handle is separated by thin necks.
            runs=np.split(xs,np.where(np.diff(xs)>2)[0]+1)
            run=max(runs,key=len)
            if len(run)>=w*.25:
                widths.append(len(run)); centres.append((run[0]+run[-1])/2+x)
        if not widths: continue
        body_w=float(np.percentile(widths,35))
        cx=float(np.median(centres))
        body_h=float(h)
        # Physical straight-sided cylinder: close to full ceramic body, not a
        # shrunken print-safe box. Artwork transparency supplies print margins.
        top=float(y+h*.08); bottom=float(y+h*.91)
        left=cx-body_w*.48; right=cx+body_w*.48
        gm=np.zeros((H,W),np.float32)
        cv2.rectangle(gm,(max(0,int(left)),max(0,int(top))),(min(W-1,int(right)),min(H-1,int(bottom))),1,-1)
        try:
            p=_canonical_surface(gm,(left,top,right,bottom),.82)
            # Override the old safe-zone fitter with physical cylinder geometry.
            rows,cols=7,13; theta=np.deg2rad(68); rad=(right-left)/2/np.sin(theta)
            xs2=cx+rad*np.sin(np.linspace(-theta,theta,cols))
            mesh=[[(float(xx),float(top+(bottom-top)*iy/(rows-1))) for xx in xs2] for iy in range(rows)]
            p.update({"mesh":mesh,"corners":(mesh[0][0],mesh[0][-1],mesh[-1][-1],mesh[-1][0]),
                      "axis":((cx,top),(cx,bottom)),"visible_fraction":.48,
                      "bbox":(left,top,right-left,bottom-top),"detector":"blank-mug-body"})
            candidates.append(p)
        except ValueError: pass
    candidates.sort(key=lambda p:p["bbox"][2]*p["bbox"][3],reverse=True)
    if expected_count: candidates=candidates[:expected_count]
    candidates.sort(key=lambda p:p["bbox"][0])
    return candidates

def _recover_canonical_bodies(rgb, found, expected_count):
    """Recover missing mugs only in constrained blank-mug scenes.

    Uses existing semantic detections as scale/vertical priors, then searches
    unexplained image regions for large bright low-saturation upright bodies.
    """
    if not expected_count or len(found)>=expected_count or not found:return found
    H,W=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    white=cv2.inRange(hsv,np.array([0,0,150],np.uint8),np.array([179,70,255],np.uint8))
    white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)),iterations=2)
    ref_h=float(np.median([p["bbox"][3] for p in found]))
    ref_w=float(np.median([p["cylinder"]["diameter"] for p in found]))
    contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        if h<ref_h*.55 or h>ref_h*1.35 or w<ref_w*.55 or w>ref_w*1.45:continue
        if h<=w*.75:continue
        bbox=(float(x),float(y),float(w),float(h))
        if any(_iou(bbox,p["bbox"])>.18 for p in found):continue
        fill=cv2.contourArea(cnt)/max(w*h,1)
        if fill<.45:continue
        # Build a synthetic canonical body mask. This fallback is permitted only
        # because generated scenes enforce upright blank straight-sided mugs.
        gm=np.zeros((H,W),np.float32)
        cv2.rectangle(gm,(x,y),(x+w,y+h),1.0,-1)
        score=min(.79,.48+.30*fill)
        try:candidates.append(_canonical_surface(gm,(x,y,x+w,y+h),score))
        except ValueError:pass
    candidates.sort(key=lambda p:p["confidence"],reverse=True)
    for p in candidates:
        if len(found)>=expected_count:break
        if not any(_iou(p["bbox"],q["bbox"])>.18 for q in found):found.append(p)
    return found

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

    found=_recover_canonical_bodies(rgb,found,expected_count)
    found.sort(key=lambda s:s["confidence"],reverse=True)
    if expected_count:found=found[:expected_count]
    found.sort(key=lambda s:s["bbox"][0])
    return found


def detect_mug_surfaces(image:Image.Image,expected_count=None):
    try:
        detect_mug_surfaces.last_error=""
        rgb=np.asarray(image.convert("RGB"))
        # Generated scenes are deliberately constrained to large blank white
        # upright mugs, so body geometry is more useful than generic COCO class.
        direct=_white_body_candidates(rgb,expected_count)
        if expected_count and len(direct)==expected_count:return direct
        semantic=_semantic_detect(image,expected_count)
        return semantic if len(semantic)>=len(direct) else direct
    except Exception as exc:
        detect_mug_surfaces.last_error=f"{type(exc).__name__}: {exc}"
        return []
detect_mug_surfaces.last_error=""
