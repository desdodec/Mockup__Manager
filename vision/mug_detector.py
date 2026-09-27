from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _runs(row: np.ndarray):
    xs=np.flatnonzero(row)
    if not len(xs): return []
    cuts=np.where(np.diff(xs)>1)[0]
    starts=np.r_[0,cuts+1]; ends=np.r_[cuts,len(xs)-1]
    return [(int(xs[a]),int(xs[b])) for a,b in zip(starts,ends)]


def _central_run(row: np.ndarray, cx: float):
    rr=_runs(row)
    if not rr:return None
    containing=[r for r in rr if r[0]<=cx<=r[1]]
    if containing:return max(containing,key=lambda r:r[1]-r[0])
    return min(rr,key=lambda r:min(abs(cx-r[0]),abs(cx-r[1])))


def _fit_cylinder(mask: np.ndarray, box, confidence: float) -> dict:
    """Fit a printable cylinder, using the instance mask only as evidence.

    The fitted cylinder is intentionally NOT allowed to follow handle
    protrusions. Width/axis come from the persistent central body and the UV
    mesh is an analytic cylindrical projection.
    """
    binary=(mask>.5).astype(np.uint8)
    H,W=binary.shape
    x1,y1,x2,y2=map(float,box); bw=x2-x1; bh=y2-y1

    # Estimate axis from distance-transform maxima in the middle of the object.
    dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    ylo=max(0,int(y1+bh*.16)); yhi=min(H,int(y1+bh*.86))
    centres=[]
    for y in range(ylo,yhi):
        row=dist[y]
        x=int(np.argmax(row))
        if row[x]>max(3,bw*.035): centres.append((y,x,float(row[x])))
    if len(centres)<8: raise ValueError("insufficient cylindrical body evidence")
    c=np.asarray(centres,float)
    # Robust axis: weighted median-like centre, then tiny linear lean fit.
    good=c[c[:,2]>=np.quantile(c[:,2],.45)]
    coef=np.polyfit(good[:,0],good[:,1],1,w=good[:,2])
    axis_x=lambda y: float(coef[0]*y+coef[1])

    # Measure central connected run per row; reject widths inflated by handle.
    samples=[]
    for y in range(ylo,yhi):
        run=_central_run(binary[y],axis_x(y))
        if not run:continue
        l,r=run; width=r-l+1
        if width>=bw*.28:samples.append((y,l,r,width))
    if len(samples)<10:raise ValueError("cannot establish mug body")
    a=np.asarray(samples,float)
    # Cylinder diameter is a robust lower-middle width. Handle attachment rows
    # tend to be wider and therefore cannot expand the fitted cylinder.
    diameter=float(np.quantile(a[:,3],.42))
    diameter=np.clip(diameter,bw*.38,bw*.78)

    # Vertical printable extent: below lip, above base. Use rows where the
    # central body is close to the fitted diameter.
    stable=a[(a[:,3]>=diameter*.88)&(a[:,3]<=diameter*1.18)]
    if len(stable)<6:stable=a
    top=float(np.quantile(stable[:,0],.08))
    bottom=float(np.quantile(stable[:,0],.94))
    top=max(top,y1+bh*.20); bottom=min(bottom,y2-bh*.10)
    if bottom-top<bh*.35:
        top=y1+bh*.24; bottom=y2-bh*.12

    # Handle side is inferred from mask mass outside the analytic cylinder.
    mid_y=(top+bottom)/2; mid_x=axis_x(mid_y); radius=diameter/2
    yy0=max(0,int(top)); yy1=min(H,int(bottom)+1)
    left_mass=binary[yy0:yy1,:max(0,int(mid_x-radius))].sum()
    right_mass=binary[yy0:yy1,min(W,int(mid_x+radius)):].sum()
    handle="left" if left_mass>right_mass*1.12 else "right" if right_mass>left_mass*1.12 else "unknown"

    # Print-safe cylinder is slightly narrower than the physical body.
    radius*=.90
    rows,cols=7,11
    # Analytic cylinder projection. Uniform angular UV samples compress toward
    # silhouette via sin(theta), unlike the old evenly-spaced screen columns.
    theta_max=np.deg2rad(72.0)
    angles=np.linspace(-theta_max,theta_max,cols)
    mesh=[]
    for iy in range(rows):
        y=top+(bottom-top)*iy/(rows-1)
        cx=axis_x(y)
        mesh.append([(cx+radius*np.sin(theta),y) for theta in angles])

    corners=(mesh[0][0],mesh[0][-1],mesh[-1][-1],mesh[-1][0])
    return {
        "corners":corners,"mesh":mesh,
        "axis":((axis_x(top),top),(axis_x(bottom),bottom)),
        "handle_side":handle,"curvature":0.0,"visible_fraction":0.40,
        "confidence":float(confidence),"bbox":(x1,y1,bw,bh),
        "detector":"fitted-cylinder",
        "cylinder":{"diameter":diameter,"top":top,"bottom":bottom},
    }


def _semantic_detect(image:Image.Image,expected_count):
    from ultralytics import YOLO
    model=YOLO("yolo11n-seg.pt")
    rgb=np.asarray(image.convert("RGB"))
    result=model.predict(source=rgb,conf=.20,imgsz=960,verbose=False)[0]
    if result.boxes is None or result.masks is None:return []
    masks=result.masks.data.cpu().numpy(); found=[]
    for i,box in enumerate(result.boxes):
        if str(result.names[int(box.cls[0].item())]).lower() not in {"cup","mug"}:continue
        xy=tuple(map(float,box.xyxy[0].tolist()))
        m=cv2.resize(masks[i],(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        found.append(_fit_cylinder(m,xy,float(box.conf[0].item())))
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
