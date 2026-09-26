from __future__ import annotations

import numpy as np
from PIL import Image


def _surface_from_box(x1: float, y1: float, x2: float, y2: float, confidence: float) -> dict:
    """Convert a semantic mug box to the central printable ceramic body.

    COCO's cup class includes the handle, so the print surface is deliberately
    inset. This is a geometry proposal; artwork remains untouched.
    """
    w=max(1.0,x2-x1); h=max(1.0,y2-y1)
    left=x1+w*0.18; right=x2-w*0.18
    top=y1+h*0.22; bottom=y2-h*0.12
    return {
        "corners":((left,top),(right,top),(right,bottom),(left,bottom)),
        "curvature":0.30,
        "visible_fraction":0.42,
        "confidence":float(confidence),
        "bbox":(float(x1),float(y1),float(w),float(h)),
        "detector":"semantic",
    }


def _semantic_detect(image: Image.Image, expected_count: int | None) -> list[dict]:
    from ultralytics import YOLO

    # COCO-pretrained segmentation recognises the semantic 'cup' category and
    # avoids the old bright-object heuristic. We use the mask-capable model so
    # the next renderer iteration can consume instance masks directly.
    model=YOLO("yolo11n-seg.pt")
    result=model.predict(source=np.asarray(image.convert("RGB")),conf=0.20,imgsz=960,verbose=False)[0]
    if result.boxes is None:
        return []

    names=result.names
    found=[]
    for box in result.boxes:
        cls_id=int(box.cls[0].item())
        label=str(names[cls_id]).lower()
        if label not in {"cup","mug"}:
            continue
        conf=float(box.conf[0].item())
        x1,y1,x2,y2=[float(v) for v in box.xyxy[0].tolist()]
        proposal=_surface_from_box(x1,y1,x2,y2,conf)
        found.append(proposal)

    found.sort(key=lambda s:s["confidence"],reverse=True)
    if expected_count:
        found=found[:expected_count]
    found.sort(key=lambda s:s["bbox"][0])
    return found


def detect_mug_surfaces(image: Image.Image, expected_count: int | None=None) -> list[dict]:
    """Detect actual cup/mug objects using a pretrained semantic model.

    The model weights download automatically on first use. If semantic
    inference cannot run, return no detections rather than inventing surfaces
    from unrelated white objects.
    """
    try:
        return _semantic_detect(image,expected_count)
    except Exception as exc:
        # Fail closed: a false positive is worse than asking for correction.
        # The UI exposes the error string for diagnosis.
        detect_mug_surfaces.last_error=f"{type(exc).__name__}: {exc}"
        return []


detect_mug_surfaces.last_error=""
