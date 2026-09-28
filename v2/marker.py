from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw

def detect_marker_mugs(scene: Image.Image):
    """Return left-to-right normalized boxes for saturated magenta mug bodies."""
    rgb=np.asarray(scene.convert("RGB"))
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([140,95,70],np.uint8),np.array([179,255,255],np.uint8))
    h,w=mask.shape
    k=max(3,int(round(min(w,h)*0.006)))
    if k%2==0: k+=1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    found=[]
    min_area=w*h*0.002
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area<min_area or bw<0.025*w or bh<0.08*h: continue
        if area/max(bw*bh,1)<0.35: continue
        found.append((x,y,bw,bh,area,i))
    found.sort(key=lambda z:z[0])
    boxes=[(x/w,y/h,(x+bw)/w,(y+bh)/h) for x,y,bw,bh,_,_ in found]
    clean=np.zeros_like(mask)
    for *_,i in found: clean[labels==i]=255
    return boxes,clean

def marker_preview(scene: Image.Image, boxes):
    out=scene.convert("RGB").copy(); d=ImageDraw.Draw(out); W,H=out.size
    width=max(3,int(min(W,H)*0.004))
    for i,(x0,y0,x1,y1) in enumerate(boxes,1):
        p=(int(x0*W),int(y0*H),int(x1*W),int(y1*H))
        d.rectangle(p,outline=(0,255,80),width=width)
        d.text((p[0]+8,p[1]+8),f"MUG {i}",fill=(0,0,0),stroke_width=3,stroke_fill=(255,255,255))
    return out
