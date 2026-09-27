from __future__ import annotations
import base64, io, json
import numpy as np
from PIL import Image
from openai import OpenAI

def _data_url(image):
    b=io.BytesIO(); image.convert("RGB").save(b,"JPEG",quality=92)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

def _mesh(m,w,h):
    p=m["printable_area"]; l=p["left"]*w; r=p["right"]*w; t=p["top"]*h; b=p["bottom"]*h
    cx=m["body_center"][0]*w; angles=np.linspace(-np.deg2rad(68),np.deg2rad(68),13)
    radius=max(8.0,(r-l)/2/np.sin(np.deg2rad(68))); xs=np.clip(cx+radius*np.sin(angles),l,r)
    mesh=[[(float(x),float(t+(b-t)*iy/6)) for x in xs] for iy in range(7)]
    q=m["body_bbox"]
    return {"corners":(mesh[0][0],mesh[0][-1],mesh[-1][-1],mesh[-1][0]),"mesh":mesh,
      "axis":((cx,t),(cx,b)),"handle_side":m.get("handle_side","unknown"),"curvature":0.0,
      "visible_fraction":.40,"confidence":float(m.get("confidence",0)),"detector":"openai-vision",
      "bbox":(q["left"]*w,q["top"]*h,(q["right"]-q["left"])*w,(q["bottom"]-q["top"])*h)}

class OpenAIMugAnalyzer:
    def __init__(self,api_key,model="gpt-5.6-luna"):
        self.client=OpenAI(api_key=api_key); self.model=model
    def analyze(self,image,expected_count):
        prompt=("Analyse this finished product photograph. Identify exactly %d blank white standard mugs, ordered left-to-right. "
        "Return normalized 0..1 coordinates relative to the complete image. For each mug distinguish the straight cylindrical CERAMIC BODY from its handle. "
        "Never include the handle in body_bbox or printable_area. printable_area must stay safely below the rim, above the curved base, and clear of handle attachments. "
        "body_center is the centre of the cylindrical body, not the whole mug. handle_side is left, right, or unknown. Do not identify background objects as mugs. "
        "Return JSON only in this shape: {\\\"mugs\\\":[{\\\"body_bbox\\\":{\\\"left\\\":0.0,\\\"top\\\":0.0,\\\"right\\\":0.0,\\\"bottom\\\":0.0},"
        "\\\"body_center\\\":[0.0,0.0],\\\"handle_side\\\":\\\"left\\\",\\\"printable_area\\\":{\\\"left\\\":0.0,\\\"top\\\":0.0,\\\"right\\\":0.0,\\\"bottom\\\":0.0},\\\"confidence\\\":0.0}]}" % expected_count)
        response=self.client.responses.create(model=self.model,input=[{"role":"user","content":[
          {"type":"input_text","text":prompt},{"type":"input_image","image_url":_data_url(image),"detail":"high"}]}])
        txt=response.output_text.strip()
        if txt.startswith("~~~"): txt=txt.split("\\n",1)[1].rsplit("~~~",1)[0]
        mugs=json.loads(txt).get("mugs",[])
        if len(mugs)!=expected_count: raise ValueError(f"Vision returned {len(mugs)} mugs; expected {expected_count}.")
        slots=[_mesh(m,*image.size) for m in mugs]
        if any(s["bbox"][2]<=0 or s["bbox"][3]<=0 for s in slots): raise ValueError("Vision returned invalid mug geometry.")
        return slots
