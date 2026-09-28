from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from v2 import Slot, Template, render

def _normalise_rect(obj:dict,w:int,h:int)->tuple[float,float,float,float]:
    left=float(obj.get("left",0)); top=float(obj.get("top",0))
    width=float(obj.get("width",0))*float(obj.get("scaleX",1))
    height=float(obj.get("height",0))*float(obj.get("scaleY",1))
    return (left/w,top/h,(left+width)/w,(top+height)/h)

def _valid(box):
    x0,y0,x1,y1=box
    return 0<=x0<x1<=1 and 0<=y0<y1<=1 and (x1-x0)>.03 and (y1-y0)>.05

def run_builder():
    st.header("Template Builder")
    st.caption("Draw one rectangle over each mug body. Avoid the handle. You only do this once per scene.")
    upload=st.file_uploader("1. Choose a background scene",type=["png","jpg","jpeg"],key="builder_scene")
    if not upload:
        st.info("Choose a scene containing one or more blank mugs.")
        return
    raw=upload.getvalue(); scene=Image.open(BytesIO(raw)).convert("RGB")
    W,H=scene.size
    maxw=1100; scale=min(1.0,maxw/W); cw=int(W*scale); ch=int(H*scale)
    display=scene.resize((cw,ch),Image.Resampling.LANCZOS)
    st.markdown("**2. Draw rectangles over the cylindrical body of every mug**")
    st.caption("Drag to draw. Click a rectangle to move or resize it. Use the toolbar to undo/delete mistakes.")
    canvas=st_canvas(fill_color="rgba(0, 120, 255, 0.16)",stroke_width=3,stroke_color="#0078ff",background_image=display,drawing_mode="rect",update_streamlit=True,height=ch,width=cw,key="mug_builder")
    objects=(canvas.json_data or {}).get("objects",[])
    rects=[o for o in objects if o.get("type")=="rect"]
    boxes=[_normalise_rect(o,cw,ch) for o in rects]
    bad=[i+1 for i,b in enumerate(boxes) if not _valid(b)]
    if not boxes:
        st.warning("Draw at least one mug rectangle.")
        return
    if bad:
        st.error("These rectangles need resizing or moving fully inside the image: "+", ".join(map(str,bad)))
        return
    st.success(f"{len(boxes)} mug area{'s' if len(boxes)!=1 else ''} ready.")
    preview=display.copy(); d=ImageDraw.Draw(preview)
    for i,b in enumerate(boxes,1):
        x0,y0,x1,y1=b; px=(int(x0*cw),int(y0*ch),int(x1*cw),int(y1*ch))
        d.rectangle(px,outline="blue",width=4); d.text((px[0]+8,px[1]+8),f"Mug {i}",fill="blue")
    st.image(preview,caption="Numbered mug areas",width="stretch")
    name=st.text_input("3. Template name",Path(upload.name).stem)
    if st.button("Test template",type="primary",width="stretch"):
        slots=tuple(Slot(f"mug_{i:02d}",b) for i,b in enumerate(boxes,1))
        # Persist source scene temporarily for this Streamlit session.
        tmp=Path(".mockup_manager_preview_scene.png"); scene.save(tmp)
        template=Template("preview",name or "New template",tmp,slots)
        cal=Image.new("RGBA",(2048,849),(255,255,255,255))
        cd=ImageDraw.Draw(cal); cd.line((512,0,512,849),fill=(255,0,0,255),width=18); cd.line((1536,0,1536,849),fill=(0,80,255,255),width=18)
        for x in range(0,2048,128): cd.line((x,0,x,849),fill=(100,100,100,255),width=3)
        st.image(render(template,{slot.id:(cal,"Front") for slot in slots}),caption="Calibration preview — red centre line should sit on the front of each mug.",width="stretch")
    payload={"version":2,"name":name or "New template","scene_filename":upload.name,"slots":[{"id":f"mug_{i:02d}","box":[round(v,6) for v in b]} for i,b in enumerate(boxes,1)]}
    st.download_button("4. Save template",json.dumps(payload,indent=2),file_name=f"{(name or 'template').replace(' ','_')}.mockup.json",mime="application/json",width="stretch")
