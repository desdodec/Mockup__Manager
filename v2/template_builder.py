from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from v2 import Slot, Template, render

def _normalise_rect(obj,w,h):
    left=float(obj.get("left",0)); top=float(obj.get("top",0))
    width=float(obj.get("width",0))*float(obj.get("scaleX",1))
    height=float(obj.get("height",0))*float(obj.get("scaleY",1))
    return (left/w,top/h,(left+width)/w,(top+height)/h)

def _valid(b):
    x0,y0,x1,y1=b
    return 0<=x0<x1<=1 and 0<=y0<y1<=1 and x1-x0>.03 and y1-y0>.05

def _adjust(b,dx=0,dy=0,dw=0,dh=0):
    x0,y0,x1,y1=b; cx=(x0+x1)/2; cy=(y0+y1)/2
    w=max(.03,x1-x0+dw); h=max(.05,y1-y0+dh)
    cx=min(1-w/2,max(w/2,cx+dx)); cy=min(1-h/2,max(h/2,cy+dy))
    return (cx-w/2,cy-h/2,cx+w/2,cy+h/2)

def _guide(scene,b):
    W,H=scene.size; x0,y0,x1,y1=b
    pad=.28; bw=x1-x0; bh=y1-y0
    l=max(0,x0-bw*pad); r=min(1,x1+bw*pad); t=max(0,y0-bh*pad); bot=min(1,y1+bh*pad)
    crop=scene.crop((int(l*W),int(t*H),int(r*W),int(bot*H))).resize((700,700),Image.Resampling.LANCZOS)
    d=ImageDraw.Draw(crop)
    sx=700/(r-l); sy=700/(bot-t)
    px0=int((x0-l)*sx); px1=int((x1-l)*sx); py0=int((y0-t)*sy); py1=int((y1-t)*sy)
    cx=(px0+px1)//2; cy=(py0+py1)//2
    d.rectangle((px0,py0,px1,py1),outline="blue",width=5)
    d.line((cx,py0,cx,py1),fill="red",width=5)
    d.line((px0,cy,px1,cy),fill="red",width=3)
    pt=int(py0+(py1-py0)*.07); pb=int(py0+(py1-py0)*.91)
    d.line((px0,pt,px1,pt),fill="orange",width=4); d.line((px0,pb,px1,pb),fill="orange",width=4)
    return crop

def run_builder():
    st.header("Template Builder")
    st.caption("Roughly mark each mug, then use Guided Calibration to centre it precisely.")
    upload=st.file_uploader("1. Choose a background scene",type=["png","jpg","jpeg"],key="builder_scene")
    if not upload:
        st.info("Choose a scene containing one or more blank mugs."); return
    scene=Image.open(BytesIO(upload.getvalue())).convert("RGB"); W,H=scene.size
    scale=min(1.0,1100/W); cw=int(W*scale); ch=int(H*scale)
    display=scene.resize((cw,ch),Image.Resampling.LANCZOS)
    st.markdown("**2. Roughly draw one rectangle over each mug body**")
    st.caption("Do not include the handle. Precision is not required here — the next step helps you centre each mug.")
    canvas=st_canvas(fill_color="rgba(0,120,255,0.16)",stroke_width=3,stroke_color="#0078ff",background_image=display,drawing_mode="rect",update_streamlit=True,height=ch,width=cw,key="mug_builder")
    rects=[o for o in ((canvas.json_data or {}).get("objects",[])) if o.get("type")=="rect"]
    raw_boxes=[_normalise_rect(o,cw,ch) for o in rects]
    if not raw_boxes:
        st.warning("Draw at least one mug rectangle."); return
    if any(not _valid(b) for b in raw_boxes):
        st.error("One or more rectangles are too small or outside the image."); return
    sig=tuple(tuple(round(v,5) for v in b) for b in raw_boxes)
    if st.session_state.get("builder_sig")!=sig:
        st.session_state.builder_sig=sig; st.session_state.builder_boxes=list(raw_boxes); st.session_state.builder_mug=0
    boxes=st.session_state.builder_boxes
    idx=min(st.session_state.get("builder_mug",0),len(boxes)-1)
    st.divider(); st.markdown(f"### 3. Guided Calibration — Mug {idx+1} of {len(boxes)}")
    st.info("The RED vertical line should run down the visual centre of the cylindrical mug body. The BLUE box should cover the body, not the handle. Orange lines show the normal print limits.")
    st.image(_guide(scene,boxes[idx]),caption=f"Mug {idx+1} enlarged guide",width=700)
    step=st.radio("Adjustment size",["Fine","Medium"],horizontal=True,key="builder_step")
    n=.002 if step=="Fine" else .006
    a,b,c,d=st.columns(4)
    if a.button("← Left",width="stretch"): boxes[idx]=_adjust(boxes[idx],dx=-n); st.rerun()
    if b.button("Right →",width="stretch"): boxes[idx]=_adjust(boxes[idx],dx=n); st.rerun()
    if c.button("↑ Up",width="stretch"): boxes[idx]=_adjust(boxes[idx],dy=-n); st.rerun()
    if d.button("Down ↓",width="stretch"): boxes[idx]=_adjust(boxes[idx],dy=n); st.rerun()
    a,b,c,d=st.columns(4)
    if a.button("Narrower",width="stretch"): boxes[idx]=_adjust(boxes[idx],dw=-2*n); st.rerun()
    if b.button("Wider",width="stretch"): boxes[idx]=_adjust(boxes[idx],dw=2*n); st.rerun()
    if c.button("Shorter",width="stretch"): boxes[idx]=_adjust(boxes[idx],dh=-2*n); st.rerun()
    if d.button("Taller",width="stretch"): boxes[idx]=_adjust(boxes[idx],dh=2*n); st.rerun()
    prev,nextc=st.columns(2)
    if prev.button("← Previous mug",disabled=idx==0,width="stretch"):
        st.session_state.builder_mug=idx-1; st.rerun()
    if nextc.button("Looks right → Next mug",disabled=idx==len(boxes)-1,width="stretch"):
        st.session_state.builder_mug=idx+1; st.rerun()
    st.divider(); name=st.text_input("4. Template name",Path(upload.name).stem)
    slots=tuple(Slot(f"mug_{i:02d}",box) for i,box in enumerate(boxes,1))
    if st.button("Test all mugs",type="primary",width="stretch"):
        tmp=Path(".mockup_manager_preview_scene.png"); scene.save(tmp)
        template=Template("preview",name or "New template",tmp,slots)
        cal=Image.new("RGBA",(2048,849),(255,255,255,255)); cd=ImageDraw.Draw(cal)
        cd.line((512,0,512,849),fill=(255,0,0,255),width=18)
        for x in range(0,2048,128): cd.line((x,0,x,849),fill=(100,100,100,255),width=3)
        st.image(render(template,{slot.id:(cal,"Front") for slot in slots}),caption="Final test — the red centre line should sit on the centre of every mug body.",width="stretch")
    payload={"version":2,"name":name or "New template","scene_filename":upload.name,"slots":[{"id":s.id,"box":[round(v,6) for v in s.box]} for s in slots]}
    st.download_button("5. Save template",json.dumps(payload,indent=2),file_name=f"{(name or 'template').replace(' ','_')}.mockup.json",mime="application/json",width="stretch")
