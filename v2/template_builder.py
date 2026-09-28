from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from v2 import Slot, Template, render
from v2.marker import detect_marker_mugs, marker_preview

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
    st.caption("Calibrate the final white-mug master once. The saved geometry is reused for every future render.")
    upload=st.file_uploader("1. Choose a background scene",type=["png","jpg","jpeg"],key="builder_scene")
    if not upload:
        st.info("Use the exact finished white-mug lifestyle photograph you want in the final mockups. No marker image is required.")
        return
    scene=Image.open(BytesIO(upload.getvalue())).convert("RGB"); W,H=scene.size

    mode="Manual rectangles"
    boxes=[]
    if mode=="Automatic marker detection":
        boxes,_=detect_marker_mugs(scene)
        if not boxes:
            st.error("No magenta marker mug bodies were detected. Use Manual rectangles for this scene, or regenerate it with saturated magenta printable bodies.")
            return
        st.success(f"{len(boxes)} printable mug{'s' if len(boxes)!=1 else ''} detected")
        st.image(marker_preview(scene,boxes),caption="Detected mugs — numbered left to right",width="stretch")
        st.caption("Green boxes are calculated from the marker regions. No centre-line judgement or hand fitting is required.")
    else:
        canvas_width=st.radio("Scene size",["Compact","Fit screen","Large"],index=0,horizontal=True,key="builder_canvas_size")
        maxw={"Compact":600,"Fit screen":760,"Large":960}[canvas_width]
        scale=min(1.0,maxw/W); cw=int(W*scale); ch=int(H*scale)
        display=scene.resize((cw,ch),Image.Resampling.LANCZOS)
        st.info("Draw one rectangle over each cylindrical PRINT AREA only. Leave handles, upper rim and the table outside. Draw all mugs; they will be ordered left to right.")
        canvas=st_canvas(fill_color="rgba(0,120,255,0.16)",stroke_width=3,stroke_color="#0078ff",background_image=display,drawing_mode="rect",update_streamlit=True,height=ch,width=cw,key="mug_builder")
        objects=((canvas.json_data or {}).get("objects",[]))
        rects=[o for o in objects if o.get("type")=="rect" or (float(o.get("width",0) or 0)>0 and float(o.get("height",0) or 0)>0 and o.get("type") not in ("image","background"))]
        boxes=sorted([_normalise_rect(o,cw,ch) for o in rects], key=lambda b:b[0])
        if not boxes:
            st.warning("Draw at least one mug rectangle.")
            return
        if any(not _valid(b) for b in boxes):
            st.error("One or more rectangles are too small or outside the image.")
            return

    st.divider()
    st.subheader("2. Template")
    name=st.text_input("Template name",Path(upload.name).stem)
    top_inset=st.slider("Top print inset",0.0,0.12,0.025,0.005,help="Leaves a natural white margin below the upper rim.")
    bottom_inset=st.slider("Bottom print inset",0.0,0.12,0.025,0.005,help="Keeps artwork above the mug/table contact edge.")
    visible=st.slider("Visible wrap angle",110,160,136,1,help="How much of the 360-degree artwork is visible across the mug face.")
    slots=tuple(Slot(f"mug_{i:02d}",box,print_top=top_inset,print_bottom=1.0-bottom_inset,marker_mask=False,visible_deg=float(visible)) for i,box in enumerate(boxes,1))

    if st.button("Preview calibration",type="primary",width="stretch"):
        tmp=Path(".mockup_manager_preview_scene.png"); scene.save(tmp)
        template=Template("preview",name or "New template",tmp,slots)
        # Build calibration marks only across the exact FRONT angular window.
        # This guarantees every slot receives the same marks regardless of full-wrap sampling.
        cal=Image.new("RGBA",(2048,849),(245,245,245,255)); cd=ImageDraw.Draw(cal)
        visible=slots[0].visible_deg if slots else 136.0
        centre=512
        half=visible/360.0*2048/2
        # Seven internal angular reference lines: centre + three each side, kept away from the projection boundaries.
        for j in range(-3,4):
            x=int(round(centre + (j/4.0)*half))
            if j==0:
                cd.line((x,0,x,849),fill=(220,0,0,255),width=12)
            else:
                cd.line((x,0,x,849),fill=(95,95,95,255),width=4)
        st.image(render(template,{slot.id:(cal,"Front") for slot in slots}),caption="Calibration preview — red is the front centre; grey lines show the cylindrical wrap. Adjust rectangles/insets until every print sits naturally inside the white ceramic body.",width="stretch")

    payload={"version":3,"name":name or "New template","detection":"calibrated-white-master","scene_filename":upload.name,"slots":[{"id":slot.id,"box":[round(v,6) for v in slot.box],"marker_mask":False,"print_top":round(slot.print_top,4),"print_bottom":round(slot.print_bottom,4),"visible_deg":slot.visible_deg} for slot in slots]}
    st.download_button("Save calibrated template",json.dumps(payload,indent=2),file_name=f"{(name or 'template').replace(' ','_')}.mockup.json",mime="application/json",width="stretch")
