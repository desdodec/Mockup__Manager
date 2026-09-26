from __future__ import annotations

import io
import json
import streamlit as st
from PIL import Image

from compositing.renderer import render_scene
from config import TEMPLATES_DIR
from scenes.registry import discover_scenes
from scripts import create_demo_template  # noqa: F401
from template_editor.calibration import draw_calibration_overlay, rectangle_corners
from template_editor.package import create_template_zip

st.set_page_config(page_title="Mockup Manager", page_icon="☕", layout="wide")
st.title("☕ Mockup Manager")
st.caption("AI creates scenes. Deterministic geometry applies your authoritative print artwork.")

render_tab, editor_tab, ai_tab = st.tabs(["Render mockups", "Template editor", "AI scene"])

with render_tab:
    scenes = discover_scenes(TEMPLATES_DIR)
    if not scenes:
        st.error("No scene templates found.")
        st.stop()
    scene_names = {s.name: s for s in scenes.values()}
    scene = scene_names[st.selectbox("Scene", list(scene_names))]
    uploads = st.file_uploader("Upload complete mug-print artwork", type=["png","jpg","jpeg"], accept_multiple_files=True, key="render-art")
    artworks = {u.name: Image.open(u).convert("RGBA") for u in uploads} if uploads else {}
    assignments, overrides = {}, {}
    left, right = st.columns([1, 1.7])
    with left:
        choices = ["— none —", *artworks.keys()]
        for slot in scene.slots:
            with st.expander(slot.label, expanded=True):
                chosen = st.selectbox("Artwork", choices, key=f"art-{slot.id}")
                face = st.radio("View", ["Front","Rear","Custom"], horizontal=True, key=f"face-{slot.id}")
                angle = 0.0 if face=="Front" else 180.0 if face=="Rear" else st.slider("Angle",0,359,int(slot.view_angle),key=f"ang-{slot.id}")
                visible = st.slider("Visible circumference",0.20,0.65,float(slot.visible_fraction),0.01,key=f"vis-{slot.id}")
                curve = st.slider("Curvature",-0.8,0.8,float(slot.curvature),0.01,key=f"cur-{slot.id}")
                opacity = st.slider("Print strength",0.0,1.0,float(slot.opacity),0.01,key=f"op-{slot.id}")
                if chosen != "— none —": assignments[slot.id]=artworks[chosen]
                overrides[slot.id]={"view_angle":angle,"visible_fraction":visible,"curvature":curve,"opacity":opacity}
    with right:
        result=render_scene(scene,assignments,slot_overrides=overrides)
        st.image(result,width="stretch")
        png=io.BytesIO(); result.save(png,"PNG")
        jpg=io.BytesIO(); result.convert("RGB").save(jpg,"JPEG",quality=95)
        c1,c2=st.columns(2)
        c1.download_button("Download PNG",png.getvalue(),f"{scene.id}-mockup.png","image/png",width="stretch")
        c2.download_button("Download JPG",jpg.getvalue(),f"{scene.id}-mockup.jpg","image/jpeg",width="stretch")

with editor_tab:
    st.subheader("Calibrate a real blank-mug scene")
    st.write("Upload a photographic scene, define each visible printable surface, then export reusable template data.")
    scene_file=st.file_uploader("Blank-mug scene",type=["png","jpg","jpeg"],key="editor-scene")
    if scene_file:
        scene_img=Image.open(scene_file).convert("RGBA")
        w,h=scene_img.size
        name=st.text_input("Template name","My mug scene")
        mug_count=st.number_input("Number of mugs",1,12,1)
        slots=[]
        controls, preview=st.columns([1,1.7])
        with controls:
            st.caption(f"Scene resolution: {w} × {h}. Coordinates below are pixels in the original image.")
            for i in range(int(mug_count)):
                with st.expander(f"Mug {i+1}",expanded=i==0):
                    cx=st.slider("Centre X",0,w,w//2,key=f"ecx{i}")
                    cy=st.slider("Centre Y",0,h,h//2,key=f"ecy{i}")
                    sw=st.slider("Print width",20,w,max(20,w//5),key=f"ew{i}")
                    sh=st.slider("Print height",20,h,max(20,h//3),key=f"eh{i}")
                    top_inset=st.slider("Top perspective inset",0,max(1,sw//3),0,key=f"eti{i}")
                    bottom_inset=st.slider("Bottom perspective inset",0,max(1,sw//3),0,key=f"ebi{i}")
                    corners=list(rectangle_corners(cx,cy,sw,sh))
                    corners[0]=(corners[0][0]+top_inset,corners[0][1]); corners[1]=(corners[1][0]-top_inset,corners[1][1])
                    corners[2]=(corners[2][0]-bottom_inset,corners[2][1]); corners[3]=(corners[3][0]+bottom_inset,corners[3][1])
                    slots.append({"corners":corners,"curvature":st.slider("Default curvature",0.0,0.8,0.28,0.01,key=f"ecur{i}"),"visible_fraction":st.slider("Visible circumference",0.2,0.65,0.42,0.01,key=f"evis{i}")})
        with preview:
            st.image(draw_calibration_overlay(scene_img,slots),width="stretch")
        filename,payload=create_template_zip(name,scene_img,slots)
        st.download_button("Export reusable template",payload,filename,"application/zip",width="stretch")
        st.info("Extract the downloaded folder into assets/templates/, restart Streamlit, and it appears in Render mockups.")

with ai_tab:
    st.subheader("AI custom scene")
    st.write("The scene-generation provider is deliberately isolated from product artwork.")
    st.info("Next integration point: generate a photorealistic blank-mug scene here, then send that scene directly into the same Template Editor for calibration.")
