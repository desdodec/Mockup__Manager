from __future__ import annotations

import io
import os
from pathlib import Path
import streamlit as st
from PIL import Image

from ai.generator import OpenAISceneGenerator
from compositing.renderer import render_scene
from scenes.runtime import make_runtime_scene
from template_editor.calibration import draw_calibration_overlay, rectangle_corners

st.set_page_config(page_title="Mockup Manager", page_icon="☕", layout="wide")
st.title("☕ AI Mug Mockup Manager")
st.caption("AI creates the photograph. Your uploaded print artwork is applied afterwards by deterministic code.")

if "generated_scene" not in st.session_state:
    st.session_state.generated_scene = None

st.subheader("1. Generate the blank-mug photograph")
prompt = st.text_area(
    "Describe the scene",
    value="3 blank white mugs on a rustic West Yorkshire farmhouse table, old stone farmhouse kitchen, soft window light, premium Etsy product photography",
    height=100,
)
g1,g2,g3 = st.columns(3)
mug_count = g1.number_input("Mugs", 1, 10, 3)
ratio = g2.selectbox("Aspect ratio", ["4:5","1:1","landscape"])
api_key = g3.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY",""), type="password")

if st.button("Generate scene", type="primary", width="stretch"):
    if not api_key:
        st.error("Enter an OpenAI API key or set OPENAI_API_KEY.")
    else:
        with st.spinner("Generating photorealistic blank-mug scene…"):
            try:
                generated = OpenAISceneGenerator(api_key).generate(prompt, int(mug_count), ratio)
                st.session_state.generated_scene = generated.image
                st.session_state.mug_count = int(mug_count)
            except Exception as exc:
                st.error(f"Scene generation failed: {exc}")

scene_img = st.session_state.generated_scene
if scene_img is not None:
    st.image(scene_img, caption="AI-generated blank-mug scene", width="stretch")
    st.divider()
    st.subheader("2. Define / correct the mug print surfaces")
    st.caption("This calibration affects geometry only. It never edits the source artwork.")
    w,h = scene_img.size
    count = int(st.session_state.get("mug_count", mug_count))
    slots=[]
    controls, overlay_col = st.columns([1,1.7])
    with controls:
        for i in range(count):
            # Spread useful starting guesses horizontally; manual correction is
            # intentionally available because arbitrary AI mug detection is not
            # yet reliable enough to trust with production artwork.
            default_x = int(w * (i+1)/(count+1))
            default_y = int(h * 0.55)
            with st.expander(f"Mug {i+1} surface", expanded=i==0):
                cx=st.slider("Centre X",0,w,default_x,key=f"cx{i}")
                cy=st.slider("Centre Y",0,h,default_y,key=f"cy{i}")
                sw=st.slider("Print width",20,w,max(20,int(w/(count+2)*0.7)),key=f"pw{i}")
                sh=st.slider("Print height",20,h,max(20,int(h*0.28)),key=f"ph{i}")
                top=st.slider("Top inset",0,max(1,sw//3),0,key=f"ti{i}")
                bottom=st.slider("Bottom inset",0,max(1,sw//3),0,key=f"bi{i}")
                corners=list(rectangle_corners(cx,cy,sw,sh))
                corners[0]=(corners[0][0]+top,corners[0][1]); corners[1]=(corners[1][0]-top,corners[1][1])
                corners[2]=(corners[2][0]-bottom,corners[2][1]); corners[3]=(corners[3][0]+bottom,corners[3][1])
                slots.append({"corners":corners,"curvature":st.slider("Curvature",0.0,0.8,0.28,0.01,key=f"cv{i}"),"visible_fraction":st.slider("Visible wrap",0.2,0.65,0.42,0.01,key=f"vf{i}")})
    with overlay_col:
        st.image(draw_calibration_overlay(scene_img,slots), caption="Blue areas are the current printable surfaces", width="stretch")

    st.divider()
    st.subheader("3. Upload and assign your production artwork")
    uploads=st.file_uploader("Complete mug-print files (front + rear on the supplied canvas)",type=["png","jpg","jpeg"],accept_multiple_files=True)
    artworks={u.name:Image.open(u).convert("RGBA") for u in uploads} if uploads else {}
    assignments,overrides={},{}
    choices=["— none —",*artworks.keys()]
    cols=st.columns(min(count,3))
    for i in range(count):
        with cols[i%len(cols)]:
            st.markdown(f"**Mug {i+1}**")
            chosen=st.selectbox("Artwork",choices,key=f"a{i}")
            face=st.radio("Face",["Front","Rear","Custom"],horizontal=True,key=f"f{i}")
            angle=0 if face=="Front" else 180 if face=="Rear" else st.slider("Rotation around mug",0,359,0,key=f"ra{i}")
            if chosen!="— none —": assignments[f"mug_{i+1:02d}"]=artworks[chosen]
            overrides[f"mug_{i+1:02d}"]={"view_angle":angle,"visible_fraction":slots[i]["visible_fraction"],"curvature":slots[i]["curvature"]}

    if assignments:
        runtime=make_runtime_scene(scene_img,slots,Path(".runtime"))
        result=render_scene(runtime,assignments,slot_overrides=overrides)
        st.divider()
        st.subheader("4. Final mockup")
        st.image(result,width="stretch")
        png=io.BytesIO(); result.save(png,"PNG")
        jpg=io.BytesIO(); result.convert("RGB").save(jpg,"JPEG",quality=96)
        d1,d2=st.columns(2)
        d1.download_button("Download PNG",png.getvalue(),"mockup.png","image/png",width="stretch")
        d2.download_button("Download JPG",jpg.getvalue(),"mockup.jpg","image/jpeg",width="stretch")
else:
    st.info("Generate a scene to begin. Your product artwork is not sent to the image-generation API.")
