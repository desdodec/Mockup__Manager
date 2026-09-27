from __future__ import annotations
import io, os, json
from pathlib import Path
import streamlit as st
from PIL import Image
from ai.generator import OpenAISceneGenerator
from compositing.renderer import render_scene
from scenes.runtime import make_runtime_scene
from template_editor.calibration import draw_calibration_overlay, rectangle_corners
from vision.mug_detector import detect_mug_surfaces

st.set_page_config(page_title="Mockup Manager",page_icon="☕",layout="wide")
st.title("☕ AI Mug Mockup Manager")
st.caption("Scene in → mugs detected → your authoritative print artwork wrapped automatically.")
st.session_state.setdefault("generated_scene",None)
st.session_state.setdefault("api_key",os.getenv("OPENAI_API_KEY",""))

st.subheader("1. Choose a scene")
source=st.radio("Scene source",["Upload my own scene","Generate with AI"],horizontal=True)
expected=st.number_input("Number of mugs",1,10,3)
if source=="Upload my own scene":
    f=st.file_uploader("Upload a photograph containing blank white mugs",type=["png","jpg","jpeg"])
    if f:
        sig=(f.name,f.size,int(expected))
        if st.session_state.get("scene_sig")!=sig:
            st.session_state.generated_scene=Image.open(f).convert("RGBA")
            st.session_state.scene_sig=sig
            st.session_state.detected=None
else:
    prompt=st.text_area("Describe the scene","3 blank white mugs on a rustic West Yorkshire farmhouse table, old stone kitchen, soft window light, premium Etsy product photography")
    c1,c2=st.columns(2)
    ratio=c1.selectbox("Aspect ratio",["4:5","1:1","landscape"])
    key=c2.text_input("OpenAI API key",value=st.session_state.api_key,type="password")
    st.session_state.api_key=key
    if st.button("Generate scene",type="primary",width="stretch"):
        if not key: st.error("Enter an API key or set OPENAI_API_KEY.")
        else:
            try:
                with st.spinner("Generating blank-mug scene…"):
                    st.session_state.generated_scene=OpenAISceneGenerator(key).generate(prompt,int(expected),ratio).image
                    st.session_state.scene_sig=("ai",prompt,int(expected),ratio)
                    st.session_state.detected=None
            except Exception as exc: st.error(f"Generation failed: {exc}")

scene=st.session_state.generated_scene
if scene is None:
    st.info("Upload or generate a blank-mug scene to begin.")
    st.stop()

st.image(scene,width="stretch")
st.subheader("2. Scene calibration")
calfile=st.file_uploader("Upload ChatGPT calibration JSON",type=["json"],key="calibration")
if calfile is not None:
    try:
        calibration=json.load(calfile)
        iw=int(calibration.get("image",{}).get("width",scene.width))
        ih=int(calibration.get("image",{}).get("height",scene.height))
        if (iw,ih)!=scene.size:
            st.error(f"Calibration is for {iw}×{ih}, but this scene is {scene.width}×{scene.height}.")
            st.stop()
        st.session_state.detected=calibration["slots"]
    except Exception as exc:
        st.error(f"Invalid calibration file: {exc}")
        st.stop()
elif st.session_state.get("detected") is None:
    st.info("Upload the calibration JSON supplied with this ChatGPT-created scene. Local detection remains available as a fallback.")
    if st.button("Try local detector",width="stretch"):
        with st.spinner("Running local fallback detector…"):
            st.session_state.detected=detect_mug_surfaces(scene,int(expected))
        st.rerun()
    st.stop()
slots=[dict(s) for s in st.session_state.detected]

if len(slots)!=int(expected):
    st.warning(f"Calibration contains {len(slots)} of {int(expected)} requested mugs. Use Adjust detection only if needed.")
    if not slots and detect_mug_surfaces.last_error:
        st.caption("The calibration is incomplete; the artwork renderer has not guessed missing mug positions.")
else:
    low=sum(s["confidence"]<0.62 for s in slots)
    if low: st.warning(f"{low} mug detection(s) have low confidence. Check the overlay before rendering.")
    else: st.success(f"Loaded {len(slots)} calibrated mug surfaces.")

st.image(draw_calibration_overlay(scene,slots),caption="Calibrated printable surfaces",width="stretch")

with st.expander("Adjust detection (fallback only)",expanded=len(slots)!=int(expected)):
    st.caption("Normal scenes should not require this. These controls are only for failed detections.")
    if st.button("Create fallback surfaces",width="stretch"):
        w,h=scene.size
        slots=[]
        for i in range(int(expected)):
            cx=int(w*(i+1)/(int(expected)+1)); cy=int(h*.55)
            slots.append({"corners":rectangle_corners(cx,cy,max(40,int(w/(int(expected)+2)*.7)),max(40,int(h*.28))),"curvature":.30,"visible_fraction":.42,"confidence":0.0})
        st.session_state.detected=slots
        st.rerun()
    for i,s in enumerate(slots):
        with st.expander(f"Mug {i+1} correction"):
            w,h=scene.size
            pts=s["corners"]; cx=int(sum(p[0] for p in pts)/4); cy=int(sum(p[1] for p in pts)/4)
            sw=int(max(p[0] for p in pts)-min(p[0] for p in pts)); sh=int(max(p[1] for p in pts)-min(p[1] for p in pts))
            nx=st.slider("Centre X",0,w,cx,key=f"x{i}"); ny=st.slider("Centre Y",0,h,cy,key=f"y{i}")
            nw=st.slider("Width",20,w,max(20,sw),key=f"w{i}"); nh=st.slider("Height",20,h,max(20,sh),key=f"h{i}")
            s["corners"]=rectangle_corners(nx,ny,nw,nh)
            s["curvature"]=st.slider("Curvature",0.0,.8,float(s.get("curvature",.3)),.01,key=f"c{i}")
            s["visible_fraction"]=st.slider("Visible wrap",.2,.65,float(s.get("visible_fraction",.42)),.01,key=f"v{i}")
    st.session_state.detected=slots

if not slots:
    st.error("No reliable mug surfaces were found. Open Adjust detection to provide fallback surfaces.")
    st.stop()

st.subheader("3. Upload artwork")
uploads=st.file_uploader("Complete mug-print canvases",type=["png","jpg","jpeg"],accept_multiple_files=True,key="art")
arts={u.name:Image.open(u).convert("RGBA") for u in uploads} if uploads else {}
assignments={}; overrides={}; choices=["— none —",*arts]
cols=st.columns(min(len(slots),3))
for i,s in enumerate(slots):
    with cols[i%len(cols)]:
        st.markdown(f"**Mug {i+1}**")
        default=i+1 if i<len(arts) else 0
        chosen=st.selectbox("Artwork",choices,index=default,key=f"a{i}")
        face=st.radio("Face",["Front","Rear"],horizontal=True,key=f"f{i}")
        if chosen!="— none —": assignments[f"mug_{i+1:02d}"]=arts[chosen]
        overrides[f"mug_{i+1:02d}"]={"view_angle":0 if face=="Front" else 180,"visible_fraction":s["visible_fraction"],"curvature":s["curvature"]}

if assignments:
    runtime=make_runtime_scene(scene,slots,Path(".runtime"))
    result=render_scene(runtime,assignments,slot_overrides=overrides)
    st.subheader("4. Final mockup")
    st.image(result,width="stretch")
    png=io.BytesIO(); result.save(png,"PNG")
    jpg=io.BytesIO(); result.convert("RGB").save(jpg,"JPEG",quality=96)
    d1,d2=st.columns(2)
    d1.download_button("Download PNG",png.getvalue(),"mockup.png","image/png",width="stretch")
    d2.download_button("Download JPG",jpg.getvalue(),"mockup.jpg","image/jpeg",width="stretch")
