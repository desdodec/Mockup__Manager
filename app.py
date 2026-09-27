from __future__ import annotations
from io import BytesIO
from pathlib import Path
import zipfile
import streamlit as st
from PIL import Image
from v2 import Slot,Template,render,render_batch

st.set_page_config(page_title="Mockup Manager V2",layout="wide")
st.title("Mockup Manager V2")
st.caption("Template-first mug mockups: choose a scene, add artwork, render. No mug detection.")

DEFAULT_SCENE=Path("H:/Downloads/ChatGPT Image Sep 27, 2026, 06_36_31 PM.png")
DEFAULT_CAL=Path("H:/Downloads/mockup_manager_cylinder_calibration_2048x849.png")

# Calibrated once for the user's 1536x1024 Yorkshire two-mug scene.
YORKSHIRE_SLOTS=(
    Slot("mug_01",(0.237,0.365,0.452,0.778),yaw_deg=-7.0),
    Slot("mug_02",(0.561,0.372,0.778,0.786),yaw_deg=7.0),
)

st.sidebar.header("Template")
scene_path=Path(st.sidebar.text_input("Scene image",str(DEFAULT_SCENE)))
if not scene_path.exists():
    st.error(f"Scene not found: {scene_path}")
    st.stop()
template=Template("yorkshire_2","Yorkshire Garden · 2 mugs",scene_path,YORKSHIRE_SLOTS)
st.sidebar.success("Yorkshire Garden · 2 mugs")
st.sidebar.caption("Geometry is stored with the template. V2 does not rediscover mugs on every render.")

scene=Image.open(scene_path).convert("RGB")
st.image(scene,caption="Template scene",width="stretch")

st.subheader("Artwork")
uploads=st.file_uploader("Upload full mug-print canvases",type=["png","jpg","jpeg"],accept_multiple_files=True)
arts=[(u.name,Image.open(u).convert("RGBA")) for u in uploads] if uploads else []
if DEFAULT_CAL.exists() and st.checkbox("Include calibration artwork",False):
    arts.insert(0,(DEFAULT_CAL.name,Image.open(DEFAULT_CAL).convert("RGBA")))

mode=st.radio("Mode",["Two-mug preview","Batch"],horizontal=True)
face=st.radio("Artwork side",["Front","Rear"],horizontal=True)

if mode=="Two-mug preview":
    if not arts:
        st.info("Upload one or two artwork files.")
        st.stop()
    cols=st.columns(2)
    assignments={}
    for i,slot in enumerate(template.slots):
        if i<len(arts):
            name,img=arts[i]
            assignments[slot.id]=(img,face)
            cols[i].image(img,caption=f"{slot.id}: {name}",width="stretch")
    if st.button("Render mockup",type="primary",width="stretch"):
        with st.spinner("Rendering locally…"):
            result=render(template,assignments)
        st.image(result,caption="Rendered mockup",width="stretch")
        b=BytesIO(); result.convert("RGB").save(b,"JPEG",quality=95,subsampling=0)
        st.download_button("Download JPG",b.getvalue(),"mockup.jpg","image/jpeg",width="stretch")
else:
    if not arts:
        st.info("Upload artwork files. V2 fills two mugs per output.")
        st.stop()
    st.write(f"{len(arts)} designs → {(len(arts)+1)//2} mockups")
    if st.button("Render batch",type="primary",width="stretch"):
        with st.spinner("Rendering batch locally…"):
            results=render_batch(template,arts,face.lower())
        z=BytesIO()
        with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as archive:
            for label,img in results:
                b=BytesIO(); img.convert("RGB").save(b,"JPEG",quality=95,subsampling=0)
                archive.writestr(f"{label}.jpg",b.getvalue())
        st.success(f"Rendered {len(results)} mockups.")
        for label,img in results[:4]: st.image(img,caption=label,width="stretch")
        st.download_button("Download all as ZIP",z.getvalue(),"mockups.zip","application/zip",width="stretch")
