from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import zipfile
import streamlit as st
from PIL import Image, ImageFilter
from v2 import Slot,Template,render,render_batch
from v2.template_builder import run_builder

st.set_page_config(page_title="Mockup Manager V2",layout="wide")
st.title("Mockup Manager V2")
st.caption("Template-first mug mockups: load a template, assign artwork to each mug, render locally.")

workspace=st.sidebar.radio("Workspace",["Mockup Generator","Template Builder"])
if workspace=="Template Builder":
    run_builder()
    st.stop()

st.sidebar.header("Template")
template_upload=st.sidebar.file_uploader("Template JSON",type=["json"],key="generator_template")
scene_upload=st.sidebar.file_uploader("Template scene image",type=["png","jpg","jpeg"],key="generator_scene")

if not template_upload or not scene_upload:
    st.info("Upload the saved .mockup.json template and the same background scene image used to create it.")
    st.stop()

try:
    payload=json.loads(template_upload.getvalue().decode("utf-8"))
    scene=Image.open(BytesIO(scene_upload.getvalue())).convert("RGB")
    marker=payload.get("detection")=="magenta-marker"
    slots=tuple(
        Slot(
            item.get("id",f"mug_{i:02d}"),
            tuple(float(v) for v in item["box"]),
            print_top=float(item.get("print_top",0.0 if marker else 0.07)),
            print_bottom=float(item.get("print_bottom",1.0 if marker else 0.91)),
            marker_mask=bool(item.get("marker_mask",marker)),
            visible_deg=float(item.get("visible_deg",136.0)),
        )
        for i,item in enumerate(payload["slots"],1)
    )
except Exception as exc:
    st.error(f"Could not load template: {exc}")
    st.stop()

# Render expects a path; cache the uploaded scene locally for this session.
scene_path=Path(".mockup_manager_active_scene.png")
scene.save(scene_path)
template=Template("uploaded",payload.get("name","Uploaded template"),scene_path,slots)

st.sidebar.success(f"{template.name} · {len(slots)} mug{'s' if len(slots)!=1 else ''}")
st.sidebar.caption("Marker templates use their detected printable mug regions automatically.")
st.sidebar.divider()
st.sidebar.subheader("Fine tune")
scale=st.sidebar.slider("Artwork vertical scale",0.80,1.15,1.00,0.01)
yshift=st.sidebar.slider("Artwork vertical position",-0.12,0.12,0.0,0.01)
xshift=st.sidebar.slider("Artwork wrap position",-0.08,0.08,0.0,0.005)
slots=tuple(
    Slot(x.id,x.box,yaw_deg=x.yaw_deg,print_top=x.print_top,print_bottom=x.print_bottom,
         visible_deg=x.visible_deg,scale=scale,offset_x=xshift,offset_y=yshift,
         ink_strength=x.ink_strength,marker_mask=x.marker_mask)
    for x in slots
)
template=Template("uploaded",template.name,scene_path,slots)

st.image(scene,caption=f"{template.name} — {len(slots)} mugs · source {scene.width}×{scene.height}px",width="stretch")

st.subheader("Export quality")
export_mode=st.radio(
    "Output size",
    ["Source resolution","3000 px long edge","4000 px long edge"],
    index=1,
    horizontal=True,
    help="Rendering uses the source scene. The downloaded final image is then high-quality upscaled if needed."
)
st.caption("For Etsy-style listing masters, 3000 px is a practical default. 4000 px gives extra room for crops/zoom but cannot invent detail missing from the source.")

def _export_image(img:Image.Image)->Image.Image:
    target={"Source resolution":0,"3000 px long edge":3000,"4000 px long edge":4000}[export_mode]
    long=max(img.size)
    if not target or long>=target:
        return img
    ratio=target/long
    size=(int(round(img.width*ratio)),int(round(img.height*ratio)))
    # LANCZOS preserves artwork edges better than browser/display scaling.
    up=img.resize(size,Image.Resampling.LANCZOS)
    # Very light post-resize sharpening to restore edge acuity without haloing.
    return up.filter(ImageFilter.UnsharpMask(radius=0.7,percent=70,threshold=3))

st.subheader("Artwork")
uploads=st.file_uploader(
    "Upload full mug-print canvases",
    type=["png","jpg","jpeg"],
    accept_multiple_files=True,
    help="Upload the original full wrap files. Front and Rear are sampled from the same production canvas.",
)
arts=[(u.name,Image.open(BytesIO(u.getvalue())).convert("RGBA")) for u in uploads] if uploads else []
if not arts:
    st.info("Upload one or more full mug-print artwork files.")
    st.stop()

mode=st.radio("Mode",["Assign mugs","Batch"],horizontal=True)
default_face=st.radio("Default side",["Front","Rear"],horizontal=True)

if mode=="Assign mugs":
    st.subheader("Mug assignments")
    names=[name for name,_ in arts]
    assignments={}
    cols=st.columns(min(len(slots),4))
    for i,slot in enumerate(slots):
        with cols[i % len(cols)]:
            st.markdown(f"**Mug {i+1}**")
            selected=st.selectbox("Artwork",names,index=min(i,len(names)-1),key=f"artwork_{slot.id}")
            img=arts[names.index(selected)][1]
            st.image(img,caption=selected,width="stretch")
            side=st.radio("Side",["Front","Rear"],index=0 if default_face=="Front" else 1,
                          horizontal=True,key=f"face_{slot.id}")
            assignments[slot.id]=(img,side)

    if st.button("Render mockup",type="primary",width="stretch"):
        with st.spinner("Rendering locally…"):
            result=render(template,assignments)
        st.image(result,caption=f"Rendered preview · export is full resolution {result.width}×{result.height}px",width="stretch")
        exported=_export_image(result)
        st.caption(f"Download size: {exported.width}×{exported.height}px")
        jpg=BytesIO(); exported.convert("RGB").save(jpg,"JPEG",quality=98,subsampling=0,dpi=(300,300))
        png=BytesIO(); exported.save(png,"PNG",dpi=(300,300))
        a,b=st.columns(2)
        a.download_button("Download JPG",jpg.getvalue(),"mockup.jpg","image/jpeg",width="stretch")
        b.download_button("Download PNG",png.getvalue(),"mockup.png","image/png",width="stretch")
else:
    st.write(f"{len(arts)} designs · {len(slots)} mugs per output · {(len(arts)+len(slots)-1)//len(slots)} mockups")
    st.caption("Batch fills mugs left-to-right using the uploaded artwork order. The Default side applies to every mug in the batch.")
    if st.button("Render batch",type="primary",width="stretch"):
        with st.spinner("Rendering batch locally…"):
            results=render_batch(template,arts,default_face.lower())
        z=BytesIO()
        with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as archive:
            for label,img in results:
                exported=_export_image(img)
                b=BytesIO(); exported.convert("RGB").save(b,"JPEG",quality=98,subsampling=0,dpi=(300,300))
                archive.writestr(f"{label}.jpg",b.getvalue())
        st.success(f"Rendered {len(results)} mockups.")
        for label,img in results[:4]:
            st.image(img,caption=label,width="stretch")
        st.download_button("Download all as ZIP",z.getvalue(),"mockups.zip","application/zip",width="stretch")
