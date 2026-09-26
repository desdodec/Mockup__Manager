from __future__ import annotations

import io
import streamlit as st
from PIL import Image

from compositing.renderer import render_scene
from config import TEMPLATES_DIR
from scenes.registry import discover_scenes
from scripts import create_demo_template  # noqa: F401

st.set_page_config(page_title="Mockup Manager", page_icon="☕", layout="wide")
st.title("☕ Mockup Manager")
st.caption("Full mug-print canvases stay authoritative; only the camera-visible wrap is sampled.")

scenes = discover_scenes(TEMPLATES_DIR)
if not scenes:
    st.error("No scene templates found.")
    st.stop()

scene_names = {s.name: s for s in scenes.values()}
scene = scene_names[st.selectbox("Scene", list(scene_names))]
uploads = st.file_uploader("Upload complete mug-print artwork", type=["png","jpg","jpeg"], accept_multiple_files=True)
artworks = {u.name: Image.open(u).convert("RGBA") for u in uploads} if uploads else {}

assignments, overrides = {}, {}
left, right = st.columns([1, 1.6])
with left:
    st.subheader("Mug slots")
    choices = ["— none —", *artworks.keys()]
    for slot in scene.slots:
        with st.expander(slot.label, expanded=True):
            chosen = st.selectbox("Artwork", choices, key=f"art-{slot.id}")
            face = st.radio("View", ["Front", "Rear", "Custom angle"], horizontal=True, key=f"face-{slot.id}")
            if face == "Front":
                angle = 0.0
            elif face == "Rear":
                angle = 180.0
            else:
                angle = st.slider("View angle", 0, 359, int(slot.view_angle), key=f"angle-{slot.id}")
            visible = st.slider("Visible circumference", 0.20, 0.65, float(slot.visible_fraction), 0.01, key=f"visible-{slot.id}")
            curve = st.slider("Curvature", -0.8, 0.8, float(slot.curvature), 0.01, key=f"curve-{slot.id}")
            opacity = st.slider("Print strength", 0.0, 1.0, float(slot.opacity), 0.01, key=f"opacity-{slot.id}")
            if chosen != "— none —":
                assignments[slot.id] = artworks[chosen]
            overrides[slot.id] = {"view_angle": angle, "visible_fraction": visible, "curvature": curve, "opacity": opacity}

with right:
    st.subheader("Preview")
    result = render_scene(scene, assignments, slot_overrides=overrides)
    st.image(result, use_container_width=True)

png = io.BytesIO(); result.save(png, "PNG")
jpg = io.BytesIO(); result.convert("RGB").save(jpg, "JPEG", quality=95)
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download PNG", png.getvalue(), f"{scene.id}-mockup.png", "image/png", use_container_width=True)
with c2:
    st.download_button("Download JPG", jpg.getvalue(), f"{scene.id}-mockup.jpg", "image/jpeg", use_container_width=True)

st.divider()
st.subheader("AI custom scene")
st.info("Scene generation remains isolated from product artwork. The uploaded print canvas is never sent through the scene generator.")
