from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from compositing.renderer import render_scene
from config import TEMPLATES_DIR
from scenes.registry import discover_scenes

st.set_page_config(page_title="Mockup Manager", page_icon="☕", layout="wide")
st.title("☕ Mockup Manager")
st.caption("AI may invent the environment. Uploaded artwork remains authoritative.")

scenes = discover_scenes(TEMPLATES_DIR)
if not scenes:
    st.error("No scene templates found in assets/templates.")
    st.stop()

scene_names = {scene.name: scene for scene in scenes.values()}
selected_name = st.selectbox("Scene", list(scene_names))
scene = scene_names[selected_name]

uploads = st.file_uploader(
    "Upload mug artwork",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

artworks: dict[str, Image.Image] = {}
if uploads:
    for upload in uploads:
        artworks[upload.name] = Image.open(upload).convert("RGBA")

assignments = {}
overrides = {}

left, right = st.columns([1, 1.6])

with left:
    st.subheader("Mug slots")
    choices = ["— none —", *artworks.keys()]

    for slot in scene.slots:
        with st.expander(slot.label, expanded=True):
            chosen = st.selectbox(
                "Artwork",
                choices,
                key=f"artwork-{slot.id}",
            )
            curvature = st.slider(
                "Curvature",
                min_value=-0.8,
                max_value=0.8,
                value=float(slot.curvature),
                step=0.01,
                key=f"curve-{slot.id}",
            )
            opacity = st.slider(
                "Print opacity",
                min_value=0.0,
                max_value=1.0,
                value=float(slot.opacity),
                step=0.01,
                key=f"opacity-{slot.id}",
            )

            if chosen != "— none —":
                assignments[slot.id] = artworks[chosen]
            overrides[slot.id] = {
                "curvature": curvature,
                "opacity": opacity,
            }

with right:
    st.subheader("Preview")
    result = render_scene(scene, assignments, slot_overrides=overrides)
    st.image(result, use_container_width=True)

png_buffer = io.BytesIO()
result.save(png_buffer, format="PNG")

jpg_buffer = io.BytesIO()
result.convert("RGB").save(jpg_buffer, format="JPEG", quality=95)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download PNG",
        data=png_buffer.getvalue(),
        file_name=f"{scene.id}-mockup.png",
        mime="image/png",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "Download JPG",
        data=jpg_buffer.getvalue(),
        file_name=f"{scene.id}-mockup.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )

st.divider()
st.subheader("AI custom scene")
st.info(
    "The provider adapter is scaffolded, but intentionally not connected yet. "
    "When enabled, it will generate only blank-mug scenes; uploaded artwork "
    "will still be applied by the deterministic renderer."
)
