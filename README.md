# Mockup Manager V2

Fast, local, deterministic mug mockup production.

## V2 rule

Artwork and mug geometry are authoritative. Runtime AI detection is not part of
the production render path.

A scene is calibrated once as a template. Every later render reuses that known
geometry, so generating another mockup is only:

1. choose template;
2. upload full mug-print artwork;
3. choose Front or Rear;
4. render/export.

The supplied V2 app defaults to the two-mug Yorkshire Garden scene on the
author's Windows workstation and supports two-artwork preview plus batch ZIP
rendering.

## Artwork

Full 2048x849 production canvases are accepted without trimming. Front is
centred at 25% of the circumference and Rear at 75%. Rendering samples the
required angular window and maps it continuously to the calibrated cylinder.

## Run

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Architecture

`v2/engine.py` is the production renderer. It has no YOLO, image-generation
API, runtime mug detector, or mesh-reconstruction dependency. V1 modules remain
in the repository for reference but are not imported by the V2 app.
