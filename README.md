# Mockup Manager

AI-assisted, deterministic mug mockup generator.

## Product rule

**AI may invent the environment, but it must never be the source of truth for the product artwork.**

Uploaded artwork is treated as authoritative. The renderer applies it using deterministic image transforms rather than asking an image model to redraw the design.

## MVP

- Upload PNG/JPG artwork
- Select a predefined scene
- Assign artwork to mug slots
- Apply cylindrical + perspective transforms
- Preserve scene lighting over the print
- Preview result
- Export PNG/JPG
- Experimental AI custom-scene adapter interface

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The starter repository includes a generated placeholder scene so the complete pipeline can run without external assets. Replace it with a real blank-mug photograph and tune the slot JSON for production-quality results.
