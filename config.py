from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
OUTPUT_DIR = ROOT / "output"

for directory in (ASSETS_DIR, TEMPLATES_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
