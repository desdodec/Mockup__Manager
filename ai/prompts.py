SYSTEM_SCENE_RULES = """
Create a photorealistic premium editorial product photograph.

The scene MUST contain exactly the requested number of ordinary white ceramic
11oz-style coffee mugs. Every mug must be physically plausible and have:
- a completely blank, undecorated, unprinted white exterior
- no logos, writing, patterns, labels, pseudo-text or embossed marks
- a clearly visible broad cylindrical body suitable for adding print later
- handles that remain visually distinct from the printable body
- realistic ceramic highlights, reflections, shadows and contact shadows
- enough separation that mugs do not overlap each other's printable bodies

Do not add any graphic design to the mugs. Do not invent product artwork.
Do not place people, hands, text overlays, watermarks or captions in the image.
Compose the mugs as the hero products, with useful visible front surfaces.
The image will later receive exact customer artwork using deterministic
computer-vision compositing, so clean mug surfaces are essential.
""".strip()


def build_scene_prompt(user_prompt: str, mug_count: int) -> str:
    return (
        f"{SYSTEM_SCENE_RULES}\n\n"
        f"EXACT MUG COUNT: {mug_count}.\n"
        f"ENVIRONMENT / ART DIRECTION: {user_prompt.strip()}\n"
        "Professional commercial product photography, natural lens behaviour, "
        "believable depth of field, high material realism."
    )
