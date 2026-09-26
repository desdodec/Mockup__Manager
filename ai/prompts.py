SYSTEM_SCENE_RULES = """
Generate product-photography environments containing blank white ceramic mugs.
Do not place logos, writing, drawings, patterns, branding, labels or pseudo-text
on the mugs. Keep printable mug bodies clearly visible and unobstructed.
The generated image is a scene asset only; real product artwork will be
composited later by deterministic code.
""".strip()


def build_scene_prompt(user_prompt: str, mug_count: int) -> str:
    return (
        f"{SYSTEM_SCENE_RULES}\n\n"
        f"Required mug count: {mug_count}.\n"
        f"Scene request: {user_prompt.strip()}"
    )
