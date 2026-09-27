SYSTEM_SCENE_RULES = """
Create a photorealistic premium product photograph.

PRODUCT GEOMETRY HAS PRIORITY OVER ART DIRECTION.

Create exactly the requested number of IDENTICAL standard straight-sided 11oz
white ceramic sublimation mugs.

Every mug MUST be:
- perfectly upright, standing on its base
- a simple straight vertical cylinder with standard 11oz proportions
- level at the rim and base
- completely plain glossy white ceramic
- free of artwork, text, logos, patterns, embossing and decoration
- fully visible in frame
- separated from every other mug
- unobstructed across the entire central body
- sharply focused
- shown with its handle clearly on the LEFT or RIGHT

Do NOT use tapered, tilted, conical, irregular, handmade or novelty mugs.
Do NOT overlap mugs.
Do NOT place flowers, hands, spoons, steam, cloth or props in front of them.
Do NOT add other cups, mugs, jugs or mug-shaped objects.

CAMERA:
Use a normal commercial product-photography viewpoint approximately level with
the centre of the mug bodies. Keep mug verticals visually vertical. Avoid
overhead views, fisheye, extreme wide angle, dramatic perspective and camera
roll. The central cylindrical faces must appear broad and easy to print onto.

The environment may be creative and atmospheric, but these product geometry
rules take priority. Preserve realistic ceramic highlights, reflections,
contact shadows and natural depth of field.
""".strip()

def build_scene_prompt(user_prompt: str, mug_count: int) -> str:
    return (
        f"{SYSTEM_SCENE_RULES}\n\n"
        f"EXACT MUG COUNT: {mug_count}.\n"
        f"ENVIRONMENT / ART DIRECTION: {user_prompt.strip()}\n\n"
        f"Final verification: exactly {mug_count} standard blank white 11oz mugs, "
        "all upright, separated and unobstructed; no additional drinkware."
    )
