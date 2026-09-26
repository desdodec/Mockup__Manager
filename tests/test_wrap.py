from PIL import Image, ImageDraw
from compositing.wrap import extract_visible_wrap


def make_wrap():
    image = Image.new("RGBA", (1000, 300), "white")
    d = ImageDraw.Draw(image)
    d.rectangle((150, 50, 350, 250), fill="red")
    d.rectangle((650, 50, 850, 250), fill="blue")
    return image


def test_front_and_rear_sample_different_faces():
    art = make_wrap()
    front = extract_visible_wrap(art, view_angle=0, visible_fraction=0.30)
    rear = extract_visible_wrap(art, view_angle=180, visible_fraction=0.30)
    assert front.getpixel((front.width // 2, 150))[:3] == (255, 0, 0)
    assert rear.getpixel((rear.width // 2, 150))[:3] == (0, 0, 255)


def test_source_is_not_modified():
    art = make_wrap()
    before = art.tobytes()
    extract_visible_wrap(art, view_angle=37, visible_fraction=0.42)
    assert art.tobytes() == before
