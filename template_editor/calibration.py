from __future__ import annotations
from PIL import Image,ImageDraw
def rectangle_corners(cx:int,cy:int,width:int,height:int):
    hw,hh=width/2,height/2; return ((cx-hw,cy-hh),(cx+hw,cy-hh),(cx+hw,cy+hh),(cx-hw,cy+hh))
def draw_calibration_overlay(image:Image.Image,slots:list[dict])->Image.Image:
    out=image.convert("RGBA").copy(); ov=Image.new("RGBA",out.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
    for n,s in enumerate(slots,1):
        if s.get("mesh"):
            mesh=s["mesh"]
            for row in mesh:d.line(row,fill=(30,144,255,155),width=2)
            for j in range(len(mesh[0])):d.line([row[j] for row in mesh],fill=(30,144,255,120),width=2)
            if s.get("axis"):d.line(s["axis"],fill=(255,190,0,255),width=4)
            x,y=mesh[0][0]
            d.rounded_rectangle((x,y-34,x+150,y-6),radius=6,fill=(0,0,0,190))
            d.text((x+8,y-29),f"Mug {n} · {s.get('handle_side','?')} handle",fill="white")
        else:
            p=[tuple(x) for x in s["corners"]]; d.polygon(p,fill=(30,144,255,45),outline=(30,144,255,255),width=4)
            x,y=p[0]; d.text((x,y-20),f"Mug {n}",fill="white")
    return Image.alpha_composite(out,ov)
def build_scene_definition(scene_id,name,image_name,slots):
    return {"id":scene_id,"name":name,"image":image_name,"slots":[]}
