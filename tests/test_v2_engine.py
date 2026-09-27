from pathlib import Path
import numpy as np
from PIL import Image
from v2.engine import Slot,Template,_sample_wrap,_project

def test_wrap_preserves_height_and_requested_width():
    a=Image.new("RGBA",(2048,849),(255,0,0,255))
    out=_sample_wrap(a,.25,136,777)
    assert out.shape==(849,777,4)

def test_projection_is_confined_to_slot():
    art=Image.new("RGBA",(2048,849),(255,0,0,255))
    slot=Slot("m",(0.25,0.2,0.75,0.8))
    layer=np.asarray(_project(art,(1000,800),slot,"front"))
    ys,xs=np.where(layer[...,3]>0)
    assert xs.min()>=250 and xs.max()<750
    assert ys.min()>=int((.2+(.8-.2)*slot.print_top)*800)-1
    assert ys.max()<=int((.2+(.8-.2)*slot.print_bottom)*800)+1

def test_front_and_rear_sample_different_source_regions():
    arr=np.zeros((20,2048,4),np.uint8); arr[...,3]=255
    arr[:,400:625,0]=255; arr[:,1420:1650,1]=255
    art=Image.fromarray(arr,"RGBA")
    front=_sample_wrap(art,.25,40,200)
    rear=_sample_wrap(art,.75,40,200)
    assert front[...,0].mean()>front[...,1].mean()
    assert rear[...,1].mean()>rear[...,0].mean()


def test_vertical_scale_keeps_projection_valid():
    art=Image.new("RGBA",(2048,849),(10,20,30,255))
    slot=Slot("m",(0.2,0.2,0.8,0.8),scale=.9)
    alpha=np.asarray(_project(art,(600,600),slot,"front"))[...,3]
    assert alpha.max()==255

def test_wrap_sampling_changes_with_centre():
    arr=np.zeros((10,2048,4),np.uint8); arr[...,3]=255
    arr[:,450:575,0]=255
    art=Image.fromarray(arr,"RGBA")
    a=_sample_wrap(art,.25,30,300)
    b=_sample_wrap(art,.35,30,300)
    assert not np.array_equal(a,b)
