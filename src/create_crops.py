import os
import glob
import numpy as np
from PIL import Image

from bb_utils import build_bb_scale, load_json_bbs


picture_dir = '../input/train/'
crop_dir = '../input/train_cropped/'
classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

def load_bb(directory='../bounding_boxes/'):
    bounding_boxes, no_boxes = build_bb_scale(load_json_bbs(directory), scale=False, picture_directory=picture_dir)
    return bounding_boxes, no_boxes

bb_dict, bad_box_imgs = load_bb(directory='../bounding_boxes/')
train_bb_id = set(bb_dict.keys())

for fish_name in classes:
    path = os.path.join(picture_dir, fish_name, '*.jpg')
    files = glob.glob(path)
    for fl in files:
        img = Image.open(fl)
        flbase = os.path.basename(fl)
        if flbase in train_bb_id:
            crop_box = tuple(int(i) for i in bb_dict[flbase])
            img = img.crop(crop_box)
            img.save(os.path.join(crop_dir, fish_name, flbase))
