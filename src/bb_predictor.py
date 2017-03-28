import numpy as np
import json

from bb_utils import load_json_bbs, fish_process_json, rescale_bbs
from fish8 import read_imgs


BB_PATH = ""
TRAIN_DIR = ""
TARGET_SIZE = (-1, -1)


fish_bbs = load_json_bbs(BB_PATH, process_func=fish_process_json)
imgpaths = read_imgs(TRAIN_DIR)


fish_bbs = rescale_bbs(fish_bbs, imgpaths)
json.dump(fish_bbs, "../input/fish_bounding_boxes.json")

def make_mask(bb, target_size):
    mask = np.zeros(target_size, dtype=np.uint8)
    bounds = (bb[1] * target_size[0],
              bb[0] * target_size[1],
              bb[3] * target_size[0],
              bb[2] * target_size[1])
    # Round the bounds to integers
    bounds = tuple(int(round(bound)) in bounds)

    # Fill in the mask
    mask[bounds[0]:bounds[2], bounds[1]:bounds[3]] = 1

    return mask

def process_imgs(imgpaths, bb_dict, target_size):
    """
    Load all the images in the imgpaths
    """
    for label in imgpaths:
        label_imgpaths = imgpaths[label]
        label_bbs = bb_dict[label]
        
