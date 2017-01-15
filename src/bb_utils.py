import os
import json
from PIL.Image import BILINEAR, open


def load_json_bbs(bb_path):
    class_lst = []
    for fname in os.listdir(bb_path):
        with open(os.path.join(bb_path, fname), 'r') as bb:
            class_lst.append(json.load(bb))
    return class_lst


#TODO depracate this version of the function and replace with a build_bb_scale
def build_bb(class_lst):
    bounding_boxes = {}
    no_boxes = []
    for i, label in enumerate(class_lst):
        for rect in label:
            if len(rect["annotations"]) > 0:
                bounding_boxes[str(rect["filename"][-13:])] = (rect["annotations"][0]["x"],
                                            rect["annotations"][0]["y"],
                                            rect["annotations"][0]["width"],
                                            rect["annotations"][0]["height"])
            # If there are no pictures
            else:
                no_boxes.append(rect["filename"][-13:])
    return bounding_boxes, no_boxes


def build_bb_scale(class_lst, scale=False, picture_directory='../input/train/'):
    bounding_boxes = {}
    no_boxes = []
    for i, label in enumerate(class_lst):
        for rect in label:
            if len(rect["annotations"]) == 1:

                bb = (rect["annotations"][0]["x"], rect["annotations"][0]["y"],
                      rect["annotations"][0]["x"] + rect["annotations"][0]["width"],
                      rect["annotations"][0]["y"] + rect["annotations"][0]["height"])

                if scale:
                    # We want to replace the rectangle coords with numbers in [0,1]
                    img = open(os.path.join(picture_directory, str(rect["filename"][-13:])))
                    img_width, img_height = img.size
                    bb = (float(bb[0]) / img_width, float(bb[1]) / img_height,
                          float(bb[2]) / img_width, float(bb[3]) / img_height)

                bounding_boxes[str(rect["filename"][-13:])] = bb
            # If there are no pictures
            else:
                no_boxes.append(rect["filename"][-13:])
    return bounding_boxes, no_boxes


def max_dim_scale(bounding_boxes, target_size):
    max_dim = max(max(bounding_boxes[img][2:]) for img in bounding_boxes)
    return float(target_size) / max_dim


def rescale_bb(bounding_boxes, scale):
    return {img: tuple([val * scale for val in bounding_boxes[img]])
            for img in bounding_boxes}