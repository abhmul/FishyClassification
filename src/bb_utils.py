import os
import json


def load_json_bbs(root, bb_path):
    class_lst = []
    for fname in os.listdir(os.path.join(root, bb_path)):
        with open(os.path.join(root, bb_path, fname), 'r') as bb:
            class_lst.append(json.load(bb))
    return class_lst


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


def max_dim_scale(bounding_boxes, target_size):
    max_dim = max(max(bounding_boxes[img][2:]) for img in bounding_boxes)
    return float(target_size) / max_dim


def rescale_bb(bounding_boxes, scale):
    return {img: tuple([val * scale for val in bounding_boxes[img]])
            for img in bounding_boxes}
