import os
import json
from PIL import Image
from fish8 import infer_classes


def load_json_bbs(bb_path, process_func=None):
    """
    Given a path, loads a dictionary that maps each class
    to its corresponding bounding box json

    Arguments:
    bb_path -- the path to the folder containing all the bounding boxes.
               Should be in the following format:
               bb_path/
                   label1/
                       label1_bounding_boxes.json
                   label2/
                       label1_bounding_boxes.json
                   ...
    process_func -- if provided, is run on the loaded json to process it

    Returns:
    A dictionary mapping label to a loaded json of bounding boxes.
    """
    classes = infer_classes(bb_path)

    def load_json_bb(bb_path_label):
        for fname in os.listdir(bb_path_label):
            with open(os.path.join(bb_path_label, fname), 'r') as bb:
                json_bb = json.load(bb)
                if process_func is not None:
                    json_bb = process_func(json_bb)
        return json_bb

    return {label: load_json_bb(os.path.join(bb_path, label)) for label in classes}


def fish_process_json(json_bb):
    """
    A function that can turn the fish json_bb structure into
    the desired structure.

    Arguments:
    json_bb -- A loaded json of bounding boxes for a specific type of fish.
               The structure is:
                   [
                       {
                           "annotations": [
                               {
                                   "class": "rect",
                                   "height": float,
                                   "width": float,
                                   "x": top left bb corner x dist (float),
                                   "y": float
                               },
                               ...
                           ],
                           "class": "image",
                           "filename": Image Filename (sometimes w/ path)
                        },
                        ...

    Returns:
    The reformated json_bb as:
        {
            "filename": Image Filename (sometimes w/ path): [
                {
                    "height": y-height of bb (float),
                    "width": x-width of bb (float),
                    "x": top left bb corner x dist (float),
                    "y": top left bb corner y dist (float)
                },
                ...
            ],
            ...
    If any negative "x" or "y" values are present, makes them 0.

    """
    reform_json = {}
    for img in json_bb:
        img_name = str(os.path.basename(img["filename"]))
        annotations = [{"height": annot["height"],
                            "width": annot["width"],
                            "x": max(0, annot["x"]),
                            "y": max(0, annot["y"])} for annot in img["annotations"]]
        reform_json[img_name] = annotations
    return reform_json


### DEPRECATED ###


def build_bb_all(class_dct, classes=None, failures=False):
    bounding_boxes = {}
    no_boxes = set([])
    if classes is None:
        classes = class_dct.keys()

    for i, label in enumerate(classes):
        json_bb = class_dct[label]
        for rect in json_bb:
            img_name = str(os.path.basename(rect["filename"]))
            if len(rect["annotations"]) >= 1:

                bb = [(ann["x"], ann["y"],
                        ann["x"] + ann["width"],
                        ann["y"] + ann["height"]) for ann in rect["annotations"]]


                bounding_boxes[img_name] = bb
            else:
                no_boxes.add(img_name)

    return bounding_boxes, no_boxes


#TODO depracate this version of the function and replace with a build_bb_scale
def build_bb(class_dct):
    bounding_boxes = {}
    no_boxes = []
    for i, (label, json_bb) in enumerate(class_dct.iteritems()):
        for rect in json_bb:
            if len(rect["annotations"]) > 0:
                bounding_boxes[str(rect["filename"][-13:])] = (rect["annotations"][0]["x"],
                                            rect["annotations"][0]["y"],
                                            rect["annotations"][0]["width"],
                                            rect["annotations"][0]["height"])
            # If there are no pictures
            else:
                no_boxes.append(rect["filename"][-13:])
    return bounding_boxes, no_boxes


def build_bb_scale(class_dct, scale=False, classes=None, picture_directory='../input/train/'):
    bounding_boxes = {}
    no_boxes = []
    if classes is None:
        classes = class_dct.keys()

    for i, label in enumerate(classes):
        json_bb = class_dct[label]
        for rect in json_bb:
            if len(rect["annotations"]) == 1:

                bb = (rect["annotations"][0]["x"], rect["annotations"][0]["y"],
                      rect["annotations"][0]["x"] + rect["annotations"][0]["width"],
                      rect["annotations"][0]["y"] + rect["annotations"][0]["height"])

                if scale:

                    # We want to replace the rectangle coords with numbers in [0,1)
                    img = Image.open(os.path.join(picture_directory, label, str(rect["filename"][-13:])))
                    img_width, img_height = img.size
                    bb = (float(bb[0]-1) / img_width, float(bb[1]-1) / img_height,
                          float(bb[2]-1) / img_width, float(bb[3]-1) / img_height)

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
