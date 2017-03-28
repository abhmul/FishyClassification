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
