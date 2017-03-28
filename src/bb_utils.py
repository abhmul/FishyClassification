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
                (
                    top left bb corner x dist (float),
                    top left bb corner y dist (float),
                    bottom right bb corner x dist (float),
                    bottom right bb corner y dist (float)
                ),
                ...
            ],
            ...
    If any negative "x" or "y" values are present, makes them 0.

    """
    reform_json = {}
    for img in json_bb:
        img_name = str(os.path.basename(img["filename"]))
        annotations = [(max(0, annot["x"]),
                        max(0, annot["y"]),
                        annot["width"] + annot["x"],
                        annot["height"] + annot["y"]) for annot in img["annotations"]]
        reform_json[img_name] = annotations
    return reform_json


def scale_bb(bb, img_size):
    """
    Rescales the bounding to 0 to 1 coordinates

    Arguments:
    bb -- The bounding box to rescale
    img_size -- The size of the image as (width, height)

    Returns:
    The bounding box rescaled to 0 to 1 coordinates
    """
    return (bb[0] / img_size[0],
            bb[1] / img_size[1],
            bb[2] / img_size[2],
            bb[3] / img_size[3])

def rescale_bbs(bb_dict, imgpaths):
    """
    Rescales a dictionary of bounding boxes for many labels

    Arguments:
    bb_dict -- A dictionary of labels mapped to a dictionary of bounding_boxes
    """
    new_bb_dict = {}
    for label, bbs in bb_dict.iteritems():
        new_bbs = {}

        for imgpath in imgpaths[label]:
            img_name = str(os.path.basename(imgpath))
            if img_name in bbs:

                new_bb =
                new_bbs[img_name] = [scale_bb(bb, bb_Image.open(imgpath).size) for bb in bbs[img_name]]

        new_bb_dict[label] = new_bbs
