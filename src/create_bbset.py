import numpy as np
import json
from keras.preprocessing.image import img_to_array
import os
import time

from bb_utils import load_json_bbs, fish_process_json, rescale_bbs
from fish8 import read_imgs, load_img

BB_PATH = "../bounding_boxes/"
TRAIN_DIR = "../input/train/"
TARGET_SIZE = (448, 448)
VERIFY = False

# Some debugging stuff
if VERIFY:
    import matplotlib.pyplot as plt
    plt.ion()


def make_mask(bb, target_size, init_mask=None):
    """
    Given a bounding box, produces a numpy mask of the image
    with dimensions target_size

    Inputs:
    bb -- The bounding box in the form (x1, y1, x2, y2) all from 0 to 1
    target_size -- The output shape of the mask

    Returns:
    The mask with True in the bounding box and False outside
    """
    # Initialize the mask
    mask = init_mask
    if mask is None:
        mask = np.zeros(target_size, dtype=bool)
    # Scale the bounds to the target size
    bounds = (bb[1] * target_size[0],
              bb[0] * target_size[1],
              bb[3] * target_size[0],
              bb[2] * target_size[1])
    # Round the bounds to integers
    bounds = tuple(int(round(bound)) for bound in bounds)

    # Fill in the mask
    mask[bounds[0]:bounds[2], bounds[1]:bounds[3]] = True

    return mask


def dict_len(d):
    """
    Given a dictionary, calculates the number of values recursively
    """
    if isinstance(d, dict):
        return sum(dict_len(v) for v in d.values())
    elif isinstance(d, list):
        return len(d)
    else:
        return 1


def process_imgs(imgpaths, bb_dict, target_size, verify=False):
    """
    Load all the images in the imgpaths
    """
    # Calculate the number of samples
    samples = dict_len(imgpaths)
    print("Dataset has %s" % samples)
    # Initialize the image and mask arrays
    img_list = np.empty((samples,) + (target_size) + (3,), dtype=np.uint8)
    mask_list = np.empty((samples,) + (target_size) + (1,), dtype=np.uint8)

    # Loop through each class
    i = 0
    for label in imgpaths:
        print("LABEL: %s" % label)
        # Loop through all the images for that class
        label_imgpaths = imgpaths[label]
        for img_fpath in label_imgpaths:
            # Get the filename from the path
            img_fname = os.path.basename(img_fpath)
            # If the we have a bounding box for that class and image
            # Create a mask for the image
            mask = np.zeros(target_size)
            if (label in bb_dict) and (img_fname in bb_dict[label]):
                label_bbs = bb_dict[label]
                # We could have multiple bounding boxes, so take the union
                for bb in label_bbs[img_fname]:
                    mask = make_mask(bb, target_size, init_mask=mask)
                try:
                    assert(np.all((mask == 1) + (mask == 0)))
                except AssertionError:
                    print(mask)
                    raise(AssertionError)

            # Add a channel axis to the mask
            mask = mask[:, :, np.newaxis].astype(np.uint8)
            # Load the actual image to the target size
            imgarr = img_to_array(load_img(img_fpath, target_size=target_size))

            # Combine the image and mask along the channel axis
            img_list[i] = imgarr.astype(np.uint8)
            mask_list[i] = mask.astype(np.uint8)
            i += 1

            # Some debugging code
            if (verify and label in bb_dict and (img_fname in bb_dict[label]) ):
                plt.imshow((imgarr.astype(np.uint8) * mask.astype(np.uint8)))
                # Wait for .5 second
                plt.pause(.5)
                print("Image %s" % i)
                pass
    # Combine the image arrays that we loaded
    return img_list, mask_list



# Load the json bounding boxes
fish_bbs = load_json_bbs(BB_PATH, process_func=fish_process_json)
# Read in the train directory structure
imgpaths = read_imgs(TRAIN_DIR)

# Rescale the bounding boxes to 0 to 1
fish_bbs = rescale_bbs(fish_bbs, imgpaths)
# Save the json for future use
with open("../input/fish_bounding_boxes.json", 'w') as json_fish:
    json.dump(fish_bbs, json_fish)

# Create the dataset
img_list, mask_list = process_imgs(imgpaths, fish_bbs, TARGET_SIZE, verify=VERIFY)
print("Training Dataset Shape: {}".format(img_list.shape))
assert(img_list.shape[:-1] == mask_list.shape[:-1])
# Save them
np.save(os.path.join(TRAIN_DIR, "train_imgs.npy"), img_list)
np.save(os.path.join(TRAIN_DIR, "train_masks.npy"), mask_list)
