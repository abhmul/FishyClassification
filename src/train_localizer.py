import numpy as np

from keras.models import Sequential, Model
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Dense, Flatten, Reshape, Input
from keras.optimizers import RMSprop

from bb_utils import load_json_bbs, build_bb_scale


# TODO Run tests to figure out whether buckets is actually buckets x buckets
def localize_net(buckets, input_size=(256, 256, 3), dropout=0., optimizer=None):

    input_img = Input(input_size)

    x = ZeroPadding2D((1, 1), input_shape=input_size)(input_img)
    x = Convolution2D(16, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output_1 = Dense(buckets**2, activation='softmax')(x)
    output_1 = Reshape((buckets, buckets), name='left_corner')(output_1)

    output_2 = Dense(buckets**2, activation='softmax')(x)
    output_2 = Reshape((buckets, buckets), name='right_corner')(output_2)

    localizer = Model(input_img, [output_1, output_2])

    if optimizer is None:
        optimizer = RMSprop()

    localizer.compile(optimizer,
                      loss={'left_corner': 'categorical_crossentropy',
                            'right_corner': 'categorical_crossentropy'})
    return localizer


def load_bb(directory='../bounding_boxes/'):
    bounding_boxes, no_boxes = build_bb_scale(load_json_bbs(directory))
    return bounding_boxes, no_boxes


def build_labels(buckets, bounding_boxes, train_id):

    nb_samples = len(bounding_boxes)
    y_left = np.zeros((nb_samples, buckets, buckets))
    y_right = np.zeros((nb_samples, buckets, buckets))

    for i, img_name in enumerate(train_id):
        left_corner = int(bounding_boxes[img_name][1]*buckets), int(bounding_boxes[img_name][0]*buckets)
        right_corner = [int(bounding_boxes[img_name][3]*buckets), int(bounding_boxes[img_name][2]*buckets)]

        if right_corner[0] < bounding_boxes[img_name][3]*buckets:
            right_corner[0] += 1
        if right_corner[1] < bounding_boxes[img_name][2]*buckets:
            right_corner[1] += 1

        y_left[i, left_corner[0], left_corner[1]] = 1.
        y_right[i, right_corner[0], right_corner[1]] = 1.

    return y_left, y_right
        

