import numpy as np
import pandas as pd
import logging
from functools import partial
import os
from sklearn.model_selection import StratifiedKFold, KFold

from keras.models import Sequential, Model
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Dense, Flatten, Reshape, Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img

from bb_utils import load_json_bbs, build_bb_scale
from transformations2 import random_rotation, random_shift, random_shear, random_zoom, compile_affine, apply_transform_coord
import fish8

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

picture_dir = '../input/train/'
plotting = False

if plotting:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def display_with_rect(im, rects):
        colors = ('r', 'b', 'g')
        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        # print im.transpose(2, 0, 1).shape
        print np.max(im)
        ax.imshow(im)

        for i, rect in enumerate(rects):
            edgecolor = colors[i]
            # Create a Rectangle patch
            rect = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=1, edgecolor=edgecolor, facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()

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
    bounding_boxes, no_boxes = build_bb_scale(load_json_bbs(directory), scale=True, picture_directory=picture_dir)
    return bounding_boxes, no_boxes


def build_labels(buckets, h, w, bounding_boxes, ids, transforms=None, images=None):

    nb_samples = len(ids)
    y_left = np.zeros((nb_samples, buckets, buckets))
    y_right = np.zeros((nb_samples, buckets, buckets))

    for i, img_name in enumerate(ids):

        top_left = (bounding_boxes[img_name][1]*h, bounding_boxes[img_name][0]*w)
        bottom_right = (bounding_boxes[img_name][3]*h, bounding_boxes[img_name][2]*w)
        bottom_left = (bounding_boxes[img_name][3]*h, bounding_boxes[img_name][0]*w)
        top_right = (bounding_boxes[img_name][1]*w, bounding_boxes[img_name][2]*w)
        if transforms is not None:
            # We need to do some affine transform to get the new bounding box coordinates
            # TODO See if hashing the transforms by image name is an ok way to do this, otherwise put in array

            top_left = apply_transform_coord(top_left, transforms[img_name], h, w)
            bottom_right = apply_transform_coord(bottom_right, transforms[img_name], h, w)
            bottom_left = apply_transform_coord(bottom_left, transforms[img_name], h, w)
            top_right = apply_transform_coord(top_right, transforms[img_name], h, w)

            # Reset  each to be top left and bottom right
            ys = top_left[0], bottom_right[0], bottom_left[0], top_right[0]
            xs = top_left[1], bottom_right[1], bottom_left[1], top_right[1]
            top_left = min(xs), min(ys)
            bottom_right = max(xs), max(ys)

        left_corner = np.clip(int(round(top_left[1]*(float(buckets)/w))), 0, buckets-1), \
                      np.clip(int(round(top_left[0]*(float(buckets)/h))), 0, buckets-1)
        right_corner = np.clip(int(round(bottom_right[1]*(float(buckets)/w))), 0, buckets-1), \
                       np.clip(int(round(bottom_right[0]*(float(buckets)/h))), 0, buckets-1)

        for j, val in enumerate(tuple(reversed(left_corner)) + tuple(reversed(right_corner))):
            if val >= buckets:
                raise ValueError('Produced index %s from %s' % (val, bounding_boxes[img_name][j]))

        y_left[i, left_corner[1], left_corner[0]] = 1.
        y_right[i, right_corner[1], right_corner[0]] = 1.

        if images is not None:
            print top_left
            print bottom_right
            y_left_coord = np.where(y_left[i] == 1.)
            y_right_coord = np.where(y_right[i] == 1.)
            y_left_coord = (y_left_coord[0][0] * (h / float(buckets)), y_left_coord[1][0] * (w / float(buckets)))
            y_right_coord = (y_right_coord[0][0] * (h / float(buckets)), y_right_coord[1][0] * (w / float(buckets)))
            print y_left_coord
            print y_right_coord
            display_with_rect(images[i], [(top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                                          (y_left_coord[1], y_left_coord[0], y_right_coord[1], y_right_coord[0])])

    return y_left, y_right


def extract_id(arr, ids, sub_ids):
    data = pd.Series(data=np.arange(len(ids)), index=ids)
    return arr[data.loc()[sub_ids].values]


def train(buckets, Xtr, Xval, bounding_boxes, tr_ids, val_ids, best_model_fname=None, generator=None, input_size=(256, 256, 3), dropout=0., optimizer=None,
          batch_size=32, nb_epoch=25, verbose=1):

    localizer = localize_net(buckets, input_size, dropout, optimizer)

    callbacks = None
    if best_model_fname is not None:
        # autosave best Model
        best_model = ModelCheckpoint(best_model_fname, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=True)
        callbacks = [best_model]

    if generator is None:
        raise NotImplementedError
        y_left, y_right = build_labels(buckets, bounding_boxes, tr_ids)
        y_left_val, y_right_val = build_labels(buckets, bounding_boxes, val_ids)
        fit = localizer.fit(Xtr, [y_left, y_right], batch_size, nb_epoch, verbose, callbacks,
                            validation_data=(Xval, [y_left_val, y_right_val]))
    else:
        fit = localizer.fit_generator(generator(Xtr, tr_ids, buckets, bounding_boxes, batch_size), samples_per_epoch=Xtr.shape[0],
                                      nb_epoch=nb_epoch, verbose=verbose, callbacks=callbacks,
                                      validation_data=generator(Xval, val_ids, buckets, bounding_boxes, batch_size),
                                      nb_val_samples=Xval.shape[0])
    return fit


def localizer_gen(X, ids, buckets, bounding_boxes, batch_size=32, transform_func=None, shuffle=True, save_to_dir=None):

    inds = np.arange(X.shape[0])
    while True:
        if shuffle:
            np.random.shuffle(inds)
        for i in xrange(0, len(inds), batch_size):
            batch_inds = inds[i:i+batch_size]
            batch_ids = ids[batch_inds]
            batch_x = X[batch_inds]
            transforms = {}
            if transform_func is not None:
                for j in xrange(batch_x.shape[0]):
                    batch_x[j], transforms[batch_ids[j]] = transform_func(batch_x[j])
                    if save_to_dir is not None:
                        img = array_to_img(batch_x[j])
                        img.save(os.path.join(save_to_dir, '{}.jpg'.format(j)))
            y_left, y_right = build_labels(buckets, X.shape[1], X.shape[2], bounding_boxes, batch_ids, transforms=transforms)
            for k in xrange(y_left.shape[0]):
                print y_left[k]
                print y_right[k]
                print batch_ids[k]
            yield batch_x, [y_left, y_right]

if __name__ == '__main__':
    Xtr, ytr, train_id = fish8.load_train_data(directory=picture_dir, target_size=(256, 256))
    bb_dict, bad_box_imgs = load_bb(directory='../bounding_boxes/')
    train_bb_id = np.sort(bb_dict.keys())

    Xtr = extract_id(Xtr, train_id, train_bb_id)
    ytr = extract_id(ytr, train_id, train_bb_id)
    print Xtr.shape
    print ytr.shape

    transform_func = compile_affine(
        [random_rotation(180.), random_shift(.2, .2), random_zoom((1 / 1.2, 1.2)), random_shear(.1)])

    for j, fish_name in enumerate(sorted(['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])):

        if fish_name == 'NoF':
            continue
        print 'Running %s' % fish_name
        inds = np.where(ytr[:, j]==1)[0]

        skf = KFold(n_splits=5, shuffle=True)
        skf.get_n_splits()
        for k, (ind_train_index, ind_test_index) in enumerate(skf.split(inds)):
            gen = partial(localizer_gen, transform_func=transform_func, shuffle=True)
            print 'Running fold {}/{}'.format(k+1, 5)
            train_index = inds[ind_train_index]
            test_index = inds[ind_test_index]
            X_train, X_val = Xtr[train_index], Xtr[test_index]
            tr_ids, val_ids = train_bb_id[train_index], train_bb_id[test_index]
            train(20, X_train, X_val, bb_dict, tr_ids, val_ids, best_model_fname='../localizer_{}_fold{}.h5'.format(fish_name, k+1), generator=gen,
                  nb_epoch=100)
