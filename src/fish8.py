import time
import os
import glob
import datetime
import logging

import pandas as pd
import numpy as np

from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical


def load_img(path, target_size=None, resampler=Image.BILINEAR):
    img = Image.open(path)
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), resample=resampler)
    return img_to_array(img)


def infer_classes(directory):
    return [subdir for subdir in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, subdir))]


def load_train_from_dir(classes, directory, target_size):
    """
    Function to load all the training images from the fish folders
    in the order of FOLDERS
    :return: 3 arrays, one of the training image arrays, another of labels, and the other of image ids
    """

    X_train = []
    train_id = []
    y_train = []
    start_time = time.time()

    if classes is None:
        classes = infer_classes(directory)

    logging.info('Found %d classes' % len(classes))

    for index, fld in enumerate(classes):
        logging.info('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(directory, fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = load_img(fl, target_size)
            X_train.append(img)
            train_id.append(flbase)
            y_train.append(index)

    logging.info('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return np.array(X_train), to_categorical(np.array(y_train), 8), np.array(train_id)


def load_test_from_dir(directory, target_size):
    files = sorted(glob.glob(directory))

    X_test = []
    test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = load_img(fl, target_size)
        X_test.append(img)
        test_id.append(flbase)

    return np.array(X_test), np.array(test_id)


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def normalize_data(X):

    print('Convert to float...')
    train_data = X.astype('float32')
    train_data /= 255.

    print('Normalizing the data')
    # train_data -= np.mean(train_data, axis=(1,2))
    # train_data /= (np.std(train_data, axis=(1,2)) + 1e-7)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data


def load_train_data(classes=None, directory='../input/train/', target_size=(256, 256)):

    Xtr, ytr, trid = load_train_from_dir(classes, directory, target_size)
    Xtr = normalize_data(Xtr)
    return Xtr, ytr, trid


def load_test_data(directory='../input/test/test_stg1', target_size=(256, 256)):

    Xte, teid = load_test_from_dir(directory, target_size)
    Xte = normalize_data(Xte)
    return Xte, teid
