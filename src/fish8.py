import time
import os
import glob
import datetime

import pandas as pd
import numpy as np

from PIL.Image import BILINEAR, open

from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical

from GLOBALS import FOLDERS, INPUT_IMGSIZE

def load_imgarr(path):
    img = open(path)
    resized = img.resize(INPUT_IMGSIZE, resample=BILINEAR)
    return img_to_array(resized)

def load_train_from_dir():
    """
    Function to load all the training images from the fish folders
    in the order of FOLDERS
    :return: 3 arrays, one of the training image arrays, another of labels, and the other of image ids
    """

    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = FOLDERS
    for index, fld in enumerate(folders):
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = load_imgarr(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return reshape_img_data(np.array(X_train)), \
           to_categorical(np.array(y_train), 8), \
           np.array(X_train_id)

def load_test_from_dir():
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = load_imgarr(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return reshape_img_data(np.array(X_test)), np.array(X_test_id)

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def reshape_img_data(X):
    return X.reshape((-1, 3,) + INPUT_IMGSIZE)

def normalize_data(X):

    print('Convert to float...')
    train_data = X.astype('float32')
    train_data /= 255

    print('Normalizing the data')
    train_data -= np.mean(train_data, axis=0)
    train_data /= (np.std(train_data, axis=0) + 1e-7)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data


def load_train_data():

    Xtr, ytr, trid = load_train_from_dir()
    Xtr = normalize_data(Xtr)
    return Xtr, ytr, trid


def load_test_data():

    Xte, teid = load_test_from_dir()
    Xte = normalize_data(Xte)
    return Xte, teid