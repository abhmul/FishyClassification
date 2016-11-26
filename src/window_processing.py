import numpy as np
from scipy.ndimage import zoom

from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import unittest

from GLOBALS import INPUT_IMGSIZE, SLIDING_WINDOW_RATIO, WINDOW_MIN_RATIO, XSTRIDE, YSTRIDE


def slide_window(X, window_size, xstride=XSTRIDE, ystride=YSTRIDE):

    for i in xrange(X.shape[-2], X.shape[-2] - window_size[0] + 1, ystride):
        for j in xrange(X.shape[-1], X.shape[-1] - window_size[1] + 1, xstride):
            yield X[:, :, i:i+window_size[0], j:j+window_size[1]]


def window_gen(X, sliding_window_ratio=SLIDING_WINDOW_RATIO, window_min_ratio=WINDOW_MIN_RATIO):

    window_size = X.shape[-2:]
    while X.shape[-2] * window_min_ratio <= window_size[0] and X.shape[-1] * window_min_ratio <= window_size[1]:
        window_size = tuple(sliding_window_ratio * dim for dim in window_size)
        for window in slide_window(X, window_size):
            yield window


def batch_generator(X, y, batch_size, shuffle, img_size=INPUT_IMGSIZE):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch,  y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def softmax(arr):

    return NotImplementedError


def normalize_multiclass(arr, axis=0):

    return arr / np.sum(arr, axis=axis).reshape(1, -1).T


def score_image_minimax(predictions, normalize_func=normalize_multiclass, eps=1e-9, testing=False):

    if testing:
        assert(predictions.shape[1:] == (32, 8))

    # predictions are WINDOWS x 32 x 8
    nof_min = np.min(predictions[:, :, 0:1], axis=0)  # 32 x 1

    fish_max = np.max(predictions[:, :, 1:], axis=0)  # 32 x 7

    return normalize_func(np.hstack([nof_min, fish_max]).clip(eps), axis=1)  # 32 x 8


def predict_window(X, model, img_size=INPUT_IMGSIZE):

    predictions = []
    num_windows = 0
    for i, window in enumerate(window_gen(X)):
        predictions.append(model.predict_on_batch(zoom(window, (1, 1,) + img_size)))
        num_windows = i
    return np.stack(predictions), num_windows


def validate_model(model, X, y, batch_size, shuffle=True, metric=log_loss):

    num_preds = X.shape[0]
    score_sum = 0
    for x_batch, y_batch in batch_generator(X, y, batch_size, shuffle):
        predictions = predict_window(x_batch, model)
        probs = score_image_minimax(predictions)
        score_sum += metric(y_batch, probs, eps=1e-9) # Using Kaggle's eps
    return score_sum / num_preds


imgen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=180,
            featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.3,
            zoom_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')


class TestWindowMethods(unittest.TestCase):

    def test_normalize_multiclass(self):

        a = np.array([[1,2,3,4],
                      [.5, .5, .5, .5],
                      [.25, .25, .25, .25],
                      [0, 0, 0, 0]])

        exp = np.array([[.1, .2, .3, .4],
                      [.25, .25, .25, .25],
                      [.25, .25, .25, .25],
                      [.25, .25, .25, .25]])

        np.testing.assert_almost_equal(normalize_multiclass(a.clip(1e-9), axis=1), exp)

    def test_score_minimax(self):

        a = np.array([[[.9, .2, .6, .4],
                      [.5, .5, .5, .5],
                      [.25, .25, .25, .25],
                      [0, 0, 0, 0]],
                      [[.3, .7, .3, .1],
                       [.2, .9, 1, 0],
                       [.2, .25, .3, .35],
                       [0, 0, 0, .1]]])

        exp = np.array([[.3, .7, .6, .4],
                      [.2, .9, 1, .5],
                      [.2, .25, .3, .35],
                      [0, 0, 0, 0.1]])
        exp.clip(1e-9)

        np.testing.assert_almost_equal(score_image_minimax(a), normalize_multiclass(exp, axis=1))



if __name__ == '__main__':
    unittest.main()

