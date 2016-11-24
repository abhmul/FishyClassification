import numpy as np
from scipy.ndimage import zoom

from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

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

def normalize_multiclass(arr):

    return arr / np.sum(arr)


def score_image_minimax(predictions, nof_ind = 0, normalize_func=normalize_multiclass):

    # predictions are WINDOWS x 32 x 8
    nof_min = np.min(predictions[:, :, nof_ind:nof_ind+1], axis=0)

    fish_max = np.max(predictions[:, :, 1:], axis=0)

    return normalize_func(np.stack([nof_min, fish_max], axis=0))


def predict_window(X, model, img_size=INPUT_IMGSIZE):

    predictions = []
    for window in window_gen(X):
        predictions.append(model.predict_on_batch(zoom(window, (1, 1,) + img_size)))
    return np.stack(predictions)

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

