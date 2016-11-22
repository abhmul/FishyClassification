import numpy as np
from scipy.ndimage import zoom

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
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        for window in window_gen(X_batch):
            yield zoom(window, (1, 1,) + img_size),  y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def validate_model(X, y, batch_size, model, img_size=INPUT_IMGSIZE):


    predictions = []
    for window in window_gen(X):
        predictions.append(model.predict_on_batch(zoom(window, (1, 1,) + img_size)))

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

model = Sequential()
model.pred
