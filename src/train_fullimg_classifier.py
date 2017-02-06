import fish8
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from models import inception_barebones

CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
TRAIN_DIR = '../input/train/'
BATCH_SIZE = 64

def to_uncategorical(y):
    return np.where(y == 1.)[1]

def pipeline1():
    best_model_file = '../fishyFullception_weights.h5'
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    X, y, trid = fish8.load_train_data(CLASSES, TRAIN_DIR, target_size=(299, 299))
    trid = None
    print("Creating Splitter")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    sss.get_n_splits()
    print("Splitting")
    train_ind, val_ind = next(sss.split(X, to_uncategorical(y)))
    print("Initializing Augmentors")
    train_aug = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=10.,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   channel_shift_range=0.,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rescale=1. / 255)
    val_aug = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rescale=1. / 255)

    print("Instantiating model")
    inception_nn = inception_barebones()
    print("Creating train gen")
    train_gen = train_aug.flow(X[train_ind], y[train_ind], batch_size=BATCH_SIZE)
    print("Creating val gen")
    val_gen = val_aug.flow(X[val_ind], y[val_ind], batch_size=BATCH_SIZE)

    inception_nn.fit_generator(train_gen, samples_per_epoch=train_ind.shape[0],
                               nb_epoch=25, callbacks=[best_model],
                               validation_data=val_gen,
                               nb_val_samples=val_ind.shape[0],
                               max_q_size=10)

pipeline1()