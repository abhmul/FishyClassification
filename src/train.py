import logging

import os
import random
import shutil

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, MaxPooling2D, AveragePooling2D, Convolution2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from kfold_utils import KFoldFromDir

random.seed(2016)
np.random.seed(2016)

root = '../input'
total_data = 'train'
train_data = 'train_split'
val_data = 'val_split'


learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_epochs = 25
batch_size = 32
nfolds = 10
FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


def inception_model(input_shape=(None, None, 3), fcn=True, test=False):

    print('Loading InceptionV3 Weights ...')
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                    input_tensor=None, input_shape=input_shape)
    # Note that the preprocessing of InceptionV3 is:
    # (x / 255 - 0.5) x 2

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)  # Shape: (1, 1, 2048)
    if fcn:
        activation = 'sigmoid' if test else 'softmax'
        output = Convolution2D(8, 1, 1, activation=activation)(output)
        if not test:
            output = Flatten(name='flatten')(output)
    else:
        output = Flatten(name='flatten')(output)
        output = Dense(8, activation='softmax', name='predictions')(output)

    InceptionV3_model = Model(InceptionV3_notop.input, output)
    # InceptionV3_model.summary()

    print('Creating optimizer and compiling')
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return InceptionV3_model

print('Initializing Augmenters')
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)


# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

kf = KFoldFromDir(nfolds, FishNames, root=root, total_data=total_data, train_data=train_data, val_data=val_data)

for (train_generator, validation_generator), (nbr_train_samples, nbr_validation_samples) in kf.fit(train_datagen,
                                                                                                   val_datagen,
                                                                                                   img_width=img_width,
                                                                                                   img_height=img_height):

    # autosave best Model
    best_model_file = '../fishyInception_weights_fold{}.h5'
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    model = inception_model((img_width, img_height, 3))
    print('Training Model...')
    model.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples,
        verbose=1,
        callbacks=[best_model])