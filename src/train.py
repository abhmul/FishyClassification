import logging

import os
import random
import shutil

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

random.seed(2016)
np.random.seed(2016)

logging.basicConfig(filename='example.log', level=logging.DEBUG)

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_epochs = 25
batch_size = 32
nfolds = 10

root = '../input'
train_total = 'train'
train_data = 'train_split'
val_data = 'val_split'

train_total_dir = os.path.join(root, train_total)
train_data_dir = os.path.join(root, train_data)
val_data_dir = os.path.join(root, val_data)

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                                input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2


print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index=-1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
# InceptionV3_model.summary()


print('Creating optimizer and compiling')
optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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

print('Running {} folds'.format(nfolds))
for i in xrange(nfolds):
    # Clean the current split
    print('Removing past splits...')
    # Remove the train and val dirs if they already exist and create empty versions
    data_dirs = set(os.listdir(root))
    if train_data in data_dirs:
        shutil.rmtree(train_data_dir)
    if val_data in data_dirs:
        shutil.rmtree(val_data_dir)
    os.mkdir(train_data_dir)
    os.mkdir(val_data_dir)

    # these are the names of the species in alphabetical order
    FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    # Initialize the the enumerators for train and val data
    nbr_train_samples = 0
    nbr_validation_samples = 0

    # make a split
    for fish in FishNames:
        # List out all of the images of species fish
        print('Performing split {} for species {}'.format(i, fish))
        total_images = np.array(os.listdir(os.path.join(train_total_dir, fish)))
        # Shuffle them if this is our first fold
        np.random.shuffle(total_images) if not i else None
        # get the validation start and end ind
        split_ind_begin = int(1. / nfolds * i * len(total_images))
        split_ind_end = int(1. / nfolds * (i+1) * len(total_images))

        # Make the directory for the species if it is not there yet (Train)
        print('Rebuilding train directory tree...')
        if fish not in os.listdir(train_data_dir):
            os.mkdir(os.path.join(train_data_dir, fish))

        # Make the train images by making an array without the fold indices
        print('Making train split...')
        train_images = np.concatenate((total_images[:split_ind_begin], total_images[split_ind_end:]))

        # copy each each image in the train set
        print('Copying train split to folder {}...'.format(fish))
        for img in train_images:
            source = os.path.join(train_total_dir, fish, img)
            target = os.path.join(train_data_dir, fish, img)
            shutil.copy(source, target)
            nbr_train_samples += 1

        # Repeat the process for the validation data (Wrap this process into a single function)
        # Make the directory for the species if it is not there yet (Validation)
        print('Rebuilding validation directory tree...')
        if fish not in os.listdir(val_data_dir):
            os.mkdir(os.path.join(val_data_dir, fish))

        # Slice out the validation images
        print('Making validation split...')
        val_images = total_images[split_ind_begin:split_ind_end]

        # copy each each image in the val set
        print('Copying validation split to folder {}...'.format(fish))
        for img in val_images:
            source = os.path.join(train_total_dir, fish, img)
            target = os.path.join(val_data_dir, fish, img)
            shutil.copy(source, target)
            nbr_validation_samples += 1

    print('All done performing split for fold {}'.format(i))

    print('Initializing Checkpointer...')
    # autosave best Model
    best_model_file = "../inception_weights_fold{}.h5".format(i)
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

    print('Creating Generators...')
    # this is the augmentation configuration we will use for validation:
    # only rescaling
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes=FishNames,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        # save_prefix = 'aug',
        classes=FishNames,
        class_mode='categorical')

    print('Training Model...')
    try:
        InceptionV3_model.fit_generator(
            train_generator,
            samples_per_epoch=nbr_train_samples,
            nb_epoch=nbr_epochs,
            validation_data=validation_generator,
            nb_val_samples=nbr_validation_samples,
            verbose=1,
            callbacks=[best_model])
    except Exception as e:  # most generic exception you can catch
        logging.exception("message")
