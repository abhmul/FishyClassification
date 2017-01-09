import random

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from kfold_utils import KFoldFromDir
from models import inception_model, resnet50_model

random.seed(2016)
np.random.seed(2016)

root = '../input'
total_data = 'train_bb/POS'
train_data = 'train_split'
val_data = 'val_split'


learning_rate = 0.0001
img_width = 224
img_height = 224
nbr_epochs = 25
batch_size = 128
nfolds = 7
FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']

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
    best_model_file = '../fishyResNet50_weights_fold{}.h5'
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    model = resnet50_model((img_width, img_height, 3), learning_rate=learning_rate)
    print('Training Model...')
    model.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples,
        verbose=1,
        callbacks=[best_model])