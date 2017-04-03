import random

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
# from image2 import ImageDataGenerator
from Transformations import RandomFlip, RandomCrop, RandomShearPIL, RandomRotationPIL, Rescale, Img2Array, Array2Img, ResizeRelativePIL
from fish8 import *
from kfold_utils import KFoldFromDir
from models import inception_model, resnet50_model

seed = np.random.randint(1, 10000)
random.seed(seed)
np.random.seed(seed)

print 'SEED: %s' % seed

root = '../input'
total_data = 'train'
train_data = 'train_split'
val_data = 'val_split'


learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_epochs = 25
batch_size = 32
nfolds = 7
FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NoF']

print("Loading the data")
Xtr, ytr, ids = load_train_from_dir(FishNames, root + total_data, (img_width, img_height))

print("Splitting the data")
Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, train_size = 0.8)

print('Initializing Augmenters')
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=1.0,
    horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = datagen.flow(Xtr, ytr)

nbr_val_aug = 1
i = 0
histories = []
# autosave best Model
best_model_file = '../fishyInception_weights_basic_fold{}_batchsize{}.h5'.format(i+1, batch_size)
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True)

model = inception_model((img_width, img_height, 3), learning_rate=learning_rate, fcn=False, classes=8)

print('Training Model...')
history = model.fit_generator(
    train_generator,
     # samples_per_epoch=nbr_train_samples,
    nb_epoch=nbr_epochs,
    validation_data=validation_generator,
    # nb_val_samples=nbr_validation_samples,
    verbose=1,
    callbacks=[best_model])

avg_loss = sum(min(history['val_loss']) for history in histories) / float(len(histories))
print 'Val_loss: %s' % avg_loss
