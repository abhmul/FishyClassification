import random

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# from image2 import ImageDataGenerator
from Transformations import RandomFlip, RandomCrop, RandomShearPIL, RandomRotationPIL, Rescale, Img2Array, Array2Img, ResizeRelativePIL

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
batch_size = 128
nfolds = 7
FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NoF']

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
# train_datagen.add(ResizeRelativePIL(.5, .5))
# train_datagen.add(Img2Array())
# train_datagen.add(RandomCrop((img_width, img_height)))
# train_datagen.add(RandomFlip())
# train_datagen.add(Array2Img(scale=False))
# train_datagen.add(RandomRotationPIL(10.))
# train_datagen.add(RandomShearPIL(.1))
# train_datagen.add(Img2Array())
# train_datagen.add(Rescale(1./255))

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(
    rescale=1./255
)
# val_datagen.add(ResizeRelativePIL(.5, .5))
# val_datagen.add(Img2Array())
# val_datagen.add(RandomCrop((img_width, img_height)))
# val_datagen.add(Rescale(1./255))

nbr_val_aug = 1

kf = KFoldFromDir(nfolds, FishNames, root=root, total_data=total_data, train_data=train_data, val_data=val_data)
i = 0
histories = []
for (train_generator, validation_generator), (nbr_train_samples, nbr_validation_samples) in kf.fit(train_datagen,
                                                                                                   val_datagen,
                                                                                                   img_width=img_width,
                                                                                                   img_height=img_height):


    # autosave best Model
    best_model_file = '../fishyInception_weights_fold{}.h5'.format(i+1)
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    model = inception_model((img_width, img_height, 3), learning_rate=learning_rate, fcn=False, classes=8)
    print 'SEED: %s' % seed
    print('Training Model...')
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples * nbr_val_aug,
        verbose=1,
        callbacks=[best_model])

    histories.append(history.history)

    print 'SEED: %s' % seed
    i += 1

avg_loss = sum(min(history['val_loss']) for history in histories) / float(len(histories))
print 'Val_loss: %s' % avg_loss