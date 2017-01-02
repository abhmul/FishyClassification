from keras.models import load_model
import os
from image2 import ImageDataGenerator
from Transformations import Rescale, RandomShift, RandomShear, RandomZoom
from models import inception_model
import numpy as np
from sklearn.metrics import log_loss
from functools import partial
import logging

from predict_utils import predict_augment, predict_kfold, predict_fcn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# img_width = 299
# img_height = 299
batch_size = 32
nfolds = 3

PosFishNames = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
NegFishNames = ['NoF']
FishNames = PosFishNames + NegFishNames

root_path = '../input'
test_data_dir = os.path.join(root_path, 'val_split')
nbr_test_samples = sum([len(files) for r, d, files in os.walk(test_data_dir)])
nbr_aug = 10

test_datagen = ImageDataGenerator()
test_datagen.add(RandomShear(.1))
test_datagen.add(RandomZoom(.1))
test_datagen.add(RandomShift(.1, .1, fast=True))
test_datagen.add(Rescale(1./255))

testgen = test_datagen.flow_from_directory(test_data_dir, target_size=(None, None),
                                     batch_size=1, shuffle=False)


model = inception_model(test=True)
predictions = np.zeros((nbr_test_samples, 8))
y = np.zeros((nbr_test_samples, 8))

for i in xrange(nfolds):
    print('Loading weights for fold {}'.format(i+1))
    best_model_file = '../fishyFCNInception_weights_fold{}.h5'.format(i + 1)
    model.load_weights(best_model_file)

    print('Predicting w/ {} rounds of augmentation for fold {}'.format(nbr_aug, nfolds))
    for k in xrange(nbr_test_samples * nbr_aug):
        if k % nbr_test_samples == 0:
            print('Performing augmentation round {}'.format(k // nbr_test_samples))
        x, y_batch = next(testgen)
        prediction = model.predict(x, 1)
        probs = np.concatenate(np.max(prediction[0, :, :, :len(PosFishNames)], axis=(0, 1)),
                               np.max(prediction[0, :, :, len(PosFishNames):], axis=(0, 1)))

        predictions[k % nbr_test_samples] += probs
        y[k % nbr_test_samples] = y_batch

print('Normalizing the predictions')
total = np.sum(predictions, axis=1).reshape(1, -1).T
predictions = predictions / np.tile(total, (1, len(FishNames)))


print('Logloss of predctions:')
print(log_loss(y, predictions, eps=1e-15))
