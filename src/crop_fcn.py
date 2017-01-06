import numpy as np
import os
import logging

from keras.preprocessing.image import load_img, img_to_array

from models import inception_model



def get_bb(activations, img, bb_size = (299, 299)):
    if len(activations.shape) != 2:
        TypeError('Input should be a 2d numpy array, given a {}d numpy array'.format(len(activations.shape)))
    inds = np.argmax(activations)
    sy, sx = [(inds[i] + 1.) / (activations.shape[i] + 1.) for i in xrange(len(activations.shape))]
    px, py = sx * img.size[0], sy * img.size[1]
    x, y = max(px - bb_size[0] // 2, 0), max(py - bb_size[1] // 2, 0)
    return (x, y,) + bb_size

TEST_DIR = '../input/test2/test_stg2/'
save_dir = '../input/preview/'
nfolds = 4

#Load the models
models = [inception_model(test=True) for i in xrange(nfolds)]
for i, model in enumerate(models):
    logging.info('Loading weights form for fold {}'.format(i+1))
    model.load_weights('../fishNoFishFCNInception_weights_fold{}.h5'.format(i + 1))

for i, img_name in enumerate(os.listdir(TEST_DIR)):
    logging.info('Running FCN Crop on {}'.format(img_name))
    img = load_img(os.path.join(TEST_DIR, img_name))
    x = img_to_array(img)
    activations = np.sum([model.predict_on_batch(x) for model in models])
    bb = get_bb(activations, img)
    cropped = img.crop(bb)
    cropped.save(os.path.join(save_dir, 'cropped_' + img_name))
    if i == 100:
        break


