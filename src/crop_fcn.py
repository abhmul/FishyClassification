import numpy as np
import os
import logging

from keras.preprocessing.image import load_img, img_to_array

from models import inception_model

logging.getLogger().setLevel(logging.INFO)


def get_bb(activations, img, bb_size = (299, 299)):
    if len(activations.shape) != 2:
        ValueError('Input should be a 2d numpy array, given a {}d numpy array'.format(len(activations.shape)))
    inds = np.unravel_index(np.argmax(activations), activations.shape)
    sy, sx = [(inds[i] + 1.) / (activations.shape[i] + 1.) for i in xrange(len(activations.shape))]
    px, py = int(round(sx * img.size[0])), int(round(sy * img.size[1]))
    x, y = max(px - bb_size[0] // 2, 0), max(py - bb_size[1] // 2, 0)
    x, y = min(x, img.size[0] - bb_size[0]), min(y, img.size[1] - bb_size[1])
    return x, y, x + bb_size[0], y + bb_size[1]

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
    x = x.reshape((1,) + x.shape)
    activations = sum([model.predict_on_batch(x) for model in models])
    print activations
    activations = activations.reshape(activations.shape[1:3])
    bb = get_bb(activations, img)
    cropped = img.crop(bb)
    cropped.save(os.path.join(save_dir, 'cropped_' + img_name))
    if i == 100:
        break


