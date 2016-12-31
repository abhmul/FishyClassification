import os
import json

import numpy as np

from image2 import ImageDataGenerator
from Transformations import ROICenter, RandomShift, CenterCrop, RandomRotation, RandomShear, RandomZoom, RandomCrop
from kfold_fcn import KFoldFromDirFCN
from train import inception_model

import keras.backend as K
from keras.callbacks import ModelCheckpoint

ROOT = '../'
BB_PATH = 'bounding_boxes'
PICS = '../input'
TOTAL_DATA = 'train_bb'
TRAIN_DATA = 'train_split'
VAL_DATA = 'val_split'


learning_rate = 0.0001
img_width = 256
img_height = 256
nbr_epochs = 25
batch_size = 32
nfolds = 10
PosFishNames = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
NegFishNames = ['NoF']

if K.image_dim_ordering() == 'tf':
    input_shape = (img_height, img_width, 3)
else:
    input_shape = (3, img_height, img_width)


def load_json_bbs():
    class_lst = []
    for fname in os.listdir(os.path.join(ROOT, BB_PATH)):
        with open(os.path.join(ROOT, BB_PATH, fname), 'r') as bb:
            class_lst.append(json.load(bb))
    return class_lst


def build_bb(class_lst):
    bounding_boxes = {}
    no_boxes = []
    for i, label in enumerate(class_lst):
        for rect in label:
            if len(rect["annotations"]) > 0:
                bounding_boxes[str(rect["filename"][-13:])] = (rect["annotations"][0]["x"],
                                            rect["annotations"][0]["y"],
                                            rect["annotations"][0]["width"],
                                            rect["annotations"][0]["height"])
            # If there are no pictures
            else:
                no_boxes.append(rect["filename"][-13:])
    return bounding_boxes, no_boxes


def train_fcn_gen(directory, pos_imgen, neg_imgen, nbr_pos, nbr_neg, nb_pos_classes, nb_neg_classes,
                  target_size=(img_height, img_width),
                  batch_size=32, color_mode='rgb'):

    if color_mode not in {'rgb', 'grayscale'}:
        raise ValueError('Invalid color mode:', color_mode,
                         '; expected "rgb" or "grayscale".')
    if pos_imgen.dim_ordering != neg_imgen.dim_ordering:
        raise ValueError('Dim orderings of augmenters must match')

    channels = 3 if color_mode == 'rgb' else 1
    samples = nbr_neg + nbr_pos
    h = target_size[0]
    w = target_size[1]

    out_shape = [0, 0, 0]
    out_shape[pos_imgen.row_index] = h
    out_shape[pos_imgen.col_index] = w
    out_shape[pos_imgen.channel_index] = channels
    out_shape = tuple([samples,] + out_shape)

    pos_dir = os.path.join(directory, 'POS')
    neg_dir = os.path.join(directory, 'NEG')
    pos_gen = pos_imgen.flow_from_directory(pos_dir, target_size, color_mode,  batch_size=batch_size, shuffle=False)
    neg_gen = pos_imgen.flow_from_directory(neg_dir, target_size, color_mode, batch_size=batch_size, shuffle=False)

    while True:
        X = np.empty(out_shape)
        y = np.zeros((samples, nb_pos_classes+nb_neg_classes))

        for i in xrange(0, nbr_neg, batch_size):
            X[i:i+batch_size], y[i:i+batch_size, nb_pos_classes:] = next(neg_gen)
        for i in xrange(nbr_neg, samples, batch_size):
            X[i:i+batch_size], y[i:i+batch_size, :nb_pos_classes] = next(pos_gen)

        np.random.shuffle(X)
        for i in xrange(0, samples, batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]


json_bb_lst = load_json_bbs()
bounding_boxes, no_boxes = build_bb(json_bb_lst)

print bounding_boxes.keys()[:10]
box_names = set(bounding_boxes.keys())

# Generator for the fish images
fish_imgen = ImageDataGenerator()
fish_imgen.add(ROICenter(bounding_boxes))
fish_imgen.add(RandomZoom(.05))
fish_imgen.add(RandomRotation(10))
fish_imgen.add(RandomShift(0.05, 0.05))
fish_imgen.add(RandomShear(.1))
fish_imgen.add(CenterCrop((img_height, img_width)))

# Generator for the NoF images
nof_imgen = ImageDataGenerator()
nof_imgen.add(RandomCrop((img_height, img_width)))


kfold = KFoldFromDirFCN(PosFishNames, NegFishNames, root=PICS, total_data=TOTAL_DATA, train_data=TRAIN_DATA, val_data=VAL_DATA)
for nbr_pos_train, nbr_pos_val, nbr_neg_train, nbr_neg_val in kfold.fit(nfolds):
    train_gen = train_fcn_gen(kfold.train_data_dir, fish_imgen, nof_imgen, nbr_pos_train, nbr_neg_train,
                              len(PosFishNames), len(NegFishNames))
    val_gen = train_fcn_gen(kfold.val_data_dir, fish_imgen, nof_imgen, nbr_pos_val, nbr_neg_val,
                              len(PosFishNames), len(NegFishNames))

    # autosave best Model
    best_model_file = '../fishyInception_weights_fold{}.h5'
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    model = inception_model(input_shape=input_shape, fcn=True, test=False)

    model.fit_generator(train_gen, samples_per_epoch=nbr_pos_train+nbr_neg_train, nbr_epochs=nbr_epochs,
                        verbose=1, callbacks=[best_model], validation_data=val_gen,
                        nb_val_samples=nbr_neg_val+nbr_pos_val)