import os
import json
# import time

import numpy as np

from image2 import ImageDataGenerator
from Transformations import ROICenter, RandomShift, CenterCrop, RandomRotation, RandomShear, RandomZoom, RandomCrop, Rescale
from kfold_fcn import KFoldFromDirFCN
from models import inception_model

import keras.backend as K
from keras.callbacks import ModelCheckpoint

ROOT = '../'
BB_PATH = 'bounding_boxes'
PICS = '../input'
TOTAL_DATA = 'train_bb'
TRAIN_DATA = 'train_split'
VAL_DATA = 'val_split'


learning_rate = 0.0005
img_width = 299
img_height = 299
nbr_epochs = 35
batch_size = 64
nfolds = 3
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


# Generates entire epoch and passes it in batches (~130s to generate entire epoch)
class TrainFCNGen(object):

    def __init__(self, directory, pos_imgen, neg_imgen, nbr_pos, nbr_neg, nb_pos_classes, nb_neg_classes,
                 target_size=(img_height, img_width), batch_size=32, color_mode='rgb', seed=None):

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        if pos_imgen.dim_ordering != neg_imgen.dim_ordering:
            raise ValueError('Dim orderings of augmenters must match')

        self.batch_size = batch_size
        self.nbr_neg = nbr_neg
        self.nbr_pos = nbr_pos
        self.nb_pos_classes = nb_pos_classes
        self.nb_neg_classes = nb_neg_classes

        channels = 3 if color_mode == 'rgb' else 1
        self.samples = nbr_neg + nbr_pos
        h = target_size[0]
        w = target_size[1]

        out_shape = [0, 0, 0]
        out_shape[pos_imgen.row_index-1] = h
        out_shape[pos_imgen.col_index-1] = w
        out_shape[pos_imgen.channel_index-1] = channels
        self.out_shape = tuple(out_shape)

        self.pos_dir = os.path.join(directory, 'POS')
        self.neg_dir = os.path.join(directory, 'NEG')
        self.pos_gen = pos_imgen.flow_from_directory(self.pos_dir, target_size, color_mode, batch_size=batch_size, shuffle=True)
        self.neg_gen = neg_imgen.flow_from_directory(self.neg_dir, target_size, color_mode, batch_size=batch_size, shuffle=True)

        # Initialize the data containers
        self.X = np.empty((self.samples,) + self.out_shape)
        self.y = np.zeros((self.samples, nb_pos_classes + nb_neg_classes))
        self.index_array = np.arange(len(self.X))

    def _reset_data(self):
        for i in xrange(self.nbr_neg, self.samples, self.batch_size):
            # profile = time.time()
            a = next(self.pos_gen)
            # print 'Took {} s to get next pos image'.format(time.time()-profile)
            # print a[0].shape
            # print a[1].shape
            start, end = i, min(i + batch_size, self.samples)
            self.X[start:end], self.y[start:end, :self.nb_pos_classes] = a
        for i in xrange(0, self.nbr_neg, self.batch_size):
            a = next(self.neg_gen)
            # print a[0].shape
            # print a[1].shape
            start, end = i, min(i + batch_size, self.nbr_neg)
            self.X[start:end], self.y[start:end, self.nb_pos_classes:] = a
        np.random.shuffle(self.index_array)

    def __iter__(self):
        while True:
            self._reset_data()
            for i in xrange(0, self.samples, self.batch_size):
                yield self.X[self.index_array[i:i + self.batch_size]], \
                      self.y[self.index_array[i:i + self.batch_size]]

# Generates on the fly (~1.5s per batch)
class TrainFCNGen2(object):

    def __init__(self, directory, pos_imgen, neg_imgen, nbr_pos, nbr_neg, nb_pos_classes, nb_neg_classes,
                 target_size=(img_height, img_width), batch_size=32, color_mode='rgb', seed=None):

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        if pos_imgen.dim_ordering != neg_imgen.dim_ordering:
            raise ValueError('Dim orderings of augmenters must match')

        self.batch_size = batch_size
        self.nbr_neg = nbr_neg
        self.nbr_pos = nbr_pos
        self.nb_pos_classes = nb_pos_classes
        self.nb_neg_classes = nb_neg_classes

        channels = 3 if color_mode == 'rgb' else 1
        self.samples = nbr_neg + nbr_pos
        h = target_size[0]
        w = target_size[1]

        out_shape = [0, 0, 0]
        out_shape[pos_imgen.row_index-1] = h
        out_shape[pos_imgen.col_index-1] = w
        out_shape[pos_imgen.channel_index-1] = channels
        self.out_shape = tuple(out_shape)

        self.pos_dir = os.path.join(directory, 'POS')
        self.neg_dir = os.path.join(directory, 'NEG')
        self.pos_gen = pos_imgen.flow_from_directory(self.pos_dir, target_size, color_mode, batch_size=1, shuffle=True)
                                                     # save_to_dir='../input/preview/')
        self.neg_gen = neg_imgen.flow_from_directory(self.neg_dir, target_size, color_mode, batch_size=1, shuffle=True)
                                                     # save_to_dir='../input/preview/')

        # Initialize the data containers
        self.index_array = np.arange(self.samples)

    def __iter__(self):
        while True:
            np.random.shuffle(self.index_array)
            for i in xrange(0, self.samples, self.batch_size):
                # a = time.time()
                inds = self.index_array[i:i+self.batch_size]
                x_batch = np.empty((len(inds),) + self.out_shape)
                y_batch = np.zeros((len(inds), self.nb_pos_classes + self.nb_neg_classes))
                for i, ind in enumerate(inds):
                    if ind >= self.nbr_neg:
                        x_batch[i:i+1], y_batch[i:i+1, :self.nb_pos_classes] = next(self.pos_gen)
                    else:
                        x_batch[i:i+1], y_batch[i:i+1, self.nb_pos_classes:] = next(self.neg_gen)
                # print('Took {} s to create batch'.format(time.time() - a))
                yield x_batch, y_batch


json_bb_lst = load_json_bbs()
bounding_boxes, no_boxes = build_bb(json_bb_lst)

# print bounding_boxes.keys()[:10]
box_names = set(bounding_boxes.keys())

# Generator for the fish images
wrg, hrg = 0.05, 0.05
fish_imgen = ImageDataGenerator()
fish_imgen.add(ROICenter(bounding_boxes, fast=True))
fish_imgen.add(CenterCrop((int(round((1+.2)*img_height)), int(round((1+.2)*img_width)))))
fish_imgen.add(RandomZoom(.05))
fish_imgen.add(RandomRotation(10))
# fish_imgen.add(RandomShift(hrg, wrg, fast=True))
fish_imgen.add(RandomShear(.1))
fish_imgen.add(CenterCrop((img_height, img_width)))
fish_imgen.add(Rescale(1./255))

# Generator for the NoF images
nof_imgen = ImageDataGenerator()
nof_imgen.add(RandomCrop((img_height, img_width)))
nof_imgen.add(Rescale(1./255))


kfold = KFoldFromDirFCN(PosFishNames, NegFishNames, root=PICS, total_data=TOTAL_DATA, train_data=TRAIN_DATA, val_data=VAL_DATA)
i = 0
for nbr_pos_train, nbr_pos_val, nbr_neg_train, nbr_neg_val in kfold.fit(nfolds):
    train_gen = TrainFCNGen2(kfold.train_data_dir, fish_imgen, nof_imgen, nbr_pos_train, nbr_neg_train,
                            len(PosFishNames), len(NegFishNames))
    val_gen = TrainFCNGen2(kfold.val_data_dir, fish_imgen, nof_imgen, nbr_pos_val, nbr_neg_val,
                          len(PosFishNames), len(NegFishNames))

    # a = iter(val_gen)
    # for i in xrange(val_gen.samples):
    #     x, y = next(a)
    #     print 'Percentages of each class:\n{}'.format(np.sum(y, axis=0) / y.shape[0])

    # autosave best Model
    best_model_file = '../fishyFCNInception_weights_fold{}.h5'.format(i+1)
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True)

    model = inception_model(input_shape=train_gen.out_shape, fcn=True, test=False, learning_rate=learning_rate)

    model.fit_generator(iter(train_gen), samples_per_epoch=train_gen.samples, nb_epoch=nbr_epochs,
                        verbose=1, callbacks=[best_model], validation_data=iter(val_gen),
                        nb_val_samples=val_gen.samples)
    i += 1