import os
import numpy as np

from keras.callbacks import ModelCheckpoint

from Transformations import RandomRotationPIL, ROICenter, CenterCrop, RandomShearPIL, Img2Array, Array2Img, RandomShift, RandomFlip, Rescale, RandomCrop
from image2 import ImageDataGenerator
from bb_utils import load_json_bbs, build_bb, max_dim_scale, rescale_bb
import time
from models import inception_model

ROOT = '../'
BB_PATH = 'bounding_boxes'
PICS = os.path.join(ROOT, 'input')
img_height = 299
img_width = 299
nb_pos_train = sum([len(files) for r, d, files in os.walk(os.path.join(PICS, 'train_split', 'POS'))])
nb_pos_val = sum([len(files) for r, d, files in os.walk(os.path.join(PICS, 'val_split', 'POS'))])

learning_rate = 0.0005
nbr_epochs = 35
batch_size = 64


# Generates on the fly (~1.5s per batch)
class TrainFCNGen3(object):

    def __init__(self, directory, pos_imgen, neg_imgen, nb_samples,
                 target_size=(img_height, img_width), batch_size=32, color_mode='rgb', seed=None):

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        if pos_imgen.dim_ordering != neg_imgen.dim_ordering:
            raise ValueError('Dim orderings of augmenters must match')

        self.batch_size = batch_size
        self.samples = nb_samples

        self.nbr_pos = self.samples // 2
        self.nbr_neg = self.samples - self.nbr_pos

        channels = 3 if color_mode == 'rgb' else 1
        h = target_size[0]
        w = target_size[1]

        out_shape = [0, 0, 0]
        out_shape[pos_imgen.row_index-1] = h
        out_shape[pos_imgen.col_index-1] = w
        out_shape[pos_imgen.channel_index-1] = channels
        self.out_shape = tuple(out_shape)

        self.pos_dir = os.path.join(directory, 'POS')
        self.neg_dir = os.path.join(directory, 'NEG')
        self.pos_gen = pos_imgen.flow_from_directory(self.pos_dir, target_size, color_mode, batch_size=1, shuffle=True,
                                                     class_mode='categorical')
                                                     # save_to_dir='../input/preview/')
        self.neg_gen = neg_imgen.flow_from_directory(self.neg_dir, target_size, color_mode, batch_size=1, shuffle=True,
                                                     class_mode = 'binary')
                                                     # save_to_dir='../input/preview/')

        # Initialize the data containers
        self.index_array = np.arange(self.samples)

    def __iter__(self):
        while True:
            np.random.shuffle(self.index_array)
            for i in xrange(0, self.samples, self.batch_size):
                a = time.time()
                inds = self.index_array[i:i+self.batch_size]
                x_batch = np.empty((len(inds),) + self.out_shape)
                y_batch = np.zeros((len(inds),))
                for i, ind in enumerate(inds):
                    if ind >= self.nbr_neg:
                        x_batch[i:i+1] = next(self.pos_gen)[0]
                        y_batch[i] = 1
                    else:
                        x_batch[i:i+1]= next(self.neg_gen)[0]
                print('Took {} s to create batch'.format(time.time() - a))
                yield x_batch, y_batch

bounding_boxes, no_boxes = build_bb(load_json_bbs(ROOT, BB_PATH))

fish_imgen = ImageDataGenerator()
fish_imgen.add(Img2Array())
fish_imgen.add(ROICenter(bounding_boxes, fast=True))
fish_imgen.add(Array2Img())
fish_imgen.add(RandomRotationPIL(90))
fish_imgen.add(RandomShearPIL(.1))
fish_imgen.add(Img2Array())
fish_imgen.add(RandomShift(.03, .03, fast=True))
fish_imgen.add(CenterCrop((299, 299)))
fish_imgen.add(RandomFlip())
fish_imgen.add(Rescale(1./255))

val_fish_imgen = ImageDataGenerator()
val_fish_imgen.add(Img2Array())
val_fish_imgen.add(ROICenter(bounding_boxes, fast=True))
val_fish_imgen.add(RandomShift(.03, .03, fast=True))
val_fish_imgen.add(CenterCrop((299, 299)))
val_fish_imgen.add(Rescale(1./255))

# Generator for the NoF images
nof_imgen = ImageDataGenerator()
nof_imgen.add(Img2Array())
nof_imgen.add(RandomCrop((img_height, img_width)))
nof_imgen.add(Rescale(1./255))

train_gen = TrainFCNGen3(os.path.join(PICS, 'train_split'), fish_imgen, nof_imgen, nb_pos_train*2, batch_size=64)
val_gen = TrainFCNGen3(os.path.join(PICS, 'val_split'), val_fish_imgen, nof_imgen, nb_pos_val*2, batch_size=64)

best_model_file = '../fishNoFishFCNInception_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True)

model = inception_model(input_shape=train_gen.out_shape, fcn=True, test=False, learning_rate=learning_rate)

fit = model.fit_generator(iter(train_gen), samples_per_epoch=train_gen.samples, nb_epoch=nbr_epochs,
                    verbose=1, callbacks=[best_model], validation_data=iter(val_gen),
                    nb_val_samples=val_gen.samples)