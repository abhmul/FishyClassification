from collections import defaultdict
import numpy as np
from PIL import Image
import os
from keras.preprocessing.image import img_to_array, ImageDataGenerator, Iterator, array_to_img
# import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from bb_utils import build_bb_all, load_json_bbs
from sklearn.model_selection import StratifiedShuffleSplit
from models import inception_barebones
# import matplotlib.pyplot as plt

CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
TRAIN_DIR = '../input/train/'
BB_PATH = '../bounding_boxes/'
TARGET_SIZE = (299, 299)
BATCH_SIZE = 32


def crop(x, crop_box, row_index=0, col_index=1):
    x = np.rollaxis(x, row_index, 0)[crop_box[1]:crop_box[3]]
    x = np.rollaxis(x, 0, row_index + 1)
    x = np.rollaxis(x, col_index, 0)[crop_box[0]:crop_box[2]]
    return np.rollaxis(x, 0, col_index + 1)


def random_crop(x, crop_size, row_index=0, col_index=1):
        h, w = x.shape[row_index], x.shape[col_index]
        rangew = (w - crop_size[1])
        rangeh = (h - crop_size[0])
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        crop_box = (offsetw, offseth, offsetw+crop_size[0], offseth+crop_size[1])
        return crop(x, crop_box, row_index=row_index, col_index=col_index)


# class ImageDataGenerator2(ImageDataGenerator):
#
#     def flow_new(self, inds, X, y=None, batch_size=32, shuffle=True, seed=None,
#              save_to_dir=None, save_prefix='', save_format='jpeg'):
#         return NumpyArrayIterator2(
#             X, y, inds, self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             seed=seed,
#             dim_ordering=self.dim_ordering,
#             save_to_dir=save_to_dir,
#             save_prefix=save_prefix,
#             save_format=save_format)
#
#
# class NumpyArrayIterator2(Iterator):
#
#     def __init__(self, x, y, inds, image_data_generator,
#                  batch_size=32, shuffle=False, seed=None,
#                  dim_ordering='default',
#                  save_to_dir=None, save_prefix='', save_format='jpeg'):
#         if y is not None and len(x) != len(y):
#             raise ValueError('X (images tensor) and y (labels) '
#                              'should have the same length. '
#                              'Found: X.shape = %s, y.shape = %s' %
#                              (np.asarray(x).shape, np.asarray(y).shape))
#         if dim_ordering == 'default':
#             dim_ordering = K.image_dim_ordering()
#         self.x = x
#         # self.x = np.asarray(x)
#         # if self.x.ndim != 4:
#         #     raise ValueError('Input data in `NumpyArrayIterator` '
#         #                      'should have rank 4. You passed an array '
#         #                      'with shape', self.x.shape)
#         # channels_axis = 3 if dim_ordering == 'tf' else 1
#         # if self.x.shape[channels_axis] not in {1, 3, 4}:
#         #     raise ValueError('NumpyArrayIterator is set to use the '
#         #                      'dimension ordering convention "' + dim_ordering + '" '
#         #                      '(channels on axis ' + str(channels_axis) + '), i.e. expected '
#         #                      'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
#         #                      'However, it was passed an array with shape ' + str(self.x.shape) +
#         #                      ' (' + str(self.x.shape[channels_axis]) + ' channels).')
#         if y is not None:
#             self.y = np.asarray(y)
#         else:
#             self.y = None
#         self.image_data_generator = image_data_generator
#         self.dim_ordering = dim_ordering
#         self.save_to_dir = save_to_dir
#         self.save_prefix = save_prefix
#         self.save_format = save_format
#         self.inds = inds
#         super(NumpyArrayIterator2, self).__init__(inds.shape[0], batch_size, shuffle, seed)
#
#
#     def next(self):
#         # for python 2.x.
#         # Keeps under lock only the mechanism which advances
#         # the indexing of each batch
#         # see http://anandology.com/blog/using-iterators-and-generators/
#         with self.lock:
#             index_array, current_index, current_batch_size = next(self.index_generator)
#         # The transformation of images is not under thread lock
#         # so it can be done in parallel
#         batch_x = np.zeros(tuple([current_batch_size] + [299, 299, 3]))
#         for i, j in enumerate(self.inds[index_array]):
#             if self.x[j].shape != (299, 299, 3):
#                 # print self.x[j].shape
#                 x = random_crop(self.x[j], (299, 299), 0, 1)
#                 # plt.imshow(array_to_img(x))
#                 # plt.show()
#                 assert(x.shape == (299, 299, 3))
#                 self.y[j] = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
#             else:
#                 x = self.x[j]
#             x = self.image_data_generator.random_transform(x.astype('float32'))
#             x = self.image_data_generator.standardize(x)
#             batch_x[i] = x
#         if self.save_to_dir:
#             for i in range(current_batch_size):
#                 img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
#                 fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                   index=current_index + i,
#                                                                   hash=np.random.randint(1e4),
#                                                                   format=self.save_format)
#                 img.save(os.path.join(self.save_to_dir, fname))
#         if self.y is None:
#             return batch_x
#         batch_y = self.y[index_array]
#         return batch_x, batch_y


def load_bb(bb_path):
    bounding_boxes, failures = build_bb_all(load_json_bbs(bb_path), failures=True)
    # for img in failures:
    #     bounding_boxes[img] = []
    return defaultdict(list, bounding_boxes)


def max_dim_crop(img, out_shape, bbox):
    max_dim = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    centerx, centery = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    new_bbox = (int(centerx - max_dim / 2), int(centery - max_dim / 2),
                int(centerx + max_dim / 2), int(centery + max_dim / 2))
    # print tuple(type(val) for val in new_bbox)
    img = img.crop(new_bbox)
    # print (type(out_shape[0]), type(out_shape[1]))

    return img.resize(out_shape, resample=Image.BICUBIC)


def save_to_dir(batch_x, save_dir='../input/preview/'):
    for i in range(batch_x.shape[0]):
        img = array_to_img(batch_x[i], scale=True)
        fname = 'i_{hash}.{format}'.format(hash=np.random.randint(0, 10000),
                                           format='jpg')
        img.save(os.path.join(save_dir, fname))

# Change this to load and resize the image to proper size
# Then turn to numpy array
def load_imgs(train_dir, classes, bounding_boxes, out_shape):
    # Load the images and their corresponding labels
    X_lst = []
    y_lst = []
    for i, label, in enumerate(classes):
        print("Loading {}".format(label))
        for img_name in os.listdir(os.path.join(train_dir, label)):
            if bounding_boxes[img_name]:
                for bbox in bounding_boxes[img_name]:
                    img = Image.open(os.path.join(train_dir, label, img_name))
                    img = max_dim_crop(img, out_shape, bbox)
                    # plt.imshow(img)
                    # plt.show()
                    X_lst.append(img_to_array(img))
                    y_lst.append(i)
            else:
                img = Image.open(os.path.join(train_dir, label, img_name))
                X_lst.append(img_to_array(img))
                y_lst.append(i)

    return X_lst, np.array(y_lst)


def roi_gen(inds, X_lst, y, augmentor, batch_size=32, shuffle=True, save_dir=None):

    while True:
        inds_of_inds = np.arange(len(inds))
        if shuffle:
            np.random.shuffle(inds_of_inds)
        batch_x = np.zeros(tuple([batch_size] + [299, 299, 3]))
        for i in xrange(0, len(inds_of_inds), batch_size):
            batch_inds = inds[inds_of_inds[i:i+batch_size]]
            batch_count = batch_inds.shape[0]
            for i, j in enumerate(batch_inds):
                x = X_lst[j]
                if x.shape != (299, 299, 3):
                    x = random_crop(x, (299, 299), 0, 1)
                    y[j] = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                x = augmentor.random_transform(x.astype('float32'))
                x = augmentor.standardize(x)
                batch_x[i] = x

            batch_y = y[batch_inds]
            if save_dir:
                save_to_dir(batch_x, save_dir=save_dir)
            yield batch_x[:batch_count], batch_y


best_model_file = '../fishyROIception_32batch_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True)


def pipeline1():
    bboxes = load_bb(BB_PATH)
    X, y = load_imgs(TRAIN_DIR, CLASSES, bboxes, TARGET_SIZE)
    print("Making Categoricals")
    y_cat = to_categorical(y)
    print("Creating Splitter")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    sss.get_n_splits()
    print("Splitting")
    train_ind, val_ind = next(sss.split(np.arange(len(X)), y))
    print("Initializing Augmentors")
    train_aug = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   rotation_range=180.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=(1/1.2, 1.2),
                                   channel_shift_range=0.,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rescale=1./255)
    val_aug = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rescale=1./255)

    print("Instantiating model")
    inception_nn = inception_barebones()
    print("Creating train gen")
    train_gen = roi_gen(train_ind, X, y_cat, train_aug, batch_size=BATCH_SIZE)
    print("Creating val gen")
    val_gen = roi_gen(val_ind, X, y_cat, val_aug, batch_size=BATCH_SIZE)

    # for i, tmp in enumerate(train_gen):
    #     if i == 1: raise StopIteration("Done making previews")

    inception_nn.fit_generator(train_gen, samples_per_epoch=train_ind.shape[0],
                               nb_epoch=25, callbacks=[best_model],
                               validation_data=val_gen,
                               nb_val_samples=val_ind.shape[0],
                               max_q_size=1)

pipeline1()
