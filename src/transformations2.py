import numpy as np
import scipy.ndimage as ndi
from functools import partial
import unittest as tst

import keras.backend as K


def apply_transform_coord(coord, transform_matrix, h=float('inf'), w=float('inf')):

    vector = np.array([[coord[0]],
                       [coord[1]],
                       [1]])
    transformed = np.dot(np.linalg.inv(transform_matrix), vector)
    return int(np.clip(transformed[0, 0], 0., h)), transformed[1, 0]


def standerdize(x_old, channel_axis):
    x = x_old
    x -= np.mean(x, axis=channel_axis, keepdims=True)
    x /= (np.std(x, axis=channel_axis, keepdims=True) + 1e-7)
    return x


def compile_affine(funcs, dim_ordering='default', fill_mode='nearest', cval=0.):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'tf', 'th'}:
        raise ValueError('dim_ordering should be "tf" (channel after row and '
                         'column) or "th" (channel before row and column). '
                         'Received arg: ', dim_ordering)
    if dim_ordering == 'th':
        channel_axis = 0
        row_axis = 1
        col_axis = 2
    if dim_ordering == 'tf':
        channel_axis = 2
        row_axis = 0
        col_axis = 1

    def transform_img(x, samplewise_center=True):
        transform_matrix = None
        h = x.shape[row_axis]
        w = x.shape[col_axis]
        for func in funcs:
            transform_matrix = func(transform_matrix, h, w)
        x_new = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        if samplewise_center:
            x_new = standerdize(x_new, channel_axis)
        return x_new, transform_matrix

    def transform_coord(coord, h, w):
        transform_matrix = None
        for func in funcs:
            transform_matrix = func(transform_matrix, h, w)
        transformed = apply_transform_coord(coord, transform_matrix, h, w)
        return transformed, transform_matrix

    def transform_func(*args):
        if len(args) == 3:
            return transform_coord(*args)
        elif len(args) == 1:
            return transform_img(*args)
        else:
            raise ValueError('Input should be an ndimage or coordinate, height, width.')

    return transform_func


def apply_rotation(theta, h, w, prev_transform=None):
    rad = np.pi / 180 * theta
    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad), 0],
                                [np.sin(rad), np.cos(rad), 0],
                                [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


def rotation(theta):
    """
    Returns a function that takes transform, height, and width of a picture
    adding on rotation transform for theta
    """
    def rot(transform, h, w):
        return apply_rotation(theta, h, w, prev_transform=transform)

    return rot


def random_rotation(rg):
    """
    Returns a function that takes transform, height, and width of a picture
    """

    def rand_rot(transform, h, w):
        return apply_rotation(np.random.uniform(-rg, rg), h, w, prev_transform=transform)

    return rand_rot


def apply_shift(tx, ty, h=None, w=None, prev_transform=None):
    #TODO figure out which shift changes which coordinate
    translation_matrix = np.array([[1, 0, ty],
                                   [0, 1, tx],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


def shift(tx_rel, ty_rel):
    """
    Returns a function that takes transform, height, and width of a picture
    adding on rotation transform for theta
    """
    def move(transform, h, w):
        return apply_shift(ty_rel*h, tx_rel*w, h, w, prev_transform=transform)

    return move


def random_shift(wrg, hrg):
    """
    Returns a function that takes transform, height, and width of a picture
    and applies a random shift to it
    """
    def rand_shift(transform, h, w):
        ty = int(np.random.uniform(-hrg, hrg)*h)
        tx = int(np.random.uniform(-wrg, wrg)*w)
        return apply_shift(tx, ty, prev_transform=transform)

    return rand_shift


def apply_shear(shear_val, h, w, prev_transform=None):
    shear_matrix = np.array([[1, -np.sin(shear_val), 0],
                             [0, np.cos(shear_val), 0],
                             [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


def shear(shear_val):
    """
        Returns a function that takes transform, height, and width of a picture
        and applies a random shear to it
        """

    def make_shear(transform, h, w):
        return apply_shear(shear_val, h, w, prev_transform=transform)

    return make_shear


def random_shear(intensity):
    """
    Returns a function that takes transform, height, and width of a picture
    and applies a random shear to it
    """
    def rand_shear(transform, h, w):
        return apply_shear(np.random.uniform(-intensity, intensity), h, w, prev_transform=transform)
    return rand_shear


def apply_zoom(zx, zy, h, w, prev_transform=None):
    # TODO see which order zx and zy should be in
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


def zoom(zx, zy):
    """
        Returns a function that takes transform, height, and width of a picture
        and applies a random zoom to it
        """

    def make_zoom(transform, h, w):
        return apply_zoom(zx, zy, h, w, prev_transform=transform)

    return make_zoom

def random_zoom(zoom_range):
    """
    Returns a function that takes transform, height, and width of a picture
    and applies a random zoom to it
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    def rand_zoom(transform, h, w):
        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        return apply_zoom(zx, zy, h, w, prev_transform=transform)
    return rand_zoom


# Not an Affine Transform
def rand_channel_shift(x, transform_mat, intensity, channel_axis=0, **kwargs):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x, transform_mat


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


class TestTransforms(tst.TestCase):

    def testRotate(self):
        transformer = compile_affine([rotation(45)])

        # Check it doesn't change the shape
        shape = (400, 200)
        loc = (int(shape[0] * .89), int(shape[1] * .6))
        test_arr = np.zeros(shape + (1,))
        test_arr[loc] = 1
        out = transformer(test_arr)[0].reshape(shape)
        # print test_arr.reshape(shape) - out
        # print loc
        # print transformer(loc, *shape)
        rounded_coords = tuple(int(round(i)) for i in (transformer(loc, *shape)[0]))
        # print rounded_coords
        self.assertEqual(out.shape, shape)
        # print out[rounded_coords]
        # print np.where(out==1.)
        self.assertEqual(out[rounded_coords], 1.)

    def testShift(self):
        transformer = compile_affine([shift(.2, .2)])

        # Check it doesn't change the shape
        shape = (8, 15)
        loc = (int(shape[0] * .5), int(shape[1] * .5))
        test_arr = np.zeros(shape + (1,))
        test_arr[loc] = 1
        out = transformer(test_arr)[0].reshape(shape)
        # print test_arr.reshape(shape) - out
        # print loc
        # print transformer(loc, *shape)
        rounded_coords = tuple(int(round(i)) for i in (transformer(loc, *shape)[0]))
        # print rounded_coords
        self.assertEqual(out.shape, shape)
        # print out[rounded_coords]
        # print np.where(out==1.)
        self.assertEqual(out[rounded_coords], 1.)

    def testRotateShift(self):
        transformer = compile_affine([rotation(90), shift(.2, .4)])
        shape = (8, 15)
        loc = (int(shape[0] * .6), int(shape[1] * .6))
        test_arr = np.zeros(shape + (1,))
        test_arr[loc] = 1
        out = transformer(test_arr)[0].reshape(shape)
        print test_arr.reshape(shape) - out
        print loc
        # print transformer(loc, *shape)
        rounded_coords = tuple(int(round(i)) for i in (transformer(loc, *shape)[0]))
        print rounded_coords
        self.assertEqual(out.shape, shape)
        # print out[rounded_coords]
        print np.where(out == 1.)
        self.assertEqual(out[rounded_coords], 1.)

if __name__ == '__main__':
    tst.main()