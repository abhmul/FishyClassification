import numpy as np
import scipy.ndimage as ndi
from functools import partial
import unittest as tst

import keras.backend as K


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

    def transform_img(x):
        transform_matrix = None
        h = x.shape[row_axis]
        w = x.shape[col_axis]
        for func in funcs:
            transform_matrix = func(transform_matrix, h, w)
        return apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    def transform_coord(coord, h, w):
        transform_matrix = None
        for func in funcs:
            transform_matrix = func(transform_matrix, h, w)
        vector = np.array([[coord[0]],
                           [coord[1]],
                           [1]])
        transformed = np.dot(np.linalg.inv(transform_matrix), vector)
        return transformed[0,0], transformed[1,0]

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


def shift(tx, ty, h=None, w=None, prev_transform=None):
    #TODO figure out which shift changes which coordinate
    translation_matrix = np.array([[1, 0, ty],
                                   [0, 1, tx],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


def random_shift(wrg, hrg):
    """
    Returns a function that takes transform, height, and width of a picture
    and applies a random shift to it
    """
    def rand_shift(transform, h, w):
        ty = int(np.random.uniform(-hrg, hrg)*h)
        tx = int(np.random.uniform(-wrg, wrg)*w)
        return shift(tx, ty, prev_transform=transform)

    return rand_shift


def shear(shear_val, h, w, prev_transform=None):
    shear_matrix = np.array([[1, -np.sin(shear_val), 0],
                             [0, np.cos(shear_val), 0],
                             [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


def random_shear(intensity):
    """
    Returns a function that takes transform, height, and width of a picture
    and applies a random shear to it
    """
    def rand_shear(transform, h, w):
        return shear(np.random.uniform(-intensity, intensity), h, w, prev_transform=transform)
    return rand_shear


def zoom(zx, zy, h, w, prev_transform=None):
    # TODO see which order zx and zy should be in
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    if prev_transform is not None:
        transform_matrix = np.dot(transform_matrix, prev_transform)
    return transform_matrix


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
        return zoom(zx, zy, h, w, prev_transform=transform)
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
        shape = (100, 200)
        loc = (int(shape[0] * .7), int(shape[1] * .4))
        test_arr = np.zeros(shape + (1,))
        test_arr[loc] = 1
        out = transformer(test_arr).reshape(shape)
        print test_arr.reshape(shape) - out
        print loc
        print transformer(loc, *shape)
        rounded_coords = tuple(int(round(i)) for i in (transformer(loc, *shape)))
        print rounded_coords
        self.assertEqual(out.shape, shape)
        print out[rounded_coords]
        print np.where(out==1.)
        self.assertEqual(out[rounded_coords], 1.)
    #
    # def testShift(self):
    #     # Check it doesn't change the shape
    #     img = input_img()
    #     img = random_rotation(20.)(img)
    #     img = random_shift(.1, .1)(img)
    #     img = random_shear(.1)(img)
    #     img = random_zoom((.8, 1.2))(img)
    #     img = random_channel_shift(.1)(img)
    #     transformer = compile_func(img)
    #
    #     test_arr = np.linspace(0.0, 1.0, 500 * 1000).reshape((500, 1000, 1))
    #     print transformer(test_arr)[1]
    #     self.assertEqual(transformer(test_arr)[0].shape, test_arr.shape)

if __name__ == '__main__':
    tst.main()