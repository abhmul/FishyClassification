import numpy as np
import scipy.ndimage as ndi
from functools import partial
import unittest as tst

import keras.backend as K


def input_img(*args, **kwargs):
    transform_mat = np.array([[1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.]])
    return lambda x, *args2, **kwargs2: (x, transform_mat)


def random_rotation(rg, fill_mode='nearest', cval=0.):
    # Take in a function, return a function that can be applied to x
    def outer(func):
        def inner(x, transform_mat=None, **kwargs):
            return partial(rand_rot, rg=rg, fill_mode=fill_mode, cval=cval, **kwargs)(*func(x, transform_mat, **kwargs))
        return inner
    return outer
    # return lambda func: \
    #     (lambda x, channel_axis, row_axis, col_axis, transform_mat=None, **kwargs:
    #      partial(rand_rot, rg=rg, fill_mode=fill_mode, cval=cval, channel_axis=channel_axis,
    #              row_axis=row_axis, col_axis=col_axis)(func(x, transform_mat, **kwargs)))


def random_shift(wrg, hrg, fill_mode='nearest', cval=0.):
    # Take in a function, return a function that can be applied to x
    return lambda func: \
        (lambda x, transform_mat=None, **kwargs:
         partial(rand_shift, wrg=wrg, hrg=hrg, fill_mode=fill_mode, cval=cval)(*func(x, transform_mat, **kwargs)))


def random_shear(intensity, fill_mode='nearest', cval=0.):
    # Take in a function, return a function that can be applied to x
    return lambda func: \
        (lambda x, transform_mat=None, channel_axis=0, row_axis=1, col_axis=2, **kwargs:
         partial(rand_shear, intensity=intensity, fill_mode=fill_mode, cval=cval, **kwargs)(*func(x, transform_mat, **kwargs)))


def random_zoom(zoom_range, fill_mode='nearest', cval=0.):
    # Take in a function, return a function that can be applied to x
    return lambda func: \
        (lambda x, transform_mat=None, **kwargs:
         partial(rand_zoom, zoom_range=zoom_range, fill_mode=fill_mode, cval=cval, **kwargs)(*func(x, transform_mat, **kwargs)))


def random_channel_shift(intensity, fill_mode='nearest', cval=0.):
    # Take in a function, return a function that can be applied to x
    return lambda func: \
        (lambda x, transform_mat=None, **kwargs:
         partial(rand_channel_shift, intensity=intensity, fill_mode=fill_mode, cval=cval, **kwargs)(*func(x, transform_mat, **kwargs)))


def compile_func(func, dim_ordering='default'):
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
    return partial(func, channel_axis=channel_axis, row_axis=row_axis, col_axis=col_axis)


def rand_rot(x, transform_mat, rg, row_axis=1, col_axis=2, channel_axis=0,
             fill_mode='nearest', cval=0., **kwargs):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    transform_matrix_scale = transform_matrix_offset_center(rotation_matrix, 1., 1.)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x, np.dot(transform_matrix_scale, transform_mat)


def rand_shift(x, transform_mat, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
               fill_mode='nearest', cval=0., **kwargs):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    ty = int(np.random.uniform(-hrg, hrg)*h)
    tx = int(np.random.uniform(-wrg, wrg)*w)
    transform_matrix_scale = np.array([[1, 0, float(ty) / h],
                                       [0, 1, float(tx) / w],
                                       [0, 0, 1]])
    translation_matrix = np.array([[1, 0, ty],
                                   [0, 1, tx],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x, np.dot(transform_matrix_scale, transform_mat)


def rand_shear(x, transform_mat, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0., **kwargs):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    transform_matrix_scale = transform_matrix_offset_center(shear_matrix, 1., 1.)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x, np.dot(transform_matrix_scale, transform_mat)


def rand_zoom(x, transform_mat, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0., **kwargs):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    transform_matrix_scale = transform_matrix_offset_center(zoom_matrix, 1., 1.)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x, np.dot(transform_matrix_scale, transform_mat)


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
        # Check it doesn't change the shape
        img = input_img()
        img = random_rotation(20.)(img)
        transformer = compile_func(img)

        test_arr = np.linspace(0.0, 1.0, 500 * 1000).reshape((500, 1000, 1))
        print transformer(test_arr)[1]
        self.assertEqual(transformer(test_arr)[0].shape, test_arr.shape)

    def testShift(self):
        # Check it doesn't change the shape
        img = input_img()
        img = random_rotation(20.)(img)
        img = random_shift(.1, .1)(img)
        img = random_shear(.1)(img)
        img = random_zoom((.8, 1.2))(img)
        img = random_channel_shift(.1)(img)
        transformer = compile_func(img)

        test_arr = np.linspace(0.0, 1.0, 500 * 1000).reshape((500, 1000, 1))
        print transformer(test_arr)[1]
        self.assertEqual(transformer(test_arr)[0].shape, test_arr.shape)

if __name__ == '__main__':
    tst.main()