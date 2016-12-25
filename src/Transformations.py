from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.ndimage as ndi

from keras import backend as K


class Transform(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, x, **kwargs):
        """Applies the transformation to some numpy array and returns the transformed array"""
        return

    @staticmethod
    def apply_transform_mat(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                             final_offset, order=0, mode=fill_mode, cval=cval) for
                          x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x

    @staticmethod
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


class RandomRotation(Transform):
    def __init__(self, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0., **kwargs):
        self.rg = rg
        self.row_index = row_index
        self.col_index = col_index
        self.channel_index = channel_index
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, **kwargs):
        theta = np.pi / 180 * np.random.uniform(-self.rg, self.rg)
        return self._rotate(x, theta)

    def _rotate(self, x, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        h, w = x.shape[self.row_index], x.shape[self.col_index]
        transform_matrix = self.transform_matrix_offset_center(rotation_matrix, h, w)
        x = self.apply_transform_mat(x, transform_matrix, self.channel_index, self.fill_mode, self.cval)
        return x


class RandomShift(Transform):
    def __init__(self, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0., **kwargs):
        self.wrg = wrg
        self.hrg = hrg
        self.row_index = row_index
        self.col_index = col_index
        self.channel_index = channel_index
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, **kwargs):
        h, w = x.shape[self.row_index], x.shape[self.col_index]
        tx = np.random.uniform(-self.hrg, self.hrg) * h
        ty = np.random.uniform(-self.wrg, self.wrg) * w
        return self._shift(x, tx, ty)

    def _shift(self, x, tx, ty):
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        transform_matrix = translation_matrix  # no need to do offset
        x = self.apply_transform_mat(x, transform_matrix, self.channel_index, self.fill_mode, self.cval)
        return x


class RandomShear(Transform):
    def __init__(self, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0., **kwargs):
        self.intensity = intensity
        self.row_index = row_index
        self.col_index = col_index
        self.channel_index = channel_index
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, **kwargs):
        shear = np.random.uniform(-self.intensity, self.intensity)
        return self._shear(x, shear)

    def _shear(self, x, shear):
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        h, w = x.shape[self.row_index], x.shape[self.col_index]
        transform_matrix = self.transform_matrix_offset_center(shear_matrix, h, w)
        x = self.apply_transform_mat(x, transform_matrix, self.channel_index, self.fill_mode, self.cval)
        return x


class RandomZoom(Transform):
    def __init__(self, zxrg, zyrg, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0., **kwargs):
        self.zoom_range = (zxrg, zyrg)
        self.row_index = row_index
        self.col_index = col_index
        self.channel_index = channel_index
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, **kwargs):
        zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        return self._zoom(x, zx, zy)

    def _zoom(self, x, zx, zy):
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        h, w = x.shape[self.row_index], x.shape[self.col_index]
        transform_matrix = self.transform_matrix_offset_center(zoom_matrix, h, w)
        x = self.apply_transform_mat(x, transform_matrix, self.channel_index, self.fill_mode, self.cval)
        return x


class RandomChannelShift(Transform):
    def __init__(self, intensity, channel_index=0, **kwargs):
        self.intensity = intensity
        self.channel_index = channel_index

    def apply(self, x, **kwargs):
        x = np.rollaxis(x, self.channel_index, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [np.clip(x_channel + np.random.uniform(-self.intensity, self.intensity), min_x, max_x)
                          for x_channel in x]
        return self._channel_shift(x, channel_images)

    def _channel_shift(self, x, channel_images):
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, self.channel_index + 1)
        return x


# TODO - DRY cropping code
# TODO - make output size = crop size (only problem when crop size > image size)
class CenterCrop(Transform):
    def __init__(self, crop_size, dim_ordering=K.image_dim_ordering(), **kwargs):
        self.crop_size = crop_size
        self.dim_ordering = dim_ordering
        if self.dim_ordering == 'tf':
            self.row_index = 0
            self.col_index = 1
        else:
            self.row_index = 1
            self.col_index = 2

    def apply(self, x, **kwargs):
        centerw, centerh = x.shape[self.row_index] // 2, x.shape[self.col_index] // 2
        halfw, halfh = self.crop_size[0] // 2, self.crop_size[1] // 2

        if self.dim_ordering == 'tf':
            return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh, :]
        else:
            return x[:, centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]


class RandomCrop(Transform):
    def __init__(self, crop_size, dim_ordering=K.image_dim_ordering(), **kwargs):
        self.crop_size = crop_size
        self.dim_ordering = dim_ordering
        if self.dim_ordering == 'tf':
            self.row_index = 0
            self.col_index = 1
        else:
            self.row_index = 1
            self.col_index = 2

    def apply(self, x, **kwargs):
        w, h = x.shape[1], x.shape[2]
        rangew = (w - self.crop_size[0]) // 2
        rangeh = (h - self.crop_size[1]) // 2
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        if self.dim_ordering == 'tf':
            return x[offsetw:offsetw + self.crop_size[0], offseth:offseth + self.crop_size[1], :]
        else:
            return x[:, offsetw:offsetw + self.crop_size[0], offseth:offseth + self.crop_size[1]]
