from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.ndimage as ndi
import unittest as tst


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
        channel_images = [ndi.affine_transform(x_channel, final_affine_matrix,
                                                             final_offset, order=0, mode=fill_mode, cval=cval) for
                          x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x

    @staticmethod
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x],
                                  [0, 1, o_y],
                                  [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x],
                                 [0, 1, -o_y],
                                 [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


class ResizeRelative(Transform):
    def __init__(self, sx, sy, interp=3,
                 fill_mode='nearest', cval=0., **kwargs):
        self.sx = sx
        self.sy = sy
        self.interp = interp
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, channel_index=0, **kwargs):
        x = np.rollaxis(x, channel_index, 0)
        channel_images = [ndi.zoom(x_channel, (self.sy, self.sx), x_channel.dtype,
                                   order=self.interp, mode=self.fill_mode, cval=self.cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x


class ResizeAbsolute(Transform):
    def __init__(self, w, h,
                    fill_mode='nearest', cval=0., **kwargs):
        self.w = float(w)
        self.h = float(h)
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        sx = self.w / x.shape[col_index]
        sy = self.h / x.shape[row_index]
        return ResizeRelative(sx, sy, **vars(self)).apply(x, channel_index=channel_index)


class Rotate(Transform):
    def __init__(self, theta,
                    fill_mode='nearest', cval=0., **kwargs):
        self.theta = theta
        self.rad = np.pi / 180 * self.theta
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        rotation_matrix = np.array([[np.cos(self.rad), -np.sin(self.rad), 0],
                                    [np.sin(self.rad), np.cos(self.rad), 0],
                                    [0, 0, 1]])

        h, w = x.shape[row_index], x.shape[col_index]
        transform_matrix = self.transform_matrix_offset_center(rotation_matrix, h, w)
        x = self.apply_transform_mat(x, transform_matrix, channel_index, self.fill_mode, self.cval)
        return x


class RandomRotation(Transform):
    def __init__(self, rg,
                    fill_mode='nearest', cval=0., **kwargs):
        self.rg = rg
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        theta = np.random.uniform(-self.rg, self.rg)
        return Rotate(theta, **vars(self)).apply(x, row_index=row_index,
                                                 col_index=col_index,
                                                 channel_index=channel_index,
                                                 **kwargs)


class Shift(Transform):
    def __init__(self, tx, ty,
                 fill_mode='nearest', cval=0., **kwargs):
        self.tx = tx
        self.ty = ty
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, channel_index=0, **kwargs):
        translation_matrix = np.array([[1, 0, self.tx],
                                       [0, 1, self.ty],
                                       [0, 0, 1]])
        transform_matrix = translation_matrix  # no need to do offset
        x = self.apply_transform_mat(x, transform_matrix, channel_index, self.fill_mode, self.cval)
        return x


class RandomShift(Transform):
    def __init__(self, wrg, hrg,
                 fill_mode='nearest', cval=0., **kwargs):
        self.wrg = wrg
        self.hrg = hrg
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        h, w = x.shape[row_index], x.shape[col_index]
        tx = np.random.uniform(-self.hrg, self.hrg) * h
        ty = np.random.uniform(-self.wrg, self.wrg) * w
        return Shift(tx, ty, **vars(self)).apply(x, row_index=row_index, col_index=col_index,
                                                 channel_index=channel_index)


class Shear(Transform):
    def __init__(self, shear,
                 fill_mode='nearest', cval=0., **kwargs):
        self.shear = shear
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        shear_matrix = np.array([[1, -np.sin(self.shear), 0],
                                 [0, np.cos(self.shear), 0],
                                 [0, 0, 1]])

        h, w = x.shape[row_index], x.shape[col_index]
        transform_matrix = self.transform_matrix_offset_center(shear_matrix, h, w)
        x = self.apply_transform_mat(x, transform_matrix, channel_index, self.fill_mode, self.cval)
        return x


class RandomShear(Transform):
    def __init__(self, intensity,
                 fill_mode='nearest', cval=0., **kwargs):
        self.intensity = intensity
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        shear = np.random.uniform(-self.intensity, self.intensity)
        return Shear(shear, **vars(self)).apply(x, row_index=row_index, col_index=col_index,
                                                channel_index=channel_index)


class Zoom(Transform):
    def __init__(self, zx, zy,
                fill_mode='nearest', cval=0., **kwargs):
        self.zx = zx
        self.zy = zy
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        zoom_matrix = np.array([[self.zx, 0, 0],
                                [0, self.zy, 0],
                                [0, 0, 1]])

        h, w = x.shape[row_index], x.shape[col_index]
        transform_matrix = self.transform_matrix_offset_center(zoom_matrix, h, w)
        x = self.apply_transform_mat(x, transform_matrix, channel_index, self.fill_mode, self.cval)
        return x


class RandomZoom(Transform):
    def __init__(self, high_z, low_z=None,
                 fill_mode='nearest', cval=0., **kwargs):
        low_z = -high_z if low_z is None else low_z
        self.zoom_range = (1+low_z, 1+high_z)
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, channel_index=0, **kwargs):
        zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        return Zoom(zx, zy, **vars(self)).apply(x, row_index=row_index, col_index=col_index,
                                                channel_index=channel_index)


class RandomChannelShift(Transform):
    def __init__(self, intensity, **kwargs):
        self.intensity = intensity

    def apply(self, x, channel_index=0, **kwargs):
        x = np.rollaxis(x, channel_index, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [np.clip(x_channel + np.random.uniform(-self.intensity, self.intensity), min_x, max_x)
                          for x_channel in x]
        return self._channel_shift(channel_images, channel_index)

    @staticmethod
    def _channel_shift(channel_images, channel_index):
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x


class Crop(Transform):
    def __init__(self, y1, y2, x1, x2, **kwargs):
        self.y = (y1, y2)
        self.x = (x1, x2)

    def apply(self, x, row_index=1, col_index=2, **kwargs):
        x = np.rollaxis(x, row_index, 0)[self.y[0]:self.y[1]]
        x = np.rollaxis(x, 0, row_index+1)
        x = np.rollaxis(x, col_index, 0)[self.x[0]:self.x[1]]
        return np.rollaxis(x, 0, col_index+1)


class CenterCrop(Transform):
    def __init__(self, crop_size, **kwargs):
        self.crop_size = crop_size

    def apply(self, x, row_index=1, col_index=2, **kwargs):
        centerh, centerw = x.shape[row_index] // 2, x.shape[col_index] // 2
        halfh, halfw = self.crop_size[0] // 2, self.crop_size[1] // 2
        return Crop(max(centerh-halfh, 0),
                    centerh+halfh,
                    max(centerw-halfw, 0),
                    centerw+halfw,
                    **vars(self)).apply(x, row_index=row_index, col_index=col_index)


class RandomCrop(Transform):
    def __init__(self, crop_size, **kwargs):
        self.crop_size = crop_size

    def apply(self, x, row_index=1, col_index=2, **kwargs):
        h, w = x.shape[row_index], x.shape[col_index]
        rangew = (w - self.crop_size[1]) // 2
        rangeh = (h - self.crop_size[0]) // 2
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        return Crop(offseth, offseth+rangeh, offsetw, offsetw+rangew, **vars(self)).apply(x, row_index=row_index,
                                                                                          col_index=col_index)


class ROI(Transform):
    def __init__(self, bounding_boxes, from_dir=True,
               fill_mode='nearest', cval=0, **kwargs):
        self.bounding_boxes = bounding_boxes
        self.from_dir = from_dir
        self.fill_mode = fill_mode
        self.cval = cval

    def apply(self, x, row_index=1, col_index=2, **kwargs):
        NotImplementedError

    def _load_bounding_box(self, filename=None, index=None):
        if self.from_dir and filename is not None:
            return self.bounding_boxes[filename]
        elif index is not None:
            return self.bounding_boxes[index]
        else:
            TypeError('Read from directory set to %s but either filename or index is None' % self.from_dir)


class ROICenter(ROI):
    def apply(self, x, row_index=1, col_index=2, channel_index=0, filename=None, index=None, **kwargs):
        x_offset, y_offset, w, h = self._load_bounding_box(filename=filename, index=index)
        centerh, centerw = x.shape[row_index] // 2, x.shape[col_index] // 2
        ty = (x_offset + w // 2) - centerw
        tx = (y_offset + h // 2) - centerh
        return Shift(tx, ty, **vars(self)).apply(x, row_index=row_index, col_index=col_index,
                                                 channel_index=channel_index)


class ROICrop(ROI):
    def apply(self, x, row_index=1, col_index=2, filename=None, index=None, **kwargs):
        x_offset, y_offset, w, h = self._load_bounding_box(filename=filename, index=index)
        return Crop(y_offset, y_offset+h, x_offset, x_offset+w, **vars(self)).apply(x, row_index=row_index,
                                                                                    col_index=col_index)


class TestTransforms(tst.TestCase):

    def testResizeRelative(self):
        # test that scale of 1 does nothing
        sx, sy = 1., 1.
        resizer = ResizeRelative(sx, sy)
        test_arr = np.linspace(0.0, 1.0, 64 ** 2).reshape((1, 64, 64))
        np.testing.assert_array_almost_equal(resizer.apply(test_arr), test_arr)

        # test that we can decrease, increase, and do both to the size
        zooms = ((1.5, 2.), (.75, .5), (4., .125))
        test_arr = np.linspace(0.0, 1.0, 64 ** 2).reshape((1, 64, 64))
        for sx, sy in zooms:
            resizer = ResizeRelative(sx, sy)
            self.assertEqual(resizer.apply(test_arr).shape, (1, int(round(sy * 64)), int(round(sx * 64))))

    def testResizeAbsolute(self):
        # test that the same size does nothing
        h, w = 64, 64
        resizer = ResizeAbsolute(w, h)
        test_arr = np.linspace(0.0, 1.0, 64 ** 2).reshape((1, 64, 64))
        np.testing.assert_array_almost_equal(resizer.apply(test_arr), test_arr)

        # test that we can decrease, increase, and do both to the size
        sizes = ((31, 51), (253, 191), (17, 412))
        test_arr = np.linspace(0.0, 1.0, 64 ** 2).reshape((1, 64, 64))
        for h, w in sizes:
            resizer = ResizeAbsolute(w, h)
            self.assertEqual(resizer.apply(test_arr).shape, (1, h, w))

    def testRotate(self):
        # Check it doesn't change the shape
        rotater = Rotate(20)
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((1, 64, 128))
        self.assertEqual(rotater.apply(test_arr).shape, test_arr.shape)

        # Check a 360 degree rotation doesn't change the matrix
        rotater = Rotate(360)
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((1, 64, 128))
        np.testing.assert_array_almost_equal(rotater.apply(test_arr), test_arr)

    def testShift(self):
        # Check it shifted properly
        shifts = ((-2, 0), (0, -2), (-1, -1))
        test_arr = np.linspace(0.0, 1.0, 4 * 4).reshape((1, 4, 4))
        for tx, ty in shifts:
            shifter = Shift(tx, ty)
            np.testing.assert_array_almost_equal(shifter.apply(test_arr)[:, 0-tx:, 0-ty:], test_arr[:, :4+tx, :4+ty])

    def testCrop(self):
        # Check if we can crop properly
        crops = ((0, 25, 0, 129), (0, 65, 14, 19), (36, 54, 100, 111))
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((1, 64, 128))
        for y1, y2, x1, x2 in crops:
            cropper = Crop(y1, y2, x1, x2)
            np.testing.assert_array_almost_equal(cropper.apply(test_arr),
                                                 test_arr[:, y1:y2, x1:x2])

        # Check if it still works when we change the channel index
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((64, 128, 1))
        for y1, y2, x1, x2 in crops:
            cropper = Crop(y1, y2, x1, x2)
            np.testing.assert_array_almost_equal(cropper.apply(test_arr, row_index=0, col_index=1, channel_index=3),
                                                 test_arr[y1:y2, x1:x2])

    def testCenterCrop(self):
        crop_size = (64, 32)
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((1, 64, 128))
        cropper = CenterCrop(crop_size)
        startx = test_arr.shape[2] // 2 - (crop_size[1] // 2)
        starty = test_arr.shape[1] // 2 - (crop_size[0] // 2)
        np.testing.assert_array_almost_equal(cropper.apply(test_arr),
                                             test_arr[:, starty:starty+crop_size[0],
                                             startx:startx+crop_size[1]])

    def testROICenter(self):
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((1, 64, 128))
        x, y, w, h = (14, 42, 50, 10)
        bounding_boxes = {'1': (x, y, w, h)}
        roi = ROICenter(bounding_boxes)
        roi_centx, roi_centy = x + w // 2, y + h // 2
        self.assertAlmostEqual(roi.apply(test_arr, filename='1')[0, 64//2, 128//2], test_arr[0, roi_centy, roi_centx])

    def testROICrop(self):
        test_arr = np.linspace(0.0, 1.0, 64 * 128).reshape((1, 64, 128))
        x, y, w, h = (14, 42, 50, 10)
        bounding_boxes = {'1': (x, y, w, h)}
        roi = ROICrop(bounding_boxes)
        np.testing.assert_array_almost_equal(roi.apply(test_arr, filename='1'),
                               test_arr[:, y:y+h, x:x+w])


if __name__ == '__main__':
    tst.main()