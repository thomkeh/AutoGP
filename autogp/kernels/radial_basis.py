import tensorflow as tf

from autogp import util
from . import kernel


class RadialBasis(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=1.0, std_dev=1.0,
                 white=0.01, input_scaling=False):
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.input_dim = input_dim
        self.white = white

    def kernel2(self, points1, points2=None):
        assert len(points1.shape) == 3
        points1 = points1 / self.lengthscale
        magnitude_square1 = tf.reduce_sum(points1**2, -1, keepdims=True)
        p1_shape = points1.shape
        if points2 is None:
            points2 = points1
            white_noise = self.white * tf.eye(points1.shape[-2].value)
            product = tf.matmul(points1, points2, transpose_b=True)
            magnitude_square2_t = tf.matrix_transpose(magnitude_square1)
        else:
            # if len(points2.shape) == 2 and len(points1.shape) == 3:
            #     points2 = points2[tf.newaxis, :, :]
            assert len(points2.shape) == 2
            points2 = points2 / self.lengthscale
            white_noise = 0.0
            product = util.matmul_br(points1, points2, transpose_b=True)
            magnitude_square2_t = tf.matrix_transpose(tf.reduce_sum(points2**2, -1, keepdims=True))
            # import ipdb; ipdb.set_trace()

        distances = magnitude_square1 - 2 * product + magnitude_square2_t
        # TODO(thomas): this seems wrong. why would we not want the covariance to go to zero no matter how far apart?
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST)

        kern = ((self.std_dev ** 2) * tf.exp(-distances / 2.0))
        return kern + white_noise

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * tf.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        points1 = points1 / self.lengthscale
        points2 = points2 / self.lengthscale
        magnitude_square1 = tf.reduce_sum(points1 ** 2, 1)[:, tf.newaxis]
        magnitude_square2 = tf.reduce_sum(points2 ** 2, 1)[:, tf.newaxis]
        distances = (magnitude_square1 - 2 * points1 @ tf.transpose(points2) +
                     tf.transpose(magnitude_square2))
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST);

        kern = ((self.std_dev ** 2) * tf.exp(-distances / 2.0))
        return kern + white_noise

    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[-2]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]
