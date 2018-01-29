import numpy as np
import tensorflow as tf

from autogp import util
from . import kernel


class RadialBasis(kernel.Kernel):
    """Kernel based on radial basis functions.

    Instance variables:
        lengthscale: Tensor(num_latent[, input_dim])
        std_dev: Tensor(num_latent)
    """
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=np.ones(1), std_dev=np.ones(1), white=np.array([.01]),
                 input_scaling=False):
        if len(lengthscale) != len(std_dev) or len(lengthscale) != len(white):
            raise ValueError("Parameters must have the same length.")
        self.num_latent = len(lengthscale)
        self.input_dim = input_dim
        self.white = tf.constant(white, dtype=tf.float32)
        self.input_scaling = input_scaling
        init_value = tf.constant(lengthscale, dtype=tf.float32)
        if input_scaling:
            self.lengthscale = tf.Variable(tf.tile(init_value[:, tf.newaxis], [1, input_dim]))
        else:
            self.lengthscale = tf.Variable(init_value)
        self.std_dev = tf.Variable(std_dev, dtype=tf.float32)

    def kernel(self, points1, points2=None):
        """
        Args:
            points1: Tensor(num_latent, num_inducing, input_dim) or Tensor(batch_size, input_dim)
            points2: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (num_latent, num_inducing, batch_size)
        """
        # reshape lengthscale so that the correct dimenions get divided in the line after
        lengthscale_br = tf.reshape(self.lengthscale, [self.num_latent, 1, self.input_dim if self.input_scaling else 1])
        points1 = points1 / lengthscale_br
        magnitude_square1 = tf.reduce_sum(points1**2, -1, keep_dims=True)
        if points2 is None:
            white_noise = self.white[:, tf.newaxis, tf.newaxis] * tf.eye(points1.shape[-2].value)
            points2 = points1
            product = tf.matmul(points1, points2, transpose_b=True)
            magnitude_square2_t = tf.matrix_transpose(magnitude_square1)
        else:
            white_noise = 0.0
            points2 = points2 / lengthscale_br
            product = util.matmul_br(points1, points2, transpose_b=True)
            magnitude_square2_t = tf.matrix_transpose(tf.reduce_sum(points2**2, -1, keep_dims=True))

        distances = magnitude_square1 - 2 * product + magnitude_square2_t
        # TODO(thomas): this seems wrong. why would we not want the covariance to go to zero no matter how far apart?
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST)

        kern = ((self.std_dev[:, tf.newaxis, tf.newaxis] ** 2) * tf.exp(-distances / 2.0))
        return kern + white_noise

    def diag_kernel(self, points):
        """
        Args:
            points: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (num_latent, batch_size)
        """
        return ((self.std_dev ** 2) + self.white)[:, tf.newaxis] * tf.ones([tf.shape(points)[-2]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

    def num_latent_functions(self):
        return self.num_latent
