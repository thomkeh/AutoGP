import numpy as np
import tensorflow as tf

from autogp import util
from . import cov


class SquaredExponential(cov.Cov):
    """Kernel based on radial basis functions.

    Instance variables:
        length_scale: Tensor(num_latent[, input_dim])
        std_dev: Tensor(num_latent)
    """
    MAX_DIST = 1e8

    def __init__(self, input_dim, output_dim=1, length_scale=1.0, std_dev=1.0, white=0.01, input_scaling=False):
        self.input_dim = input_dim
        self.num_latent = output_dim
        self.white = tf.constant(white, dtype=tf.float32)
        self.input_scaling = input_scaling
        init_value = tf.constant_initializer(length_scale, dtype=tf.float32)
        with tf.variable_scope("radial_basis_parameters"):
            if input_scaling:
                self.length_scale = tf.get_variable("lengthscale", [output_dim, input_dim], initializer=init_value)
            else:
                self.length_scale = tf.get_variable("lengthscale", [output_dim], initializer=init_value)
            self.std_dev = tf.get_variable("std_dev", [output_dim],
                                           initializer=tf.constant_initializer(std_dev, dtype=tf.float32))

    def cov_func(self, points1, points2=None):
        """
        Args:
            points1: Tensor(num_latent, num_inducing, input_dim) or Tensor(batch_size, input_dim)
            points2: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (num_latent, num_inducing, batch_size)
        """
        # reshape lengthscale so that the correct dimenions get divided in the line after
        lengthscale_br = tf.reshape(self.length_scale, [self.num_latent, 1, self.input_dim if self.input_scaling else 1])
        points1 = points1 / lengthscale_br
        magnitude_square1 = tf.reduce_sum(points1**2, -1, keepdims=True)
        if points2 is None:
            white_noise = self.white * tf.eye(tf.shape(points1)[-2])
            points2 = points1
            product = tf.matmul(points1, points2, transpose_b=True)
            magnitude_square2_t = tf.matrix_transpose(magnitude_square1)
        else:
            white_noise = 0.0
            points2 = points2 / lengthscale_br
            product = util.matmul_br(points1, points2, transpose_b=True)
            magnitude_square2_t = tf.matrix_transpose(tf.reduce_sum(points2**2, -1, keepdims=True))

        distances = magnitude_square1 - 2 * product + magnitude_square2_t
        # TODO(thomas): this seems wrong. why would we not want the covariance to go to zero no matter how far apart?
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST)

        kern = ((self.std_dev[:, tf.newaxis, tf.newaxis] ** 2) * tf.exp(-distances / 2.0))
        return kern + white_noise

    def diag_cov_func(self, points):
        """
        Args:
            points: Tensor(batch_size, input_dim)
        Returns:
            Tensor of shape (num_latent, batch_size)
        """
        return ((self.std_dev ** 2) + self.white)[:, tf.newaxis] * tf.ones([tf.shape(points)[-2]])

    def get_params(self):
        return [self.length_scale, self.std_dev]

    def num_latent_functions(self):
        return self.num_latent
