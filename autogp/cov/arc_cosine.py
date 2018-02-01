import numpy as np
import tensorflow as tf

from . import cov


class ArcCosine(cov.Cov):
    """Kernel based on radial basis functions.

    Instance variables:
        lengthscale: Tensor(num_latent[, input_dim])
        std_dev: Tensor(num_latent)
    """
    def __init__(self, input_dim, output_dim=1, degree=0, depth=1, length_scale=np.ones(1), std_dev=np.ones(1),
                 white=np.array([1e-4]), input_scaling=False):
        self.num_latent = output_dim
        self.degree = degree
        self.depth = depth
        self.white = tf.constant(white, dtype=tf.float32)[:, tf.newaxis]
        init_value = tf.constant_initializer(length_scale, dtype=tf.float32)
        with tf.variable_scope("radial_basis_parameters"):
            if input_scaling:
                self.length_scale_raw = tf.get_variable("length_scale", [output_dim, input_dim], initializer=init_value)
            else:
                self.length_scale_raw = tf.get_variable("length_scale", [output_dim], initializer=init_value)
            self.std_dev = tf.get_variable("std_dev", [output_dim],
                                           initializer=tf.constant_initializer(std_dev, dtype=tf.float32))

        # reshape the parameters for easier use later
        self.lengthscale = tf.reshape(self.length_scale_raw, [self.num_latent, 1, input_dim if input_scaling else 1])

    def cov_func(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white[..., tf.newaxis] * tf.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        kern = self.recursive_kernel(points1 / self.lengthscale, points2 / self.lengthscale, self.depth)
        return (self.std_dev[:, tf.newaxis, tf.newaxis] ** 2) * kern + white_noise

    def recursive_kernel(self, points1, points2, depth):
        if depth == 1:
            mag_sqr1 = tf.reduce_sum(points1 ** 2, 1, keep_dims=True)
            mag_sqr2 = tf.reduce_sum(points2 ** 2, 1, keep_dims=True)
            point_prod = tf.matmul(points1, points2, transpose_b=True)  # points1 @ points2.T
        else:
            mag_sqr1 = tf.expand_dims(self.diag_recursive_kernel(points1, depth - 1), 1)
            mag_sqr2 = tf.expand_dims(self.diag_recursive_kernel(points2, depth - 1), 1)
            point_prod = self.recursive_kernel(points1, points2, depth - 1)

        mag_prod = tf.sqrt(mag_sqr1) * tf.matrix_transpose(tf.sqrt(mag_sqr2))
        cos_angles = (2 * point_prod) / (tf.sqrt(1 + 2 * mag_sqr1) * tf.matrix_transpose(tf.sqrt(1 + 2 * mag_sqr2)))

        return (((mag_prod ** self.degree) / np.pi) * self.angular_func(cos_angles))

    def diag_cov_func(self, points):
        return self.std_dev[:, tf.newaxis]**2 * self.diag_recursive_kernel(points / self.lengthscale, self.depth) + (
                self.white)

    # TODO(karl): Add a memoize decorator.
    # @util.memoize
    def diag_recursive_kernel(self, points, depth):
        # TODO(karl): Consider computing this in closed form.
        if depth == 1:
            mag_sqr = tf.reduce_sum(points ** 2, 1)
        else:
            mag_sqr = self.diag_recursive_kernel(points, depth - 1)

        return ((mag_sqr ** self.degree) * self.angular_func(2 * mag_sqr / (1 + 2 * mag_sqr)) / np.pi)

    def angular_func(self, cos_angles):
        angles = tf.acos(cos_angles)
        sin_angles = tf.sin(angles)
        pi_diff = np.pi - angles
        if self.degree == 0:
            return pi_diff
        elif self.degree == 1:
            return sin_angles + pi_diff * cos_angles
        elif self.degree == 2:
            return 3 * sin_angles * cos_angles + pi_diff * (1 + 2 * cos_angles ** 2)
        else:
            assert False

    def get_params(self):
        return [self.std_dev, self.length_scale_raw]

    def num_latent_functions(self):
        return self.num_latent
