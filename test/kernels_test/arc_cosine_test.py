import unittest

import numpy as np
import tensorflow as tf

from autogp import cov


SIG_FIGS = 5


class TestArcCosine(unittest.TestCase):
    def cov_func(self, points1, points2=None, degree=0, depth=1):
        tf.reset_default_graph()
        arc_cosine = cov.ArcCosine(degree, depth, white=[0.0])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if points2 is not None:
            return sess.run(arc_cosine.cov_func(tf.constant(points1, dtype=tf.float32),
                                              tf.constant(points2, dtype=tf.float32)))
        else:
            return sess.run(arc_cosine.cov_func(tf.constant(points1, dtype=tf.float32)))

    def diag_cov_func(self, points, degree=0, depth=1):
        tf.reset_default_graph()
        arc_cosine = cov.ArcCosine(degree, depth, white=[0.0])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess.run(arc_cosine.diag_cov_func(tf.constant(points, dtype=tf.float32)))

    def test_simple_kern(self):
        kern = self.cov_func([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        np.testing.assert_almost_equal(kern, [[1.0, 0.5, 0.5],
                                              [0.5, 1.0, 0.5],
                                              [0.5, 0.5, 1.0]])

    def test_parallel_kern(self):
        kern = self.cov_func([[3.0, 5.0, 2.0],
                            [-3.0, -5.0, -2.0],
                            [6.0, 10.0, 4.0]])
        np.testing.assert_almost_equal(kern, [[1.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0],
                                              [1.0, 0.0, 1.0]])
