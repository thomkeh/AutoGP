"""
Graph for exact inference.
"""
import numpy as np
import tensorflow as tf

from autogp import util, inf, cov, lik


class Exact(inf.Inference):
    def __init__(self, cov_func: cov.Cov, lik_func: lik.Likelihood) -> None:
        self.cov = cov_func
        self.lik = lik_func

    def inference(self, raw_weights, raw_means, raw_covars, X, train_inputs, train_outputs, num_train, X_star, y):
        """Build graph for computing predictive mean and variance and negative log marginal likelihood.

        Args:
            X: inputs
            y: targets
            X_star: test inputs
        Returns:
            negative log marginal likelihood and predictive mean and variance
        """
        num_inducing = X.shape[-2].value
        sn = self.lik.get_params()[0]
        # Kxx (num_latent, num_train, num_train)
        Kxx = self.cov.cov_func(X) + sn**2 * tf.eye(num_inducing)
        # L (same size as Kxx)
        L = tf.cholesky(Kxx)
        # alpha = Kxx \ y
        # b = A \ y means A @ b = y
        # α (num_latent, num_train, 1)
        α = tf.cholesky_solve(L, tf.matrix_transpose(y)[..., tf.newaxis])
        return (-self._build_log_marginal_likelihood(y, L, α, num_inducing), None,
                self._build_prediction(X, X_star, L, α))

    def _build_prediction(self, X, X_star, L, alpha):
        # Kxx_star (num_latent, num_train, num_test)
        Kxx_star = self.cov.cov_func(X, X_star)
        # f_star_mean (num_latent, num_test, 1)
        f_star_mean = tf.matmul(Kxx_star, alpha, transpose_a=True)
        # Kx_star_x_star (num_latent, num_test, num_test)
        Kx_star_x_star = self.cov.cov_func(X_star)
        # v (num_latent, num_train, num_test)
        v = tf.cholesky_solve(L, Kxx_star)
        # var_f_star (same shape as Kx_star_x_star)
        var_f_star = Kx_star_x_star - tf.reduce_sum(v**2, -2)
        return tf.transpose(tf.squeeze(f_star_mean, -1)), tf.transpose(var_f_star)

    @staticmethod
    def _build_log_marginal_likelihood(y, L, α, num_train):
        n_dim = tf.constant(num_train, dtype=tf.float32)
        # contract the batch dimension, quad_form (num_latent,)
        quad_form = tf.einsum('bl,lb->l', y, tf.squeeze(α, 2))
        log_trace = tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), -1)
        # log_marginal_likelihood (num_latent,)
        log_marginal_likelihood = -0.5 * quad_form - log_trace - 0.5 * n_dim * tf.log(np.pi)
        # sum over num_latent in the end to get a scalar, this corresponds to mutliplying the marginal likelihoods
        # of all the latent functions
        return tf.reduce_sum(log_marginal_likelihood)
