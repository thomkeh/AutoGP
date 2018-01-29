import numpy as np
import tensorflow as tf

from . import kernels
from . import likelihoods
from . import util


class GaussianProcess:
    """
    The class representing the AutoGP model.

    Parameters
    ----------
    likelihood_func : subclass of likelihoods.Likelihood
        An object representing the likelihood function p(y|f).
    kernel_funcs : list of subclasses of kernels.Kernel
        A list of one kernel per latent function.
    inducing_inputs : ndarray
        An array of initial inducing input locations. Dimensions: num_inducing * input_dim.
    num_components : int
        The number of mixture of Gaussian components.
    diag_post : bool
        True if the mixture of Gaussians uses a diagonal covariance, False otherwise.
    num_samples : int
        The number of samples to approximate the expected log likelihood of the posterior.
    """
    def __init__(self,
                 likelihood_func,
                 kernel_func,
                 inducing_inputs,
                 num_components=1,
                 diag_post=False,
                 num_samples=100):
        # Get the actual functions if they were initialized as strings.
        self.likelihood = likelihood_func
        self.kernel = kernel_func
        assert isinstance(self.kernel, kernels.Kernel)

        # Save whether our posterior is diagonal or not.
        self.diag_post = diag_post

        # Repeat the inducing inputs for all latent processes if we haven't been given individually
        # specified inputs per process.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :], [self.kernel.num_latent_functions(), 1, 1])

        # Initialize all model dimension constants.
        self.num_components = num_components
        self.num_latent = self.kernel.num_latent_functions()
        self.num_samples = num_samples
        self.num_inducing = inducing_inputs.shape[1]
        self.input_dim = inducing_inputs.shape[2]

        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions.
        self.raw_weights = tf.get_variable("raw_weights", [self.num_components], initializer=tf.zeros_initializer())
        self.raw_means = tf.get_variable("raw_means", [self.num_components, self.num_latent, self.num_inducing],
                                         initializer=tf.zeros_initializer())
        if self.diag_post:
            self.raw_covars = tf.get_variable("raw_covars", [self.num_components, self.num_latent, self.num_inducing],
                                              initializer=tf.ones_initializer())
        else:
            self.raw_covars = tf.get_variable("raw_covars", [self.num_components, self.num_latent] +
                                              util.tri_vec_shape(self.num_inducing), initializer=tf.zeros_initializer())
        self.raw_inducing_inputs = tf.get_variable("raw_inducing_inputs",
                                                   initializer=tf.constant(inducing_inputs, dtype=tf.float32))
        self.raw_likelihood_params = self.likelihood.get_params()
        self.raw_kernel_params = self.kernel.get_params()

        # Define placeholder variables for training and predicting.
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                           name="train_inputs")
        self.train_outputs = tf.placeholder(tf.float32, shape=[None, None],
                                            name="train_outputs")
        self.test_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                          name="test_inputs")

        # Now build our computational graph.
        self.nelbo, self.loo_loss, self.predictions = self._build_graph(self.raw_weights,
                                                                        self.raw_means,
                                                                        self.raw_covars,
                                                                        self.raw_inducing_inputs,
                                                                        self.train_inputs,
                                                                        self.train_outputs,
                                                                        self.num_train,
                                                                        self.test_inputs)

        # config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        # Do all the tensorflow bookkeeping.
        self.session = tf.Session()
        self.optimizer = None
        self.train_step = None

    def fit(self, data, optimizer, loo_steps=10, var_steps=10, epochs=200,
            batch_size=None, display_step=1, hyper_with_elbo=True,
            optimize_inducing=True, test=None, loss=None):
        """
        Fit the Gaussian process model to the given data.

        Parameters
        ----------
        data : subclass of datasets.DataSet
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        loo_steps : int
            Number of steps    to update hyper-parameters using loo objective
        var_steps : int
            Number of steps to update    variational parameters using variational objective (elbo).
        epochs : int
            The number of epochs to optimize the model for.
        batch_size : int
            The number of datapoints to use per mini-batch when training. If batch_size is None,
            then we perform batch gradient descent.
        display_step : int
            The frequency at which the objective values are printed out.
        hyper_with_elbo: bool
            True to optimize hyper-parameters using the elbo objective, in addition to using loo objective
            False to optimizer only variational posterior parameters with elbo objective
        optimize_inducing: bool
            True to optimizr inducing inputs
        """
        num_train = data.num_examples
        if batch_size is None:
            batch_size = num_train

        var_param = [self.raw_means, self.raw_covars, self.raw_weights]  # variational parameters
        hyper_param = self.raw_kernel_params + self.raw_likelihood_params  # hyperparameters
        if optimize_inducing:
            hyper_param = hyper_param + [self.raw_inducing_inputs]

        if self.optimizer != optimizer:
            self.optimizer = optimizer
            self.loo_train_step = optimizer.minimize(self.loo_loss, var_list=hyper_param)
            if (hyper_with_elbo) is True:
                self.train_step = optimizer.minimize(self.nelbo, var_list=var_param + hyper_param)
            else:
                self.train_step = optimizer.minimize(self.nelbo, var_list=var_param)

            self.session.run(tf.global_variables_initializer())

        iter = 0
        while data.epochs_completed < epochs:
            var_iter = 0
            while (var_iter < var_steps):
                batch = data.next_batch(batch_size)
                self.session.run(self.train_step, feed_dict={self.train_inputs: batch[0],
                                                             self.train_outputs: batch[1],
                                                             self.num_train: num_train})
                if var_iter % display_step == 0:
                    self._print_state(data, test, loss, num_train, iter)
                var_iter += 1
                iter += 1

            loo_iter = 0
            while (loo_iter < loo_steps):
                batch = data.next_batch(batch_size)
                self.session.run(self.loo_train_step, feed_dict={self.train_inputs: batch[0],
                                                                 self.train_outputs: batch[1],
                                                                 self.num_train: num_train})
                if loo_iter % display_step == 0:
                    self._print_state(data, test, loss, num_train, iter)
                loo_iter += 1
                iter += 1

    def predict(self, test_inputs, batch_size=None):
        """
        Predict outputs given inputs.

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test * input_dim.
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the
            entire test set at once.

        Returns
        -------
        ndarray
            The predicted mean of the test inputs. Dimensions: num_test * output_dim.
        ndarray
            The predicted variance of the test inputs. Dimensions: num_test * output_dim.
        """
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = util.ceil_divide(test_inputs.shape[0], batch_size)

        test_inputs = np.array_split(test_inputs, num_batches)
        pred_means = [0.0] * num_batches
        pred_vars = [0.0] * num_batches
        for i in range(num_batches):
            pred_means[i], pred_vars[i] = self.session.run(
                self.predictions, feed_dict={self.test_inputs: test_inputs[i]})

        return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

    def _print_state(self, data, test, loss, num_train, iter):
        if num_train <= 100000:
            nelbo = self.session.run(self.nelbo, feed_dict={self.train_inputs: data.X,
                                                            self.train_outputs: data.Y,
                                                            self.num_train: num_train})
            loo = self.session.run(self.loo_loss, feed_dict={self.train_inputs: data.X,
                                                             self.train_outputs: data.Y,
                                                             self.num_train: num_train})
            print(f"iter={iter!r} [epoch={data.epochs_completed!r}] nelbo={nelbo!r}", end=" ")
            print(f"loo={loo!r}")

        if loss is not None:
            ypred = self.predict(test.X)[0]
            print(f"iter={iter!r} [epoch={data.epochs_completed!r}] current {loss.get_name()}={loss.eval(test.Y, ypred):.4}")

    def _build_graph(self, raw_weights, raw_means, raw_covars, raw_inducing_inputs,
                     train_inputs, train_outputs, num_train, test_inputs):
        # First transform all raw variables into their internal form.
        # Use softmax(raw_weights) to keep all weights normalized.
        weights = tf.exp(raw_weights) / tf.reduce_sum(tf.exp(raw_weights))

        if self.diag_post:
            # Use exp(raw_covars) so as to guarantee the diagonal matrix remains positive definite.
            covars = tf.exp(raw_covars)
        else:
            # Use vec_to_tri(raw_covars) so as to only optimize over the lower triangular portion.
            # We note that we will always operate over the cholesky space internally.
            covars_list = [None] * self.num_components
            for i in range(self.num_components):
                mat = util.vec_to_tri(raw_covars[i, :, :])
                diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
                exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
                covars_list[i] = mat - diag_mat + exp_diag_mat
            covars = tf.stack(covars_list, 0)
        # Both inducing inputs and the posterior means can vary freely so don't change them.
        means = raw_means
        inducing_inputs = raw_inducing_inputs

        # Build the matrices of covariances between inducing inputs.
        kernel_chol = tf.cholesky(self.kernel.kernel(inducing_inputs))

        # Now build the objective function.
        entropy = self._build_entropy(weights, means, covars)
        cross_ent = self._build_cross_ent(weights, means, covars, kernel_chol)
        ell = self._build_ell(weights, means, covars, inducing_inputs,
                              kernel_chol, train_inputs, train_outputs)
        batch_size = tf.to_float(tf.shape(train_inputs)[0])
        nelbo = -((batch_size / num_train) * (entropy + cross_ent) + ell)

        # Build the leave one out loss function.
        loo_loss = self._build_loo_loss(weights, means, covars, inducing_inputs,
                                        kernel_chol, train_inputs, train_outputs)

        # Finally, build the prediction function.
        predictions = self._build_predict(weights, means, covars, inducing_inputs, kernel_chol, test_inputs)

        return nelbo, loo_loss, predictions

    def _build_loo_loss(self, weights, means, covars, inducing_inputs, kernel_chol, train_inputs, train_outputs):
        """Construct leave out one loss

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            inducing_inputs: (num_latent, num_inducing, input_dim)
            kernel_chol: (num_latent, num_inducing, num_inducing)
            test_inputs: (batch_size, input_dim)
            train_outputs: (batch_size, output_dim)
        Returns:
            LOO loss
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, train_inputs)
        loss = 0
        latent_samples = self._build_samples(kern_prods, kern_sums, means, covars)
        # output of log_cond_prob: (num_components, num_samples, batch_size, output_dim)
        loss_by_component = tf.reduce_mean(1.0 / (tf.exp(self.likelihood.log_cond_prob(
            train_outputs, latent_samples)) + 1e-7), axis=1)
        loss = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis] * loss_by_component, axis=0)
        return tf.reduce_sum(tf.log(loss))

    def _build_predict(self, weights, means, covars, inducing_inputs, kernel_chol, test_inputs):
        """Construct predictive distribution

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            inducing_inputs: (num_latent, num_inducing, input_dim)
            kernel_chol: (num_latent, num_inducing, num_inducing)
            test_inputs: (batch_size, input_dim)
        Returns:
            means and variances of the predictive distribution
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, test_inputs)
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        pred_means, pred_vars = self.likelihood.predict(sample_means, sample_vars)

        # Compute the mean and variance of the gaussian mixture from their components.
        # weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weights = weights[:, tf.newaxis, tf.newaxis]
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)
        return weighted_means, weighted_vars

    def _build_entropy(self, weights, means, covars):
        """Construct entropy.

        Args:
            weights: shape: (num_components)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
        Returns:
            Entropy (scalar)
        """
        # First build a square matrix of normals.
        if self.diag_post:
            # construct normal distributions for all combinations of compontents
            normal = util.DiagNormal(means, covars[tf.newaxis, ...] + covars[:, tf.newaxis, ...])
        else:
            # TODO(karl): Can we just stay in cholesky space somehow?
            square = util.mat_square(covars)
            covars_sum = tf.cholesky(square[tf.newaxis, ...] + square[:, tf.newaxis, ...])
            normal = util.CholNormal(means, covars_sum)
        # compute log probability of all means in all normal distributions
        # then sum over all latent functions
        # shape of log_normal_probs: (num_components, num_components)
        log_normal_probs = tf.reduce_sum(normal.log_prob(means[:, tf.newaxis, ...]), axis=-1)

        # Now compute the entropy.
        # broadcast `weights` into dimension 1, then do `logsumexp` in that dimension
        weighted_logsumexp_probs = tf.reduce_logsumexp(tf.log(weights) + log_normal_probs, 1)
        # multiply with weights again and then sum over it all
        return -tf.tensordot(weights, weighted_logsumexp_probs, 1)

    def _build_cross_ent(self, weights, means, covars, kernel_chol):
        """Construct the cross-entropy.

        Args:
            weights: shape: (num_components)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            kernel_chol: shape: (num_latent, num_inducing, num_inducing)
        Returns:
            Cross entropy as scalar
        """
        if self.diag_post:
            # TODO(karl): this is a bit inefficient since we're not making use of the fact
            # that covars is diagonal. A solution most likely involves a custom tf op.

            # shape of trace: (num_components, num_latent)
            trace = tf.trace(util.cholesky_solve_br(kernel_chol, tf.matrix_diag(covars)))
        else:
            trace = tf.reduce_sum(util.diag_mul(util.cholesky_solve_br(kernel_chol, covars),
                                                tf.matrix_transpose(covars)), axis=-1)

        # sum_val has the same shape as weights
        sum_val = tf.reduce_sum(util.CholNormal(means, kernel_chol).log_prob(0.0) - 0.5 * trace, -1)

        # dot product of weights and sum_val
        cross_ent = tf.tensordot(weights, sum_val, 1)

        return cross_ent

    def _build_ell(self, weights, means, covars, inducing_inputs, kernel_chol, train_inputs, train_outputs):
        """Construct the Expected Log Likelihood

        Args:
            weights: (num_components,)
            means: shape: (num_components, num_latent, num_inducing)
            covars: shape: (num_components, num_latent, num_inducing[, num_inducing])
            inducing_inputs: (num_latent, num_inducing, input_dim)
            kernel_chol: (num_latent, num_inducing, num_inducing)
            train_inputs: (batch_size, input_dim)
            train_outputs: (batch_size, output_dim)
        Returns:
            Expected log likelihood as scalar
        """
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, train_inputs)
        # shape of `latent_samples`: (num_components, num_samples, batch_size, num_latent)
        latent_samples = self._build_samples(kern_prods, kern_sums, means, covars)
        ell_by_compontent = tf.reduce_sum(self.likelihood.log_cond_prob(train_outputs, latent_samples), axis=[1, 2])

        # dot product
        ell = tf.tensordot(weights, ell_by_compontent, 1)
        return ell / self.num_samples

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        """Helper function for `_build_ell`

        Args:
            kernel_chol: Tensor(num_latent, num_inducing, num_inducing)
            inducing_inputs: Tensor(num_latent, num_inducing, input_dim)
            train_inputs: Tensor(batch_size, input_dim)
        Returns:
            `kern_prods` (num_latent, batch_size, num_inducing) and `kern_sums` (num_latent, batch_size)
        """
        # shape of ind_train_kern: (num_latent, num_inducing, batch_size)
        ind_train_kern = self.kernel.kernel(inducing_inputs, train_inputs)
        # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
        kern_prods = tf.matrix_transpose(tf.cholesky_solve(kernel_chol, ind_train_kern))
        # We only need the diagonal components.
        kern_sums = (self.kernel.diag_kernel(train_inputs) - util.diag_mul(kern_prods, ind_train_kern))

        return kern_prods, kern_sums

    def _build_samples(self, kern_prods, kern_sums, means, covars):
        """can handle 4D input

        Args:
            kern_prods: (num_latent, batch_size, num_inducing)
            kern_sums: (num_latent, batch_size)
            means: (num_components, num_latent, num_inducing)
            covars: (num_components, num_latent, num_inducing[, num_inducing])
        Returns:
        """
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        batch_size = tf.shape(sample_means)[-2]
        return (sample_means[:, tf.newaxis, ...] + tf.sqrt(sample_vars)[:, tf.newaxis, ...] *
                tf.random_normal([self.num_components, self.num_samples, batch_size, self.num_latent]))

    def _build_sample_info(self, kern_prods, kern_sums, means, covars):
        """Get means and variances of a distribution by sampling

        Args:
            kern_prods: (num_latent, batch_size, num_inducing)
            kern_sums: (num_latent, batch_size)
            means: (num_components, num_latent, num_inducing)
            covars: (num_components, num_latent, num_inducing[, num_inducing])
        Returns:
            sample_means (num_components, batch_size, num_latent), sample_vars (num_components, batch_size, num_latent)
        """
        if self.diag_post:
            quad_form = util.diag_mul(kern_prods * tf.expand_dims(covars, -2), tf.transpose(kern_prods, [0, 2, 1]))
        else:
            full_covar = util.mat_square(covars)  # same shape as covars
            quad_form = util.diag_mul(util.matmul_br(kern_prods, full_covar), tf.matrix_transpose(kern_prods))
        sample_means = util.matmul_br(kern_prods, means[..., tf.newaxis])  # (num_components, num_latent, batch_size, 1)
        sample_vars = tf.matrix_transpose(kern_sums + quad_form)  # (num_components, x, num_latent)
        return tf.matrix_transpose(tf.squeeze(sample_means, -1)), sample_vars
