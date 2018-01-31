import numpy as np
import tensorflow as tf

from . import cov
from . import lik
from . import util
from . import inf


class GaussianProcess:
    """
    The class representing the AutoGP model.

    Instance variables:
        num_components:
        num_latent:
        num_samples:
        num_inducing:
        input_dim:
    """
    def __init__(self,
                 inducing_inputs,
                 cov_func,
                 inf_func,
                 # mean_func=mean.ZeroOffset(),
                 lik_func,
                 num_components=1,
                 diag_post=False,
                 inducing_outputs=None):
        """
        Args:
            lik_func: subclass of likelihoods.Likelihood
                An object representing the likelihood function p(y|f).
            cov_func: list of subclasses of kernels.Kernel
                A list of one kernel per latent function.
            inducing_inputs: ndarray
                An array of initial inducing input locations. Dimensions: num_inducing * input_dim.
            num_components: int
                The number of mixture of Gaussian components.
            diag_post: bool
                True if the mixture of Gaussians uses a diagonal covariance, False otherwise.
            num_samples: int
                The number of samples to approximate the expected log likelihood of the posterior.
        """
        # Get the actual functions if they were initialized as strings.
        self.inf = inf_func
        assert isinstance(self.inf, inf.Inference)
        num_latent = cov_func.num_latent_functions()

        # Repeat the inducing inputs for all latent processes if we haven't been given individually
        # specified inputs per process.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :], [num_latent, 1, 1])

        # Initialize all model dimension constants.
        num_inducing = inducing_inputs.shape[-2]
        self.input_dim = inducing_inputs.shape[-1]

        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions.
        self.raw_weights = tf.get_variable("raw_weights", [num_components], initializer=tf.zeros_initializer())
        self.raw_means = tf.get_variable("raw_means", [num_components, num_latent, num_inducing],
                                         initializer=tf.zeros_initializer())
        if diag_post:
            self.raw_covars = tf.get_variable("raw_covars", [num_components, num_latent, num_inducing],
                                              initializer=tf.ones_initializer())
        else:
            self.raw_covars = tf.get_variable("raw_covars", [num_components, num_latent] +
                                              util.tri_vec_shape(num_inducing), initializer=tf.zeros_initializer())
        self.raw_inducing_inputs = tf.get_variable("raw_inducing_inputs",
                                                   initializer=tf.constant(inducing_inputs, dtype=tf.float32))
        self.raw_likelihood_params = lik_func.get_params()
        self.raw_kernel_params = cov_func.get_params()
        if inducing_outputs is not None:
            self.inducing_outputs = tf.constant(inducing_outputs, dtype=tf.float32)

        # Define placeholder variables for training and predicting.
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                           name="train_inputs")
        self.train_outputs = tf.placeholder(tf.float32, shape=[None, None],
                                            name="train_outputs")
        self.test_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                          name="test_inputs")

        # Now build our computational graph.
        is_exact_inference = True
        if is_exact_inference:
            # this exact inference code has the following assumptions
            # the train inputs have been passed to this class as inducing inputs
            # the train outputs have been passed to this class as inducing outputs
            # batching has been switched off
            self.nelbo, self.predictions = inf.exact.build_graph(self.raw_inducing_inputs,
                                                                 self.inducing_outputs,
                                                                 self.test_inputs,
                                                                 num_inducing,
                                                                 cov_func.num_latent_functions(),
                                                                 cov_func,
                                                                 lik_func)
        else:
            self.nelbo, self.loo_loss, self.predictions = self.inf.inference(self.raw_weights,
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

    def fit(self, data, optimizer, loo_steps=10, var_steps=10, epochs=200, batch_size=None, display_step=1,
            hyper_with_elbo=True, optimize_inducing=True, test=None, loss=None, only_hyper=False):
        """
        Fit the Gaussian process model to the given data.

        Args:
            data: subclass of datasets.DataSet
                The train inputs and outputs.
            optimizer: TensorFlow optimizer
                The optimizer to use in the fitting process.
            loo_steps: int
                Number of steps to update hyper-parameters using loo objective
            var_steps: int
                Number of steps to update    variational parameters using variational objective (elbo).
            epochs: int
                The number of epochs to optimize the model for.
            batch_size: int
                The number of datapoints to use per mini-batch when training. If batch_size is None,
                then we perform batch gradient descent.
            display_step: int
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
            if only_hyper:
                self.train_step = optimizer.minimize(self.nelbo, var_list=hyper_param)
            else:
                loo_train_step = optimizer.minimize(self.loo_loss, var_list=hyper_param)
                if hyper_with_elbo:
                    self.train_step = optimizer.minimize(self.nelbo, var_list=var_param + hyper_param)
                else:
                    self.train_step = optimizer.minimize(self.nelbo, var_list=var_param)

            self.session.run(tf.global_variables_initializer())

        step = 0
        while data.epochs_completed < epochs:
            var_iter = 0
            while var_iter < var_steps:
                batch = data.next_batch(batch_size)
                self.session.run(self.train_step, feed_dict={self.train_inputs: batch[0],
                                                             self.train_outputs: batch[1],
                                                             self.num_train: num_train})
                if var_iter % display_step == 0:
                    self._print_state(data, test, loss, num_train, step, only_hyper)
                var_iter += 1
                step += 1

            loo_iter = 0
            while loo_iter < loo_steps and not only_hyper:
                batch = data.next_batch(batch_size)
                self.session.run(loo_train_step, feed_dict={self.train_inputs: batch[0],
                                                            self.train_outputs: batch[1],
                                                            self.num_train: num_train})
                if loo_iter % display_step == 0:
                    self._print_state(data, test, loss, num_train, step, False)
                loo_iter += 1
                step += 1

    def predict(self, test_inputs, batch_size=None):
        """
        Predict outputs given inputs.

        Args:
            test_inputs: ndarray
                Points on which we wish to make predictions. Dimensions: num_test * input_dim.
            batch_size: int
                The size of the batches we make predictions on. If batch_size is None, predict on the
                entire test set at once.

        Returns:
            ndarray
                The predicted mean of the test inputs. Dimensions: (num_test, output_dim).
            ndarray
                The predicted variance of the test inputs. Dimensions: (num_test, output_dim).
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

    def _print_state(self, data, test, loss, num_train, step, only_hyper):
        if num_train <= 100000:
            nelbo = self.session.run(self.nelbo, feed_dict={self.train_inputs: data.X,
                                                            self.train_outputs: data.Y,
                                                            self.num_train: num_train})
            print(f"step={step!r} [epoch={data.epochs_completed!r}] nelbo={nelbo!r}", end=" ")
            if not only_hyper:
                loo = self.session.run(self.loo_loss, feed_dict={self.train_inputs: data.X,
                                                                 self.train_outputs: data.Y,
                                                                 self.num_train: num_train})
                print(f"loo={loo!r}", end="")
            print("")

        if loss is not None:
            ypred = self.predict(test.X)[0]
            print(f"step={step!r} [epoch={data.epochs_completed!r}] current {loss.get_name()}={loss.eval(test.Y, ypred):.4}")
