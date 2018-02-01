import sklearn.cluster
import autogp
from autogp import lik
from autogp import cov
import tensorflow as tf
from autogp import datasets
from autogp import losses
from autogp import util


def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


if __name__ == '__main__':
    FLAGS = util.util.get_flags()
    BATCH_SIZE = FLAGS.batch_size
    LEARNING_RATE = FLAGS.learning_rate
    DISPLAY_STEP = FLAGS.display_step
    EPOCHS = FLAGS.n_epochs
    NUM_SAMPLES = FLAGS.mc_train
    NUM_INDUCING = FLAGS.n_inducing
    IS_ARD = FLAGS.is_ard

    data, test, _ = datasets.import_mnist()

    # Setup initial values for the model.
    likelihood = lik.Softmax()
    kern = cov.SquaredExponential(data.X.shape[1], length_scale=[10] * 10, std_dev=[1] * 10, white=[.01] * 10,
                                  input_scaling=IS_ARD)
    # kern = [kernels.ArcCosine(X.shape[1], 2, 3, 5.0, 1.0, input_scaling=True) for i in range(10)]
    # RadialBasis(X.shape[1], input_scaling=True) for i in xrange(10)]

    Z = init_z(data.X, NUM_INDUCING)
    m = autogp.GaussianProcess(likelihood, kern, Z, num_samples=NUM_SAMPLES)

    # setting up loss to be reported during training
    error_rate = losses.ZeroOneLoss(data.Dout)

    import time
    otime = time.time()
    o = tf.train.RMSPropOptimizer(LEARNING_RATE)
    start = time.time()
    m.fit(data, o, loo_steps=50, var_steps=50, epochs=EPOCHS,
          batch_size=BATCH_SIZE, display_step=DISPLAY_STEP, test=test,
          loss=error_rate)
    print(time.time() - start)
    print(time.time() - otime)

    ypred = m.predict(test.X)[0]
    print(f"Final {error_rate.get_name()}={error_rate.eval(test.Y, ypred):.4}")
