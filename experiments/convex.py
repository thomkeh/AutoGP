import autogp
from autogp import datasets
from autogp import cov
from autogp import lik
from autogp import losses
from autogp import util
import numpy as np
import os
import subprocess
import pandas as pd
import sklearn.cluster
import sklearn.preprocessing
import tensorflow as tf

DATA_DIR = "experiments/data/"
TRAIN_PATH = DATA_DIR + "convex_train.amat"
TEST_PATH = DATA_DIR + "50k/convex_test.amat"


def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


def get_convex_data():
    print("Getting convex data ...")
    os.chdir('experiments/data')
    subprocess.call(["../scripts/get_convex_data.sh"])
    os.chdir("../../")
    print("done")


def normalize_features(xtrain, xtest):
    mean = np.mean(xtrain, axis=0)
    std = np.std(xtrain, axis=0)
    xtrain = (xtrain-mean)/std
    xtest = (xtest-mean)/std
    return xtrain, xtest


# Gettign the data
if os.path.exists(TRAIN_PATH) is False:  # directory does not exist, download the data
    get_convex_data()

FLAGS = util.get_flags()
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
DISPLAY_STEP = FLAGS.display_step
EPOCHS = FLAGS.n_epochs
NUM_SAMPLES =  FLAGS.mc_train
NUM_INDUCING = FLAGS.n_inducing
IS_ARD = FLAGS.is_ard
LENGTHSCALE = FLAGS.lengthscale
VAR_STEPS = FLAGS.var_steps
LOOCV_STEPS = FLAGS.loocv_steps
NUM_COMPONENTS = FLAGS.num_components
DEVICE_NAME = FLAGS.device_name
KERNEL = FLAGS.kernel
DEGREE = FLAGS.kernel_degree
DEPTH  = FLAGS.kernel_depth
HYPER_WITH_ELBO = FLAGS.hyper_with_elbo
OPTIMIZE_INDUCING = FLAGS.optimize_inducing
LATENT_NOISE = FLAGS.latent_noise

print(FLAGS.__flags)

# Read in and scale the data.
train_data = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None)
test_data = pd.read_csv(TEST_PATH, sep=r"\s+", header=None)
train_X = train_data.values[:, :-1]
train_Y = train_data.values[:, -1:]
test_X = test_data.values[:, :-1]
test_Y = test_data.values[:, -1:]

# feature normalization if requested
if (FLAGS.normalize_features is True):
    train_X, test_X = normalize_features(train_X, test_X)

data = datasets.DataSet(train_X, train_Y)
test = datasets.DataSet(test_X, test_Y)

Z = init_z(data.X, NUM_INDUCING)
likelihood = lik.Logistic()  # Setup initial values for the model.

if KERNEL == 'arccosine':
    kern = [cov.ArcCosine(data.X.shape[1], degree=DEGREE, depth=DEPTH, lengthscale=LENGTHSCALE, std_dev=1.0,
                          input_scaling=IS_ARD, white=LATENT_NOISE) for i in range(1)]
else:
    kern = [cov.SquaredExponential(data.X.shape[1], length_scale=LENGTHSCALE, input_scaling=IS_ARD,
                                   white=LATENT_NOISE) for i in range(1)]


m = autogp.GaussianProcess(likelihood, kern, Z, num_samples=NUM_SAMPLES, num_components=NUM_COMPONENTS)
error_rate = losses.ZeroOneLoss(data.Dout)
o = tf.train.AdamOptimizer(LEARNING_RATE)
m.fit(data, o, loo_steps=LOOCV_STEPS, var_steps=VAR_STEPS, epochs=EPOCHS, batch_size=BATCH_SIZE, display_step=DISPLAY_STEP,
      hyper_with_elbo=HYPER_WITH_ELBO, optimize_inducing=OPTIMIZE_INDUCING, test=test, loss=error_rate)

ypred = m.predict(test.X)[0]
print("Final " + error_rate.get_name() + "=" + "%.4f" % error_rate.eval(test.Y, ypred))
