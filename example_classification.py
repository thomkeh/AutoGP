import autogp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def _gaussian_generator(mean, cov, label, n):
    distribution = multivariate_normal(mean=mean, cov=cov)
    X = distribution.rvs(n)
    y = np.ones(n, dtype=float) * label
    return distribution, X, y


def _generate_feature(n, disc_factor):
    """Generate the non-sensitive features randomly"""
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1 = _gaussian_generator(mu1, sigma1, 1, n)  # positive class
    nv2, X2, y2 = _gaussian_generator(mu2, sigma2, 0, n)  # negative class

    # join the positive and negative class clusters
    inputs = np.vstack((X1, X2))
    outputs = np.hstack((y1, y2))

    rotation = np.array([[np.cos(disc_factor), -np.sin(disc_factor)],
                         [np.sin(disc_factor), np.cos(disc_factor)]])
    inputs_aux = inputs @ rotation

    #### Generate the sensitive feature here ####
    sensi_attr = []  # this array holds the sensitive feature value
    for i in range(len(inputs)):
        x = inputs_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s

        r = np.random.uniform()  # generate a random number from 0 to 1

        if r < p1:  # the first cluster is the positive class
            sensi_attr.append(1.0)  # 1.0 means its male
        else:
            sensi_attr.append(0.0)  # 0.0 -> female

    sensi_attr = np.array(sensi_attr)

    return inputs, outputs[:, np.newaxis], sensi_attr[:, np.newaxis]


"""Synthetic data with bias"""
SEED = 123
np.random.seed(SEED)  # set the random seed, which can be reproduced again

N_all = 250
disc_factor = np.pi / 5.0  # discrimination in the data -- decrease it to generate more discrimination
inputs, outputs, sensi_attr = _generate_feature(N_all, disc_factor)

num_train = 200

# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx)
xtrain = inputs[idx[: num_train]]
ytrain = outputs[idx[: num_train]]
strain = sensi_attr[idx[: num_train]]

data = autogp.datasets.DataSet(xtrain, ytrain, strain)
xtest = inputs[idx[num_train:]]
ytest = outputs[idx[num_train:]]
stest = sensi_attr[idx[num_train:]]

# Initialize the Gaussian process.
likelihood = autogp.lik.Logistic()
kernel = autogp.cov.SquaredExponential(1, white=1e-1)
inference = autogp.inf.VariationalInference(kernel, likelihood)

inducing_inputs = xtrain
model = autogp.GaussianProcess(inducing_inputs, kernel, inference, likelihood, inducing_outputs=ytrain)

# Train the model.

# model.fit(data, optimizer, batch_size=1, loo_steps=10, var_steps=10, epochs=100, optimize_inducing=False)
model.fit(data, batch_size=None, loo_steps=0, var_steps=1, epochs=500, optimize_inducing=False,
          only_hyper=False)

# Predict new inputs.
ypred, _ = model.predict(xtest)




