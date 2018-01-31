import autogp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Generate synthetic data.
N_all = 200
N = 50
inputs = 5 * np.linspace(0, 1, num=N_all)[:, np.newaxis]
outputs = np.sin(inputs)

# selects training and test
idx = np.arange(N_all)
np.random.shuffle(idx)
xtrain = inputs[idx[:N]]
ytrain = outputs[idx[:N]]
data = autogp.datasets.DataSet(xtrain, ytrain)
xtest = inputs[idx[N:]]
ytest = outputs[idx[N:]]

# Initialize the Gaussian process.
likelihood = autogp.lik.Gaussian()
kernel = autogp.cov.SquaredExponential(1)
inference = autogp.inf.VariationalInference(kernel, likelihood)
inducing_inputs = xtrain
model = autogp.GaussianProcess(inducing_inputs, kernel, inference, likelihood, inducing_outputs=ytrain)

# Train the model.
optimizer = tf.train.RMSPropOptimizer(0.005)
model.fit(data, optimizer, batch_size=None, loo_steps=0, var_steps=1, epochs=100, optimize_inducing=False,
          hyper_with_elbo=True)

# Predict new inputs.
ypred, _ = model.predict(xtest)
plt.plot(xtrain, ytrain, '.', mew=2)
plt.plot(xtest, ytest, 'o', mew=2)
plt.plot(xtest, ypred, 'x', mew=2)
plt.show()
