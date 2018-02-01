from abc import ABC, abstractmethod


class Inference(ABC):
    @abstractmethod
    def inference(self, raw_weights, raw_means, raw_covars, raw_inducing_inputs,
                  train_inputs, train_outputs, num_train, test_inputs):
        pass
