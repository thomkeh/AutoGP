from abc import ABC, abstractmethod


class Cov(ABC):
    @abstractmethod
    def cov_func(self, inputs1, inputs2=None):
        pass

    @abstractmethod
    def diag_cov_func(self, inputs1):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def num_latent_functions(self):
        pass
