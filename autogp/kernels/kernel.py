from abc import ABC, abstractmethod


class Kernel(ABC):
    @abstractmethod
    def kernel(self, inputs1, inputs2=None):
        pass

    @abstractmethod
    def diag_kernel(self, inputs1, inputs2=None):
        pass

    @abstractmethod
    def get_params(self):
        pass
