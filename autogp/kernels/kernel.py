from __future__ import print_function, absolute_import, division
import abc


class Kernel:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def kernel(self, inputs1, inputs2=None):
        pass

    @abc.abstractmethod
    def diag_kernel(self, inputs1, inputs2=None):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass
