import unittest

from autogp import util


class TestInitList(unittest.TestCase):
    def test_empty(self):
        self.assertEquals(0, 0)
    # TODO(thomas): test matmul_br and broadcast
