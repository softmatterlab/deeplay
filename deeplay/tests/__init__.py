import unittest
from unittest.mock import patch
import torch

_patcher = None


def setUpModule():
    global _patcher
    _patcher = patch("torch.backends.mps.is_available", return_value=False)
    _patcher.start()


def tearDownModule():
    global _patcher
    _patcher.stop()


class TestMyModule(unittest.TestCase):

    def test_feature_one(self):
        self.assertFalse(torch.backends.mps.is_available())
