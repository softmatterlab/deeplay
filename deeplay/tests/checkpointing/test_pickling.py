import unittest
import deeplay as dl
import dill
import torch


class TestPickling(unittest.TestCase):

    def test_pickling_direct(self):
        net = dl.MultiLayerPerceptron(2, [1], 1)
        s = dill.dumps(net)
        net2 = dill.loads(s)

        self.assertFalse(net._has_built)
        self.assertFalse(net2._has_built)

        self.assertEqual(len(net2.blocks), 2)
        self.assertEqual(net2.in_features, 2)
        self.assertEqual(net2.hidden_features, [1])
        self.assertEqual(net2.out_features, 1)

    def test_pickling_build(self):

        x = torch.randn(10, 2)

        net = dl.MultiLayerPerceptron(2, [1], 1)
        net.build()
        print(net._config_tape)

        s = dill.dumps(net)
        net2 = dill.loads(s)

        self.assertTrue(net._has_built)
        self.assertTrue(net2._has_built)

        y = net(x)
        y2 = net2(x)

        yequal = torch.allclose(y, y2)
        self.assertTrue(yequal)

    def test_configure_base(self):

        net = dl.MultiLayerPerceptron(4, [1], 1)
        print("conf1")
        net.configure(in_features=4)
        print("conf2")
        net.configure(in_features=8)
        # net.build()

        print(net._config_tape)

        s = dill.dumps(net)
        net2 = dill.loads(s)

        x = torch.randn(10, 4)
        y = net(x)
        y2 = net2(x)

        yequal = torch.allclose(y, y2)
        self.assertTrue(yequal)
