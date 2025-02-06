import unittest
import deeplay as dl
import dill
import torch


class TestPickling(unittest.TestCase):

    def assertSameOutput(self, net, x):
        net2 = dill.loads(dill.dumps(net))
        y = net(x)
        y2 = net2(x)
        yequal = torch.allclose(y, y2)
        self.assertTrue(yequal)

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

        self.assertSameOutput(net, x)

    def test_configure_base(self):

        net = dl.MultiLayerPerceptron(2, [1], 1)
        net.configure(in_features=4)
        net.build()

        x = torch.randn(10, 4)
        self.assertSameOutput(net, x)

    def test_configure_child(self):

        net = dl.MultiLayerPerceptron(2, [1], 1)
        net.blocks[0].configure(in_features=4)
        net.build()

        x = torch.randn(10, 4)

        self.assertSameOutput(net, x)

    def test_sequential(self):

        net = dl.Sequential(
            dl.MultiLayerPerceptron(2, [1], 1),
            dl.MultiLayerPerceptron(1, [1], 2),
        )
        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_sequential_configure_input(self):

        n1 = dl.MultiLayerPerceptron(2, [1], 1)
        n1.configure(hidden_features=[4])

        net = dl.Sequential(
            n1,
            dl.MultiLayerPerceptron(1, [1], 2),
        )

        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_sequential_build_input(self):

        n1 = dl.MultiLayerPerceptron(2, [1], 1)
        n1.configure(hidden_features=[4])
        n1.build()

        net = dl.Sequential(
            n1,
            dl.MultiLayerPerceptron(1, [1], 2),
        )

        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_setattr_scalar(self):
        class Net(dl.DeeplayModule):
            def __init__(self):
                super().__init__()
                self.mulf = 1
                self.net = dl.MultiLayerPerceptron(2, [1], 1)

            def forward(self, x):
                return self.net(x) * self.mulf

        net = Net()
        net.mulf = 2
        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_setattr_scalar_2(self):
        class Net(dl.DeeplayModule):
            def __init__(self):
                super().__init__()
                self.mulf = 1
                self.net = dl.MultiLayerPerceptron(2, [1], 1)

            def forward(self, x):
                return self.net(x) * self.mulf

        net = Net()
        net.set("mulf", 2)
        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_setattr_module(self):
        class Net(dl.DeeplayModule):
            def __init__(self):
                super().__init__()
                self.mulf = 1
                self.net = dl.MultiLayerPerceptron(2, [1], 1)

            def forward(self, x):
                return self.net(x) * self.mulf

        child = dl.MultiLayerPerceptron(2, [4], 1)

        net = Net()
        net.net = child

        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_setattr_module_2(self):
        class Net(dl.DeeplayModule):
            def __init__(self):
                super().__init__()
                self.mulf = 1
                self.net = dl.MultiLayerPerceptron(2, [1], 1)

            def forward(self, x):
                return self.net(x) * self.mulf

        child = dl.MultiLayerPerceptron(2, [4], 1)

        net = Net()
        net.set("net", child)
        net.build()

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)

    def test_setattr_submodule(self):
        class Net(dl.DeeplayModule):
            def __init__(self):
                super().__init__()
                self.mulf = 1
                self.net = dl.MultiLayerPerceptron(2, [1], 1)

            def forward(self, x):
                return self.net(x) * self.mulf

        child = dl.MultiLayerPerceptron(2, [1], 1)
        child.configure(hidden_features=[4])

        net = Net()
        net.net = child.blocks[0]
        net.build()

        self.assertEqual(net.net.layer.out_features, 4)

        x = torch.randn(10, 2)
        self.assertSameOutput(net, x)
