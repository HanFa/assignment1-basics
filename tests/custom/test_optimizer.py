import unittest

import torch.nn

from cs336_basics.optimizers import SGD


class TestOptimizer(unittest.TestCase):

    def test_simple_training_with_sgd_optimizer(self):
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

        optimizer = SGD([weights], lr=1e2)

        for step in range(100):
            optimizer.zero_grad()

            loss = (weights ** 2).mean()

            print(f"loss {loss.cpu().item()}, step {step}")

            loss.backward()
            optimizer.step()
