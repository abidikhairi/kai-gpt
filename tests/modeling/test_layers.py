import unittest

import torch

from kai_gpt.modeling import (
    FeedForward
)


class FeedForwardModuleTest(unittest.TestCase):
    def test_feedforward(self):
        module = FeedForward(16, 32, ff_dropout_probs=0.1)
        hidden_states = torch.randn((3, 5, 16))
        
        output = module(hidden_states)
        
        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual((3, 5, 16), output.shape)
