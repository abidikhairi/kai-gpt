import unittest

import torch

from kai_gpt.modeling import (
    FeedForward,
    TransformerLayer
)


class FeedForwardModuleTest(unittest.TestCase):
    def test_feedforward(self):
        module = FeedForward(16, 32, ff_dropout_probs=0.1)
        hidden_states = torch.randn((3, 5, 16))
        
        output = module(hidden_states)
        
        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual((3, 5, 16), output.shape)


class TransformerLayerTest(unittest.TestCase):
    def test_transformer_layer(self):
        
        layer = TransformerLayer(
            hidden_size=16,
            num_attention_heads=4,
            intermediate_size=32,
            attn_dropout_probs=0.1,
            ff_dropout_probs=0.1,
            rms_eps=6e-10
        )
        
        hidden_states = torch.randn((3, 5, 16))
        attention_mask = torch.randn((3, 4, 5, 5))
        
        output = layer(hidden_states=hidden_states, attention_mask=attention_mask, return_attentions=True)
        
        self.assertIsNotNone(output)
        self.assertEqual(2, len(output))
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertEqual((3, 5, 16), output[0].shape)
        self.assertEqual((3, 4, 5, 5), output[1].shape)
    