from typing import Tuple, Union
import unittest
import torch

from kai_gpt.modeling import NativeAttention

class NativeAttentionImplTest(unittest.TestCase):
    
    def test_native_attention_without_attention(self):
        bs = 3
        seq_len = 5
        attention = NativeAttention(hidden_size=16, num_attention_heads=2, attn_dropout_probs=0.1)
        hidden_states = torch.randn((bs, seq_len, 16))
        
        output = attention(hidden_states, return_attentions=False)[0]
        
        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual((bs, seq_len, 16), output.shape)


    def test_native_attention_with_attention_mask(self):
        bs = 3
        seq_len = 5
        num_heads = 2
        
        attention = NativeAttention(hidden_size=16, num_attention_heads=num_heads, attn_dropout_probs=0.1)
        
        hidden_states = torch.randn((bs, seq_len, 16))
        attention_mask = torch.randn((bs, num_heads, seq_len, seq_len))
        
        output = attention(hidden_states, attention_mask, return_attentions=True)
        
        self.assertIsNotNone(output)
        self.assertEqual(2, len(output))
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertIsInstance(output[1], torch.Tensor)
        
        self.assertEqual((bs, seq_len, 16), output[0].shape)
        self.assertEqual((bs, num_heads, seq_len, seq_len), output[1].shape)
