import unittest
import torch

from kai_gpt.modeling.configuration import TransformerConfiguration
from kai_gpt.modeling.model import Transformer


class TransformerModelTest(unittest.TestCase):
    def test_build_4d_causal_mask(self):
        config = TransformerConfiguration()
        model = Transformer(config)
        
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        
        causal_attention_mask = model._build_4d_causal_mask(attention_mask=attention_mask, num_heads=4)
        
        self.assertIsNotNone(causal_attention_mask)
        self.assertEqual(6, (causal_attention_mask[0][0] == 0).sum().item()) 
        self.assertEqual(10, (causal_attention_mask[1][0] == 0).sum().item()) 
    
    
    def test_model_end2end(self):
        config = TransformerConfiguration()
        model = Transformer(config)
        
        input_ids = torch.tensor([[1, 4 , 5, 2, 0], [1, 4, 2, 0, 0]]).long()
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        
        
        output = model(input_ids, attention_mask)
        
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.last_hidden_states)
        self.assertEqual((2, 5, 16), output.last_hidden_states.shape)

    def test_generate_function(self):
        config = TransformerConfiguration()
        model = Transformer(config)
        
        generated_text = model.generate(max_new_tokens=20, temperature=1, top_k=250, strategy='random')
        
        self.assertEqual((1, 20 + 1), generated_text.shape) # +1 for bos_token
