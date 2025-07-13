import unittest
import torch

from kai_gpt.modeling import Embedding

class EmbeddingTableTest(unittest.TestCase):
    
    def test_lookup_table(self):
        input_ids = torch.tensor([1, 2, 3, 4]).long()
        
        embeddings = Embedding(num_embeddings=5, embedding_dim=32, padding_idx=0)
        
        output = embeddings(input_ids)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual((4, 32), output.shape)
    
    def test_padding_should_return_zeros(self):
        input_ids = torch.zeros((1, 4)).long()
        
        embeddings = Embedding(num_embeddings=5, embedding_dim=32, padding_idx=0)
        
        output = embeddings(input_ids)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(0, torch.sum(output))
    