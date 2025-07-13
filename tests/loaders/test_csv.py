import unittest
import pandas as pd

from kai_gpt.loaders import CsvTextDataset
from kai_gpt.tokenization import GptTokenizerFast


class CsvDatasetTest(unittest.TestCase):
    def _prepare_dataset(self):
        text = "Okay, so I need to find the radius of this circle Γ that touches both arcs AC and BD externally and also touches the side CD of the unit square ABCD."
        dataset_dir = 'data/temp/toys'
        data = [{"text": text, "index": i} for i in range(16)]
        df = pd.DataFrame(data)
        df.to_csv(f'{dataset_dir}/train.csv', index=False)
        df.to_csv(f'{dataset_dir}/validation.csv', index=False)
        df.to_csv(f'{dataset_dir}/test.csv', index=False)
        
        return dataset_dir

    def test_train_dataloader(self):
        dataset_root = self._prepare_dataset()
        tokenizer = GptTokenizerFast.from_pretrained('data/tokenizer') # always exists (even in github)
        
        dataset = CsvTextDataset(
            dataset_dir=dataset_root,
            tokenizer=tokenizer,
            text_column="text",
            max_seq_len=128,
            batch_size=4
        )
        
        dataset.setup(stage='fit')
        train_loader = dataset.train_dataloader()
        
        self.assertEqual(4, len(train_loader))
        
        batch = next(iter(train_loader))
        self.assertIsNotNone(batch)
        self.assertIsNotNone(batch['input_ids'])
        self.assertIsNotNone(batch['attention_mask'])
        self.assertEqual((4, 128), batch['input_ids'].shape)
        self.assertEqual((4, 128), batch['attention_mask'].shape)
    
    def test_validation_dataloader(self):
        dataset_root = self._prepare_dataset()
        tokenizer = GptTokenizerFast.from_pretrained('data/tokenizer') # always exists (even in github)
        
        dataset = CsvTextDataset(
            dataset_dir=dataset_root,
            tokenizer=tokenizer,
            text_column="text",
            max_seq_len=128,
            batch_size=4
        )
        
        dataset.setup(stage='fit')
        valid_loader = dataset.val_dataloader()
        
        self.assertEqual(4, len(valid_loader))
        
        batch = next(iter(valid_loader))
        self.assertIsNotNone(batch)
        self.assertIsNotNone(batch['input_ids'])
        self.assertIsNotNone(batch['attention_mask'])
        self.assertEqual((4, 128), batch['input_ids'].shape)
        self.assertEqual((4, 128), batch['attention_mask'].shape)
    