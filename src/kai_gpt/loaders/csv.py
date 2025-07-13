import os.path as osp
from pytorch_lightning import LightningDataModule
from datasets import Dataset
from torch.utils.data import DataLoader

from kai_gpt.tokenization import GptTokenizerFast


SPLITS_NAMES = ['train', 'validation', 'test']

class CsvTextDataset(LightningDataModule):
    """A PyTorch Lightning DataModule for loading and processing text data from CSV files.

    This module handles:
    - Loading CSV files for train/validation/test splits
    - Tokenizing text data
    - Creating batched datasets for training
    - Parallel processing of data

    Args:
        dataset_dir (str): Directory containing CSV files (train.csv, val.csv, test.csv)
        tokenizer (GptTokenizerFast): Tokenizer for processing text
        text_column (str, optional): Name of column containing text data. Default: 'text'
        batch_size (int, optional): Batch size for DataLoaders. Default: 4
        max_seq_len (int, optional): Maximum sequence length for tokenization. Default: 512
        num_proc (int, optional): Number of processes for parallel processing. Default: 4
        file_ext (str, optional): File extension for dataset files. Default: 'csv'

    Attributes:
        splits (dict): Mapping of split names to file paths
        datasets (dict): Processed datasets for each split
        tokenizer (GptTokenizerFast): Text tokenizer
        max_seq_len (int): Maximum sequence length
        text_column (str): Name of text column in CSV
        batch_size (int): Batch size
        num_proc (int): Number of processing workers

    Example:
        >>> tokenizer = GptTokenizerFast.from_pretrained('gpt2')
        >>> dataset = CsvTextDataset(
        ...     dataset_dir='data/',
        ...     tokenizer=tokenizer,
        ...     batch_size=32
        ... )
        >>> dataset.setup()
        >>> train_loader = dataset.train_dataloader()
    """
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: GptTokenizerFast,
        text_column: str = 'text',
        batch_size: int = 4, 
        max_seq_len: int = 512,
        num_proc: int = 4,
        file_ext: str = 'csv'
    ) -> None:
        super().__init__()
        
        self.splits = {}
        for split in SPLITS_NAMES:
            if osp.exists(f'{dataset_dir}/{split}.{file_ext}'):
                self.splits[split] = f'{dataset_dir}/{split}.{file_ext}'
            else:
                # TODO: log warning here
                pass
            
        self.datasets = {}
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_column = text_column
        self.batch_size = batch_size
        self.num_proc = num_proc

        
    def setup(self, stage: str) -> None:
        for split, filepath in self.splits.items():
            self.datasets[split] = self._prepare_dataset(Dataset.from_csv(filepath)) # type: ignore
        
        return super().setup(stage)
    
    def _tokenize_batch(self, examples):
        return self.tokenizer(examples[self.text_column], padding="max_length", max_length=self.max_seq_len)
    
    def _prepare_dataset(self, dataset: Dataset):
        return dataset.map(self._tokenize_batch, batched=True, batch_size=512, num_proc=self.num_proc) \
            .select_columns(['input_ids', 'attention_mask']) \
            .with_format('torch')
    
    def _get_dataloader(self, dataset: DataLoader, shuffle: bool = True):
        return DataLoader(
            dataset=dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_proc
        )
    
    def train_dataloader(self):
        return self._get_dataloader(self.datasets['train'], True) # type: ignore

    
    def val_dataloader(self):
        return self._get_dataloader(self.datasets['validation'], True) # type: ignore

    
    def test_dataloader(self):
        return self._get_dataloader(self.datasets['test'], False) # type: ignore

