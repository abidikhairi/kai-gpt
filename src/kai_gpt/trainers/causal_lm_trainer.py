from typing import Literal, Optional, Tuple
from torch import optim
from pytorch_lightning import LightningModule
import torch

from kai_gpt.modeling.configuration import TransformerConfiguration
from kai_gpt.modeling.model import Transformer
from kai_gpt.tokenization import GptTokenizerFast


class CausalLmModel(LightningModule):
    def __init__(
        self, 
        config: TransformerConfiguration,
        tokenizer: Optional[GptTokenizerFast] = None,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.99, 0.98),
        warmup_steps: int = 2000,
        max_training_steps: int = 10000,
    ) -> None:
        super().__init__()
        
        self.config = config
        self.model = Transformer(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = tokenizer
        
        self.learning_rate = learning_rate
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_training_steps = max_training_steps
    
        self.save_hyperparameters()
    
    def linear_warmup_decay_scheduler(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return max(0.0, (total_steps - current_step) / (total_steps - warmup_steps))
    
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=self.betas)
        scheduler = self.linear_warmup_decay_scheduler(optimizer, self.warmup_steps, self.max_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tokens_embeds: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens_embeds=tokens_embeds,
            return_hidden_states=return_hidden_states,
            return_attentions=return_attentions
        )


    def training_step(self, batch, batch_idx):
        labels = batch['input_ids'].clone()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        labels = labels[:, 1:]
        
        # input_ids[input_ids == self.tokenizer.pad_token_id] = -100 # type: ignore
        labels[labels == self.tokenizer.pad_token_id] = -100 # type: ignore
        
        batch_size, seq_len = input_ids.shape
        
        output = self(input_ids=input_ids, attention_mask=attention_mask)        
        
        loss = self.loss_fn(
            output.logits.view(batch_size * seq_len, -1),
            labels.contiguous().view(-1)
        )
        
        self.log('train/loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch['input_ids'].clone()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        labels = labels[:, 1:]
        
        # input_ids[input_ids == self.tokenizer.pad_token_id] = -100 # type: ignore
        labels[labels == self.tokenizer.pad_token_id] = -100 # type: ignore
        
        batch_size, seq_len = input_ids.shape
        
        output = self(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = self.loss_fn(
            output.logits.view(batch_size * seq_len, -1),
            labels.contiguous().view(-1)
        )
        
        self.log('valid/loss', loss)
        self.log('valid/perplexity', loss.exp())
        
        return loss

    @torch.no_grad()
    def generate(
        self,
        max_new_tokens: int = 20,
        temperature: float = 1,
        top_k: int = 250,
        strategy: Literal['greedy', 'random'] = 'greedy'
    ):
        return self.model.generate(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            strategy=strategy
        )
