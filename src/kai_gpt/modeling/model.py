from typing import Literal, Optional
import torch
from torch import nn

from kai_gpt.modeling.modeling_outputs import TransformerOutput
from kai_gpt.modeling.configuration import TransformerConfiguration

from kai_gpt.modeling.embedding import TokenEmbedding
from kai_gpt.modeling.layers import (
    TransformerLayer,
    LMHead
)

class Transformer(nn.Module):
    """A complete Transformer model implementing encoder/decoder architecture.

    This model combines:
    1. Token embeddings with positional encodings
    2. Multiple transformer layers
    3. Language modeling head

    Args:
        config (TransformerConfiguration): Configuration object containing all model parameters.
            Expected attributes:
            - vocab_size: int
            - hidden_size: int
            - pad_token_id: int
            - max_seq_len: int
            - token_dropout_probs: float
            - num_attention_heads: int
            - intermediate_size: int
            - attn_dropout_probs: float
            - ff_dropout_probs: float
            - rms_norm: float
            - num_hidden_layers: int

    Attributes:
        embed_tokens (TokenEmbedding): Token embedding layer
        layers (nn.ModuleList): List of transformer layers
        lm_head (LMHead): Language model head for prediction
        config (TransformerConfiguration): Model configuration

    Example:
        >>> config = TransformerConfiguration(
        ...     vocab_size=32000,
        ...     hidden_size=768,
        ...     pad_token_id=0,
        ...     max_seq_len=512,
        ...     num_attention_heads=12,
        ...     num_hidden_layers=6
        ... )
        >>> model = Transformer(config)
        >>> input_ids = torch.randint(0, 32000, (1, 128))
        >>> outputs = model(input_ids)
    """
    def __init__(self, config: TransformerConfiguration) -> None:
        super().__init__()
        
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            max_seq_len=config.max_seq_len,
            token_dropout_probs=config.token_dropout_probs
        )
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=config.hidden_size, num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size, attn_dropout_probs=config.attn_dropout_probs,
                ff_dropout_probs=config.ff_dropout_probs, rms_eps=config.rms_norm
            )
            for _ in range(config.num_hidden_layers)        
        ])
        
        self.lm_head = LMHead(hidden_size=config.hidden_size, vocab_size=config.vocab_size)
        self.config = config
    
    def _build_4d_causal_mask(self, attention_mask: torch.Tensor, num_heads: int):
        bs, seq_len = attention_mask.size()
    
        # (1, seq_len, seq_len) lower-triangular causal mask
        causal = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device))

        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, seq_len)
        causal_mask = causal.unsqueeze(0).unsqueeze(1) * pad_mask  # (bs, 1, seq_len, seq_len)
        causal_mask = causal_mask * pad_mask.transpose(-1, -2)  # apply to keys axis
        
        causal_mask = (1 - causal_mask).float()
        causal_mask = causal_mask * torch.finfo(torch.float16).min
        
        # 0: attend to, -inf: do not attend
        return causal_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tokens_embeds: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ) -> TransformerOutput:
        
        if (input_ids is None and tokens_embeds is None) or (input_ids is not None and tokens_embeds is not None):
            raise ValueError("You must provide exactly one of `input_ids` or `tokens_embeds`, not both.")

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = tokens_embeds
        
        if attention_mask is None:
            # create default attention_mask
            attention_mask = torch.ones_like(tokens_embeds) # type: ignore
        
        if attention_mask.ndim == 2:
            attention_mask = self._build_4d_causal_mask(attention_mask, self.config.num_attention_heads)
        
        all_hidden_states = (hidden_states,)
        all_attentions = ()
        for layer in self.layers:
            output = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                return_attentions=return_attentions
            )
            
            if return_attentions:
                all_attentions += (output[1],)
            if return_hidden_states:
                all_hidden_states += output[0]
            
            hidden_states = output[0]
            
        logits = self.lm_head(hidden_states)
        return TransformerOutput(
            attentions=all_attentions if return_attentions else None,
            hidden_states=all_hidden_states if return_hidden_states else None,
            last_hidden_states=hidden_states,
            logits=logits,
        )

    @torch.no_grad()
    def generate(
        self,
        max_new_tokens: int,
        temperature: float = 1,
        top_k: int = 250,
        strategy: Literal['greedy', 'random'] = 'greedy'
    ):
        """Generates token sequences using the language model.

        Args:
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float, optional): Controls randomness of predictions by scaling logits.
                Lower = more deterministic, higher = more random. Default: 1.0
            top_k (int, optional): When strategy='random', restricts sampling to top-k most likely tokens. 
                Default: 250
            strategy (Literal['greedy', 'random'], optional): Generation strategy:
                - 'greedy': Always selects the most likely next token
                - 'random': Samples from the probability distribution
                Default: 'greedy'

        Returns:
            torch.Tensor: Generated token sequence of shape (seq_len,)

        Raises:
            ValueError: If unknown generation strategy is provided.

        Notes:
            - Generation stops when EOS or PAD token is generated
            - Uses teacher-forcing (autoregressive generation)
            - Runs in inference mode (no gradients)

        Example:
            >>> model = Transformer(config)
            >>> tokens = model.generate(max_new_tokens=50, temperature=0.7, strategy='random')
            >>> print(tokenizer.decode(tokens))
        """
        generated_tokens = torch.tensor([self.config.bos_token_id]).long().unsqueeze(0)
        attention_mask = torch.ones_like(generated_tokens).long()
        
        for _ in range(max_new_tokens):
            output = self(input_ids=generated_tokens, attention_mask=attention_mask)
            logits = output.logits[:, -1, :]  # (1, vocab_size)
            logits = logits / temperature

            if strategy == 'greedy':
                next_token = torch.argmax(logits, dim=-1)
            elif strategy == 'random':
                logits = torch.topk(logits, k=top_k, dim=-1)
                values, indices = logits.values, logits.indices
                probs = torch.softmax(values, dim=-1)
                next_token = indices.gather(-1, torch.multinomial(probs, num_samples=1))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            next_token = next_token
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            attention_mask = torch.ones_like(generated_tokens)

            if next_token.item() == self.config.eos_token_id or next_token.item() == self.config.pad_token_id:
                break

        return generated_tokens
