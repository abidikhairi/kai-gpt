import math
from typing import Optional, Tuple, Union
import torch
from torch import nn

from kai_gpt.modeling.embedding import RotaryEmbedding


class NativeAttention(nn.Module):
    """Implements scaled dot-product attention with rotary position embeddings.
    
    This attention mechanism performs multi-head attention with rotary position embeddings
    (RoPE) applied to queries and keys before computing attention scores. The implementation
    includes optional attention masking and dropout.

    Args:
        hidden_size (int): The dimension of the input and output features.
        num_attention_heads (int): Number of attention heads.
        attn_dropout_probs (float, optional): Dropout probability for attention scores. Default: 0.1.

    Raises:
        AssertionError: If hidden_size is not divisible by num_attention_heads.

    Attributes:
        head_size (int): Dimension of each attention head.
        w_q (nn.Linear): Linear projection for queries.
        w_k (nn.Linear): Linear projection for keys.
        w_v (nn.Linear): Linear projection for values.
        output (nn.Linear): Final linear projection layer.
        rotary_embedding (RotaryEmbedding): Rotary position embedding module.

    Examples:
        >>> attention = NativeAttention(hidden_size=512, num_attention_heads=8)
        >>> hidden_states = torch.randn(2, 16, 512)  # (batch, seq_len, hidden_size)
        >>> output = attention(hidden_states)
        >>> output_with_attn = attention(hidden_states, return_attentions=True)
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attn_dropout_probs: float = 0.1
    ) -> None:
        super().__init__()
        
        assert hidden_size // num_attention_heads != 0, "hidden_size must be divisible by num_attention_heads"
        
        self.head_size = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.attn_dropout_probs = attn_dropout_probs
        
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, False)
        
        self.output = nn.Linear(self.hidden_size, self.hidden_size, False)
        
        self.rotary_embedding = RotaryEmbedding(hidden_size=self.hidden_size)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False 
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bs, seq_len, _ = hidden_states.shape

        # [bs, seq_len, h] -> [bs, seq_len, num_heads, head_size]        
        value_states = self.w_v(hidden_states).view(bs, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # [bs, seq_len, h] -> [bs, seq_len, h]        
        query_states = self.w_q(hidden_states)
        key_states = self.w_k(hidden_states)
        
        # Apply rotary position embeddings
        query_states, key_states = self.rotary_embedding(query_states, key_states)
        
        # [bs, seq_len, h] -> [bs, seq_len, num_heads, head_size]        
        key_states = key_states.view(bs, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        query_states = query_states.view(bs, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # scaled dot product attention
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Model forward() function)
            scores = scores + attention_mask # NOTE: do not `.unsqueeze(1)` broadcast as it can break batching
        
        # Normalize and calculate context vectors
        scores = torch.softmax(scores, dim=-1)
        context = torch.matmul(scores, value_states)

        # Reshape to desired output shape
        context = context.transpose(1, 2).contiguous()
        context = context.view(bs, seq_len, -1)

        # Projection to output space
        context = self.output(context)
        
        result = (context,)
        if return_attentions:
            result += (scores, )

        return result
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, ) -> None:
        super().__init__()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor], 
    ):
        pass
    
