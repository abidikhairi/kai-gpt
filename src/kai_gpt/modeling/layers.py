from typing import Optional, Tuple, Union
import torch
from torch import nn

from kai_gpt.modeling.attention import NativeAttention


class GatedFeedForward(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ff_dropout_probs: float = 0.1
    ) -> None:
        super().__init__()
        
        self.dropout_probs = ff_dropout_probs
        self.linear1 = nn.Linear(hidden_size, intermediate_size, False)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(intermediate_size, hidden_size, False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.linear1(hidden_states))
        hidden_states = torch.dropout(hidden_states, self.dropout_probs, train=self.training)
        
        return self.linear2(hidden_states)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attn_dropout_probs: float = 0.1,
        ff_dropout_probs: float = 0.1,
        rms_eps: float = 6e-10,
    ) -> None:
        super().__init__()
        
        self.pre_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        
        self.attention = NativeAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attn_dropout_probs=attn_dropout_probs
        )
        
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        
        self.feedforward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ff_dropout_probs=ff_dropout_probs
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        hidden_states = self.pre_attention_layernorm(hidden_states)        
        output = self.attention(hidden_states, attention_mask, return_attentions)
        hidden_states = output[0] + hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feedforward(hidden_states) + hidden_states
        
        result = (hidden_states,)
        
        if return_attentions:
            result += (output[1],)
            
        return result
