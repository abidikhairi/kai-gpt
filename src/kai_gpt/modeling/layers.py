import torch
from torch import nn


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
