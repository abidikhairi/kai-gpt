from typing import Iterable, Optional
from pydantic import BaseModel, ConfigDict
import torch


class TransformerOutput(BaseModel):
    last_hidden_states: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Iterable[torch.Tensor]] = None
    attentions: Optional[Iterable[torch.Tensor]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

