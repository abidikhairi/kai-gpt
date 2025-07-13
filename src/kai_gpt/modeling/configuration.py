from pydantic import BaseModel


class TransformerConfiguration(BaseModel):
    vocab_size: int = 32768
    num_hidden_layers: int = 12
    hidden_size: int = 16
    intermediate_size: int = 32
    num_attention_heads: int = 4
    attn_dropout_probs: float = 0.1
    ff_dropout_probs: float = 0.1
    token_dropout_probs: float = 0.0 # by default no dropout
    max_seq_len: int = 1024
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    rms_norm: float = 8e-14
