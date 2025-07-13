import math
import torch
from torch import nn

class Embedding(nn.Embedding):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.
    
    This module is a subclass of `torch.nn.Embedding` and maintains all its functionality.
    It is used to store word embeddings and retrieve them using indices. The input to the module
    is a list of indices, and the output is the corresponding word embeddings.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int, optional): If specified, the entries at `padding_idx` do not contribute to
            the gradient; therefore, the embedding vector at `padding_idx` is not updated during
            training, i.e. it remains as a fixed "pad". Default: None.
        max_norm (float, optional): If given, each embedding vector with norm larger than `max_norm`
            is renormalized to have norm `max_norm`. Default: None.
        norm_type (float, optional): The p of the p-norm to compute for the `max_norm` option. Default: 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of
            frequency of the words in the mini-batch. Default: False.
        sparse (bool, optional): If True, gradient w.r.t. `weight` matrix will be a sparse tensor.
            Default: False.
        _weight (torch.Tensor, optional): Pre-defined embedding matrix. Default: None.
        _freeze (bool, optional): If True, the tensor does not get updated in the learning process.
            Default: False.
        device (torch.device, optional): The desired device of the returned tensor. Default: None.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default: None.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
            initialized from :math:`\mathcal{N}(0, 1)`. # type: ignore

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract.
        - Output: :math:`(*, H)`, where `*` is the input shape and `H` is `embedding_dim`.

    Examples::
        >>> embedding = Embedding(10, 3)
        >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],
                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, max_norm: float | None = None, norm_type: float = 2, scale_grad_by_freq: bool = False, sparse: bool = False, _weight: torch.Tensor | None = None, _freeze: bool = False, device=None, dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)     # even
        pe[:, 1::2] = torch.cos(position * div_term)     # odd
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, hidden_size)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len] # type: ignore
    
class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        max_seq_len: int,
        token_dropout_probs: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(hidden_size=hidden_size, max_len=max_seq_len)
        self.token_dropout_probs = token_dropout_probs
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.dropout(
            self.embedding(input_ids),
            p=self.token_dropout_probs,
            train=self.training
        )
        
        return self.positional_encoding(hidden_states)

        
def rotate_half(x):
    """Rotates half the hidden dims of the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor to be rotated. Last dimension must be even-sized.
        
    Returns:
        torch.Tensor: Tensor with the second half of its channels rotated by 90 degrees.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Applies rotary position embeddings to the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor to apply position embeddings to.
        cos (torch.Tensor): Cosine component of the rotary embedding.
        sin (torch.Tensor): Sine component of the rotary embedding.
        
    Returns:
        torch.Tensor: Tensor with rotary position embeddings applied.
    """
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(torch.nn.Module):
    """Implements rotary position embeddings (RoPE) for transformer models.
    
    Rotary position embeddings provide relative position information to transformer models by
    rotating query and key vectors based on their positions. This implementation is based on
    the RoFormer architecture.
    
    Args:
        dim (int): The dimensionality of the embeddings (must be even).
        
    Attributes:
        inv_freq (torch.Tensor): Inverse frequency buffer for position encoding.
        _seq_len_cached (int): Cached sequence length for cos/sin tables.
        _cos_cached (torch.Tensor): Cached cosine table.
        _sin_cached (torch.Tensor): Cached sine table.
        
    Example:
        >>> rope = RotaryEmbedding(dim=64)
        >>> q = torch.randn(1, 8, 32, 64)  # (batch, heads, seq_len, dim)
        >>> k = torch.randn(1, 8, 32, 64)
        >>> q_embed, k_embed = rope(q, k)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.int64).float() / hidden_size))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached).to(dtype=q.dtype),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached).to(dtype=k.dtype),
        )
    