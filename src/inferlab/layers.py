"""Building blocks of the Qwen2 decoder, written as plain functions over tensors."""

import torch


def embed(token_ids: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
    """Map token ids to vectors.

    token_ids: (batch, seq) int64
    embedding_weight: (vocab_size, hidden_size)
    returns: (batch, seq, hidden_size), same dtype as embedding_weight

    Do not use torch.nn.Embedding or torch.nn.functional.embedding.
    """
    raise NotImplementedError


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root mean square normalization over the last dimension, then scale by `weight`.

    x: (..., hidden_size), any float dtype
    weight: (hidden_size,)
    returns: same shape as x, same dtype as x

    Must match transformers' Qwen2RMSNorm, including how it handles low-precision
    inputs such as float16.
    """
    raise NotImplementedError
