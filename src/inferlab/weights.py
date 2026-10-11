"""Loading Qwen2.5 config and weights from a Hugging Face snapshot directory."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool


def load_config(model_dir: Path) -> ModelConfig:
    """Read `config.json` from `model_dir` into a ModelConfig.

    Hint: not every field above appears in config.json under that exact name.
    Some must be derived from other fields.
    """
    raise NotImplementedError


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from the `*.safetensors` files in `model_dir`.

    Return a flat dict mapping the tensor name exactly as stored in the file
    (e.g. "model.layers.0.self_attn.q_proj.weight") to the tensor, in the dtype
    it was saved in. Do not use transformers for this.
    """
    raise NotImplementedError


def count_parameters(state_dict: dict[str, torch.Tensor], config: ModelConfig) -> int:
    """Total number of unique parameters in the model.

    Must equal what Hugging Face reports for the same model. Think about which
    weights are shared, and whether they are stored once or twice on disk.
    """
    raise NotImplementedError
