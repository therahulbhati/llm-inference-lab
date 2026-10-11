"""Concept 01: model anatomy. Embedding lookup and RMSNorm."""

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from inferlab.layers import embed, rms_norm

HIDDEN = 896
EPS = 1e-6


def _hf_rms_norm(weight: torch.Tensor) -> Qwen2RMSNorm:
    norm = Qwen2RMSNorm(weight.shape[0], eps=EPS).to(weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


def test_embed_shape_and_dtype():
    weight = torch.randn(100, 8)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = embed(ids, weight)
    assert out.shape == (2, 3, 8)
    assert out.dtype == weight.dtype


@pytest.mark.model
def test_embed_matches_hf(hf_model):
    ids = torch.tensor([[9707, 11, 1879, 0], [151643, 0, 42, 7]])
    weight = hf_model.model.embed_tokens.weight
    with torch.no_grad():
        expected = hf_model.model.embed_tokens(ids)
    torch.testing.assert_close(embed(ids, weight), expected)


def test_rms_norm_float32_matches_hf():
    torch.manual_seed(0)
    x = torch.randn(2, 5, HIDDEN)
    weight = torch.randn(HIDDEN)
    with torch.no_grad():
        expected = _hf_rms_norm(weight)(x)
    torch.testing.assert_close(rms_norm(x, weight, EPS), expected)


def test_rms_norm_output_is_normalized():
    torch.manual_seed(0)
    x = torch.randn(3, HIDDEN) * 50
    out = rms_norm(x, torch.ones(HIDDEN), EPS)
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(3), atol=1e-4, rtol=0)


def test_rms_norm_float16_large_activations():
    """Activations this large are real: some residual stream values reach hundreds."""
    torch.manual_seed(0)
    x = (torch.randn(2, 5, HIDDEN) * 300).half()
    weight = torch.randn(HIDDEN).half()
    with torch.no_grad():
        expected = _hf_rms_norm(weight)(x)
    out = rms_norm(x, weight, EPS)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0, "output is all zeros: what is 300**2 in float16?"
    torch.testing.assert_close(out, expected)


@pytest.mark.model
def test_rms_norm_with_real_weights(hf_model):
    torch.manual_seed(0)
    layer = hf_model.model.layers[0].input_layernorm
    x = torch.randn(1, 4, HIDDEN)
    with torch.no_grad():
        expected = layer(x)
    torch.testing.assert_close(rms_norm(x, layer.weight, layer.variance_epsilon), expected)
