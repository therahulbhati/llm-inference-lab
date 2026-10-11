"""Concept 01: model anatomy. Config, weight files and parameter count."""

import pytest
import torch

from inferlab.weights import count_parameters, load_config, load_state_dict

pytestmark = pytest.mark.model


def test_config_matches_hf(model_dir, hf_model):
    config = load_config(model_dir)
    ref = hf_model.config
    assert config.vocab_size == ref.vocab_size
    assert config.hidden_size == ref.hidden_size
    assert config.intermediate_size == ref.intermediate_size
    assert config.num_hidden_layers == ref.num_hidden_layers
    assert config.num_attention_heads == ref.num_attention_heads
    assert config.num_key_value_heads == ref.num_key_value_heads
    assert config.rms_norm_eps == ref.rms_norm_eps
    assert config.max_position_embeddings == ref.max_position_embeddings
    assert config.tie_word_embeddings is True


def test_config_derived_fields(model_dir):
    config = load_config(model_dir)
    assert config.head_dim == 64
    assert config.rope_theta == 1_000_000.0


def test_state_dict_names_and_shapes(model_dir, hf_model):
    state_dict = load_state_dict(model_dir)
    ref = hf_model.state_dict()
    for name, tensor in state_dict.items():
        assert name in ref, f"unexpected tensor {name}"
        assert tensor.shape == ref[name].shape, name
    # One reference tensor is not in the file. Your note should explain why.
    missing = set(ref) - set(state_dict)
    assert missing == {"lm_head.weight"}


def test_state_dict_keeps_stored_dtype(model_dir):
    state_dict = load_state_dict(model_dir)
    dtypes = {t.dtype for t in state_dict.values()}
    assert dtypes == {torch.bfloat16}


def test_state_dict_values_match_hf(model_dir, hf_model):
    state_dict = load_state_dict(model_dir)
    ref = hf_model.state_dict()
    for name in ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.bias", "model.norm.weight"]:
        torch.testing.assert_close(state_dict[name].float(), ref[name])


def test_parameter_count_matches_hf(model_dir, hf_model):
    config = load_config(model_dir)
    state_dict = load_state_dict(model_dir)
    expected = sum(p.numel() for p in hf_model.parameters())
    assert count_parameters(state_dict, config) == expected
