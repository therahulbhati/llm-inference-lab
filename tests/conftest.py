from pathlib import Path

import pytest
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def model_dir() -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(MODEL_ID, local_files_only=True))
    except Exception:
        pytest.skip(f"{MODEL_ID} not downloaded. Run: uv run hf download {MODEL_ID}")


@pytest.fixture(scope="session")
def hf_model(model_dir):
    """Reference model from transformers, in float32 on CPU."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    return model.eval()
