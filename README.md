# InferLab

A minimal LLM inference engine built from scratch, checked token-for-token against Hugging Face Transformers, with every optimization measured.

Model: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B). Baseline hardware: Apple M1, 8 GB (CPU and MPS).

> Work in progress. No benchmark results yet.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1 | Own forward pass and greedy decode, token-exact vs HF | In progress |
| M2 | Streaming server and load generator (TTFT, TPOT, P50/P95, goodput) | Planned |
| M3 | KV cache and continuous batching | Planned |
| M4 | INT8/INT4 weight-only quantization with perplexity checks | Planned |
| M5 | GPU profiling, Triton kernel, vLLM comparison | Planned |
| M6 | Docker, metrics, CI | Planned |

## Setup

```bash
uv sync
uv run hf download Qwen/Qwen2.5-0.5B
uv run pytest
```
