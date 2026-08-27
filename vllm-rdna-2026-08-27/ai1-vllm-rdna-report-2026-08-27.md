# AI1 vLLM RDNA lane — benchmark report (2026-08-27)

**Host:** ai1 (192.168.10.12) — Fedora 44, ROCm 7.14.0, 2× Radeon AI PRO R9700 (gfx1201, 32 GiB each), Core Ultra 7 265K, 93 GiB RAM, Python 3.14.7.
**Stack:** official AMD RDNA wheels (torch 2.11.0+rocm7.14.0 device-gfx1201, flash-attn 2.8.3 RDNA, **vLLM 0.23.1.dev1+rocm7.14.0.g9ddef7117.d20260715**) — no community fork.
**Model:** `TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ` — W4A16 INT4 (AWQ scaling + GPTQ, compressed-tensors `pack-quantized`), 25.1 GB, qwen3_5 GDN hybrid arch, BF16 vision tower + BF16 MTP head included.
**Placement:** single GPU (HIP_VISIBLE_DEVICES=1, GPU1), tensor_parallel_size=1. GPU0 carried the production llama-swap Vulkan lane throughout — untouched (27.4 GiB steady before/during/after).

## Results (standard harness)

Harness: ~200-tok prompt / 128 out / temp 0 / thinking off (`chat_template_kwargs {"enable_thinking": false}`), nonce'd prefill (no prefix-cache hits), 3 reps, median. Script: `vllm_bench.py` (stdlib-only).

| Context | PP (tok/s) | TG (tok/s) | TTFT (ms) |
|---|---:|---:|---:|
| 4K (BF16 KV) | 968 | 8.5 | 156 |
| 40K (fp8 KV) | 967 | 8.3 | 156 |

VRAM: 28.6 GiB used on GPU1 at 0.90 gpu-memory-utilization (23.4 GiB weights + KV + CUDA graphs; ~3.2 GiB headroom).

## Comparison vs llama.cpp single-card Q4 lane (same model family)

| Metric | vLLM W4A16 | llama.cpp Q4 (targets) |
|---|---:|---:|
| Prefill | 967-968 | ~700-1350 (Q4 4K bench 1346; Q8 @40K 682; Q4 MTP5 655) |
| Decode | 8.3-8.5 | no-spec ~30 · MTP5 36.5 · ROCmFPX-Q4 MTP5 42.1 |
| TTFT | 156 ms | not tracked |

**Verdict:** prefill is competitive; decode is a **3.5-5× regression**. Root cause (log-confirmed): `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation` — the official RDNA wheel's decode runs on Triton-fallback kernels for this model/quant on gfx1201, plus the compressed-tensors W4A16 dequant path. Same disease class seen on ai2 CUDA (W4A16 slowest vLLM GDN path) and mattbucci's R9700 SGLang note (native AWQ 21.6 vs compressed-tensors AWQ 3.6 t/s).

**Status: experimental lane, not production.** llama.cpp/llama-swap stays the ai1 single-stream lane. vLLM's untested upside on this box: batching/concurrency (max-num-seqs 32 lane) and MTP speculative decoding (head ships in-repo, BF16).

## Untested follow-ups (official stack only)

1. INT8 W8A16 A/B (`lued/Qwen3.8-27B-INT8-W8A16-MTP`, 29.4 GB) — INT8 beat W4A16 ~4.7× on ai2 CUDA GDN; 27.4 GiB fits one card at ≤4K.
2. MTP speculative: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (BF16 head already in the repo).
3. TP=2 — known R9700 hang risk (vllm-project/vllm#40980, PR #46190); validate as timed experiment only.

Full recipe + load-failure fixes: `ai1-vllm-rdna-recipe-2026-08-27.md` (companion doc).
