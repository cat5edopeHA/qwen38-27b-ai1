# vLLM Radiance fork lane on ai1 — 2026-08-27

Community-fork test on ai1 (2× Radeon AI PRO R9700, gfx1201, ROCm 7.14 host, Fedora 44). User directive: no fork until official-stack numbers existed; FP8 skipped (8-bit outlier, doesn't fit one card).

## Stack (pinned)

- Image: `docker.io/stilldeadcode/vllm-radiance:0.9.3` (digest `bebc00145d7d`, 9.5 GB on disk) — vLLM 0.27.1 + PyTorch 2.11 + Triton 3.6 + AITER 0.1.17, ROCm 7.14 **bundled in-image**, Ubuntu 24.04, Python 3.12. Source: `magiccodingman/vllm-radiance` (Codeberg fork of StillDeadcode/vllm-radiance); libr4d hand-written gfx1201 kernels (attention/GDN/vision/AR) + Radiance FP8 GEMMs + dynamic MTP draft.
- Tested on TP=2 + FP8 by upstream; **single-GPU + non-FP8 untested** — this run is the untested-corner probe.

## Launch recipe (rootful podman — required)

Three environmental traps, all hit in the field:

1. **Rootless podman fails**: HSA `Memory critical error ... Reason: Memory in use` at first GPU alloc (also in the startup rocm-bandwidth-test). Root cause: memlock `ulimit -l` = 8 MiB (soft AND hard) for user mike — HSA/NCCL needs large pinned host buffers. Rootless cannot raise above the hard limit. **Fix: run rootful (`sudo -n podman`) with `--ulimit memlock=-1:-1`** (mike has passwordless sudo on ai1).
2. **SELinux (Fedora Enforcing)**: bind mounts need `:z` (`-v /srv/ai/models:/models:ro,z`, `-v .../radiance-cache:/cache:z`) or the container gets EACCES despite 777.
3. **Short-name resolution**: rootless `podman pull` over non-TTY errors — use `docker.io/...` fully-qualified.

Serving flags (Qwen3.8-27B INT4, single card, GPU1):

```bash
sudo -n podman run -d --name vllm-radiance \
  --ulimit memlock=-1:-1 --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --shm-size 4g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /srv/ai/models:/models:ro,z -v /srv/ai/wip/radiance-cache:/cache:z \
  -p 8002:8000 \
  -e HIP_VISIBLE_DEVICES=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  docker.io/stilldeadcode/vllm-radiance:0.9.3 \
  /models/qwen38-27b-int4-awq \
  --served-model-name qwen38-27b-int4-awq --language-model-only \
  --max-model-len 4096 --max-num-seqs 32 --gpu-memory-utilization 0.95 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --enable-prefix-caching --mamba-cache-mode align --no-async-scheduling \
  [--speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"ROCM_AITER_UNIFIED_ATTN","disable_padded_drafter_batch":true}'] \
  --host 0.0.0.0 --port 8000
```

Health: GET `http://127.0.0.1:8002/health` (host port 8002 → container 8000). First load 75-135 s (AITER JIT, cached in /cache).

## Load-failure ladder (all hit)

1. Rootless + memlock 8 MiB → HSA "Memory in use" (see above) → **rootful + `--ulimit memlock=-1:-1`**.
2. Cache volume EACCES → **`:z`** on mounts.
3. `No available memory for the cache blocks` at 0.90 with vision tower → **`--language-model-only`** (text bench; frees ~0.8-3 GiB) + 0.95.
4. MTP depth 8 OOM → **depth 4 + 0.97 + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** (graph estimator reserves worst-case draft+verify graphs at all capture sizes; off recovers the headroom, at mild runtime-OOM risk).
5. FP8 checkpoint (Qwen/Qwen3.8-27B-FP8, 30.9 GB) does NOT fit one card: weights ~28 GiB + workspace > 0.97 budget → "No available memory" even no-spec at max-num-seqs 8. **Skipped per user directive** (8-bit outlier, not q4-class). Its FP8-quantized `mtp.safetensors` (0.48 GB) also fails the drafter loader (image expects BF16-style head).
6. KV-ceiling with MTP4: 9,438 tokens (draft buffers eat the card) → 40K needs the no-MTP lane (46,080-token KV at fp8).

## Measured (standard harness: ~200-tok prompt / 128 out / temp 0 / thinking off, 3 reps median; `~/ai-runtime/experiments/vllm_bench.py`)

| Config | ctx | PP | TG | TTFT | KV cache |
|---|---|---|---:|---:|---:|
| Official 0.23.1 INT4 (reference) | 4K | 968 | 8.5 | 156 ms | — |
| Official 0.23.1 INT4 | 40K fp8 | 967 | 8.3 | 156 ms | 57,344 |
| **Radiance INT4 no-spec** | 4K | 1255 | 15.9 | 175 ms | — |
| **Radiance INT4 no-spec** | 40K fp8 | 1252 | 15.8 | 176 ms | 46,080 |
| **Radiance INT4 MTP4** | 4K | 1194 | 24.7 | 230 ms | 9,438 |

SpecDecoding (MTP4): mean acceptance length 2.7-2.8, per-position 0.75/0.49/0.36/0.21, avg acceptance 47-50%.

vs llama.cpp Q4 single-card targets (PP ~1350 · TG 30 no-spec / 36.5 MTP5 / 42.1 ROCmFPX-MTP5): radiance INT4+MTP = **PP -11%, TG -32%**; still the closest any vLLM stack has come on this card. Official wheel was TG -80% vs llama.cpp; radiance MTP closes to -32%.

## Verdict

- Radiance's RDNA4 kernels + AITER attention work on **single GPU + non-FP8** (untested territory) — decode 8.5 → 15.9 t/s (+87%) vs the official wheel, PP +30%; MTP4 adds another +55% (24.7 t/s).
- Dedicated `rdna_hybrid_w4a16` kernels exist in-image (INT4 is a first-class path, not a fallback).
- llama.cpp MTP5 (30-42 t/s) still wins single-stream decode; radiance wins the "vLLM on ai1" crown and is the candidate if batched serving ever matters. TP=2 (their tested config) is a natural follow-up but needs both GPUs (prod owns GPU0) and carries the known R9700 TP hang risk.
- Evidence: ai1 `/srv/ai/wip/vllm-serve-*.log`, container logs via `sudo podman logs`, bench at `~/ai-runtime/experiments/vllm_bench.py`; local `/home/hermes/ai1-vllm-bench/` (report/recipe/run scripts).
