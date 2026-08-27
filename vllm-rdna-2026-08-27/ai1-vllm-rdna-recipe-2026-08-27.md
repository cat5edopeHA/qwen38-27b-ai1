# AI1 vLLM RDNA lane — install & serve recipe (2026-08-27)

Verified on ai1: Fedora 44, ROCm 7.14.0, 2× Radeon AI PRO R9700 (gfx1201, 32 GiB), Python 3.14.7. Official AMD RDNA wheels only (per https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html, ROCm 7.14 / Radeon / pip section).

## 1. Install

```bash
# Host prereq: Triton's hip_utils compile needs Python headers
sudo dnf install -y python3-devel

python3.14 -m venv ~/ai-runtime/experiments/vllm-rdna
source ~/ai-runtime/experiments/vllm-rdna/bin/activate
python -m pip install --upgrade pip
python -m pip install uv

uv pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1201]==2.11.0+rocm7.14.0" \
  "torchvision[device-gfx1201]==0.26.0+rocm7.14.0" \
  "torchaudio==2.11.0+rocm7.14.0"

uv pip install https://rocm.frameworks.amd.com/whl-multi-arch/vllm-rdna/flash-attn/flash_attn-2.8.3-py3-none-any.whl
uv pip install "https://rocm.frameworks.amd.com/whl-multi-arch/vllm-rdna/vllm/vllm-0.23.1.dev1%2Brocm7.14.0.g9ddef7117.d20260715-cp314-cp314-linux_x86_64.whl"

# Required env (every shell that runs vllm):
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

Verify: `python -c "import vllm, torch, flash_attn; print(vllm.__version__, torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"`

Note: the RDNA wheel index only carries vLLM 0.23.1.dev1 for ROCm 7.14 (checked 2026-08-27). AMD documents a long-warmup known issue for vLLM 0.21-0.25 on Radeon; fixed in ≥0.26.

## 2. Model

```bash
hf download TelperionAI/Qwen3.8-27B-INT4-AWQ-GPTQ --local-dir /srv/ai/models/qwen38-27b-int4-awq
# 25.1 GB; public repo (works anonymous). W4A16, qwen3_5 GDN hybrid, vision tower + BF16 MTP head included.
```

## 3. Serve (pin to one GPU)

```bash
export HIP_VISIBLE_DEVICES=1   # GPU1 = second R9700; GPU0 = production llama-swap lane

# 4K bench / short-ctx lane:
vllm serve /srv/ai/models/qwen38-27b-int4-awq \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen38-27b-int4-awq

# 40K lane (fp8 KV required — default BF16 KV OOMs at 40K):
vllm serve /srv/ai/models/qwen38-27b-int4-awq \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 40960 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
  --served-model-name qwen38-27b-int4-awq
```

Health: `curl -sf http://127.0.0.1:8002/health` (GET; vLLM returns 405 to HEAD). First load ~4 min (Triton compile, cached after).

## 4. Load-failure fixes (all hit in the field)

1. **`Python.h: No such file or directory`** during model inspection — Triton's hip_utils.c compile fails → "Model architectures ['Qwen3_5ForConditionalGeneration'] failed to be inspected". Fix: `sudo dnf install python3-devel`.
2. **`max_num_seqs (256) exceeds available Mamba cache blocks (42)`** — GDN hybrid's recurrent-state cache caps concurrent sequences; CUDA-graph capture refuses. Fix: `--max-num-seqs 32`.
3. **40K OOMs KV** — needs 2.68 GiB vs 2.05 available (est. max len 30576). Fix: `--kv-cache-dtype fp8` (57,344-token KV cache, 1.4× concurrency at 40K; also apples-to-apples with llama.cpp q8 KV).
4. **Decode slowness (8.3-8.5 t/s)** — log: "Cannot use ROCm custom paged attention kernel, falling back to Triton implementation". Known wheel limitation; no flag fix found within the official stack.

## 5. Benchmark

`vllm_bench.py` (companion, stdlib-only) — standard ai1 harness shape:

```bash
python3 vllm_bench.py http://127.0.0.1:8002 qwen38-27b-int4-awq 3 128
```

## 6. Teardown

```bash
pgrep -af '[v]llm serve' | grep -v 'bash -c' | awk '{print $1}' | xargs -r kill -TERM
# verify: rocm-smi --showmeminfo vram → GPU1 back to ~73 MB idle; GPU0 untouched
```
